from pathlib import Path
import argparse, json, csv, yaml, hashlib
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import binary_erosion
import SimpleITK as sitk
from model.single_edge_image_unet import SingleEdgeImageUNet
from loss import SSIM_Loss
from metrics import exact_metrics
from protocol import foreground_mask, postprocess_score, scoring_profile, validate_calibration
from case_lists import read_names

def read_volume(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--config', default='configs/testing.yaml')
    a.add_argument('--checkpoint')
    a.add_argument('--output', required=True)
    a.add_argument('--device', default='cuda:0')
    a.add_argument('--threshold-file')
    a.add_argument('--calibrate', action='store_true')
    a.add_argument('--save-reconstructions', action='store_true')
    a.add_argument('--describe', action='store_true')
    args = a.parse_args()
    c = yaml.safe_load(open(args.config))
    fixed_settings = {'SSIM_window': 11, 'SSIM_sigma': 1.5, 'SSIM_C1': 0.01 ** 2, 'SSIM_C2': 0.03 ** 2, 'SSIM_data_range': 1, 'SSIM_padding': 'zero_5', 'normalization': 'clip_0_1', 'score': 'L1_times_3D_SSIM_loss'}
    if any((c.get(k) != v for k, v in fixed_settings.items())):
        raise ValueError('Unsupported score configuration; this runner implements the recorded fixed SSIM and mask settings')
    foreground_mask(np.ones((3, 3, 3)), c['mask_protocol'])
    if c['postprocessing'] not in ('raw', 'min3_mean9'):
        raise ValueError('Unsupported postprocessing')
    if args.describe:
        print(json.dumps({'input_grid': c['input_grid_zyx'], 'output_grid': c['output_grid_zyx'], 'postprocessing': c['postprocessing'], 'training': False}))
        return
    if sum([bool(args.threshold_file), args.calibrate]) != 1:
        a.error('Choose validation calibration or an existing validation threshold file')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new, empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(c['seed'])
    from model.factory import create_model
    model = create_model(c)
    checkpoint_path = Path(args.checkpoint or c['checkpoint'])
    hasher = hashlib.sha256()
    with checkpoint_path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            hasher.update(block)
    checkpoint_sha = hasher.hexdigest()
    profile = scoring_profile(c)
    if args.threshold_file:
        calibration = json.loads(Path(args.threshold_file).read_text())
        validate_calibration(calibration, profile, checkpoint_sha, [])
    checkpoint = torch.load(args.checkpoint or c['checkpoint'], map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint.get('state_dict', checkpoint))
    model.to(args.device).eval()
    ssim = SSIM_Loss(11, False, True).to(args.device)
    smooth = torch.nn.Sequential(torch.nn.MaxPool3d(3, 1, 1), torch.nn.AvgPool3d(9, 1, 4))
    names = read_names(c['case_list'], c['image_dir'], c['label_dir'])
    assert len(names) == len(set(names)) and names
    if args.calibrate and (c.get('dataset_role') != 'validation' or len(names) != 64):
        raise ValueError('Calibration requires a 64-scan validation list')
    if args.threshold_file and c.get('dataset_role') != 'testing':
        raise ValueError('Fixed-threshold evaluation requires dataset_role=testing')
    if args.threshold_file:
        validate_calibration(calibration, profile, checkpoint_sha, names)
    records = []
    zero_negative = zero_positive = 0
    fg_total = 0
    with torch.no_grad():
        for index, name in enumerate(names):
            assert Path(name).name == name, 'case lists must contain basenames only'
            x = read_volume(Path(c['image_dir']) / name)
            y = read_volume(Path(c['label_dir']) / name) > 0
            assert x.shape == y.shape and np.isfinite(x).all()
            x = torch.as_tensor(np.clip(x, 0, 1), dtype=torch.float32, device=args.device)[None, None]
            x = F.interpolate(x, c['input_grid_zyx'], mode='trilinear', align_corners=False)
            rec = model(x)
            x = F.interpolate(x, c['output_grid_zyx'], mode='trilinear', align_corners=False)
            rec = F.interpolate(rec, c['output_grid_zyx'], mode='trilinear', align_corners=False)
            if args.save_reconstructions:
                folder = out / 'reconstructions'
                folder.mkdir(exist_ok=True)
                sitk.WriteImage(sitk.GetImageFromArray(rec[0, 0].cpu().numpy()), str(folder / name))
            lab = F.interpolate(torch.as_tensor(y, dtype=torch.float32, device=args.device)[None, None], c['output_grid_zyx'], mode='nearest')[0, 0].cpu().numpy() > 0
            ss = ssim(rec, x)
            score = (x - rec).abs() * ss
            score = postprocess_score(score, c['postprocessing'])
            mask = foreground_mask(x[0, 0].cpu().numpy(), c['mask_protocol'])
            s = score[0, 0].cpu().numpy()[mask].astype(np.float32)
            label = lab[mask].astype(np.uint8)
            assert np.isfinite(s).all() and (s >= 0).all() and (len(s) > 0)
            np.save(out / ('%04d_scores.npy' % index), s)
            np.save(out / ('%04d_labels.npy' % index), label)
            bp = int(lab[~mask].sum())
            bg = int((~mask).sum())
            zero_positive += bp
            zero_negative += bg - bp
            records.append({'case': name, 'input_sha256': file_sha(Path(c['image_dir']) / name), 'GT_sha256': file_sha(Path(c['label_dir']) / name), 'output_grid_zyx': c['output_grid_zyx'], 'n': len(s), 'positives': int(lab.sum()), 'background_positive': bp, 'background': bg})
            fg_total += len(s)
    (out / 'private_case_metrics.json').write_text(json.dumps(records, indent=2))
    scores = np.lib.format.open_memmap(out / 'pooled_scores.npy', mode='w+', dtype=np.float32, shape=(fg_total,))
    labels = np.lib.format.open_memmap(out / 'pooled_labels.npy', mode='w+', dtype=np.uint8, shape=(fg_total,))
    offset = 0
    for i, r in enumerate(records):
        end = offset + r['n']
        scores[offset:end] = np.load(out / ('%04d_scores.npy' % i), allow_pickle=False)
        labels[offset:end] = np.load(out / ('%04d_labels.npy' % i), allow_pickle=False)
        offset = end
    scores.flush()
    labels.flush()
    metric = exact_metrics(scores, labels, zero_negative, zero_positive)
    if args.calibrate:
        result = {'threshold_raw': metric['Threshold'], 'selection': 'pooled F1 on this development list only', 'scans': len(names), 'config': c, 'scoring_profile': profile, 'checkpoint_sha256': checkpoint_sha, 'development_case_basename_sha256': [hashlib.sha256(name.encode()).hexdigest() for name in names]}
        (out / 'development_threshold.json').write_text(json.dumps(result, indent=2))
        return
    theta = calibration['threshold_raw']
    dice = []
    for i, r in enumerate(records):
        s = np.load(out / ('%04d_scores.npy' % i), allow_pickle=False)
        y = np.load(out / ('%04d_labels.npy' % i), allow_pickle=False)
        prediction = s >= theta
        tp = int(y[prediction].sum())
        pred = int(prediction.sum())
        if theta <= 0:
            tp += r['background_positive']
            pred += r['background']
        r['predicted_positive_voxels'] = pred
        denominator = pred + r['positives']
        r['empty_both'] = denominator == 0
        dice.append(2 * tp / denominator if denominator else 1.0)
    (out / 'metrics.json').write_text(json.dumps({'AP': metric['AP'], 'DSC_scan_mean': float(np.mean(dice)), 'threshold': theta, 'scans': len(records), 'protocol': 'validation fixed threshold', 'config': c, 'empty_GT_scans': sum((r['positives'] == 0 for r in records)), 'empty_prediction_scans': sum((r['predicted_positive_voxels'] == 0 for r in records)), 'empty_both_scans': sum((r['empty_both'] for r in records)), 'empty_both_DSC': 1.0}, indent=2))
    with (out / 'private_per_scan_DSC.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['case', 'DSC'])
        w.writerows(zip(names, dice))
if __name__ == '__main__':
    main()
