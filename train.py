import argparse
import glob
import json
import logging
import os
from pathlib import Path
import sys
from datetime import datetime
import yaml
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / 'configs/train_main.yaml'

def load_settings(config_path=DEFAULT_CONFIG):
    with Path(config_path).open() as handle:
        config = yaml.safe_load(handle)
    if config['model'] != 'SingleEdgeImageUNet' or config['adain']:
        raise ValueError('this entrypoint requires SingleEdgeImageUNet with adain=false')
    if config['input_noise'] not in (['local_dae'], ['none']):
        raise ValueError('supported inputs: local_dae or none')
    if config['local_dae_shape_mode'] != 'ellipsoid' or config['local_dae_coverage_mode'] != 'current':
        raise ValueError('local DAE supports ellipsoid/current only')
    if not 0 < config['local_dae_target_coverage'] <= 1:
        raise ValueError('local DAE coverage must be in (0,1]')
    if config['in_channels'] != 1 or config['out_channels'] != 1:
        raise ValueError('the local DAE experiment requires single-channel images')
    factor = 2 ** (len(config['filters']) - 1)
    if any((size % factor for size in config['input_size'])):
        raise ValueError('input size must be divisible by the encoder downsampling factor')
    if config.get('local_dae_noise_normalization') != 'per_example_spatial_minmax_signed':
        raise ValueError('Paper noise requires per-example spatial min-max normalization to [-1,1]')
    if config.get('local_dae_noise_sigma') != 0.2 or config.get('local_dae_noise_amplitude') != 0.6:
        raise ValueError('Paper noise settings require sigma=0.2 and amplitude=0.6')
    if config.get('train_continue'):
        raise ValueError('this new experiment starts from scratch; no implicit old-model checkpoint loading')
    return config

def describe(config):
    from model.factory import create_model
    model = create_model(config)
    counts = {name: sum((p.numel() for p in module.parameters())) if module is not None else 0 for name, module in [('edge_encoder', model.edge_encoder), ('image_encoder', model.image_encoder), ('decoder', model.decoder)]}
    counts['total'] = sum((p.numel() for p in model.parameters()))
    print(json.dumps({'status': 'prepared_not_trained', 'parameters': counts, 'branch_mode': config.get('branch_mode', 'both'), 'capacity_input_pair': config.get('capacity_input_pair'), 'fusion_type': config['fusion_type'], 'training_call': 'not executed', 'noise_sigma': config['local_dae_noise_sigma'], 'noise_amplitude': config['local_dae_noise_amplitude'], 'noise_normalization': config['local_dae_noise_normalization']}))
    return counts

def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])
    import torch
    import utils
    from trainer import Trainer
    if not torch.cuda.is_available():
        raise RuntimeError('the existing training framework requires CUDA')
    if len(str(config['gpu']).split(',')) != 1:
        raise ValueError('the current training framework supports one GPU')
    for field in ['train_img_dir', 'test_img_dir', 'test_lab_dir']:
        if not Path(config[field]).is_dir():
            raise FileNotFoundError(config[field])
    from case_lists import read_names, require_disjoint
    train_names = read_names(config['train_case_list'], config['train_img_dir'])
    monitor_names = read_names(config['monitor_case_list'], config['test_img_dir'], config['test_lab_dir'])
    require_disjoint(train_names, monitor_names)
    if len(monitor_names) != config['test_num']:
        raise ValueError('Monitor list length mismatch')
    args = argparse.Namespace(**config)
    args.main_path = str(Path(__file__).resolve())
    args.num_gpu = 1
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    name = f'{stamp}-{args.model}-localDAE{args.local_dae_target_coverage}-e{args.max_epochs}-bs{args.batch_size}'
    parent = Path(args.save_dir)
    if not parent.is_absolute():
        parent = ROOT / parent
    args.save_dir = str(parent / stamp[:6] / name)
    scripts = glob.glob(str(ROOT / '*.py')) + [str(ROOT / p) for p in ['model', 'loss', 'data', 'configs']]
    utils.create_exp_dir(args.save_dir, scripts_to_save=scripts)
    logger = logging.getLogger('single_edge_image')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    for handler in [logging.StreamHandler(sys.stdout), logging.FileHandler(Path(args.save_dir) / 'log.txt')]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    utils.write_to_yaml(args, str(Path(args.save_dir) / 'config.yaml'))
    utils.write_to_yaml(args, str(Path(args.save_dir) / 'scripts/config.yaml'))
    logger.info('args = %s', args)
    utils.seed_everything(args.seed)
    Trainer(args, logger).run()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--train', action='store_true', help='explicitly launch training')
    mode.add_argument('--describe', action='store_true', help='print settings and architecture without training')
    parser.add_argument('--ssim-weight', type=float, help='0 for L1 only; default YAML uses L1 plus SSIM with weight 1')
    args = parser.parse_args()
    config = load_settings(args.config)
    if args.ssim_weight is not None:
        if args.ssim_weight < 0:
            parser.error('--ssim-weight must be nonnegative')
        config['w_dict']['l_ssim'] = args.ssim_weight
    if args.train:
        train(config)
    else:
        describe(config)
if __name__ == '__main__':
    main()
