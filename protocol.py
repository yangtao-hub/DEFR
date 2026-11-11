import numpy as np
import torch
from scipy.ndimage import binary_erosion

def foreground_mask(x, protocol):
    mask = np.asarray(x) > 0
    if protocol == 'native_foreground':
        return mask
    if protocol == 'eroded3':
        return binary_erosion(mask, np.ones((3, 3, 3)))
    raise ValueError('mask_protocol must be native_foreground or eroded3')

def postprocess_score(score, mode):
    if mode == 'raw':
        return score.abs()
    if mode == 'min3_mean9':
        smooth = torch.nn.Sequential(torch.nn.MaxPool3d(3, 1, 1), torch.nn.AvgPool3d(9, 1, 4))
        return (-smooth(-score)).abs()
    raise ValueError('postprocessing must be raw or min3_mean9')

def scoring_profile(c):
    ignored = {'image_dir', 'label_dir', 'case_list', 'checkpoint', 'dataset_role'}
    return {k: v for k, v in c.items() if k not in ignored}

def validate_calibration(calibration, profile, checkpoint_sha, names):
    import hashlib
    if calibration['scoring_profile'] != profile or calibration['checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('development/test scoring profile or checkpoint mismatch')
    if {hashlib.sha256(n.encode()).hexdigest() for n in names} & set(calibration['development_case_basename_sha256']):
        raise ValueError('Development/test case overlap')
