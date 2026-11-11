import numpy as np
import torch
import torch.nn.functional as F

def _local_dae_fraction_mask_torch(image: torch.Tensor, target_coverage: float, region_count: int) -> torch.Tensor:
    foreground = image[0] > 0
    coordinates = torch.nonzero(foreground, as_tuple=False)
    foreground_voxels = int(coordinates.shape[0])
    if foreground_voxels == 0:
        raise ValueError('local DAE requires a non-empty foreground')
    target_voxels = min(foreground_voxels, max(1, int(round(float(target_coverage) * foreground_voxels))))
    if target_voxels == foreground_voxels:
        return foreground.unsqueeze(0)
    coordinates_float = coordinates.to(dtype=image.dtype)
    lower = coordinates_float.amin(dim=0)
    upper = coordinates_float.amax(dim=0) + 1.0
    foreground_shape = torch.clamp(upper - lower, min=1.0)
    scores = torch.full((foreground_voxels,), torch.inf, device=image.device, dtype=image.dtype)
    center_indices = np.random.choice(foreground_voxels, size=min(int(region_count), foreground_voxels), replace=False)
    for center_index in np.atleast_1d(center_indices):
        center = coordinates_float[int(center_index)]
        radii = torch.as_tensor(np.random.uniform(0.12, 0.32, size=3), device=image.device, dtype=image.dtype) * foreground_shape
        radii = torch.clamp(radii, min=1.0)
        distance = ((coordinates_float - center) / radii).square().sum(dim=1)
        scores = torch.minimum(scores, distance)
    selected = torch.topk(scores, target_voxels, largest=False).indices
    mask = torch.zeros_like(image, dtype=torch.bool)
    chosen = coordinates[selected]
    mask[0, chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
    return mask

def signed_minmax_noise(noise):
    if noise.ndim != 4 or noise.shape[0] != 1:
        raise ValueError('Expected one single-channel example [1,D,H,W]')
    low, high = (noise.amin(), noise.amax())
    span = high - low
    if not torch.isfinite(noise).all():
        raise ValueError('Nonfinite noise')
    if span.item() == 0:
        return torch.zeros_like(noise)
    return 2 * (noise - low) / span - 1

def apply_local_corruption(image, normalized_noise, region_mask, amplitude):
    return (image + amplitude * region_mask.to(image.dtype) * normalized_noise).clamp(0, 1)

def local_noise_torch(x: torch.Tensor, shape_mode: str='ellipsoid', coverage_mode: str='current', target_coverage: float=0.25, return_region_mask: bool=False, noise_sigma: float=0.2, noise_amplitude: float=0.6):
    if x.ndim != 5 or x.shape[1] != 1:
        raise ValueError(f'local DAE expects B,C,D,H,W input, got shape={tuple(x.shape)}')
    if not x.is_floating_point():
        raise TypeError('local DAE expects a floating-point tensor')
    if shape_mode != 'ellipsoid':
        raise ValueError("local DAE requires shape_mode='ellipsoid'")
    if coverage_mode != 'current':
        raise ValueError("local DAE requires coverage_mode='current'")
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError('target_coverage must be in (0, 1]')
    if not np.isfinite(noise_sigma) or noise_sigma <= 0:
        raise ValueError('noise_sigma must be finite and positive')
    if not np.isfinite(noise_amplitude) or noise_amplitude < 0:
        raise ValueError('noise_amplitude must be finite and nonnegative')
    augmented = []
    region_masks = []
    for image in x:
        requested_regions = int(np.random.randint(2, 5))
        region_mask = _local_dae_fraction_mask_torch(image, target_coverage, requested_regions)
        low_shape = (1, image.shape[0], max(1, image.shape[1] // 8), max(1, image.shape[2] // 8), max(1, image.shape[3] // 8))
        local_noise = torch.empty(low_shape, device=x.device, dtype=x.dtype).normal_(mean=0.0, std=noise_sigma)
        local_noise = F.interpolate(local_noise, size=image.shape[-3:], mode='trilinear', align_corners=False).squeeze(0)
        local_noise = signed_minmax_noise(local_noise)
        result = apply_local_corruption(image, local_noise, region_mask, noise_amplitude)
        augmented.append(result)
        region_masks.append(region_mask)
    augmented_tensor = torch.stack(augmented)
    residual = torch.abs(augmented_tensor - x)
    if return_region_mask:
        return (augmented_tensor, residual, torch.stack(region_masks))
    return (augmented_tensor, residual)
