import numpy as np

def resize_for_network(image, label=None, size=(112, 160, 160)):
    import torch
    import torch.nn.functional as F
    tensor = torch.from_numpy(np.clip(image, 0, 1).astype(np.float32))[None, None]
    resized = F.interpolate(tensor, size=size, mode='trilinear', align_corners=False)[0, 0].numpy()
    if label is None:
        return resized, None
    mask = torch.from_numpy((label > 0).astype(np.float32))[None, None]
    mask = F.interpolate(mask, size=size, mode='nearest')[0, 0].numpy().astype(np.uint8)
    return resized, mask
