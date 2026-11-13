import numpy as np
from .common import fresh, aligned
from .labels import binarize

def crop_plan(image, shape=(160, 224, 160)):
    coords = np.nonzero(image)
    if not len(coords[0]):
        raise ValueError("Empty brain foreground")
    lo = np.array([c.min() for c in coords])
    hi = np.array([c.max() + 1 for c in coords])
    if hi[2] - lo[2] > shape[2]:
        extra = hi[2] - lo[2] - shape[2]
        right = int(extra / 2) + 1
        hi[2] -= right
        lo[2] += extra - right
    size = hi - lo
    if np.any(size > np.array(shape)):
        raise ValueError("Foreground exceeds crop size in z/y; inspect rather than discard it")
    pad = np.array(shape) - size
    padding = tuple((int(n // 2), int(n - n // 2)) for n in pad)
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    return slices, padding

def apply_crop(array, plan):
    return np.pad(array[plan[0]], plan[1])

def crop(a):
    import SimpleITK as sitk
    out = fresh(a.output)
    image = sitk.ReadImage(a.image)
    array = sitk.GetArrayFromImage(image)
    plan = crop_plan(array)
    cropped = apply_crop(array, plan)
    foreground = cropped != 0
    cropped[foreground] = np.maximum(cropped[foreground], 1e-3)
    sitk.WriteImage(sitk.GetImageFromArray(cropped.astype(np.float32)), str(out/'image.nii.gz'))
    if a.label:
        label = sitk.ReadImage(a.label)
        if not aligned(image, label):
            raise ValueError("Image and lesion label geometry mismatch")
        mask = binarize(apply_crop(sitk.GetArrayFromImage(label), plan))
        sitk.WriteImage(sitk.GetImageFromArray(mask.astype(np.uint8)), str(out/'label.nii.gz'))
