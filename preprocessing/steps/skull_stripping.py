import numpy as np
from .common import execute, aligned

def infer_mask(tool, image, destination, model):
    execute([tool, '-i', image, '-o', destination, '-md', model])
    return destination/'predictions/testApiSession/predictions/Segm.nii.gz'

def apply_mask(image_path, mask_path, destination):
    import SimpleITK as sitk
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    if not aligned(image, mask):
        raise ValueError('Skull-stripping mask geometry mismatch')
    array = sitk.GetArrayFromImage(image) * (sitk.GetArrayFromImage(mask) > 0)
    result = sitk.GetImageFromArray(array.astype(np.float32))
    result.CopyInformation(image)
    sitk.WriteImage(result, str(destination))
