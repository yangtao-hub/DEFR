"""T2 brain MRI preprocessing with explicit, dataset-independent inputs."""
import argparse
from pathlib import Path
import numpy as np
from steps.common import fresh, aligned
from steps.orientation import reorient
from steps.bias_correction import correct_reference
from steps.registration import estimate_rigid, apply_rigid
from steps.skull_stripping import infer_mask, apply_mask
from steps.crop_padding import crop_plan, apply_crop, crop
from steps.intensity import finalize_intensity, normalize

def register(a):
    import SimpleITK as sitk
    out = fresh(a.output)
    tools = Path(a.captk_bin)
    def tool(name):
        p = tools / name
        return p if p.exists() else tools / (name + '.exe')
    fixed = sitk.ReadImage(a.atlas)
    if not np.allclose(fixed.GetSpacing(), (1, 1, 1)):
        raise ValueError("Provide the 1-mm SRI24 template")
    reference = a.reference or a.image
    inputs = {'image': a.image, 'reference': reference}
    if a.label:
        inputs['label'] = a.label
    if a.brain_mask:
        inputs['brain_mask'] = a.brain_mask
    image_header = sitk.ReadImage(a.image)
    for name, path in inputs.items():
        if not aligned(image_header, sitk.ReadImage(path)):
            raise ValueError(f"{name} must be aligned to the input image before registration")
        reorient(tool('Utilities'), path, out/(name+'_rai.nii.gz'))
    corrected = out / 'reference_n4.nii.gz'
    correct_reference(tool('Preprocessing'), out/'reference_rai.nii.gz', corrected)
    transform = out/'rigid.mat'
    estimate_rigid(tool('greedy'), a.atlas, corrected, transform)
    for name in inputs:
        apply_rigid(tool('greedy'), a.atlas, out/(name+'_rai.nii.gz'),
                    out/(name+'_sri.nii.gz'), transform, name in ('label', 'brain_mask'))
    if a.brain_mask:
        brain_mask = out/'brain_mask_sri.nii.gz'
    else:
        if not a.skull_model:
            raise ValueError("Provide a brain mask or a CaPTk DeepMedic skull-stripping model")
        brain_mask = infer_mask(tool('DeepMedic'), out/'reference_sri.nii.gz', out/'skull', a.skull_model)
    apply_mask(out/'image_sri.nii.gz', brain_mask, out/'brain.nii.gz')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='stage', required=True)
    p = subs.add_parser('register')
    for key in ['image', 'atlas', 'captk-bin', 'output']:
        p.add_argument('--'+key, required=True)
    for key in ['reference', 'label', 'brain-mask', 'skull-model']:
        p.add_argument('--'+key)
    p.set_defaults(func=register)
    p = subs.add_parser('crop')
    p.add_argument('--image', required=True)
    p.add_argument('--label')
    p.add_argument('--output', required=True)
    p.set_defaults(func=crop)
    p = subs.add_parser('normalize')
    p.add_argument('--images', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--nyul-bin', default='nyul-normalize')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--fit-training', action='store_true')
    mode.add_argument('--histogram')
    p.set_defaults(func=normalize)
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
