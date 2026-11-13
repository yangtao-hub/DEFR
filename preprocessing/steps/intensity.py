from pathlib import Path
import numpy as np
from .common import execute, fresh

def finalize_intensity(normalized, original):
    if normalized.shape != original.shape or not np.isfinite(normalized).all():
        raise ValueError("Invalid normalized array")
    image = np.maximum(normalized, 1e-3) / 100.0
    image[original <= 0] = 0
    return np.clip(image, 0, 1).astype(np.float32)

def normalize(a):
    import SimpleITK as sitk
    out = fresh(a.output)
    normalized = fresh(out/'nyul')
    command = [a.nyul_bin, a.images, '-o', normalized, '-mo', 't2',
               '--output-min-value', '1', '--output-max-value', '100',
               '--min-percentile', '1', '--max-percentile', '99',
               '--percentile-after-min', '10', '--percentile-before-max', '90',
               '--percentile-step', '10']
    if a.fit_training:
        command += ['-ssh', out/'training_histogram.npy']
    else:
        command += ['-lsh', a.histogram]
    execute(command)
    ready = fresh(out/'images')
    files = sorted(Path(a.images).glob('*.nii*'))
    if not files:
        raise ValueError("Input directory contains no NIfTI files")
    for source in files:
        stem = source.name[:-7] if source.name.endswith('.nii.gz') else source.stem
        candidates = [normalized/(stem+'_nyul.nii.gz'), normalized/(stem+'_nyul.nii')]
        matches = [p for p in candidates if p.exists()]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one Nyul output for {source.name}")
        original = sitk.ReadImage(str(source))
        mapped = sitk.GetArrayFromImage(sitk.ReadImage(str(matches[0])))
        final = finalize_intensity(mapped, sitk.GetArrayFromImage(original))
        result = sitk.GetImageFromArray(final)
        result.CopyInformation(original)
        sitk.WriteImage(result, str(ready/source.name))
