# DEFR

**A Dual-Encoder Feature Fusion Framework for Pseudo-Healthy Reconstruction in 3D Brain MRI Anomaly Detection**

DEFR combines structural and appearance features through discrepancy-guided adaptive fusion (DGAF) to reconstruct pseudo-healthy images.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Data preparation

Provide preprocessed, aligned T2-weighted NIfTI volumes (`.nii.gz`) with zero background and intensities in [0, 1]. The input tensor size is 112 x 160 x 160. Perform registration, skull stripping and intensity standardization beforehand.

Set your local paths in `configs/`.

## Training

```bash
python train.py --config configs/train_main.yaml --train
```

## Validation and evaluation

Select the threshold on the 64 validation scans, then keep it fixed for testing:

```bash
python evaluate.py --config configs/validation.yaml --calibrate --output outputs/validation
python evaluate.py --config configs/testing.yaml --threshold-file outputs/validation/development_threshold.json --output outputs/testing --save-reconstructions
```
