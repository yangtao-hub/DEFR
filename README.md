# DEFR

**A Dual-Encoder Feature Fusion Framework for Pseudo-Healthy Reconstruction in 3D Brain MRI Anomaly Detection**

DEFR combines structural and appearance features through discrepancy-guided adaptive fusion (DGAF) to reconstruct pseudo-healthy images. Anomaly maps combine L1 reconstruction error and 3D SSIM.

## Installation

Python 3.8 and a CUDA-compatible GPU are required for training.

```bash
python -m pip install -r requirements.txt
```

## Data preparation

Provide preprocessed, aligned T2-weighted NIfTI volumes (`.nii.gz`) with zero background and intensities in [0, 1]. The input tensor size is 112 x 160 x 160. Perform registration, skull stripping and intensity standardization beforehand.

Set your local paths in `configs/`. Case lists contain one filename per line; labels use matching filenames and positive values for lesions. Use disjoint training, 64-scan validation and deduplicated test sets. Data, case lists and pretrained weights are not included.

## Training

In `train_main.yaml`, `test_img_dir`, `test_lab_dir` and `monitor_case_list` refer to the validation set.

```bash
python train.py --config configs/train_main.yaml --train
```

The configuration uses normalized local noise, DGAF temperature 0.1, L1 + SSIM loss, Adam (learning rate 0.0001), batch size 2 and 100 epochs. Set `checkpoint` in both evaluation configurations to the selected `best_ap_*.pth` checkpoint.

## Validation and evaluation

Select the threshold on the 64 validation scans, then keep it fixed for testing:

```bash
python evaluate.py --config configs/validation.yaml --calibrate --output outputs/validation
python evaluate.py --config configs/testing.yaml --threshold-file outputs/validation/development_threshold.json --output outputs/testing --save-reconstructions
```

Use new output directories and the same checkpoint and scoring settings in both stages. Evaluation requires lesion labels and reports AP and DSC. Saved reconstructions use the tensor grid rather than the original physical image grid.

Code-level synthetic checks have passed; reproduction of the manuscript tables has not yet been verified.