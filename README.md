# DEFR

Implementation accompanying **A Dual-Encoder Feature Fusion Framework for Pseudo-Healthy Reconstruction in 3D Brain MRI Anomaly Detection**.

DEFR combines an edge-based structural encoder and an appearance encoder through discrepancy-guided adaptive fusion (DGAF). The reconstructed pseudo-healthy image is used to compute an anomaly map from L1 error and 3D SSIM.

## Installation

Use Python 3.8 and a CUDA-compatible GPU for training. The dependencies pin PyTorch 2.0.1 with CUDA 11.8 wheels.

```bash
python -m pip install -r requirements.txt
```

## Inputs and configuration

Prepare preprocessed, spatially aligned T2-weighted NIfTI volumes (`.nii.gz`) with zero background and intensities in [0, 1]. The network uses a 112 x 160 x 160 tensor grid. Registration, skull stripping and intensity standardization must be performed before running this code.

Each case-list file contains one image basename per line. Validation and test labels must use matching basenames; positive label values identify lesions. Supply your own disjoint training, 64-scan validation and deduplicated test lists. No images, labels, case lists or trained weights are included.

Edit the generic paths in the configuration files for your local inputs and output directory. In `configs/train_main.yaml`, `test_img_dir`, `test_lab_dir` and `monitor_case_list` refer to **validation**, not the final test set.

## Training

```bash
python train.py --config configs/train_main.yaml --describe
python train.py --config configs/train_main.yaml --train
```

The main configuration uses five encoder scales, DGAF temperature 0.1, L1 + SSIM loss, Adam with learning rate 0.0001, batch size 2 and 100 epochs. Gaussian noise is upsampled, min-max normalized to [-1, 1], scaled by 0.6 and applied within a local mask; corrupted intensities are clipped to [0, 1]. Checkpoint selection uses validation AP. Set `checkpoint` in both evaluation configurations to the selected `best_ap_*.pth` file from your run.

## Validation and inference

Calibrate a threshold using the 64 validation scans:

```bash
python evaluate.py --config configs/validation.yaml --calibrate --output outputs/validation
```

Apply that fixed threshold to the separate test set:

```bash
python evaluate.py --config configs/testing.yaml --threshold-file outputs/validation/development_threshold.json --output outputs/testing --save-reconstructions
```

Use a new, empty output directory for each run. The checkpoint and scoring settings must match between validation and testing. This evaluation entry point requires lesion labels to report AP and DSC; the model itself accepts image tensors without labels. Saved reconstructions are on the tensor grid, not the original physical image grid.

The implementation has passed synthetic code checks. End-to-end reproduction of the manuscript tables has not been verified.