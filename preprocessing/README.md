# DEFR preprocessing

Preprocessing utilities for T2-weighted brain MRI.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

Run only the stages needed for your dataset. For images already skull-stripped and registered to SRI24, start with cropping. Use new output directories; inputs are not overwritten.

```bash
python preprocess.py register --image input/T2.nii.gz --reference input/T1.nii.gz --atlas assets/SRI24_T1.nii.gz --captk-bin tools/captk/bin --skull-model assets/skull_model --output work/registered
python preprocess.py crop --image work/registered/brain.nii.gz --label input/label_sri.nii.gz --output work/cropped
python preprocess.py normalize --images cropped/train_images --fit-training --output work/train_normalized
python preprocess.py normalize --images cropped/eval_images --histogram work/train_normalized/training_histogram.npy --output work/eval_normalized
```

Replace example paths with local inputs. Use `--brain-mask` instead of `--skull-model` for an existing aligned mask. Add `--label` to registration to transform annotations jointly; use its `label_sri.nii.gz` output for cropping. Omit label arguments when unavailable. Inputs to registration must share a physical grid. Omit `--reference` when registering T2 directly and use a matching T2 atlas.

The modules perform RAI reorientation, rigid registration, brain masking, shared image/label cropping, and Nyul intensity standardization. N4 corrects the reference used to estimate registration; the transform is applied to the uncorrected, reoriented image. Fit the Nyul histogram on normal training images only and reuse it for validation/testing.

Cropping produces model-grid arrays of shape 160 × 224 × 160 (z, y, x), not native-space images. DEFR's loader resizes them to 112 × 160 × 160.
