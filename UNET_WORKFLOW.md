# BEV U-Net Mask Detection

This pipeline trains a small U-Net to detect injected black-mask faults in
LiDAR/radar BEV grids.

## What The Network Sees

Each training input is a six-channel BEV tensor:

1. `lidar_density`
2. `lidar_height`
3. `radar_density`
4. `radar_velocity`
5. `radar_range_min`
6. `radar_rcs_max`

The target is a one-channel binary mask:

- `1` = faulty/injected masked cell
- `0` = normal cell

## How Training Examples Are Created

`make_unet_dataset.py` starts from a clean BEV `.npz` and creates many synthetic
examples by placing random rectangular black masks at different positions and
sizes.

For each sample:

- input = BEV with one random rectangle zeroed out
- target = actual rectangle mask

So the first version trains on many masks generated from one BEV frame. For a
stronger thesis experiment, generate BEV files for many matched LiDAR/radar
frames and extend the dataset script to sample across all of them.

## Run Everything In PyCharm

Set the PyCharm interpreter to the environment that has NumPy, Open3D, and
PyTorch installed. Then run:

```powershell
run_unet_pipeline.py
```

Useful parameters:

```text
--num-samples 200
--epochs 10
--batch-size 4
--min-mask-occupied-cells 50
--max-mask-area-fraction 0.08
--lr 0.0003
```

`--min-mask-occupied-cells` rejects random mask locations that cover mostly empty
space. This keeps training examples focused on regions where the clean BEV had
LiDAR/radar content before masking.

`--max-mask-area-fraction 0.08` caps masks at 8% of the full BEV image area.

## Output

Prediction overlays are saved to:

```text
C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\unet_predictions\
```

By default, `predict_unet.py` writes 10 predicted-vs-actual overlay images.

## Visual Random Validation

Run:

```powershell
validate_unet_random.py
```

Each run creates one new random mask on the clean BEV, predicts that same masked
input with both saved U-Net models, and writes PNG overlays to:

```text
C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\unet_random_validation\
```

Use this for visual inspection after training. Leave `--seed` unset if you want a
different mask every run.

The comparison image places the BCE+IoU prediction on the left and the
BCE+Tversky prediction on the right. Both sides use the same actual mask.

Overlay colors:

- red = actual injected mask
- blue = predicted mask
- magenta = overlap

The model is saved to:

```text
C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\models\unet_bev_mask.pt
```

## Comparing Loss Functions

Run:

```powershell
compare_loss_functions.py
```

This trains two models on the same dataset:

- `bce_iou`: binary cross entropy plus IoU/Jaccard loss
- `bce_tversky`: binary cross entropy plus Tversky loss

Metrics are written to:

```text
C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\models\metrics_bce_iou.csv
C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\models\metrics_bce_tversky.csv
```

How to choose the better model:

- higher validation IoU = better overlap
- higher validation F1 = better balance between predicted and actual mask
- lower false positive cells = less extra blue area
- lower false negative cells = less missed red area

For the thesis, report validation IoU and F1 as primary metrics, then use false
positive and false negative cell counts to explain whether one loss overpredicts
or underpredicts the faulty area.
