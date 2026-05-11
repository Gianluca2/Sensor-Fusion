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
--epochs 15
--batch-size 4
```

## Output

The prediction overlay is saved to:

```text
C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\unet_predictions\sample_000000_overlay.png
```

Overlay colors:

- red = actual injected mask
- blue = predicted mask
- magenta = overlap

The model is saved to:

```text
C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\models\unet_bev_mask.pt
```
