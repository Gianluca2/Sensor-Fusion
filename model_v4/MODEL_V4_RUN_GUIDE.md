# Model V4

Model V4 keeps the radar-conditioned Model V3-style input, but changes the target and loss.

## Goal

Input:

```text
faulty 13-channel LiDAR/radar BEV + known fault type + known severity
```

Output:

```text
1-channel soft LiDAR unreliability map
```

Unlike V3, V4 is not trained only as a hard binary mask segmenter. The target is a continuous clean-vs-faulty BEV damage map:

```text
soft_target = clip(max(abs(clean - faulty)) / target_threshold, 0, 1)
binary_target = max(abs(clean - faulty)) > target_threshold
```

The soft target is used for reliability-map training. The binary target is kept for IoU, precision, and recall reporting.

## Input Channels

```text
lidar_density
lidar_height
lidar_height_spread
binary_occupancy
range_from_sensor
expected_lidar_density_by_range_angle
lidar_density_expected_residual
local_density_residual
temporal_density_consistency
radar_occupancy
radar_density
radar_abs_velocity
radar_supported_missing_lidar
```

`radar_supported_missing_lidar` is a derived channel:

```text
radar_supported_missing_lidar = radar_occupancy * (1 - lidar_binary_occupancy)
```

It highlights cells where radar sees structure but LiDAR is missing occupancy, which gives the model a direct clue for sparse regions that may be LiDAR failures rather than naturally empty space.

`expected_lidar_density_by_range_angle` is a heuristic LiDAR prior computed from range-angle bins in the current BEV. It gives the model context for what density is locally expected at similar geometry.

`lidar_density_expected_residual` highlights cells where the observed LiDAR density is lower than the expected range-angle density:

```text
lidar_density_expected_residual = max(expected_lidar_density_by_range_angle - lidar_density, 0)
```

This is intended to help distinguish normal sparse range behavior from suspicious missing LiDAR support.

## Loss

```text
L = soft_weight * SmoothL1(predicted_unreliability, soft_target)
  + bce_weight  * BCEWithLogits(logits, binary_target)
  + dice_weight * Dice(predicted_unreliability, binary_target)
```

The default weights are:

```text
soft_weight = 1.0
bce_weight = 0.5
dice_weight = 1.0
range_loss_weight = 0.25
```

The range loss weight applies:

```text
spatial_weight = 1 + range_loss_weight * range_from_sensor
```

## Architecture Change

V4 uses GroupNorm instead of BatchNorm. This is more stable for small or changing batch sizes.

## Separate LiDAR And Radar Aggregation

V4 can aggregate different numbers of LiDAR and radar frames:

```bash
--lidar-aggregate-scans 3
--radar-aggregate-scans 20
```

This means each training sample uses 3 motion-compensated Aeva LiDAR frames, but 20 motion-compensated Continental radar frames. The reference frame is the current frame `t`, and aggregation uses only past/current frames: LiDAR `[t-2, t-1, t]` and radar `[t-19, ..., t]`. If these flags are omitted, both default to `--aggregate-scans`.

## Linux 10k Test

```bash
cd /mnt/3D10B36523559581/Gianluca/Sensor-Fusion
source .venv_model_v2/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=/mnt/3D10B36523559581/Gianluca/model_v4_outputs_10k_soft_reliability

python model_v4/generate_model_v4_samples_controlled.py \
  --repo-dir /mnt/3D10B36523559581/Gianluca/Sensor-Fusion \
  --data-root /mnt/3D10B36523559581/HeRCULES \
  --dataset-dir "$OUT/dataset" \
  --target-samples 10000 \
  --chunk-size 500 \
  --lidar-aggregate-scans 3 \
  --radar-aggregate-scans 20

python model_v4/train_model_v4.py \
  --dataset-dir "$OUT/dataset" \
  --model-path "$OUT/models/model_v4_10k_best.pt" \
  --latest-model-path "$OUT/models/model_v4_10k_latest.pt" \
  --metrics-path "$OUT/models/model_v4_10k_metrics.csv" \
  --epochs 50 \
  --batch-size 18 \
  --loader-workers 8 \
  --channel-normalization dataset \
  --normalization-samples 4096 \
  --threshold 0.65 \
  --soft-weight 1.0 \
  --bce-weight 0.5 \
  --dice-weight 0.5 \
  --range-loss-weight 0.25 \
  --range-channel-index 4 \
  --base-channels 48 \
  --early-stop-patience 50

python model_v4/predict_model_v4.py \
  --model-path "$OUT/models/model_v4_10k_best.pt" \
  --sample-dir "$OUT/dataset" \
  --output-dir "$OUT/predictions" \
  --num-outputs 30 \
  --threshold 0.65
```
