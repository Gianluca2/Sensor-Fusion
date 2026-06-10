# Model V3

Model V3 keeps the direct fault-mask segmentation goal from Model V2, but changes the input representation and the fault-injection stage.

## Goal

Input:

```text
faulty 10-channel LiDAR/radar BEV + known fault type + known severity
```

Output:

```text
1-channel fault probability mask
```

The model does not reconstruct clean BEV. It predicts where LiDAR is unreliable.

## Key Differences From Model V2

1. V3 uses 10 BEV channels:

```text
lidar_density
lidar_height
lidar_height_spread
binary_occupancy
range_from_sensor
local_density_residual
temporal_density_consistency
radar_occupancy
radar_density
radar_abs_velocity
```

The radar channels are conditioning channels. They are not faulted; they give the network a weak expectation of where structure or occupancy may exist even when LiDAR becomes unreliable.

2. Faults are injected before BEV projection:

```text
raw Aeva point cloud -> fault injector -> motion compensation -> BEV projection
```

This means the faulty BEV is produced from faulty LiDAR points, instead of applying an artificial mask after BEV creation.

3. The default loss is range-weighted BCE + Dice:

```text
loss = weighted BCEWithLogits + Dice
negative_weight = 2.2
positive_weight = 2.0
range_loss_weight = 1.0
```

The negative weight is slightly higher than the positive weight to reduce false positives without making false negatives cheap. The range weighting multiplies the per-cell mask loss by `1 + range_loss_weight * range_from_sensor`, so farther cells matter more during optimization.

Because the input changed from 7 channels to 10 channels, regenerate Model V3 samples in a new dataset folder before training. Do not mix old 7-channel samples with new radar-conditioned samples.

## Linux Quick Run

```bash
cd /mnt/3D10B36523559581/Gianluca/Sensor-Fusion
source .venv_model_v2/bin/activate

python model_v3/run_model_v3_4090.py --quick --compressed-samples
```

## Linux 75k-Sample 75-Epoch Run

```bash
cd /mnt/3D10B36523559581/Gianluca/Sensor-Fusion
source .venv_model_v2/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python model_v3/run_model_v3_75k_4090.py
```

The 75k run generates samples in 2,500-sample chunks. If sample generation crashes,
rerun the same command; existing `sample_*.npz` files are skipped and the run
continues filling the missing chunks.

This writes:

```text
/mnt/3D10B36523559581/Gianluca/model_v3_outputs_75k/model_v3_dataset_75k
/mnt/3D10B36523559581/Gianluca/model_v3_outputs_75k/models/model_v3_75k_best.pt
/mnt/3D10B36523559581/Gianluca/model_v3_outputs_75k/models/model_v3_75k_latest.pt
/mnt/3D10B36523559581/Gianluca/model_v3_outputs_75k/models/model_v3_75k_training_metrics.csv
/mnt/3D10B36523559581/Gianluca/model_v3_outputs_75k/model_v3_75k_predictions
```

`model_v3_75k_latest.pt` is overwritten after every epoch. `model_v3_75k_best.pt` keeps the best validation-loss checkpoint.
If training is interrupted, run the same command again. The launcher resumes from
`model_v3_75k_latest.pt` automatically and continues until epoch 75. Use
`--restart-training` only if you intentionally want to ignore the latest checkpoint.

## Linux 5000-Sample Test

```bash
cd /mnt/3D10B36523559581/Gianluca/Sensor-Fusion
source .venv_model_v2/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python model_v3/rewrite_model_v3_samples.py \
  --data-root /mnt/3D10B36523559581/HeRCULES \
  --dataset-dir /mnt/3D10B36523559581/Gianluca/model_v3_outputs/model_v3_dataset_5k \
  --num-samples 5000 \
  --aggregate-scans 3 \
  --compressed-samples

python model_v3/train_model_v3.py \
  --dataset-dir /mnt/3D10B36523559581/Gianluca/model_v3_outputs/model_v3_dataset_5k \
  --model-path /mnt/3D10B36523559581/Gianluca/model_v3_outputs/models/model_v3_5k.pt \
  --metrics-path /mnt/3D10B36523559581/Gianluca/model_v3_outputs/models/model_v3_5k_metrics.csv \
  --batch-size 8 \
  --loader-workers 4 \
  --epochs 20 \
  --channel-normalization dataset \
  --normalization-samples 1024 \
  --threshold 0.65 \
  --positive-weight 2.0 \
  --negative-weight 2.2 \
  --range-loss-weight 1.0 \
  --range-channel-index 4

python model_v3/predict_model_v3.py \
  --model-path /mnt/3D10B36523559581/Gianluca/model_v3_outputs/models/model_v3_5k.pt \
  --sample-dir /mnt/3D10B36523559581/Gianluca/model_v3_outputs/model_v3_dataset_5k \
  --output-dir /mnt/3D10B36523559581/Gianluca/model_v3_outputs/model_v3_5k_predictions \
  --num-outputs 10 \
  --threshold 0.65
```
