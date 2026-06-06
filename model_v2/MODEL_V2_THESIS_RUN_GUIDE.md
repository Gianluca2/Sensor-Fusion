# Model V2 Thesis Run Guide

This file lists the scripts needed to run the Model V2 LiDAR fault-restoration experiment.

## Goal

Model V2 receives a faulty LiDAR BEV representation and the known injected fault metadata:

```text
faulty BEV + known fault type + known severity -> reconstructed clean BEV
```

The clean BEV is not used as an input during inference. During training, the clean BEV is the target used to compute the reconstruction loss.

After reconstruction, the predicted faulty region is derived by comparing:

```text
abs(faulty BEV - reconstructed clean BEV)
```

The resulting reconstruction-error map is thresholded to create the predicted fault mask.

## Required Scripts

Core Model V2 scripts:

```text
model_v2\Model_V2.py
model_v2\train_model_v2.py
model_v2\rewrite_model_v2_samples.py
model_v2\run_model_v2_4090.sh
model_v2\run_model_v2_4090_quick.sh
model_v2\predict_model_v2_reconstruction_error.py
model_v2\predict_model_v2.py
```

Shared helper scripts:

```text
bev_fault_visualization.py
bev_projection.py
```

Existing data-generation scripts used before Model V2:

```text
build_bev_dataset.py
make_autoencoder_dataset.py
hercules_lidar_faults.py
```

## Data Locations

Current BEV pool:

```text
C:\Users\gianl\ThesisOutputs\HerculesFiles\outputs\bev_autoencoder
```

Current paired faulty/clean sample dataset:

```text
C:\Users\gianl\ThesisOutputs\HerculesFiles\outputs\autoencoder_dataset
```

Each `sample_*.npz` file should contain:

```text
input          faulty BEV tensor
clean          actual clean BEV tensor
target         actual fault mask
metadata_json  injected fault type and severity
```

## Python Environment

Use the Python 3.11 virtual environment:

```powershell
C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\MaskingTest\.venv311\Scripts\python.exe
```

Quick CUDA check:

```powershell
cd "C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\MaskingTest"
.\.venv311\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

## Rewrite Samples From The Full HeRCULES Dataset

On the Linux training machine, use `rewrite_model_v2_samples.py` to rebuild the BEV pool from every available LiDAR frame and then rewrite paired faulty/clean samples.

Example using your shown HeRCULES folder:

```bash
cd /mnt/3D10B36523559581/Gianluca/Sensor-Fusion
python3 model_v2/rewrite_model_v2_samples.py \
  --data-root /mnt/3D10B36523559581/Gianluca/HeRCULES \
  --bev-dir /mnt/3D10B36523559581/Gianluca/model_v2_outputs/bev_all_frames \
  --dataset-dir /mnt/3D10B36523559581/Gianluca/model_v2_outputs/model_v2_dataset \
  --rewrite-bev \
  --num-workers 8
```

This uses:

```text
stride = 1
aggregate_scans = 3
fault types = laser, photodetector, scanning, optical, window, mounting
severities = mild, moderate, severe
```

By default, every BEV frame receives every fault/severity combination once:

```text
samples = BEV frame count * 6 fault types * 3 severities
```

For a quick test, cap the BEV frames per scene:

```bash
python3 model_v2/rewrite_model_v2_samples.py \
  --data-root /mnt/3D10B36523559581/Gianluca/HeRCULES \
  --frames-per-scene-cap 20 \
  --num-workers 2
```

After rewriting samples, train with:

```bash
python3 model_v2/train_model_v2.py \
  --dataset-dir /mnt/3D10B36523559581/Gianluca/model_v2_outputs/model_v2_dataset \
  --model-path /mnt/3D10B36523559581/Gianluca/model_v2_outputs/models/model_v2.pt \
  --metrics-path /mnt/3D10B36523559581/Gianluca/model_v2_outputs/models/model_v2_training_metrics.csv
```

## RTX 4090 Machine Launchers

For the Linux RTX 4090 machine, the repository includes ready-to-run launchers.

Quick smoke test:

```bash
cd /mnt/3D10B36523559581/Gianluca/Sensor-Fusion
bash model_v2/run_model_v2_4090_quick.sh
```

Full run:

```bash
cd /mnt/3D10B36523559581/Gianluca/Sensor-Fusion
bash model_v2/run_model_v2_4090.sh
```

Default full-run settings:

```text
data root = /mnt/3D10B36523559581/Gianluca/HeRCULES
output root = /mnt/3D10B36523559581/Gianluca/model_v2_outputs
sample generation workers = 12
training batch size = 8
training DataLoader workers = 8
epochs = 100
```

You can override settings without editing the script:

```bash
BATCH_SIZE=16 EPOCHS=200 SAMPLE_WORKERS=16 bash model_v2/run_model_v2_4090.sh
```

If the BEV pool has already been created and you only want to rewrite fault samples and retrain:

```bash
REWRITE_BEV=0 bash model_v2/run_model_v2_4090.sh
```

## Train Model V2

Run:

```powershell
cd "C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\MaskingTest"
.\.venv311\Scripts\python.exe model_v2\train_model_v2.py
```

The default training script reads existing samples from:

```text
C:\Users\gianl\ThesisOutputs\HerculesFiles\outputs\autoencoder_dataset
```

It saves the model to:

```text
C:\Users\gianl\ThesisOutputs\HerculesFiles\outputs\models\model_v2.pt
```

It saves training metrics to:

```text
C:\Users\gianl\ThesisOutputs\HerculesFiles\outputs\models\model_v2_training_metrics.csv
```

The metrics CSV tracks both training and validation rows for every epoch. Important columns include:

```text
loss
reconstruction_loss
learning_rate
iou
precision
recall
f1
f2
mean_error
```

The learning rate is adaptive. `train_model_v2.py` uses `ReduceLROnPlateau`, which monitors validation loss and reduces the learning rate when validation loss stalls.

Default adaptive learning-rate settings:

```text
initial lr = 1e-3
lr reduce patience = 2 epochs
lr reduce factor = 0.5
```

You can override them:

```powershell
.\.venv311\Scripts\python.exe model_v2\train_model_v2.py --lr 0.001 --lr-reduce-patience 3 --lr-reduce-factor 0.5
```

## Visualize Model V2

For the current conditioned reconstruction-error Model V2, run:

```powershell
cd "C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\MaskingTest"
.\.venv311\Scripts\python.exe model_v2\predict_model_v2_reconstruction_error.py
```

Output folder:

```text
C:\Users\gianl\ThesisOutputs\HerculesFiles\outputs\model_v2_reconstruction_error_predictions
```

The generated four-panel image shows:

```text
Clean BEV
Reconstructed clean BEV
Faulty BEV
Actual mask vs reconstruction-error predicted mask
```

Overlay colors:

```text
red      actual fault mask
blue     predicted fault mask
magenta  overlap between actual and predicted
```

## Legacy Direct-Mask Visualizer

The file `predict_model_v2.py` is kept for compatibility with older direct-mask Model V2 checkpoints.

It automatically detects whether the checkpoint is the newer conditioned reconstruction-error model. If so, it redirects to `predict_model_v2_reconstruction_error.py`.

Run:

```powershell
.\.venv311\Scripts\python.exe model_v2\predict_model_v2.py
```

## Important Interpretation

For Model V2:

```text
Clean BEV is the training target, not the model input.
```

The model learns restoration from:

```text
faulty BEV + injected fault metadata
```

The fault metadata is provided by the injection process:

```text
fault type
severity
```

The model is not trained to classify the fault type or severity. It receives that information as conditioning so it can learn how each known fault category differs from the corresponding clean BEV.

## Recommended Supercomputer Run

For a serious training run, increase:

```text
number of samples
number of BEV frames
number of epochs
```

The local 5-epoch run is only a sanity check. It verifies that the model produces a reconstruction attempt and that the reconstruction-error mask pipeline works. It is not enough to evaluate final restoration quality.
