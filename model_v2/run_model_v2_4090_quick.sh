#!/usr/bin/env bash
set -euo pipefail

# Quick path/pipeline smoke test for the RTX 4090 machine.
# It caps BEV generation to 20 frames per scene and trains for 2 epochs.

export REPO_DIR="${REPO_DIR:-/mnt/3D10B36523559581/Gianluca/Sensor-Fusion}"
export DATA_ROOT="${DATA_ROOT:-/mnt/3D10B36523559581/Gianluca/HeRCULES}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/3D10B36523559581/Gianluca/model_v2_outputs_quick}"
export SAMPLE_WORKERS="${SAMPLE_WORKERS:-4}"
export LOADER_WORKERS="${LOADER_WORKERS:-4}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export EPOCHS="${EPOCHS:-2}"

cd "$REPO_DIR"

python3 model_v2/rewrite_model_v2_samples.py \
  --data-root "$DATA_ROOT" \
  --bev-dir "$OUTPUT_ROOT/bev_quick" \
  --dataset-dir "$OUTPUT_ROOT/model_v2_dataset_quick" \
  --rewrite-bev \
  --frames-per-scene-cap 20 \
  --num-workers "$SAMPLE_WORKERS"

python3 model_v2/train_model_v2.py \
  --dataset-dir "$OUTPUT_ROOT/model_v2_dataset_quick" \
  --model-path "$OUTPUT_ROOT/models/model_v2_quick.pt" \
  --metrics-path "$OUTPUT_ROOT/models/model_v2_quick_metrics.csv" \
  --batch-size "$BATCH_SIZE" \
  --loader-workers "$LOADER_WORKERS" \
  --epochs "$EPOCHS"

python3 model_v2/predict_model_v2_reconstruction_error.py \
  --model-path "$OUTPUT_ROOT/models/model_v2_quick.pt" \
  --sample-dir "$OUTPUT_ROOT/model_v2_dataset_quick" \
  --output-dir "$OUTPUT_ROOT/model_v2_predictions_quick" \
  --num-outputs 5

echo "Quick test complete: $OUTPUT_ROOT"
