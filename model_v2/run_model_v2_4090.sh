#!/usr/bin/env bash
set -euo pipefail

# RTX 4090 / i9-13900KS training launcher for Model V2.
# Run from the repository root:
#   bash model_v2/run_model_v2_4090.sh

REPO_DIR="${REPO_DIR:-/mnt/3D10B36523559581/Gianluca/Sensor-Fusion}"
DATA_ROOT="${DATA_ROOT:-/mnt/3D10B36523559581/Gianluca/HeRCULES}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/3D10B36523559581/Gianluca/model_v2_outputs}"

BEV_DIR="${BEV_DIR:-$OUTPUT_ROOT/bev_all_frames}"
DATASET_DIR="${DATASET_DIR:-$OUTPUT_ROOT/model_v2_dataset}"
MODEL_DIR="${MODEL_DIR:-$OUTPUT_ROOT/models}"
PREDICTION_DIR="${PREDICTION_DIR:-$OUTPUT_ROOT/model_v2_predictions}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
REWRITE_BEV="${REWRITE_BEV:-1}"
SAMPLE_WORKERS="${SAMPLE_WORKERS:-12}"
LOADER_WORKERS="${LOADER_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-100}"

mkdir -p "$OUTPUT_ROOT" "$MODEL_DIR" "$PREDICTION_DIR"
cd "$REPO_DIR"

echo "Repository: $REPO_DIR"
echo "HeRCULES data: $DATA_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Python: $PYTHON_BIN"

echo
echo "=== Environment check ==="
"$PYTHON_BIN" - <<'PY'
import sys
print("python:", sys.version)
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
except Exception as exc:
    print("torch check failed:", exc)
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

rewrite_args=()
if [[ "$REWRITE_BEV" == "1" ]]; then
  rewrite_args+=(--rewrite-bev)
fi

echo
echo "=== 1. Build/rewrite full Model V2 dataset ==="
"$PYTHON_BIN" model_v2/rewrite_model_v2_samples.py \
  --data-root "$DATA_ROOT" \
  --bev-dir "$BEV_DIR" \
  --dataset-dir "$DATASET_DIR" \
  --num-workers "$SAMPLE_WORKERS" \
  "${rewrite_args[@]}"

echo
echo "=== 2. Train Model V2 ==="
"$PYTHON_BIN" model_v2/train_model_v2.py \
  --dataset-dir "$DATASET_DIR" \
  --model-path "$MODEL_DIR/model_v2.pt" \
  --metrics-path "$MODEL_DIR/model_v2_training_metrics.csv" \
  --batch-size "$BATCH_SIZE" \
  --loader-workers "$LOADER_WORKERS" \
  --epochs "$EPOCHS"

echo
echo "=== 3. Write validation visualizations ==="
"$PYTHON_BIN" model_v2/predict_model_v2_reconstruction_error.py \
  --model-path "$MODEL_DIR/model_v2.pt" \
  --sample-dir "$DATASET_DIR" \
  --output-dir "$PREDICTION_DIR" \
  --num-outputs 10

echo
echo "Done."
echo "Model: $MODEL_DIR/model_v2.pt"
echo "Metrics: $MODEL_DIR/model_v2_training_metrics.csv"
echo "Predictions: $PREDICTION_DIR"
