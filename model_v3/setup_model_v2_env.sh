#!/usr/bin/env bash
set -euo pipefail

# Non-invasive local Python environment setup for Model V2.
# Run from the repository root:
#   bash model_v2/setup_model_v2_env.sh

REPO_DIR="${REPO_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv_model_v2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

cd "$REPO_DIR"

echo "Repository: $REPO_DIR"
echo "Virtual env: $VENV_DIR"
echo "Python: $PYTHON_BIN"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Install CUDA PyTorch first so pip resolves the GPU build from the PyTorch index.
python -m pip install torch --index-url "$TORCH_INDEX_URL"
python -m pip install -r model_v2/requirements_model_v2.txt

python - <<'PY'
import sys
import numpy
from PIL import Image
import torch

print("python executable:", sys.executable)
print("numpy:", numpy.__version__)
print("pillow:", Image.__version__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
print("cuda version:", torch.version.cuda)
PY

cat <<EOF

Setup complete.

In VS Code:
  1. Press Ctrl+Shift+P
  2. Choose "Python: Select Interpreter"
  3. Select:
     $VENV_DIR/bin/python

To run the full pipeline with this environment:
  source "$VENV_DIR/bin/activate"
  python model_v2/run_model_v2_4090.py

EOF
