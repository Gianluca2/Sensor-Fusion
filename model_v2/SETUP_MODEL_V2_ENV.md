# Model V2 Dependency Setup

Use this on the Linux RTX 4090 machine from the repository root:

```bash
cd /mnt/3D10B36523559581/Gianluca/Sensor-Fusion
bash model_v2/setup_model_v2_env.sh
```

The script creates a local virtual environment:

```text
/mnt/3D10B36523559581/Gianluca/Sensor-Fusion/.venv_model_v2
```

It installs:

```text
torch with CUDA 12.1 wheels
numpy
pillow
```

After setup, select the interpreter in VS Code:

```text
Ctrl+Shift+P
Python: Select Interpreter
.venv_model_v2/bin/python
```

Then run the full pipeline:

```bash
source .venv_model_v2/bin/activate
python model_v2/run_model_v2_4090.py
```

Quick smoke test:

```bash
source .venv_model_v2/bin/activate
python model_v2/run_model_v2_4090.py --quick
```

Check CUDA:

```bash
source .venv_model_v2/bin/activate
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
PY
```

If VS Code still reports `import torch could not be resolved`, it is using the wrong interpreter. Select `.venv_model_v2/bin/python`.
