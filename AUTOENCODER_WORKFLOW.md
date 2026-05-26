# BEV Autoencoder Fault Detection

This project now uses a LiDAR-only BEV autoencoder for synthetic black-patch fault detection.

The pipeline is:

1. Build LiDAR BEV tensors from HeRCULES Aeva frames.
2. Create synthetic corrupted samples by inserting a black rectangular mask into occupied BEV regions.
3. Train an autoencoder to reconstruct the clean BEV tensor from the corrupted input.
4. Compute reconstruction error between the autoencoder output and corrupted input.
5. Threshold the highest-error cells to estimate the faulty region.
6. Save PNG overlays where red is the actual mask, blue is the predicted error region, and magenta is overlap.

Main command:

```powershell
.\.venv311\Scripts\python.exe run_autoencoder_1500_even.py
```

Core files:

- `autoencoder_model.py`: convolutional BEV autoencoder.
- `make_autoencoder_dataset.py`: synthetic masked-sample creation.
- `train_autoencoder.py`: reconstruction training and metric logging.
- `predict_autoencoder.py`: reconstruction-error prediction and PNG output.
- `validate_autoencoder_random.py`: fresh random-mask validation.
- `run_autoencoder_pipeline.py`: full configurable pipeline runner.
- `run_autoencoder_1500_even.py`: current thesis experiment preset.
