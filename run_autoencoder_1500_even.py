from pathlib import Path
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs")

BEV_DIR = OUTPUT_ROOT / "bev_autoencoder_1500_even_30_epochs"
DATASET_DIR = OUTPUT_ROOT / "autoencoder_1500_even_30_epochs"
MODEL_PATH = OUTPUT_ROOT / "models" / "bev_autoencoder_1500_even_30_epochs.pt"
PREDICTION_DIR = OUTPUT_ROOT / "autoencoder_1500_even_30_epochs_predictions"


def run_step(label: str, command: list[str]):
    print(f"\n=== {label} ===")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def main():
    python = sys.executable

    run_step(
        "Run autoencoder pipeline with 1500 evenly spaced BEV frames",
        [
            python,
            str(PROJECT_DIR / "run_autoencoder_pipeline.py"),
            "--epochs",
            "30",
            "--batch-size",
            "1",
            "--num-samples",
            "5000",
            "--val-fraction",
            "0.20",
            "--bev-frames-per-scene",
            "500",
            "--bev-stride",
            "10",
            "--evenly-spaced-bev",
            "--aggregate-scans",
            "3",
            "--depth",
            "4",
            "--dropout",
            "0.10",
            "--lr",
            "0.0003",
            "--loss",
            "l1_mse",
            "--error-percentile",
            "98.0",
            "--min-mask-occupied-cells",
            "50",
            "--min-mask-area-fraction",
            "0.01",
            "--max-mask-area-fraction",
            "0.03",
            "--bev-dir",
            str(BEV_DIR),
            "--dataset-dir",
            str(DATASET_DIR),
            "--model-path",
            str(MODEL_PATH),
            "--prediction-dir",
            str(PREDICTION_DIR),
            "--num-prediction-outputs",
            "10",
        ],
    )

    print("\nAutoencoder test complete.")
    print(f"Model: {MODEL_PATH}")
    print(f"Predictions: {PREDICTION_DIR}")
    print("BEV frames: up to 1500 = 500 evenly spaced frames per scene * 3 scenes")
    print("Training samples: 4000 = 5000 samples * 0.80 train split")
    print("Validation samples: 1000 = 5000 samples * 0.20 validation split")
    print("Epochs: 30")


if __name__ == "__main__":
    main()
