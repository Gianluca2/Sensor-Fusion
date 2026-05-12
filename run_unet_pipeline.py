from pathlib import Path
import argparse
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\unet_dataset"
)
DEFAULT_MODEL_PATH = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\models"
    r"\unet_bev_mask.pt"
)
DEFAULT_PREDICTION_OUTPUT = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs"
    r"\unet_predictions\sample_000000_overlay.png"
)
DEFAULT_PREDICTION_DIR = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs"
    r"\unet_predictions"
)


def run_step(label: str, command: list[str]):
    print(f"\n=== {label} ===")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run BEV projection, U-Net dataset creation, training, and prediction."
    )
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-mask-occupied-cells", type=int, default=50)
    parser.add_argument("--max-mask-area-fraction", type=float, default=0.08)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prediction-output", default=DEFAULT_PREDICTION_OUTPUT)
    parser.add_argument("--prediction-dir", default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--num-prediction-outputs", type=int, default=10)
    parser.add_argument(
        "--skip-bev",
        action="store_true",
        help="Skip BEV generation and use the existing bev_match_000000.npz.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and use the existing saved model.",
    )
    args = parser.parse_args()

    python = sys.executable

    if not args.skip_bev:
        run_step(
            "1. Create BEV projection",
            [python, str(PROJECT_DIR / "bev_projection.py")],
        )

    run_step(
        "2. Create synthetic U-Net training samples",
        [
            python,
            str(PROJECT_DIR / "make_unet_dataset.py"),
            "--num-samples",
            str(args.num_samples),
            "--output-dir",
            args.dataset_dir,
            "--min-mask-occupied-cells",
            str(args.min_mask_occupied_cells),
            "--max-mask-area-fraction",
            str(args.max_mask_area_fraction),
        ],
    )

    if not args.skip_training:
        run_step(
            "3. Train small U-Net",
            [
                python,
                str(PROJECT_DIR / "train_unet.py"),
                "--dataset-dir",
                args.dataset_dir,
                "--model-path",
                args.model_path,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
            ],
        )

    run_step(
        "4. Predict mask and save red/blue overlay",
        [
            python,
            str(PROJECT_DIR / "predict_unet.py"),
            "--model-path",
            args.model_path,
            "--sample-dir",
            args.dataset_dir,
            "--output-dir",
            args.prediction_dir,
            "--num-outputs",
            str(args.num_prediction_outputs),
        ],
    )

    print("\nPipeline complete.")
    print(f"Prediction overlays: {args.prediction_dir}")
    print("Colors: red=actual mask, blue=predicted mask, magenta=overlap")


if __name__ == "__main__":
    main()
