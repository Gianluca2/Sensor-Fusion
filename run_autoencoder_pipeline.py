from pathlib import Path
import argparse
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\autoencoder_dataset"
)
DEFAULT_BEV_DIR = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\bev_autoencoder"
)
DEFAULT_MODEL_PATH = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\models"
    r"\bev_autoencoder.pt"
)
DEFAULT_PREDICTION_DIR = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs"
    r"\autoencoder_predictions"
)


def run_step(label: str, command: list[str]):
    print(f"\n=== {label} ===")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run LiDAR-only BEV creation, masked-sample creation, autoencoder training, and prediction."
    )
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--loss", default="l1_mse", choices=["l1", "mse", "l1_mse"])
    parser.add_argument("--error-threshold", type=float, default=None)
    parser.add_argument("--error-percentile", type=float, default=98.0)
    parser.add_argument("--min-mask-occupied-cells", type=int, default=50)
    parser.add_argument("--min-mask-area-fraction", type=float, default=0.01)
    parser.add_argument("--max-mask-area-fraction", type=float, default=0.03)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--bev-dir", default=DEFAULT_BEV_DIR)
    parser.add_argument("--bev-frames-per-scene", type=int, default=500)
    parser.add_argument("--bev-stride", type=int, default=10)
    parser.add_argument("--evenly-spaced-bev", action="store_true")
    parser.add_argument("--aggregate-scans", type=int, default=3)
    parser.add_argument("--z-min", type=float, default=-4.0)
    parser.add_argument("--z-max", type=float, default=6.0)
    parser.add_argument("--z-resolution", type=float, default=0.5)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prediction-dir", default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--num-prediction-outputs", type=int, default=10)
    parser.add_argument("--skip-bev", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    python = sys.executable

    if not args.skip_bev:
        bev_command = [
            python,
            str(PROJECT_DIR / "build_bev_dataset.py"),
            "--output-dir",
            args.bev_dir,
            "--frames-per-scene",
            str(args.bev_frames_per_scene),
            "--stride",
            str(args.bev_stride),
            "--aggregate-scans",
            str(args.aggregate_scans),
            "--z-min",
            str(args.z_min),
            "--z-max",
            str(args.z_max),
            "--z-resolution",
            str(args.z_resolution),
        ]
        if args.evenly_spaced_bev:
            bev_command.append("--evenly-spaced")
        run_step("1. Create BEV projection", bev_command)

    run_step(
        "2. Create synthetic masked reconstruction samples",
        [
            python,
            str(PROJECT_DIR / "make_autoencoder_dataset.py"),
            "--num-samples",
            str(args.num_samples),
            "--output-dir",
            args.dataset_dir,
            "--bev-dir",
            args.bev_dir,
            "--min-mask-occupied-cells",
            str(args.min_mask_occupied_cells),
            "--min-mask-area-fraction",
            str(args.min_mask_area_fraction),
            "--max-mask-area-fraction",
            str(args.max_mask_area_fraction),
        ],
    )

    if not args.skip_training:
        run_step(
            "3. Train BEV autoencoder",
            [
                python,
                str(PROJECT_DIR / "train_autoencoder.py"),
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
                "--dropout",
                str(args.dropout),
                "--depth",
                str(args.depth),
                "--val-fraction",
                str(args.val_fraction),
                "--loss",
                args.loss,
                "--error-percentile",
                str(args.error_percentile),
            ] + (
                ["--error-threshold", str(args.error_threshold)]
                if args.error_threshold is not None
                else []
            ),
        )

    run_step(
        "4. Detect mask from reconstruction error and save overlays",
        [
            python,
            str(PROJECT_DIR / "predict_autoencoder.py"),
            "--model-path",
            args.model_path,
            "--sample-dir",
            args.dataset_dir,
            "--output-dir",
            args.prediction_dir,
            "--num-outputs",
            str(args.num_prediction_outputs),
            "--error-percentile",
            str(args.error_percentile),
        ] + (
            ["--error-threshold", str(args.error_threshold)]
            if args.error_threshold is not None
            else []
        ),
    )

    print("\nAutoencoder pipeline complete.")
    print(f"Prediction overlays: {args.prediction_dir}")
    print("Colors: red=actual mask, blue=predicted error region, magenta=overlap")


if __name__ == "__main__":
    main()
