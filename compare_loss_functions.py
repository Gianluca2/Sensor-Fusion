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
DEFAULT_OUTPUT_DIR = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\models"
)


def run_step(label: str, command: list[str]):
    print(f"\n=== {label} ===")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def build_lidar_bevs(args):
    if args.skip_bev:
        return

    command = [
        sys.executable,
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
        command.append("--evenly-spaced")

    run_step("Build LiDAR-only BEV files", command)


def build_lidar_dataset(args):
    if args.skip_dataset:
        return

    run_step(
        "Build LiDAR-only autoencoder samples",
        [
            sys.executable,
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


def run_training(loss_name: str, args):
    output_dir = Path(args.output_dir)
    model_path = output_dir / f"bev_autoencoder_{loss_name}.pt"
    metrics_path = output_dir / f"autoencoder_metrics_{loss_name}.csv"

    command = [
        sys.executable,
        str(PROJECT_DIR / "train_autoencoder.py"),
        "--dataset-dir",
        args.dataset_dir,
        "--model-path",
        str(model_path),
        "--metrics-path",
        str(metrics_path),
        "--loss",
        loss_name,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--base-channels",
        str(args.base_channels),
        "--dropout",
        str(args.dropout),
        "--depth",
        str(args.depth),
        "--lr",
        str(args.lr),
        "--val-fraction",
        str(args.val_fraction),
        "--seed",
        str(args.seed),
        "--error-percentile",
        str(args.error_percentile),
    ]

    print(f"\n=== Training autoencoder with {loss_name} reconstruction loss ===")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_DIR, check=True)

    return model_path, metrics_path


def main():
    parser = argparse.ArgumentParser(
        description="Build LiDAR-only data and compare autoencoder reconstruction losses."
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--bev-dir", default=DEFAULT_BEV_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--error-percentile", type=float, default=98.0)
    parser.add_argument("--min-mask-occupied-cells", type=int, default=50)
    parser.add_argument("--min-mask-area-fraction", type=float, default=0.01)
    parser.add_argument("--max-mask-area-fraction", type=float, default=0.03)
    parser.add_argument("--bev-frames-per-scene", type=int, default=500)
    parser.add_argument("--bev-stride", type=int, default=10)
    parser.add_argument("--evenly-spaced-bev", action="store_true")
    parser.add_argument("--aggregate-scans", type=int, default=3)
    parser.add_argument("--z-min", type=float, default=-4.0)
    parser.add_argument("--z-max", type=float, default=6.0)
    parser.add_argument("--z-resolution", type=float, default=0.5)
    parser.add_argument(
        "--losses",
        nargs="+",
        default=["l1", "mse", "l1_mse"],
        choices=["l1", "mse", "l1_mse"],
        help="Reconstruction losses to train and compare.",
    )
    parser.add_argument("--skip-bev", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    build_lidar_bevs(args)
    build_lidar_dataset(args)

    results = [run_training(loss_name, args) for loss_name in args.losses]

    print("\nDone. Compare these metrics files:")
    for model_path, metrics_path in results:
        print(f"  model: {model_path}")
        print(f"  metrics: {metrics_path}")

    print("\nFor the autoencoder, lower validation reconstruction loss is the main training signal.")
    print("The IoU/F1 columns are computed after thresholding reconstruction error.")


if __name__ == "__main__":
    main()
