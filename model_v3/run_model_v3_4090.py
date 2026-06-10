from pathlib import Path
import argparse
import subprocess
import sys


DEFAULT_REPO_DIR = Path("/mnt/3D10B36523559581/Gianluca/Sensor-Fusion")
DEFAULT_DATA_ROOT = Path("/mnt/3D10B36523559581/HeRCULES")
DEFAULT_OUTPUT_ROOT = Path("/mnt/3D10B36523559581/Gianluca/model_v3_outputs")


def run_step(name: str, command: list[str], cwd: Path):
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def check_environment(python_exe: str, repo_dir: Path):
    run_step(
        "Environment check",
        [
            python_exe,
            "-c",
            (
                "import sys; "
                "print('python:', sys.version); "
                "import torch; "
                "print('torch:', torch.__version__); "
                "print('cuda available:', torch.cuda.is_available()); "
                "print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
            ),
        ],
        repo_dir,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the full Model V3 pipeline on the RTX 4090 Linux machine."
    )
    parser.add_argument("--repo-dir", default=str(DEFAULT_REPO_DIR))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--compressed-samples", action="store_true")
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--aggregate-scans", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--positive-weight", type=float, default=2.0)
    parser.add_argument("--negative-weight", type=float, default=2.2)
    parser.add_argument("--range-loss-weight", type=float, default=1.0)
    parser.add_argument("--range-channel-index", type=int, default=4)
    parser.add_argument("--quick", action="store_true", help="Run a small smoke test.")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    if args.quick:
        output_root = Path(str(output_root) + "_quick")
        args.loader_workers = min(args.loader_workers, 4)
        args.batch_size = min(args.batch_size, 4)
        args.epochs = 2
        args.num_samples = 200

    dataset_dir = output_root / "model_v3_dataset"
    model_dir = output_root / "models"
    prediction_dir = output_root / "model_v3_predictions"
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repository: {repo_dir}")
    print(f"HeRCULES data: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Python: {args.python}")

    check_environment(args.python, repo_dir)

    rewrite_command = [
        args.python,
        str(repo_dir / "model_v3" / "rewrite_model_v3_samples.py"),
        "--data-root",
        str(data_root),
        "--dataset-dir",
        str(dataset_dir),
        "--num-samples",
        str(args.num_samples),
        "--aggregate-scans",
        str(args.aggregate_scans),
    ]
    if args.compressed_samples:
        rewrite_command.append("--compressed-samples")

    run_step("1. Build/rewrite full Model V3 dataset", rewrite_command, repo_dir)

    run_step(
        "2. Train Model V3",
        [
            args.python,
            str(repo_dir / "model_v3" / "train_model_v3.py"),
            "--dataset-dir",
            str(dataset_dir),
            "--model-path",
            str(model_dir / "model_v3.pt"),
            "--metrics-path",
            str(model_dir / "model_v3_training_metrics.csv"),
            "--batch-size",
            str(args.batch_size),
            "--loader-workers",
            str(args.loader_workers),
            "--epochs",
            str(args.epochs),
            "--threshold",
            str(args.threshold),
            "--positive-weight",
            str(args.positive_weight),
            "--negative-weight",
            str(args.negative_weight),
            "--range-loss-weight",
            str(args.range_loss_weight),
            "--range-channel-index",
            str(args.range_channel_index),
        ],
        repo_dir,
    )

    run_step(
        "3. Write validation visualizations",
        [
            args.python,
            str(repo_dir / "model_v3" / "predict_model_v3.py"),
            "--model-path",
            str(model_dir / "model_v3.pt"),
            "--sample-dir",
            str(dataset_dir),
            "--output-dir",
            str(prediction_dir),
            "--num-outputs",
            "10",
            "--threshold",
            str(args.threshold),
        ],
        repo_dir,
    )

    print("\nDone.")
    print(f"Model: {model_dir / 'model_v3.pt'}")
    print(f"Metrics: {model_dir / 'model_v3_training_metrics.csv'}")
    print(f"Predictions: {prediction_dir}")


if __name__ == "__main__":
    main()


