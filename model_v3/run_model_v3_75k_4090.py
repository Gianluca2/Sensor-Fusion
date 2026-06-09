from pathlib import Path
import argparse
import subprocess
import sys


DEFAULT_REPO_DIR = Path("/mnt/3D10B36523559581/Gianluca/Sensor-Fusion")
DEFAULT_DATA_ROOT = Path("/mnt/3D10B36523559581/HeRCULES")
DEFAULT_OUTPUT_ROOT = Path("/mnt/3D10B36523559581/Gianluca/model_v3_outputs_75k")


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


def build_samples_in_chunks(args, repo_dir: Path, data_root: Path, dataset_dir: Path):
    remaining = args.num_samples
    start_index = 0
    while remaining > 0:
        chunk_samples = min(args.sample_chunk_size, remaining)
        rewrite_command = [
            args.python,
            str(repo_dir / "model_v3" / "rewrite_model_v3_samples.py"),
            "--data-root",
            str(data_root),
            "--dataset-dir",
            str(dataset_dir),
            "--num-samples",
            str(chunk_samples),
            "--start-index",
            str(start_index),
            "--aggregate-scans",
            str(args.aggregate_scans),
            "--keep-existing",
            "--skip-existing",
        ]
        if not args.uncompressed_samples:
            rewrite_command.append("--compressed-samples")

        chunk_end = start_index + chunk_samples - 1
        run_step(
            f"1. Build/rewrite Model V3 sample chunk {start_index}-{chunk_end}",
            rewrite_command,
            repo_dir,
        )
        start_index += chunk_samples
        remaining -= chunk_samples


def training_command(args, repo_dir: Path, dataset_dir: Path, best_model_path: Path, latest_model_path: Path, metrics_path: Path):
    command = [
        args.python,
        str(repo_dir / "model_v3" / "train_model_v3.py"),
        "--dataset-dir",
        str(dataset_dir),
        "--model-path",
        str(best_model_path),
        "--latest-model-path",
        str(latest_model_path),
        "--metrics-path",
        str(metrics_path),
        "--batch-size",
        str(args.batch_size),
        "--loader-workers",
        str(args.loader_workers),
        "--epochs",
        str(args.epochs),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--channel-normalization",
        "dataset",
        "--normalization-samples",
        str(args.normalization_samples),
        "--threshold",
        str(args.threshold),
        "--positive-weight",
        str(args.positive_weight),
        "--negative-weight",
        str(args.negative_weight),
    ]
    if latest_model_path.exists() and not args.restart_training:
        command.extend(["--resume-from", str(latest_model_path)])
    return command


def main():
    parser = argparse.ArgumentParser(
        description="Run the 75k-sample, 75-epoch Model V3 experiment on the RTX 4090 Linux machine."
    )
    parser.add_argument("--repo-dir", default=str(DEFAULT_REPO_DIR))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--num-samples", type=int, default=75000)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--aggregate-scans", type=int, default=3)
    parser.add_argument("--sample-chunk-size", type=int, default=2500)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--positive-weight", type=float, default=3.0)
    parser.add_argument("--negative-weight", type=float, default=1.5)
    parser.add_argument("--normalization-samples", type=int, default=4096)
    parser.add_argument("--num-prediction-outputs", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=75)
    parser.add_argument(
        "--restart-training",
        action="store_true",
        help="Ignore model_v3_75k_latest.pt and train from epoch 1.",
    )
    parser.add_argument(
        "--uncompressed-samples",
        action="store_true",
        help="Write uncompressed .npz samples. Default is compressed to reduce disk usage.",
    )
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    dataset_dir = output_root / "model_v3_dataset_75k"
    model_dir = output_root / "models"
    prediction_dir = output_root / "model_v3_75k_predictions"
    best_model_path = model_dir / "model_v3_75k_best.pt"
    latest_model_path = model_dir / "model_v3_75k_latest.pt"
    metrics_path = model_dir / "model_v3_75k_training_metrics.csv"

    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repository: {repo_dir}")
    print(f"HeRCULES data: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Dataset: {dataset_dir}")
    print(f"Best model: {best_model_path}")
    print(f"Latest epoch model: {latest_model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {prediction_dir}")
    print(f"Python: {args.python}")

    check_environment(args.python, repo_dir)

    build_samples_in_chunks(args, repo_dir, data_root, dataset_dir)

    run_step(
        "2. Train Model V3 for 75 epochs",
        training_command(args, repo_dir, dataset_dir, best_model_path, latest_model_path, metrics_path),
        repo_dir,
    )

    run_step(
        "3. Write example prediction plots",
        [
            args.python,
            str(repo_dir / "model_v3" / "predict_model_v3.py"),
            "--model-path",
            str(best_model_path),
            "--sample-dir",
            str(dataset_dir),
            "--output-dir",
            str(prediction_dir),
            "--num-outputs",
            str(args.num_prediction_outputs),
            "--threshold",
            str(args.threshold),
        ],
        repo_dir,
    )

    print("\nDone.")
    print(f"Best model: {best_model_path}")
    print(f"Latest model: {latest_model_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Prediction PNGs: {prediction_dir}")


if __name__ == "__main__":
    main()
