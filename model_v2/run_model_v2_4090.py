from pathlib import Path
import argparse
import subprocess
import sys


DEFAULT_REPO_DIR = Path("/mnt/3D10B36523559581/Gianluca/Sensor-Fusion")
DEFAULT_DATA_ROOT = Path("/mnt/3D10B36523559581/HeRCULES")
DEFAULT_OUTPUT_ROOT = Path("/mnt/3D10B36523559581/Gianluca/model_v2_outputs")


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
        description="Run the full Model V2 pipeline on the RTX 4090 Linux machine."
    )
    parser.add_argument("--repo-dir", default=str(DEFAULT_REPO_DIR))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--rewrite-bev", action="store_true", default=True)
    parser.add_argument(
        "--reuse-bev",
        action="store_true",
        help="Reuse existing BEV files instead of rebuilding the BEV pool.",
    )
    parser.add_argument("--sample-workers", type=int, default=12)
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--aggregate-scans", type=int, default=3)
    parser.add_argument("--frames-per-scene-cap", type=int, default=1_000_000)
    parser.add_argument("--quick", action="store_true", help="Run a small smoke test.")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    if args.quick:
        output_root = Path(str(output_root) + "_quick")
        args.sample_workers = min(args.sample_workers, 4)
        args.loader_workers = min(args.loader_workers, 4)
        args.batch_size = min(args.batch_size, 4)
        args.epochs = 2
        args.frames_per_scene_cap = 20

    bev_dir = output_root / "bev_all_frames"
    dataset_dir = output_root / "model_v2_dataset"
    model_dir = output_root / "models"
    prediction_dir = output_root / "model_v2_predictions"
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repository: {repo_dir}")
    print(f"HeRCULES data: {data_root}")
    print(f"Output root: {output_root}")
    print(f"Python: {args.python}")

    check_environment(args.python, repo_dir)

    rewrite_command = [
        args.python,
        str(repo_dir / "model_v2" / "rewrite_model_v2_samples.py"),
        "--data-root",
        str(data_root),
        "--bev-dir",
        str(bev_dir),
        "--dataset-dir",
        str(dataset_dir),
        "--num-workers",
        str(args.sample_workers),
        "--aggregate-scans",
        str(args.aggregate_scans),
        "--frames-per-scene-cap",
        str(args.frames_per_scene_cap),
    ]
    if not args.reuse_bev:
        rewrite_command.append("--rewrite-bev")

    run_step("1. Build/rewrite full Model V2 dataset", rewrite_command, repo_dir)

    run_step(
        "2. Train Model V2",
        [
            args.python,
            str(repo_dir / "model_v2" / "train_model_v2.py"),
            "--dataset-dir",
            str(dataset_dir),
            "--model-path",
            str(model_dir / "model_v2.pt"),
            "--metrics-path",
            str(model_dir / "model_v2_training_metrics.csv"),
            "--batch-size",
            str(args.batch_size),
            "--loader-workers",
            str(args.loader_workers),
            "--epochs",
            str(args.epochs),
        ],
        repo_dir,
    )

    run_step(
        "3. Write validation visualizations",
        [
            args.python,
            str(repo_dir / "model_v2" / "predict_model_v2_reconstruction_error.py"),
            "--model-path",
            str(model_dir / "model_v2.pt"),
            "--sample-dir",
            str(dataset_dir),
            "--output-dir",
            str(prediction_dir),
            "--num-outputs",
            "10",
        ],
        repo_dir,
    )

    print("\nDone.")
    print(f"Model: {model_dir / 'model_v2.pt'}")
    print(f"Metrics: {model_dir / 'model_v2_training_metrics.csv'}")
    print(f"Predictions: {prediction_dir}")


if __name__ == "__main__":
    main()
