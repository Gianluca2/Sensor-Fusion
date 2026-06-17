from pathlib import Path
import argparse
import subprocess
import sys


DEFAULT_REPO_DIR = Path(r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\MaskingTest")
DEFAULT_DATA_ROOT = Path(r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\Data")
DEFAULT_OUTPUT_ROOT = Path(r"C:\Users\gianl\ThesisOutputs\HerculesFiles\outputs\model_v5_initial_5k")


def run_step(name: str, command: list[str], cwd: Path):
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the initial local Model V5 reliability-map test.")
    parser.add_argument("--repo-dir", default=str(DEFAULT_REPO_DIR))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--target-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--lidar-aggregate-scans", type=int, default=3)
    parser.add_argument("--radar-aggregate-scans", type=int, default=50)
    parser.add_argument("--heatmap-blur-iterations", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    dataset_dir = output_root / "dataset"
    model_dir = output_root / "models"
    prediction_dir = output_root / "predictions"
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    run_step(
        "Environment check",
        [
            args.python,
            "-c",
            (
                "import sys; print('python:', sys.executable); "
                "import torch; print('torch:', torch.__version__); "
                "print('cuda:', torch.cuda.is_available()); "
                "print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
            ),
        ],
        repo_dir,
    )

    if not args.skip_generation:
        run_step(
            "1. Generate V5 regional-reliability samples",
            [
                args.python,
                str(repo_dir / "model_v5" / "generate_model_v5_samples_controlled.py"),
                "--repo-dir",
                str(repo_dir),
                "--data-root",
                str(data_root),
                "--dataset-dir",
                str(dataset_dir),
                "--target-samples",
                str(args.target_samples),
                "--chunk-size",
                str(args.chunk_size),
                "--lidar-aggregate-scans",
                str(args.lidar_aggregate_scans),
                "--radar-aggregate-scans",
                str(args.radar_aggregate_scans),
                "--heatmap-blur-iterations",
                str(args.heatmap_blur_iterations),
            ],
            repo_dir,
        )

    if not args.skip_training:
        run_step(
            "2. Train V5 reliability model",
            [
                args.python,
                str(repo_dir / "model_v5" / "train_model_v5.py"),
                "--dataset-dir",
                str(dataset_dir),
                "--model-path",
                str(model_dir / "model_v5_best.pt"),
                "--latest-model-path",
                str(model_dir / "model_v5_latest.pt"),
                "--metrics-path",
                str(model_dir / "model_v5_metrics.csv"),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--loader-workers",
                "0",
                "--channel-normalization",
                "dataset",
                "--normalization-samples",
                "2048",
                "--threshold",
                "0.65",
                "--soft-weight",
                "1.0",
                "--bce-weight",
                "0.5",
                "--dice-weight",
                "1.0",
                "--range-loss-weight",
                "0.25",
                "--range-channel-index",
                "4",
                "--base-channels",
                str(args.base_channels),
                "--early-stop-patience",
                str(args.epochs),
            ],
            repo_dir,
        )

    run_step(
        "3. Write V5 prediction PNGs",
        [
            args.python,
            str(repo_dir / "model_v5" / "predict_model_v5.py"),
            "--model-path",
            str(model_dir / "model_v5_best.pt"),
            "--sample-dir",
            str(dataset_dir),
            "--output-dir",
            str(prediction_dir),
            "--num-outputs",
            "30",
            "--threshold",
            "0.65",
        ],
        repo_dir,
    )


if __name__ == "__main__":
    main()
