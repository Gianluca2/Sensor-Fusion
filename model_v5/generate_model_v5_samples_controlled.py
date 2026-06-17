from pathlib import Path
import argparse
import subprocess
import sys
import time


DEFAULT_REPO_DIR = Path("/mnt/3D10B36523559581/Gianluca/Sensor-Fusion")
DEFAULT_DATA_ROOT = Path("/mnt/3D10B36523559581/HeRCULES")
DEFAULT_DATASET_DIR = Path("/mnt/3D10B36523559581/Gianluca/model_v5_outputs/model_v5_dataset")


def count_samples(dataset_dir: Path) -> int:
    return sum(1 for _ in dataset_dir.glob("sample_*.npz"))


def run_chunk(command: list[str], cwd: Path) -> bool:
    print(" ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Generate Model V5 soft-reliability samples in controlled resumable chunks."
    )
    parser.add_argument("--repo-dir", default=str(DEFAULT_REPO_DIR))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--target-samples", type=int, default=75000)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument(
        "--aggregate-scans",
        type=int,
        default=3,
        help="Backward-compatible default used for both LiDAR and radar aggregation unless overridden.",
    )
    parser.add_argument("--lidar-aggregate-scans", type=int, default=None)
    parser.add_argument("--radar-aggregate-scans", type=int, default=None)
    parser.add_argument("--heatmap-blur-iterations", type=int, default=6)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--uncompressed-samples",
        action="store_true",
        help="Write uncompressed .npz files. Default is compressed to reduce disk usage.",
    )
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir)
    data_root = Path(args.data_root)
    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if args.lidar_aggregate_scans is None:
        args.lidar_aggregate_scans = args.aggregate_scans
    if args.radar_aggregate_scans is None:
        args.radar_aggregate_scans = args.aggregate_scans

    print(f"Repository: {repo_dir}", flush=True)
    print(f"HeRCULES data: {data_root}", flush=True)
    print(f"Dataset dir: {dataset_dir}", flush=True)
    print(f"Target samples: {args.target_samples}", flush=True)
    print(f"Chunk size: {args.chunk_size}", flush=True)
    print(f"LiDAR aggregate scans: {args.lidar_aggregate_scans}", flush=True)
    print(f"Radar aggregate scans: {args.radar_aggregate_scans}", flush=True)
    print(f"Heatmap blur iterations: {args.heatmap_blur_iterations}", flush=True)
    print(f"Existing samples before start: {count_samples(dataset_dir)}", flush=True)

    for start_index in range(args.start_index, args.target_samples, args.chunk_size):
        chunk_size = min(args.chunk_size, args.target_samples - start_index)
        chunk_end = start_index + chunk_size - 1
        first_path = dataset_dir / f"sample_{start_index:06d}.npz"
        last_path = dataset_dir / f"sample_{chunk_end:06d}.npz"

        if first_path.exists() and last_path.exists():
            print(
                f"Skipping completed-looking chunk {start_index}-{chunk_end}; "
                f"existing count={count_samples(dataset_dir)}",
                flush=True,
            )
            continue

        command = [
            args.python,
            str(repo_dir / "model_v5" / "rewrite_model_v5_samples.py"),
            "--data-root",
            str(data_root),
            "--dataset-dir",
            str(dataset_dir),
            "--num-samples",
            str(chunk_size),
            "--start-index",
            str(start_index),
            "--aggregate-scans",
            str(args.aggregate_scans),
            "--lidar-aggregate-scans",
            str(args.lidar_aggregate_scans),
            "--radar-aggregate-scans",
            str(args.radar_aggregate_scans),
            "--heatmap-blur-iterations",
            str(args.heatmap_blur_iterations),
            "--keep-existing",
            "--skip-existing",
        ]
        if not args.uncompressed_samples:
            command.append("--compressed-samples")

        print(f"\n=== Generating chunk {start_index}-{chunk_end} ===", flush=True)
        for attempt in range(1, args.max_retries + 1):
            print(f"Attempt {attempt}/{args.max_retries}", flush=True)
            if run_chunk(command, repo_dir):
                break
            if attempt == args.max_retries:
                existing = count_samples(dataset_dir)
                raise RuntimeError(
                    f"Chunk {start_index}-{chunk_end} failed after {args.max_retries} attempts. "
                    f"Existing sample count is {existing}. Rerun this script to continue."
                )
            time.sleep(args.retry_sleep_seconds)

        print(
            f"Completed chunk {start_index}-{chunk_end}; "
            f"existing count={count_samples(dataset_dir)}",
            flush=True,
        )

    final_count = count_samples(dataset_dir)
    print(f"\nDone. Existing sample count: {final_count}/{args.target_samples}", flush=True)
    if final_count < args.target_samples:
        print("Some samples are still missing. Rerun the same command to continue.", flush=True)


if __name__ == "__main__":
    main()


