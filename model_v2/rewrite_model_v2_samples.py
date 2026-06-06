from pathlib import Path
import argparse
import json
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "outputs"
DEFAULT_BEV_DIR = DEFAULT_OUTPUT_ROOT / "bev_model_v2_all_frames"
DEFAULT_DATASET_DIR = DEFAULT_OUTPUT_ROOT / "model_v2_dataset"
FAULT_TYPES = ["laser", "photodetector", "scanning", "optical", "window", "mounting"]
SEVERITIES = ["mild", "moderate", "severe"]


def run_step(name: str, command: list[str]):
    print(f"\n=== {name} ===")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def count_bev_files(bev_dir: Path) -> int:
    return sum(1 for path in bev_dir.rglob("*.npz") if path.name != "manifest.npz")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite Model V2 training samples from a full HeRCULES dataset. "
            "This builds clean BEVs from every LiDAR frame, then injects "
            "HeRCULES LiDAR faults into those BEVs."
        )
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root HeRCULES folder containing scene folders such as Bridge01_Day.",
    )
    parser.add_argument("--bev-dir", default=str(DEFAULT_BEV_DIR))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument(
        "--rewrite-bev",
        action="store_true",
        help="Rebuild the BEV pool even if BEV files already exist.",
    )
    parser.add_argument(
        "--aggregate-scans",
        type=int,
        default=3,
        help="Number of LiDAR timesteps to motion-compensate and aggregate per BEV.",
    )
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=80.0)
    parser.add_argument("--y-min", type=float, default=-40.0)
    parser.add_argument("--y-max", type=float, default=40.0)
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--z-min", type=float, default=-4.0)
    parser.add_argument("--z-max", type=float, default=6.0)
    parser.add_argument("--z-resolution", type=float, default=0.5)
    parser.add_argument(
        "--samples-per-combination",
        type=int,
        default=1,
        help=(
            "How many times to repeat each BEV/fault/severity combination. "
            "1 means every BEV gets every fault type and severity once."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help=(
            "Override total sample count. Default is "
            "BEV count * 6 fault types * 3 severities * samples-per-combination."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--compressed-samples",
        action="store_true",
        help="Write compressed .npz samples. Slower but smaller.",
    )
    parser.add_argument(
        "--target-threshold",
        type=float,
        default=0.05,
        help="Channel-difference threshold used to create the target fault mask.",
    )
    parser.add_argument(
        "--frames-per-scene-cap",
        type=int,
        default=1_000_000,
        help="High cap used with stride=1. Lower this for quick tests.",
    )
    args = parser.parse_args()

    python_exe = sys.executable
    bev_dir = Path(args.bev_dir)
    dataset_dir = Path(args.dataset_dir)

    existing_bev_count = count_bev_files(bev_dir) if bev_dir.exists() else 0
    if args.rewrite_bev or existing_bev_count == 0:
        run_step(
            "1. Build BEV pool from every available LiDAR frame",
            [
                python_exe,
                str(PROJECT_DIR / "build_bev_dataset.py"),
                "--data-root",
                args.data_root,
                "--output-dir",
                str(bev_dir),
                "--frames-per-scene",
                str(args.frames_per_scene_cap),
                "--stride",
                "1",
                "--aggregate-scans",
                str(args.aggregate_scans),
                "--x-min",
                str(args.x_min),
                "--x-max",
                str(args.x_max),
                "--y-min",
                str(args.y_min),
                "--y-max",
                str(args.y_max),
                "--resolution",
                str(args.resolution),
                "--z-min",
                str(args.z_min),
                "--z-max",
                str(args.z_max),
                "--z-resolution",
                str(args.z_resolution),
            ],
        )
    else:
        print(f"Reusing existing BEV pool: {bev_dir}")
        print(f"Existing BEV files: {existing_bev_count}")

    bev_count = count_bev_files(bev_dir)
    if bev_count == 0:
        raise RuntimeError(f"No BEV files were found in {bev_dir}")

    combination_count = len(FAULT_TYPES) * len(SEVERITIES)
    num_samples = args.num_samples
    if num_samples is None:
        num_samples = bev_count * combination_count * args.samples_per_combination

    make_dataset_command = [
        python_exe,
        str(PROJECT_DIR / "make_autoencoder_dataset.py"),
        "--bev-dir",
        str(bev_dir),
        "--output-dir",
        str(dataset_dir),
        "--num-samples",
        str(num_samples),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--realistic-target-threshold",
        str(args.target_threshold),
        "--balanced-fault-grid",
    ]
    for fault_type in FAULT_TYPES:
        make_dataset_command.extend(["--realistic-fault", fault_type])
    for severity in SEVERITIES:
        make_dataset_command.extend(["--realistic-fault-severity", severity])
    if args.compressed_samples:
        make_dataset_command.append("--compressed-samples")

    run_step("2. Rewrite faulty/clean paired Model V2 samples", make_dataset_command)

    summary = {
        "data_root": args.data_root,
        "bev_dir": str(bev_dir),
        "dataset_dir": str(dataset_dir),
        "bev_count": bev_count,
        "fault_types": FAULT_TYPES,
        "severities": SEVERITIES,
        "fault_severity_combinations_per_bev": combination_count,
        "samples_per_combination": args.samples_per_combination,
        "num_samples": num_samples,
        "aggregate_scans": args.aggregate_scans,
        "stride": 1,
    }
    summary_path = dataset_dir / "model_v2_rewrite_summary.json"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote rewrite summary: {summary_path}")
    print(f"BEV files used: {bev_count}")
    print(f"Samples written: {num_samples}")


if __name__ == "__main__":
    main()
