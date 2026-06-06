from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import random
from types import SimpleNamespace

import numpy as np

from bev_projection import project_lidar_bev
from build_bev_dataset import (
    invert_transform,
    load_poses,
    nearest_by_timestamp,
    pose_to_transform,
    transform_xyz,
)
from hercules_lidar_faults import (
    SUPPORTED_HERCULES_LIDAR_FAULTS,
    apply_hercules_lidar_fault,
)
from build_bev_dataset import read_aeva_bin


FAST_OUTPUT_ROOT = r"C:\Users\gianl\ThesisOutputs\HerculesFiles\outputs"
DEFAULT_BEV = str(Path(FAST_OUTPUT_ROOT) / "bev" / "bev_match_000000.npz")
DEFAULT_OUTPUT_DIR = str(Path(FAST_OUTPUT_ROOT) / "autoencoder_dataset")
DEFAULT_LAYERS = [
    "lidar_density",
    "lidar_height",
    "lidar_occupied_voxel_count",
    "lidar_height_spread",
    "lidar_height_bin_occupancy_ratio",
]
DEFAULT_REALISTIC_FAULTS = ["laser", "photodetector", "scanning", "optical", "window", "mounting"]
DEFAULT_REALISTIC_SEVERITIES = ["mild", "moderate", "severe"]
FAULT_CLASSES = ["laser", "photodetector", "scanning", "optical", "window", "mounting"]
LIDAR_FRAME_CACHE = {}


def load_clean_bev(path: Path, layers):
    with np.load(path) as data:
        missing = [layer for layer in layers if layer not in data.files]
        if missing:
            raise ValueError(f"Missing layers in {path}: {missing}")

        channels = [data[layer].astype(np.float32) for layer in layers]
        metadata = {}
        if "metadata_json" in data.files:
            metadata = json.loads(str(data["metadata_json"].item()))

    return np.stack(channels, axis=0), metadata


def find_bev_files(bev_dir: Path):
    return sorted(
        path for path in bev_dir.rglob("*.npz")
        if path.name != "manifest.npz"
    )


def read_cached_aeva_bin(path: Path):
    cache_key = str(path)
    if cache_key not in LIDAR_FRAME_CACHE:
        LIDAR_FRAME_CACHE[cache_key] = read_aeva_bin(path)
    return LIDAR_FRAME_CACHE[cache_key]


def scene_pose_file(source_metadata: dict) -> Path:
    scene_root = Path(source_metadata["scene_root"])
    pose_path = scene_root / "PR_GT" / "Aeva_gt.txt"
    if not pose_path.exists():
        raise FileNotFoundError(f"Could not find Aeva pose file: {pose_path}")
    return pose_path


def load_faulted_aggregated_lidar(source_metadata: dict, fault_type: str, severity: str, rng):
    if "aggregated_lidar_frames" in source_metadata:
        frames = source_metadata["aggregated_lidar_frames"]
    else:
        frames = [{
            "timestamp": source_metadata["reference_lidar_timestamp"],
            "path": source_metadata["reference_lidar_path"],
        }]

    poses = load_poses(scene_pose_file(source_metadata))
    reference_timestamp = int(source_metadata["reference_lidar_timestamp"])
    reference_pose = nearest_by_timestamp(reference_timestamp, poses)
    reference_to_world = pose_to_transform(reference_pose)
    world_to_reference = invert_transform(reference_to_world)
    aggregated = []

    for frame in frames:
        frame_path = Path(frame["path"])
        frame_timestamp = int(frame["timestamp"])
        frame_pose = nearest_by_timestamp(frame_timestamp, poses)
        sensor_to_world = pose_to_transform(frame_pose)
        sensor_to_reference = world_to_reference @ sensor_to_world

        points = read_cached_aeva_bin(frame_path)
        faulted = apply_hercules_lidar_fault(
            points,
            fault_type=fault_type,
            severity=severity,
            rng=rng,
        )
        transformed_xyz = transform_xyz(faulted[:, :3], sensor_to_reference).astype(np.float32)
        aggregated.append(transformed_xyz)

    return np.vstack(aggregated)


def stack_layers(bev_layers: dict) -> np.ndarray:
    return np.stack([bev_layers[layer].astype(np.float32) for layer in DEFAULT_LAYERS], axis=0)


def difference_target(clean: np.ndarray, faulty: np.ndarray, threshold: float) -> np.ndarray:
    difference = np.max(np.abs(clean - faulty), axis=0)
    return (difference > threshold).astype(np.float32)


def make_hercules_lidar_fault_sample(clean, source_metadata, sample_index: int, args):
    if getattr(args, "balanced_fault_grid", False):
        combo_count = len(args.realistic_faults) * len(args.realistic_fault_severities)
        combo_index = sample_index % combo_count
        fault_type = args.realistic_faults[combo_index // len(args.realistic_fault_severities)]
        severity = args.realistic_fault_severities[combo_index % len(args.realistic_fault_severities)]
    else:
        fault_type = random.choice(args.realistic_faults)
        severity = random.choice(args.realistic_fault_severities)
    rng = np.random.default_rng(args.seed + sample_index)
    faulted_xyz = load_faulted_aggregated_lidar(
        source_metadata,
        fault_type=fault_type,
        severity=severity,
        rng=rng,
    )
    x_range = tuple(source_metadata["x_range_m"])
    y_range = tuple(source_metadata["y_range_m"])
    z_range = tuple(source_metadata["z_range_m"])
    resolution = float(source_metadata["resolution_m_per_cell"])
    z_resolution = float(source_metadata["z_resolution_m_per_voxel"])

    faulty_layers = project_lidar_bev(
        faulted_xyz,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
        z_range=z_range,
        z_resolution=z_resolution,
    )
    faulty = stack_layers(faulty_layers)
    target = difference_target(clean, faulty, args.realistic_target_threshold)

    metadata = {
        "sample_index": sample_index,
        "fault_source": "hercules_lidar",
        "fault_type": fault_type,
        "fault_severity": severity,
        "fault_target_cells": int(np.count_nonzero(target)),
        "fault_target_fraction": float(np.count_nonzero(target) / target.size),
    }
    return faulty, target, metadata


def write_sample(sample_path: Path, faulty, clean, target, sample_metadata, compressed: bool):
    writer = np.savez_compressed if compressed else np.savez
    writer(
        sample_path,
        input=faulty.astype(np.float32),
        clean=clean.astype(np.float32),
        target=target.astype(np.float32),
        metadata_json=json.dumps(sample_metadata, indent=2),
    )


def generate_sample_record(index: int, bev_files: list[str], args_dict: dict):
    args = SimpleNamespace(**args_dict)
    random.seed(args.seed + index)
    np.random.seed(args.seed + index)

    if getattr(args, "balanced_fault_grid", False):
        combo_count = len(args.realistic_faults) * len(args.realistic_fault_severities)
        source_bev = Path(bev_files[(index // combo_count) % len(bev_files)])
    else:
        source_bev = Path(bev_files[index % len(bev_files)])
    clean, source_metadata = load_clean_bev(source_bev, DEFAULT_LAYERS)
    faulty, target, sample_metadata = make_hercules_lidar_fault_sample(
        clean,
        source_metadata,
        index,
        args,
    )

    sample_path = Path(args.output_dir) / f"sample_{index:06d}.npz"
    sample_metadata["source_bev"] = str(source_bev)
    sample_metadata["source_metadata"] = source_metadata
    write_sample(
        sample_path,
        faulty,
        clean,
        target,
        sample_metadata,
        compressed=args.compressed_samples,
    )
    return {
        "path": str(sample_path),
        **sample_metadata,
    }


def clear_existing_samples(output_dir: Path):
    removed = 0
    for path in output_dir.glob("sample_*.npz"):
        path.unlink()
        removed += 1

    return removed


def existing_dataset_is_usable(output_dir: Path, requested_samples: int, args) -> bool:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return False

    sample_count = sum(1 for _ in output_dir.glob("sample_*.npz"))
    if sample_count < requested_samples:
        return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    manifest_severities = manifest.get("realistic_fault_severities")
    if manifest_severities is None and "realistic_fault_severity" in manifest:
        manifest_severities = [manifest["realistic_fault_severity"]]

    return (
        manifest.get("fault_source") == "hercules_lidar"
        and sorted(manifest.get("realistic_faults", [])) == sorted(args.realistic_faults)
        and sorted(manifest_severities or []) == sorted(args.realistic_fault_severities)
        and manifest.get("compressed_samples") == args.compressed_samples
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create synthetic masked BEV samples for autoencoder reconstruction training."
    )
    parser.add_argument("--bev", default=DEFAULT_BEV)
    parser.add_argument(
        "--bev-dir",
        default=None,
        help="Directory tree of clean BEV .npz files. If provided, samples are drawn across all files.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(2, os.cpu_count() or 1),
        help="Parallel workers used for sample generation. Use 1 for sequential generation.",
    )
    parser.add_argument(
        "--compressed-samples",
        action="store_true",
        help="Use compressed .npz samples. Default is faster uncompressed .npz.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse an existing sample dataset instead of deleting and regenerating it.",
    )
    parser.add_argument(
        "--balanced-fault-grid",
        action="store_true",
        help="Cycle every BEV through every selected fault type and severity combination.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--realistic-fault",
        dest="realistic_faults",
        action="append",
        choices=SUPPORTED_HERCULES_LIDAR_FAULTS,
        help="HeRCULES LiDAR fault type to sample. Repeat to mix faults.",
    )
    parser.add_argument(
        "--realistic-fault-severity",
        dest="realistic_fault_severities",
        action="append",
        choices=["mild", "moderate", "severe"],
        help="Fault severity to sample. Repeat to mix severities. Defaults to all severities.",
    )
    parser.add_argument(
        "--realistic-target-threshold",
        type=float,
        default=0.05,
        help="BEV channel-difference threshold used to label realistic fault target cells.",
    )
    args = parser.parse_args()
    if args.realistic_faults is None:
        args.realistic_faults = list(DEFAULT_REALISTIC_FAULTS)
    if args.realistic_fault_severities is None:
        args.realistic_fault_severities = list(DEFAULT_REALISTIC_SEVERITIES)

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_existing and existing_dataset_is_usable(output_dir, args.num_samples, args):
        print(f"Reusing existing samples in {output_dir}")
        print(f"Requested samples: {args.num_samples}")
        print("Use the pipeline option --rebuild-dataset if you want fresh fault samples.")
        return

    removed_samples = clear_existing_samples(output_dir)
    if removed_samples:
        print(f"Removed {removed_samples} stale sample files from {output_dir}")

    if args.bev_dir is not None:
        bev_files = find_bev_files(Path(args.bev_dir))
        if not bev_files:
            raise FileNotFoundError(f"No BEV .npz files found under {args.bev_dir}")
    else:
        bev_files = [Path(args.bev)]

    manifest = {
        "source_bev": str(Path(args.bev)) if args.bev_dir is None else None,
        "source_bev_dir": str(Path(args.bev_dir)) if args.bev_dir is not None else None,
        "source_bev_count": len(bev_files),
        "num_samples": args.num_samples,
        "layers": DEFAULT_LAYERS,
        "fault_classes": FAULT_CLASSES,
        "fault_source": "hercules_lidar",
        "realistic_faults": args.realistic_faults,
        "realistic_fault_severities": args.realistic_fault_severities,
        "realistic_target_threshold": args.realistic_target_threshold,
        "compressed_samples": args.compressed_samples,
        "balanced_fault_grid": args.balanced_fault_grid,
        "num_workers": args.num_workers,
        "samples": [],
    }

    bev_file_strings = [str(path) for path in bev_files]
    args.output_dir = str(output_dir)
    args_dict = vars(args).copy()
    if args.num_workers <= 1:
        for index in range(args.num_samples):
            manifest["samples"].append(generate_sample_record(index, bev_file_strings, args_dict))
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(generate_sample_record, index, bev_file_strings, args_dict)
                for index in range(args.num_samples)
            ]
            for completed_count, future in enumerate(as_completed(futures), start=1):
                manifest["samples"].append(future.result())
                if completed_count % 100 == 0 or completed_count == args.num_samples:
                    print(f"Generated {completed_count}/{args.num_samples} samples")
        manifest["samples"].sort(key=lambda row: row["sample_index"])

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {args.num_samples} samples to {output_dir}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Source BEV files: {len(bev_files)}")
    print("Fault source: hercules_lidar")
    print(f"Sample compression: {'compressed' if args.compressed_samples else 'uncompressed'}")
    print(f"Sample generation workers: {args.num_workers}")
    print(f"Realistic fault types: {', '.join(args.realistic_faults)}")
    print(f"Realistic fault severities: {', '.join(args.realistic_fault_severities)}")


if __name__ == "__main__":
    main()
