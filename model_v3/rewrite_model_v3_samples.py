from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from build_bev_dataset import (
    discover_scene_roots,
    invert_transform,
    load_frames,
    load_poses,
    nearest_by_timestamp,
    pose_to_transform,
    read_aeva_bin,
    transform_xyz,
)
from hercules_lidar_faults import apply_hercules_lidar_fault


DEFAULT_DATA_ROOT = Path("/mnt/3D10B36523559581/HeRCULES")
DEFAULT_OUTPUT_DIR = Path("/mnt/3D10B36523559581/Gianluca/model_v3_outputs/model_v3_dataset")
FAULT_TYPES = ["laser", "photodetector", "scanning", "optical", "window", "mounting"]
SEVERITIES = ["mild", "moderate", "severe"]
V3_LAYERS = [
    "lidar_density",
    "lidar_height",
    "lidar_height_spread",
    "binary_occupancy",
    "range_from_sensor",
    "local_density_residual",
    "temporal_density_consistency",
    "expected_density_by_range",
]


def metric_to_grid(xyz: np.ndarray, x_range, y_range, resolution: float):
    x_min, x_max = x_range
    y_min, y_max = y_range
    height = int(np.ceil((x_max - x_min) / resolution))
    width = int(np.ceil((y_max - y_min) / resolution))
    valid = (
        (xyz[:, 0] >= x_min)
        & (xyz[:, 0] < x_max)
        & (xyz[:, 1] >= y_min)
        & (xyz[:, 1] < y_max)
    )
    xyz_valid = xyz[valid]
    cols = np.floor((xyz_valid[:, 1] - y_min) / resolution).astype(np.int32)
    rows_from_bottom = np.floor((xyz_valid[:, 0] - x_min) / resolution).astype(np.int32)
    rows = height - 1 - rows_from_bottom
    in_grid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    return xyz_valid[in_grid], rows[in_grid], cols[in_grid], height, width


def normalize_by_max(grid: np.ndarray) -> np.ndarray:
    max_value = float(np.max(grid))
    if max_value <= 0.0:
        return np.zeros_like(grid, dtype=np.float32)
    return (grid / max_value).astype(np.float32)


def normalize_occupied(grid: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    output = np.zeros_like(grid, dtype=np.float32)
    if not np.any(occupied):
        return output
    values = grid[occupied]
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value == min_value:
        output[occupied] = 1.0
    else:
        output[occupied] = (values - min_value) / (max_value - min_value)
    return output


def local_mean_3x3(grid: np.ndarray) -> np.ndarray:
    padded = np.pad(grid, 1, mode="edge")
    total = np.zeros_like(grid, dtype=np.float32)
    for row_offset in range(3):
        for col_offset in range(3):
            total += padded[row_offset:row_offset + grid.shape[0], col_offset:col_offset + grid.shape[1]]
    return total / 9.0


def make_range_channels(height: int, width: int, x_range, y_range, resolution: float):
    x_min, x_max = x_range
    y_min, y_max = y_range
    row_indices = np.arange(height, dtype=np.float32)
    col_indices = np.arange(width, dtype=np.float32)
    x_centers = x_min + (height - 1 - row_indices + 0.5) * resolution
    y_centers = y_min + (col_indices + 0.5) * resolution
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="ij")
    metric_range = np.sqrt(x_grid * x_grid + y_grid * y_grid).astype(np.float32)
    max_range = max(float(np.max(metric_range)), 1e-6)
    range_from_sensor = metric_range / max_range
    expected_density_by_range = np.exp(-metric_range / (0.5 * max_range)).astype(np.float32)
    return range_from_sensor.astype(np.float32), expected_density_by_range


def occupancy_grid(xyz: np.ndarray, x_range, y_range, resolution: float):
    xyz_valid, rows, cols, height, width = metric_to_grid(xyz, x_range, y_range, resolution)
    occupied = np.zeros((height, width), dtype=np.float32)
    if len(xyz_valid) > 0:
        occupied[rows, cols] = 1.0
    return occupied


def project_lidar_bev_v3(
    scan_xyz_list: list[np.ndarray],
    x_range,
    y_range,
    resolution: float,
):
    lidar_xyz = np.vstack(scan_xyz_list).astype(np.float32)
    xyz, rows, cols, height, width = metric_to_grid(lidar_xyz, x_range, y_range, resolution)

    density = np.zeros((height, width), dtype=np.float32)
    max_height = np.full((height, width), -np.inf, dtype=np.float32)
    min_height = np.full((height, width), np.inf, dtype=np.float32)
    if len(xyz) > 0:
        np.add.at(density, (rows, cols), 1.0)
        np.maximum.at(max_height, (rows, cols), xyz[:, 2])
        np.minimum.at(min_height, (rows, cols), xyz[:, 2])

    occupied = density > 0
    density_normalized = normalize_by_max(np.log1p(density))
    height_normalized = normalize_occupied(max_height, occupied)
    height_spread = np.zeros((height, width), dtype=np.float32)
    height_spread[occupied] = max_height[occupied] - min_height[occupied]
    height_spread = normalize_by_max(height_spread)
    binary_occupancy = occupied.astype(np.float32)
    local_density_residual = normalize_by_max(np.abs(density_normalized - local_mean_3x3(density_normalized)))
    range_from_sensor, expected_density_by_range = make_range_channels(
        height,
        width,
        x_range,
        y_range,
        resolution,
    )

    temporal_stack = [
        occupancy_grid(scan_xyz, x_range, y_range, resolution)
        for scan_xyz in scan_xyz_list
        if len(scan_xyz) > 0
    ]
    if temporal_stack:
        temporal_density_consistency = np.mean(np.stack(temporal_stack, axis=0), axis=0).astype(np.float32)
    else:
        temporal_density_consistency = np.zeros((height, width), dtype=np.float32)

    return {
        "lidar_density": density_normalized,
        "lidar_height": height_normalized,
        "lidar_height_spread": height_spread,
        "binary_occupancy": binary_occupancy,
        "range_from_sensor": range_from_sensor,
        "local_density_residual": local_density_residual,
        "temporal_density_consistency": temporal_density_consistency,
        "expected_density_by_range": expected_density_by_range,
    }


def stack_layers(bev_layers: dict) -> np.ndarray:
    return np.stack([bev_layers[layer].astype(np.float32) for layer in V3_LAYERS], axis=0)


def difference_target(clean: np.ndarray, faulty: np.ndarray, threshold: float) -> np.ndarray:
    difference = np.max(np.abs(clean - faulty), axis=0)
    return (difference > threshold).astype(np.float32)


def transform_scan_to_reference(points: np.ndarray, frame_timestamp: int, poses, world_to_reference) -> np.ndarray:
    pose = nearest_by_timestamp(frame_timestamp, poses)
    sensor_to_world = pose_to_transform(pose)
    sensor_to_reference = world_to_reference @ sensor_to_world
    return transform_xyz(points[:, :3], sensor_to_reference).astype(np.float32)


def build_clean_and_faulty_scan_lists(scene, start_index: int, scan_count: int, fault_type: str, severity: str, rng):
    lidar_frames = scene["lidar_frames"]
    poses = scene["lidar_poses"]
    reference = lidar_frames[start_index]
    reference_pose = nearest_by_timestamp(reference["timestamp"], poses)
    reference_to_world = pose_to_transform(reference_pose)
    world_to_reference = invert_transform(reference_to_world)
    clean_scans = []
    faulty_scans = []
    frame_metadata = []

    for frame in lidar_frames[start_index:start_index + scan_count]:
        aeva_points = read_aeva_bin(frame["path"])
        faulted_points = apply_hercules_lidar_fault(
            aeva_points,
            fault_type=fault_type,
            severity=severity,
            rng=rng,
        )
        clean_scans.append(
            transform_scan_to_reference(aeva_points, frame["timestamp"], poses, world_to_reference)
        )
        faulty_scans.append(
            transform_scan_to_reference(faulted_points, frame["timestamp"], poses, world_to_reference)
        )
        frame_metadata.append({
            "timestamp": frame["timestamp"],
            "path": str(frame["path"]),
            "delta_ms_from_reference": (frame["timestamp"] - reference["timestamp"]) / 1_000_000.0,
        })

    return clean_scans, faulty_scans, reference, frame_metadata


def prepare_scenes(data_root: Path, aggregate_scans: int):
    scenes = []
    for scene in discover_scene_roots(data_root):
        lidar_frames = load_frames(scene["aeva_dir"])
        if len(lidar_frames) < aggregate_scans:
            continue
        scene["lidar_frames"] = lidar_frames
        scene["lidar_poses"] = load_poses(scene["aeva_gt"])
        scene["max_start"] = len(lidar_frames) - aggregate_scans
        scenes.append(scene)
    if not scenes:
        raise FileNotFoundError(f"No valid HeRCULES scenes found under {data_root}")
    return scenes


def make_sample(index: int, scenes, args):
    combo_count = len(FAULT_TYPES) * len(SEVERITIES)
    combo_index = index % combo_count
    fault_type = FAULT_TYPES[combo_index // len(SEVERITIES)]
    severity = SEVERITIES[combo_index % len(SEVERITIES)]
    scene = scenes[index % len(scenes)]
    scene_cycle_index = index // len(scenes)
    start_index = scene_cycle_index % (scene["max_start"] + 1)
    rng = np.random.default_rng(args.seed + index)

    clean_scans, faulty_scans, reference, frame_metadata = build_clean_and_faulty_scan_lists(
        scene,
        start_index,
        args.aggregate_scans,
        fault_type,
        severity,
        rng,
    )
    x_range = (args.x_min, args.x_max)
    y_range = (args.y_min, args.y_max)
    clean_layers = project_lidar_bev_v3(clean_scans, x_range, y_range, args.resolution)
    faulty_layers = project_lidar_bev_v3(faulty_scans, x_range, y_range, args.resolution)
    clean = stack_layers(clean_layers)
    faulty = stack_layers(faulty_layers)
    target = difference_target(clean, faulty, args.target_threshold)

    metadata = {
        "sample_index": index,
        "model_version": "v3",
        "scene": scene["name"],
        "scene_root": str(scene["root"]),
        "reference_lidar_timestamp": reference["timestamp"],
        "reference_lidar_path": str(reference["path"]),
        "aggregated_lidar_frames": frame_metadata,
        "fault_source": "hercules_lidar_before_bev_projection",
        "fault_type": fault_type,
        "fault_severity": severity,
        "fault_target_cells": int(np.count_nonzero(target)),
        "fault_target_fraction": float(np.count_nonzero(target) / target.size),
        "layers": V3_LAYERS,
        "x_range_m": list(x_range),
        "y_range_m": list(y_range),
        "resolution_m_per_cell": args.resolution,
        "aggregate_scans": args.aggregate_scans,
    }
    return faulty, clean, target, metadata


def write_sample(path: Path, faulty, clean, target, metadata, compressed: bool):
    writer = np.savez_compressed if compressed else np.savez
    writer(
        path,
        input=faulty.astype(np.float32),
        clean=clean.astype(np.float32),
        target=target.astype(np.float32),
        metadata_json=json.dumps(metadata, indent=2),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create Model V3 samples with faults injected into raw Aeva LiDAR "
            "points before BEV projection."
        )
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--aggregate-scans", type=int, default=3)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=80.0)
    parser.add_argument("--y-min", type=float, default=-40.0)
    parser.add_argument("--y-max", type=float, default=40.0)
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--target-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--compressed-samples", action="store_true")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete existing sample_*.npz files before writing.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.keep_existing:
        removed = 0
        for path in output_dir.glob("sample_*.npz"):
            path.unlink()
            removed += 1
        if removed:
            print(f"Removed {removed} stale samples from {output_dir}")

    scenes = prepare_scenes(Path(args.data_root), args.aggregate_scans)
    print("Discovered scenes:")
    for scene in scenes:
        print(f"  {scene['name']}: {len(scene['lidar_frames'])} LiDAR frames")
    print(f"V3 layers: {', '.join(V3_LAYERS)}")
    print("Fault injection: raw Aeva point cloud -> fault injector -> motion compensation -> BEV")

    manifest = {
        "model_version": "v3",
        "data_root": str(Path(args.data_root)),
        "dataset_dir": str(output_dir),
        "num_samples": args.num_samples,
        "aggregate_scans": args.aggregate_scans,
        "layers": V3_LAYERS,
        "fault_types": FAULT_TYPES,
        "severities": SEVERITIES,
        "target_threshold": args.target_threshold,
        "fault_injection_stage": "before_bev_projection",
        "samples": [],
    }

    for index in range(args.num_samples):
        faulty, clean, target, metadata = make_sample(index, scenes, args)
        sample_path = output_dir / f"sample_{index:06d}.npz"
        write_sample(sample_path, faulty, clean, target, metadata, args.compressed_samples)
        manifest["samples"].append({"path": str(sample_path), **metadata})
        if (index + 1) % 25 == 0 or index + 1 == args.num_samples:
            print(f"Generated {index + 1}/{args.num_samples} samples", flush=True)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.num_samples} samples to {output_dir}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
