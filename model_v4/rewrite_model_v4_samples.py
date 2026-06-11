from pathlib import Path
import argparse
import json
import random
import struct
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
DEFAULT_OUTPUT_DIR = Path("/mnt/3D10B36523559581/Gianluca/model_v4_outputs/model_v4_dataset")
CONTINENTAL_RECORD_SIZE_BYTES = 29
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
    "radar_occupancy",
    "radar_density",
    "radar_abs_velocity",
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


def make_range_channel(height: int, width: int, x_range, y_range, resolution: float):
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
    return range_from_sensor.astype(np.float32)


def occupancy_grid(xyz: np.ndarray, x_range, y_range, resolution: float):
    xyz_valid, rows, cols, height, width = metric_to_grid(xyz, x_range, y_range, resolution)
    occupied = np.zeros((height, width), dtype=np.float32)
    if len(xyz_valid) > 0:
        occupied[rows, cols] = 1.0
    return occupied


def find_first_dir_named(root: Path, name: str):
    name = name.lower()
    candidates = [
        path for path in root.rglob("*")
        if path.is_dir() and path.name.lower() == name and list(path.glob("*.bin"))
    ]
    return sorted(candidates, key=lambda path: len(path.parts))[0] if candidates else None


def find_first_file_named(root: Path, name: str):
    matches = sorted(root.rglob(name), key=lambda path: len(path.parts))
    return matches[0] if matches else None


def read_continental_bin(path: Path) -> np.ndarray:
    points = []
    with open(path, "rb") as file:
        while True:
            data = file.read(CONTINENTAL_RECORD_SIZE_BYTES)
            if len(data) == 0:
                break
            if len(data) != CONTINENTAL_RECORD_SIZE_BYTES:
                raise ValueError(
                    f"Incomplete Continental radar record in {path}: "
                    f"expected {CONTINENTAL_RECORD_SIZE_BYTES} bytes, got {len(data)}"
                )

            x, y, z, velocity, radar_range = struct.unpack("fffff", data[:20])
            rcs = struct.unpack("B", data[20:21])[0]
            azimuth, elevation = struct.unpack("ff", data[21:29])
            points.append([x, y, z, velocity, radar_range, rcs, azimuth, elevation])

    return np.asarray(points, dtype=np.float32)


def load_lidar_to_radar_transform(calibration_path: Path):
    with open(calibration_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("Tr_lidar_to_radar:"):
                values = line.split(":", maxsplit=1)[1].split()
                numbers = np.asarray([float(value) for value in values], dtype=np.float64)
                if len(numbers) != 12:
                    raise ValueError(
                        f"Expected 12 calibration values in {calibration_path}, got {len(numbers)}"
                    )
                transform = np.eye(4, dtype=np.float64)
                transform[:3, :] = numbers.reshape(3, 4)
                return transform

    raise ValueError(f"Could not find Tr_lidar_to_radar in {calibration_path}")


def transform_radar_to_reference_lidar(
    radar_points: np.ndarray,
    radar_to_lidar: np.ndarray,
    lidar_pose,
    world_to_reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(radar_points) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    radar_xyz_lidar = transform_xyz(radar_points[:, :3], radar_to_lidar)
    lidar_to_world = pose_to_transform(lidar_pose)
    lidar_to_reference = world_to_reference @ lidar_to_world
    radar_xyz_reference = transform_xyz(radar_xyz_lidar, lidar_to_reference).astype(np.float32)
    return radar_xyz_reference, radar_points[:, 3].astype(np.float32)


def transform_radar_to_reference_from_pose(
    radar_points: np.ndarray,
    radar_pose,
    world_to_reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(radar_points) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    radar_to_world = pose_to_transform(radar_pose)
    radar_to_reference = world_to_reference @ radar_to_world
    radar_xyz_reference = transform_xyz(radar_points[:, :3], radar_to_reference).astype(np.float32)
    return radar_xyz_reference, radar_points[:, 3].astype(np.float32)


def project_radar_bev_v3(
    radar_xyz_list: list[np.ndarray],
    radar_velocity_list: list[np.ndarray],
    x_range,
    y_range,
    resolution: float,
):
    if radar_xyz_list:
        radar_xyz = np.vstack(radar_xyz_list).astype(np.float32)
        radar_velocity = np.concatenate(radar_velocity_list).astype(np.float32)
    else:
        height = int(np.ceil((x_range[1] - x_range[0]) / resolution))
        width = int(np.ceil((y_range[1] - y_range[0]) / resolution))
        empty = np.zeros((height, width), dtype=np.float32)
        return {
            "radar_occupancy": empty,
            "radar_density": empty,
            "radar_abs_velocity": empty,
        }

    x_min, x_max = x_range
    y_min, y_max = y_range
    height = int(np.ceil((x_max - x_min) / resolution))
    width = int(np.ceil((y_max - y_min) / resolution))
    valid = (
        (radar_xyz[:, 0] >= x_min)
        & (radar_xyz[:, 0] < x_max)
        & (radar_xyz[:, 1] >= y_min)
        & (radar_xyz[:, 1] < y_max)
    )
    xyz = radar_xyz[valid]
    velocity = np.abs(radar_velocity[valid])
    cols = np.floor((xyz[:, 1] - y_min) / resolution).astype(np.int32)
    rows_from_bottom = np.floor((xyz[:, 0] - x_min) / resolution).astype(np.int32)
    rows = height - 1 - rows_from_bottom
    in_grid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows = rows[in_grid]
    cols = cols[in_grid]
    velocity = velocity[in_grid]

    radar_occupancy = np.zeros((height, width), dtype=np.float32)
    radar_density = np.zeros((height, width), dtype=np.float32)
    radar_abs_velocity = np.zeros((height, width), dtype=np.float32)

    if len(rows) > 0:
        radar_occupancy[rows, cols] = 1.0
        np.add.at(radar_density, (rows, cols), 1.0)
        np.maximum.at(radar_abs_velocity, (rows, cols), velocity)

    return {
        "radar_occupancy": radar_occupancy,
        "radar_density": normalize_by_max(np.log1p(radar_density)),
        "radar_abs_velocity": normalize_by_max(radar_abs_velocity),
    }


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
    range_from_sensor = make_range_channel(
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
    }


def stack_layers(bev_layers: dict) -> np.ndarray:
    return np.stack([bev_layers[layer].astype(np.float32) for layer in V3_LAYERS], axis=0)


def damage_target(clean: np.ndarray, faulty: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    difference = np.max(np.abs(clean - faulty), axis=0)
    soft_target = np.clip(difference / max(threshold, 1e-6), 0.0, 1.0).astype(np.float32)
    binary_target = (difference > threshold).astype(np.float32)
    return soft_target, binary_target


def transform_scan_to_reference(points: np.ndarray, frame_timestamp: int, poses, world_to_reference) -> np.ndarray:
    pose = nearest_by_timestamp(frame_timestamp, poses)
    sensor_to_world = pose_to_transform(pose)
    sensor_to_reference = world_to_reference @ sensor_to_world
    return transform_xyz(points[:, :3], sensor_to_reference).astype(np.float32)


def nearest_index_by_timestamp(timestamp: int, rows) -> int:
    timestamps = [row["timestamp"] for row in rows]
    insert_at = np.searchsorted(timestamps, timestamp)
    best_index = max(0, min(insert_at, len(rows) - 1))
    for index in (insert_at - 1, insert_at):
        if 0 <= index < len(rows):
            if abs(rows[index]["timestamp"] - timestamp) < abs(rows[best_index]["timestamp"] - timestamp):
                best_index = index
    return best_index


def build_clean_faulty_and_radar_scan_lists(
    scene,
    start_index: int,
    lidar_scan_count: int,
    radar_scan_count: int,
    fault_type: str,
    severity: str,
    rng,
):
    lidar_frames = scene["lidar_frames"]
    poses = scene["lidar_poses"]
    reference = lidar_frames[start_index]
    reference_pose = nearest_by_timestamp(reference["timestamp"], poses)
    reference_to_world = pose_to_transform(reference_pose)
    world_to_reference = invert_transform(reference_to_world)
    clean_scans = []
    faulty_scans = []
    radar_scans = []
    radar_velocities = []
    frame_metadata = []
    radar_metadata = []

    for frame in lidar_frames[start_index:start_index + lidar_scan_count]:
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

    if scene.get("radar_frames") and radar_scan_count > 0:
        nearest_radar_index = nearest_index_by_timestamp(reference["timestamp"], scene["radar_frames"])
        radar_start_index = min(nearest_radar_index, max(0, len(scene["radar_frames"]) - radar_scan_count))
        selected_radar_frames = scene["radar_frames"][radar_start_index:radar_start_index + radar_scan_count]

        for radar_frame in selected_radar_frames:
            radar_points = read_continental_bin(radar_frame["path"])
            if scene.get("radar_poses"):
                radar_pose = nearest_by_timestamp(radar_frame["timestamp"], scene["radar_poses"])
                radar_xyz, radar_velocity = transform_radar_to_reference_from_pose(
                    radar_points,
                    radar_pose,
                    world_to_reference,
                )
            elif scene.get("radar_to_lidar") is not None:
                lidar_pose = nearest_by_timestamp(radar_frame["timestamp"], poses)
                radar_xyz, radar_velocity = transform_radar_to_reference_lidar(
                    radar_points,
                    scene["radar_to_lidar"],
                    lidar_pose,
                    world_to_reference,
                )
            else:
                continue

            radar_scans.append(radar_xyz)
            radar_velocities.append(radar_velocity)
            radar_metadata.append({
                "timestamp": radar_frame["timestamp"],
                "path": str(radar_frame["path"]),
                "delta_ms_from_reference_lidar": (radar_frame["timestamp"] - reference["timestamp"]) / 1_000_000.0,
            })

    return clean_scans, faulty_scans, radar_scans, radar_velocities, reference, frame_metadata, radar_metadata


def prepare_scenes(data_root: Path, lidar_aggregate_scans: int):
    scenes = []
    for scene in discover_scene_roots(data_root):
        lidar_frames = load_frames(scene["aeva_dir"])
        if len(lidar_frames) < lidar_aggregate_scans:
            continue
        scene["lidar_frames"] = lidar_frames
        scene["lidar_poses"] = load_poses(scene["aeva_gt"])
        scene["max_start"] = len(lidar_frames) - lidar_aggregate_scans
        radar_dir = find_first_dir_named(scene["root"], "continental")
        radar_gt = find_first_file_named(scene["root"], "Continental_gt.txt")
        calibration = find_first_file_named(scene["root"], "Continental_LiDAR.txt")
        scene["radar_dir"] = radar_dir
        scene["radar_gt"] = radar_gt
        scene["calibration"] = calibration
        if radar_dir and radar_gt and calibration:
            scene["radar_frames"] = load_frames(radar_dir)
            scene["radar_poses"] = load_poses(radar_gt)
            scene["radar_to_lidar"] = invert_transform(load_lidar_to_radar_transform(calibration))
        else:
            scene["radar_frames"] = []
            scene["radar_poses"] = []
            scene["radar_to_lidar"] = None
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

    clean_scans, faulty_scans, radar_scans, radar_velocities, reference, frame_metadata, radar_metadata = build_clean_faulty_and_radar_scan_lists(
        scene,
        start_index,
        args.lidar_aggregate_scans,
        args.radar_aggregate_scans,
        fault_type,
        severity,
        rng,
    )
    x_range = (args.x_min, args.x_max)
    y_range = (args.y_min, args.y_max)
    clean_layers = project_lidar_bev_v3(
        clean_scans,
        x_range,
        y_range,
        args.resolution,
    )
    faulty_layers = project_lidar_bev_v3(
        faulty_scans,
        x_range,
        y_range,
        args.resolution,
    )
    radar_layers = project_radar_bev_v3(
        radar_scans,
        radar_velocities,
        x_range,
        y_range,
        args.resolution,
    )
    clean_layers.update(radar_layers)
    faulty_layers.update(radar_layers)
    clean = stack_layers(clean_layers)
    faulty = stack_layers(faulty_layers)
    target, binary_target = damage_target(clean, faulty, args.target_threshold)

    metadata = {
        "sample_index": index,
        "model_version": "v4",
        "scene": scene["name"],
        "scene_root": str(scene["root"]),
        "reference_lidar_timestamp": reference["timestamp"],
        "reference_lidar_path": str(reference["path"]),
        "aggregated_lidar_frames": frame_metadata,
        "aggregated_radar_frames": radar_metadata,
        "fault_source": "hercules_lidar_before_bev_projection",
        "fault_type": fault_type,
        "fault_severity": severity,
        "target_type": "soft_damage_unreliability",
        "binary_target_threshold": args.target_threshold,
        "fault_target_cells": int(np.count_nonzero(binary_target)),
        "fault_target_fraction": float(np.count_nonzero(binary_target) / binary_target.size),
        "soft_target_mean": float(np.mean(target)),
        "layers": V3_LAYERS,
        "radar_conditioning": bool(radar_scans),
        "radar_layers": ["radar_occupancy", "radar_density", "radar_abs_velocity"],
        "radar_calibration_path": str(scene["calibration"]) if scene.get("calibration") else None,
        "x_range_m": list(x_range),
        "y_range_m": list(y_range),
        "resolution_m_per_cell": args.resolution,
        "aggregate_scans": args.aggregate_scans,
        "lidar_aggregate_scans": args.lidar_aggregate_scans,
        "radar_aggregate_scans": args.radar_aggregate_scans,
    }
    return faulty, clean, target, binary_target, metadata


def write_sample(path: Path, faulty, clean, target, binary_target, metadata, compressed: bool):
    writer = np.savez_compressed if compressed else np.savez
    writer(
        path,
        input=faulty.astype(np.float32),
        clean=clean.astype(np.float32),
        target=target.astype(np.float32),
        binary_target=binary_target.astype(np.float32),
        metadata_json=json.dumps(metadata, indent=2),
    )


def load_existing_metadata(path: Path):
    with np.load(path) as data:
        metadata = {}
        if "metadata_json" in data.files:
            metadata = json.loads(str(data["metadata_json"].item()))
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create Model V4 samples with faults injected into raw Aeva LiDAR "
            "points before BEV projection."
        )
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First global sample index to write. Useful for chunked/resumable generation.",
    )
    parser.add_argument(
        "--aggregate-scans",
        type=int,
        default=3,
        help="Backward-compatible default used for both LiDAR and radar aggregation unless overridden.",
    )
    parser.add_argument(
        "--lidar-aggregate-scans",
        type=int,
        default=None,
        help="Number of consecutive Aeva LiDAR frames to aggregate per sample.",
    )
    parser.add_argument(
        "--radar-aggregate-scans",
        type=int,
        default=None,
        help="Number of consecutive Continental radar frames to aggregate per sample.",
    )
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
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sample files that already exist instead of rewriting them.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.lidar_aggregate_scans is None:
        args.lidar_aggregate_scans = args.aggregate_scans
    if args.radar_aggregate_scans is None:
        args.radar_aggregate_scans = args.aggregate_scans
    output_dir = Path(args.dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.keep_existing:
        removed = 0
        for path in output_dir.glob("sample_*.npz"):
            path.unlink()
            removed += 1
        if removed:
            print(f"Removed {removed} stale samples from {output_dir}")

    scenes = prepare_scenes(Path(args.data_root), args.lidar_aggregate_scans)
    print("Discovered scenes:")
    for scene in scenes:
        print(f"  {scene['name']}: {len(scene['lidar_frames'])} LiDAR frames")
    print(f"V4 layers: {', '.join(V3_LAYERS)}")
    print("Fault injection: raw Aeva point cloud -> fault injector -> motion compensation -> BEV")
    print("Target: soft clean-vs-faulty BEV damage map plus binary target for metrics")
    print(f"LiDAR aggregate scans per sample: {args.lidar_aggregate_scans}")
    print(f"Radar aggregate scans per sample: {args.radar_aggregate_scans}")

    manifest = {
        "model_version": "v4",
        "data_root": str(Path(args.data_root)),
        "dataset_dir": str(output_dir),
        "num_samples": args.num_samples,
        "start_index": args.start_index,
        "end_index_exclusive": args.start_index + args.num_samples,
        "aggregate_scans": args.aggregate_scans,
        "lidar_aggregate_scans": args.lidar_aggregate_scans,
        "radar_aggregate_scans": args.radar_aggregate_scans,
        "layers": V3_LAYERS,
        "fault_types": FAULT_TYPES,
        "severities": SEVERITIES,
        "target_type": "soft_damage_unreliability",
        "binary_target_threshold": args.target_threshold,
        "fault_injection_stage": "before_bev_projection",
        "samples": [],
    }

    for offset, index in enumerate(range(args.start_index, args.start_index + args.num_samples)):
        sample_path = output_dir / f"sample_{index:06d}.npz"
        if args.skip_existing and sample_path.exists():
            metadata = load_existing_metadata(sample_path)
            if not metadata:
                metadata = {"sample_index": index, "status": "existing_metadata_missing"}
            manifest["samples"].append({"path": str(sample_path), **metadata})
            if (offset + 1) % 25 == 0 or offset + 1 == args.num_samples:
                print(
                    f"Generated/skipped {offset + 1}/{args.num_samples} samples "
                    f"(global index {index})",
                    flush=True,
                )
            continue

        faulty, clean, target, binary_target, metadata = make_sample(index, scenes, args)
        write_sample(sample_path, faulty, clean, target, binary_target, metadata, args.compressed_samples)
        manifest["samples"].append({"path": str(sample_path), **metadata})
        if (offset + 1) % 25 == 0 or offset + 1 == args.num_samples:
            print(
                f"Generated/skipped {offset + 1}/{args.num_samples} samples "
                f"(global index {index})",
                flush=True,
            )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {args.num_samples} samples to {output_dir}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
