from pathlib import Path
import argparse
import bisect
import json
import struct

import numpy as np

from bev_projection import (
    make_rgb_preview,
    project_lidar_bev,
    save_bev,
    write_image,
)


DEFAULT_DATA_ROOT = r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\Data"
DEFAULT_OUTPUT_DIR = r"C:\Users\gianl\ThesisOutputs\HerculesFiles\outputs\bev_multi"
AEVA_RECORD_SIZE_BYTES = 29


def read_aeva_bin(path: Path) -> np.ndarray:
    points = []
    with open(path, "rb") as file:
        while True:
            data = file.read(AEVA_RECORD_SIZE_BYTES)
            if len(data) == 0:
                break
            if len(data) != AEVA_RECORD_SIZE_BYTES:
                raise ValueError(
                    f"Incomplete Aeva record in {path}: "
                    f"expected {AEVA_RECORD_SIZE_BYTES} bytes, got {len(data)}"
                )

            x, y, z, reflectivity, velocity = struct.unpack("fffff", data[:20])
            time_offset_ns = struct.unpack("I", data[20:24])[0]
            line_index = struct.unpack("B", data[24:25])[0]
            intensity = struct.unpack("f", data[25:29])[0]
            points.append([x, y, z, reflectivity, velocity, time_offset_ns, line_index, intensity])

    return np.asarray(points, dtype=np.float32)


def timestamp_from_path(path: Path) -> int:
    return int(path.stem)


def find_first_file(root: Path, name: str):
    matches = sorted(root.rglob(name), key=lambda path: len(path.parts))
    return matches[0] if matches else None


def find_aeva_dir(root: Path):
    candidates = []
    for path in root.rglob("*"):
        if path.is_dir() and path.name.lower() == "aeva" and list(path.glob("*.bin")):
            candidates.append(path)
    return sorted(candidates, key=lambda path: len(path.parts))[0] if candidates else None


def discover_scene_roots(data_root: Path):
    scenes = []
    for aeva_gt in data_root.rglob("Aeva_gt.txt"):
        scene_root = aeva_gt.parent.parent
        aeva_dir = find_aeva_dir(scene_root)

        if aeva_dir and aeva_gt:
            scenes.append({
                "name": scene_root.name,
                "root": scene_root,
                "aeva_dir": aeva_dir,
                "aeva_gt": aeva_gt,
            })

    unique = {}
    for scene in scenes:
        key = str(scene["root"]).lower()
        unique[key] = scene

    sorted_scenes = sorted(unique.values(), key=lambda item: len(item["root"].parts))
    filtered = []
    for scene in sorted_scenes:
        root = scene["root"]
        if any(parent["root"] in root.parents for parent in filtered):
            continue
        filtered.append(scene)

    return sorted(filtered, key=lambda item: str(item["root"]).lower())


def load_frames(folder: Path):
    frames = [{"timestamp": timestamp_from_path(path), "path": path} for path in folder.glob("*.bin")]
    return sorted(frames, key=lambda row: row["timestamp"])


def load_poses(path: Path):
    poses = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 8:
                continue

            timestamp = int(parts[0])
            values = [float(value) for value in parts[1:]]
            poses.append({
                "timestamp": timestamp,
                "position": np.asarray(values[:3], dtype=np.float64),
                "quat_xyzw": np.asarray(values[3:], dtype=np.float64),
            })

    return sorted(poses, key=lambda row: row["timestamp"])


def nearest_by_timestamp(timestamp: int, rows):
    timestamps = [row["timestamp"] for row in rows]
    insert_at = bisect.bisect_left(timestamps, timestamp)
    best = None

    for index in (insert_at - 1, insert_at):
        if index < 0 or index >= len(rows):
            continue

        row = rows[index]
        delta = row["timestamp"] - timestamp
        if best is None or abs(delta) < abs(best["delta"]):
            best = {**row, "delta": delta}

    return best


def quat_xyzw_to_matrix(quat):
    x, y, z, w = quat
    norm = np.linalg.norm(quat)
    if norm == 0:
        return np.eye(3, dtype=np.float64)

    x, y, z, w = quat / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def pose_to_transform(pose):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_xyzw_to_matrix(pose["quat_xyzw"])
    transform[:3, 3] = pose["position"]
    return transform


def invert_transform(transform):
    inverse = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def transform_xyz(xyz, transform):
    xyz_h = np.ones((len(xyz), 4), dtype=np.float64)
    xyz_h[:, :3] = xyz
    return (transform @ xyz_h.T).T[:, :3]


def aggregate_lidar(lidar_frames, lidar_poses, start_index, scan_count):
    reference = lidar_frames[start_index]
    reference_pose = nearest_by_timestamp(reference["timestamp"], lidar_poses)
    reference_to_world = pose_to_transform(reference_pose)
    world_to_reference = invert_transform(reference_to_world)
    aggregated = []
    frame_metadata = []

    for frame in lidar_frames[start_index:start_index + scan_count]:
        pose = nearest_by_timestamp(frame["timestamp"], lidar_poses)
        sensor_to_world = pose_to_transform(pose)
        sensor_to_reference = world_to_reference @ sensor_to_world
        points = read_aeva_bin(frame["path"])[:, :3]
        aggregated.append(transform_xyz(points, sensor_to_reference).astype(np.float32))
        frame_metadata.append({
            "timestamp": frame["timestamp"],
            "path": str(frame["path"]),
            "delta_ms_from_reference": (frame["timestamp"] - reference["timestamp"]) / 1_000_000.0,
        })

    return np.vstack(aggregated), reference, frame_metadata


def choose_start_indices(frame_count: int, scan_count: int, frames_per_scene: int):
    max_start = frame_count - scan_count
    if max_start < 0:
        return []

    requested = min(frames_per_scene, max_start + 1)
    if requested <= 1:
        return [0]

    indices = np.linspace(0, max_start, num=requested)
    return sorted({int(round(index)) for index in indices})


def build_scene_bevs(scene, args):
    lidar_frames = load_frames(scene["aeva_dir"])
    lidar_poses = load_poses(scene["aeva_gt"])

    if len(lidar_frames) < args.aggregate_scans:
        print(f"Skipping {scene['name']}: not enough LiDAR frames")
        return 0

    scene_output_dir = Path(args.output_dir) / scene["name"]
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    x_range = (args.x_min, args.x_max)
    y_range = (args.y_min, args.y_max)

    if args.evenly_spaced:
        start_indices = choose_start_indices(
            len(lidar_frames),
            args.aggregate_scans,
            args.frames_per_scene,
        )
    else:
        max_start = len(lidar_frames) - args.aggregate_scans
        start_indices = range(0, max_start + 1, args.stride)
    written = 0

    for start_index in start_indices:
        if written >= args.frames_per_scene:
            break

        lidar_xyz, reference, frame_metadata = aggregate_lidar(
            lidar_frames,
            lidar_poses,
            start_index,
            args.aggregate_scans,
        )

        bev_layers = {}
        bev_layers.update(
            project_lidar_bev(
                lidar_xyz,
                x_range,
                y_range,
                args.resolution,
                z_range=(args.z_min, args.z_max),
                z_resolution=args.z_resolution,
            )
        )
        rgb = make_rgb_preview(bev_layers)

        stem = f"{scene['name']}_bev_{written:06d}"
        npz_path = scene_output_dir / f"{stem}.npz"
        image_path = scene_output_dir / f"{stem}.png"
        metadata = {
            "scene": scene["name"],
            "scene_root": str(scene["root"]),
            "reference_lidar_timestamp": reference["timestamp"],
            "reference_lidar_path": str(reference["path"]),
            "aggregated_lidar_frames": frame_metadata,
            "sensor_mode": "lidar_only",
            "aggregate_scans": args.aggregate_scans,
            "aggregation": "t_to_t_plus_scans_minus_1_motion_compensated_to_reference_lidar",
            "x_range_m": list(x_range),
            "y_range_m": list(y_range),
            "resolution_m_per_cell": args.resolution,
            "z_range_m": [args.z_min, args.z_max],
            "z_resolution_m_per_voxel": args.z_resolution,
            "grid_shape": list(rgb.shape[:2]),
        }

        save_bev(npz_path, bev_layers, rgb, metadata)
        write_image(image_path, rgb)
        written += 1

    print(f"{scene['name']}: wrote {written} BEV files to {scene_output_dir}")
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Build multi-scene, motion-compensated LiDAR-only BEV files."
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames-per-scene", type=int, default=30)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument(
        "--evenly-spaced",
        action="store_true",
        help="Sample --frames-per-scene start frames evenly across each scene timeline.",
    )
    parser.add_argument("--aggregate-scans", type=int, default=3)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=80.0)
    parser.add_argument("--y-min", type=float, default=-40.0)
    parser.add_argument("--y-max", type=float, default=40.0)
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--z-min", type=float, default=-4.0)
    parser.add_argument("--z-max", type=float, default=6.0)
    parser.add_argument("--z-resolution", type=float, default=0.5)
    args = parser.parse_args()

    scenes = discover_scene_roots(Path(args.data_root))
    if not scenes:
        raise FileNotFoundError(f"No valid HeRCULES scenes found under {args.data_root}")

    print("Discovered scenes:")
    for scene in scenes:
        print(f"  {scene['name']}: {scene['root']}")

    total = 0
    for scene in scenes:
        total += build_scene_bevs(scene, args)

    print(f"Total BEV files written: {total}")
    manifest_path = Path(args.output_dir) / "manifest.json"
    manifest = {
        "data_root": str(Path(args.data_root)),
        "output_dir": str(Path(args.output_dir)),
        "aggregate_scans": args.aggregate_scans,
        "frames_per_scene": args.frames_per_scene,
        "stride": args.stride,
        "evenly_spaced": args.evenly_spaced,
        "total_bev_files": total,
        "scenes": [
            {
                "name": scene["name"],
                "root": str(scene["root"]),
                "aeva_dir": str(scene["aeva_dir"]),
                "aeva_gt": str(scene["aeva_gt"]),
            }
            for scene in scenes
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
