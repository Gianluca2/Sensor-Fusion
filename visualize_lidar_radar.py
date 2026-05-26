from pathlib import Path
import argparse

import numpy as np
import open3d as o3d

from visualize_lidar import read_aeva_bin


DEFAULT_DATA_ROOT = r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\Data"
LIDAR_TIMESTEP_COLORS = [
    [0.35, 0.35, 0.35],
    [0.62, 0.62, 0.62],
    [0.88, 0.88, 0.88],
]


def find_first_lidar_bin(data_root: Path) -> Path:
    candidates = sorted(
        path for path in data_root.rglob("*.bin")
        if path.parent.name.lower() == "aeva"
    )
    if not candidates:
        raise FileNotFoundError(f"No Aeva LiDAR .bin files found under {data_root}")

    return candidates[0]


def load_frames(folder: Path):
    frames = [{"timestamp": int(path.stem), "path": path} for path in folder.glob("*.bin")]
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
    insert_at = np.searchsorted(timestamps, timestamp)
    best = None

    for index in (insert_at - 1, insert_at):
        if index < 0 or index >= len(rows):
            continue

        row = rows[index]
        delta = row["timestamp"] - timestamp
        if best is None or abs(delta) < abs(best["delta"]):
            best = {**row, "delta": delta}

    if best is None:
        raise ValueError(f"No timestamp match found for {timestamp}")

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


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def transform_xyz(xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    xyz_h = np.ones((len(xyz), 4), dtype=np.float64)
    xyz_h[:, :3] = xyz
    transformed = (transform @ xyz_h.T).T
    return transformed[:, :3]


def find_scene_file(sensor_path: Path, relative_path: Path) -> Path:
    for parent in sensor_path.parents:
        candidate = parent / relative_path
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find {relative_path} above {sensor_path}")


def find_frame_index(frames, frame_path: Path):
    for index, frame in enumerate(frames):
        if frame["path"].name == frame_path.name:
            return index

    raise ValueError(f"Could not find frame {frame_path.name} in {frame_path.parent}")


def transform_points_between_poses(xyz: np.ndarray, source_pose, reference_pose) -> np.ndarray:
    reference_to_world = pose_to_transform(reference_pose)
    world_to_reference = invert_transform(reference_to_world)
    source_to_world = pose_to_transform(source_pose)
    source_to_reference = world_to_reference @ source_to_world
    return transform_xyz(xyz, source_to_reference)


def make_point_cloud(xyz: np.ndarray, color, voxel_size: float | None = None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.paint_uniform_color(color)

    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    return pcd


def load_aggregated_lidar_clouds(lidar_path: Path, scan_count: int):
    lidar_frames = load_frames(lidar_path.parent)
    lidar_index = find_frame_index(lidar_frames, lidar_path)
    lidar_poses = load_poses(find_scene_file(lidar_path, Path("PR_GT") / "Aeva_gt.txt"))

    reference_frame = lidar_frames[lidar_index]
    reference_pose = nearest_by_timestamp(reference_frame["timestamp"], lidar_poses)
    lidar_clouds = []

    for offset in range(scan_count):
        frame_index = lidar_index + offset
        if frame_index >= len(lidar_frames):
            break

        lidar_frame = lidar_frames[frame_index]
        lidar_pose = nearest_by_timestamp(lidar_frame["timestamp"], lidar_poses)
        lidar_xyz = read_aeva_bin(lidar_frame["path"])[:, :3]

        lidar_clouds.append({
            "offset": offset,
            "timestamp": lidar_frame["timestamp"],
            "path": lidar_frame["path"],
            "xyz": transform_points_between_poses(lidar_xyz, lidar_pose, reference_pose),
        })

    return lidar_clouds


def visualize_timestep_clouds(lidar_clouds, lidar_voxel_size):
    geometries = []

    for cloud in lidar_clouds:
        color = LIDAR_TIMESTEP_COLORS[min(cloud["offset"], len(LIDAR_TIMESTEP_COLORS) - 1)]
        geometries.append(
            make_point_cloud(
                cloud["xyz"],
                color=color,
                voxel_size=lidar_voxel_size,
            )
        )

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    geometries.append(axes)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR Timesteps", width=1280, height=720)
    for geometry in geometries:
        vis.add_geometry(geometry)

    render_options = vis.get_render_option()
    render_options.point_size = 2.0
    render_options.background_color = np.asarray([0.02, 0.02, 0.02])

    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize one HeRCULES Aeva LiDAR frame or motion-compensated LiDAR timesteps."
    )
    parser.add_argument(
        "--lidar-bin",
        default=None,
        help="Path to one Aeva LiDAR .bin file. Defaults to the first Aeva frame under --data-root.",
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Root folder used to find a default Aeva LiDAR frame.",
    )
    parser.add_argument(
        "--lidar-voxel-size",
        type=float,
        default=0.15,
        help="Voxel size for LiDAR downsampling. Use 0 to disable.",
    )
    parser.add_argument(
        "--aggregate-scans",
        type=int,
        default=3,
        help="Number of consecutive LiDAR scans to visualize.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Load and transform frames without opening the Open3D viewer.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    lidar_path = Path(args.lidar_bin) if args.lidar_bin else find_first_lidar_bin(data_root)
    if not lidar_path.exists():
        raise FileNotFoundError(f"LiDAR .bin file not found: {lidar_path}")

    lidar_clouds = load_aggregated_lidar_clouds(lidar_path, args.aggregate_scans)

    print(f"Reference LiDAR path: {lidar_path}")
    print("\nAggregated LiDAR timestep clouds:")
    for cloud in lidar_clouds:
        xyz = cloud["xyz"]
        print(
            f"  LiDAR t+{cloud['offset']}: {len(xyz)} points, "
            f"timestamp={cloud['timestamp']}, {cloud['path']}"
        )
        print(f"    xyz min: {np.min(xyz, axis=0)}")
        print(f"    xyz max: {np.max(xyz, axis=0)}")

    print("LiDAR colors: t=dark gray, t+1=medium gray, t+2=light gray")

    if not args.no_window:
        visualize_timestep_clouds(
            lidar_clouds,
            lidar_voxel_size=args.lidar_voxel_size,
        )


if __name__ == "__main__":
    main()
