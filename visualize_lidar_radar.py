from pathlib import Path
import argparse
import csv

import numpy as np
import open3d as o3d

from visualize_lidar import read_aeva_bin
from visualize_radar import read_continental_bin


DEFAULT_MATCH_INDEX = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs"
    r"\lidar_radar_matches.csv"
)
DEFAULT_CALIBRATION = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\Data"
    r"\Day_1_Parking\Calibration\Continental_LiDAR.txt"
)
DEFAULT_DATA_ROOT = r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\Data"
LIDAR_TIMESTEP_COLORS = [
    [0.35, 0.35, 0.35],
    [0.62, 0.62, 0.62],
    [0.88, 0.88, 0.88],
]
RADAR_TIMESTEP_COLORS = [
    [0.45, 0.02, 0.02],
    [0.80, 0.05, 0.03],
    [1.00, 0.18, 0.08],
]


def load_match(match_index: Path, match_row: int):
    with open(match_index, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader):
            if index == match_row:
                return row

    raise IndexError(f"Match row {match_row} was not found in {match_index}")


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


def resolve_existing_path(path: Path, data_root: Path) -> Path:
    if path.exists():
        return path

    candidates = sorted(data_root.rglob(path.name), key=lambda candidate: len(candidate.parts))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"Could not find {path.name} under {data_root}")


def find_scene_calibration(lidar_path: Path, fallback: Path) -> Path:
    for parent in lidar_path.parents:
        calibration = parent / "Calibration" / "Continental_LiDAR.txt"
        if calibration.exists():
            return calibration

    return fallback


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


def make_point_cloud(xyz: np.ndarray, color, voxel_size: float | None = None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.paint_uniform_color(color)

    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    return pcd


def transform_points_between_poses(xyz: np.ndarray, source_pose, reference_pose) -> np.ndarray:
    reference_to_world = pose_to_transform(reference_pose)
    world_to_reference = invert_transform(reference_to_world)
    source_to_world = pose_to_transform(source_pose)
    source_to_reference = world_to_reference @ source_to_world
    return transform_xyz(xyz, source_to_reference)


def load_aggregated_timestep_clouds(lidar_path: Path, radar_path: Path, scan_count: int):
    lidar_frames = load_frames(lidar_path.parent)
    radar_frames = load_frames(radar_path.parent)
    lidar_index = find_frame_index(lidar_frames, lidar_path)
    lidar_poses = load_poses(find_scene_file(lidar_path, Path("PR_GT") / "Aeva_gt.txt"))
    radar_poses = load_poses(find_scene_file(radar_path, Path("PR_GT") / "Continental_gt.txt"))

    reference_frame = lidar_frames[lidar_index]
    reference_pose = nearest_by_timestamp(reference_frame["timestamp"], lidar_poses)
    lidar_clouds = []
    radar_clouds = []

    for offset in range(scan_count):
        frame_index = lidar_index + offset
        if frame_index >= len(lidar_frames):
            break

        lidar_frame = lidar_frames[frame_index]
        radar_frame = nearest_by_timestamp(lidar_frame["timestamp"], radar_frames)
        lidar_pose = nearest_by_timestamp(lidar_frame["timestamp"], lidar_poses)
        radar_pose = nearest_by_timestamp(radar_frame["timestamp"], radar_poses)

        lidar_xyz = read_aeva_bin(lidar_frame["path"])[:, :3]
        radar_points = read_continental_bin(radar_frame["path"])
        radar_xyz = radar_points[:, :3]

        lidar_clouds.append({
            "offset": offset,
            "timestamp": lidar_frame["timestamp"],
            "path": lidar_frame["path"],
            "xyz": transform_points_between_poses(lidar_xyz, lidar_pose, reference_pose),
        })
        radar_clouds.append({
            "offset": offset,
            "timestamp": radar_frame["timestamp"],
            "path": radar_frame["path"],
            "delta_ms": (radar_frame["timestamp"] - lidar_frame["timestamp"]) / 1_000_000.0,
            "xyz": transform_points_between_poses(radar_xyz, radar_pose, reference_pose),
        })

    return lidar_clouds, radar_clouds


def visualize(lidar_xyz, radar_xyz_lidar_frame, lidar_voxel_size, radar_point_size):
    lidar_pcd = make_point_cloud(
        lidar_xyz,
        color=[0.65, 0.65, 0.65],
        voxel_size=lidar_voxel_size,
    )
    radar_pcd = make_point_cloud(
        radar_xyz_lidar_frame,
        color=[1.0, 0.1, 0.05],
    )
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR + Radar Overlay", width=1280, height=720)
    vis.add_geometry(lidar_pcd)
    vis.add_geometry(radar_pcd)
    vis.add_geometry(axes)

    render_options = vis.get_render_option()
    render_options.point_size = radar_point_size
    render_options.background_color = np.asarray([0.02, 0.02, 0.02])

    vis.run()
    vis.destroy_window()


def visualize_timestep_clouds(lidar_clouds, radar_clouds, lidar_voxel_size, radar_point_size):
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

    for cloud in radar_clouds:
        color = RADAR_TIMESTEP_COLORS[min(cloud["offset"], len(RADAR_TIMESTEP_COLORS) - 1)]
        geometries.append(make_point_cloud(cloud["xyz"], color=color))

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    geometries.append(axes)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3-Scan LiDAR + Radar Overlay", width=1280, height=720)
    for geometry in geometries:
        vis.add_geometry(geometry)

    render_options = vis.get_render_option()
    render_options.point_size = radar_point_size
    render_options.background_color = np.asarray([0.02, 0.02, 0.02])

    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="Overlay a matched HeRCULES Aeva LiDAR frame and Continental radar frame."
    )
    parser.add_argument(
        "--match-index",
        default=DEFAULT_MATCH_INDEX,
        help="CSV produced by match_frames.py.",
    )
    parser.add_argument(
        "--match-row",
        type=int,
        default=0,
        help="Zero-based row index from the match CSV to visualize.",
    )
    parser.add_argument(
        "--calibration",
        default=DEFAULT_CALIBRATION,
        help="Continental_LiDAR.txt calibration file.",
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Root folder used to resolve stale match paths after reorganizing data.",
    )
    parser.add_argument(
        "--lidar-voxel-size",
        type=float,
        default=0.15,
        help="Voxel size for LiDAR downsampling. Use 0 to disable.",
    )
    parser.add_argument(
        "--radar-point-size",
        type=float,
        default=5.0,
        help="Open3D point size. This applies globally in the simple visualizer.",
    )
    parser.add_argument(
        "--aggregate-scans",
        type=int,
        default=3,
        help="Number of consecutive scans to visualize. Use 1 for the old single-frame view.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Load and transform frames without opening the Open3D viewer.",
    )
    args = parser.parse_args()

    match = load_match(Path(args.match_index), args.match_row)
    data_root = Path(args.data_root)
    lidar_path = resolve_existing_path(Path(match["lidar_path"]), data_root)
    radar_path = resolve_existing_path(Path(match["radar_path"]), data_root)

    lidar_points = read_aeva_bin(lidar_path)
    radar_points = read_continental_bin(radar_path)

    lidar_xyz = lidar_points[:, :3]
    radar_xyz = radar_points[:, :3]

    calibration_path = find_scene_calibration(lidar_path, Path(args.calibration))
    lidar_to_radar = load_lidar_to_radar_transform(calibration_path)
    radar_to_lidar = invert_transform(lidar_to_radar)
    radar_xyz_lidar_frame = transform_xyz(radar_xyz, radar_to_lidar)

    print(f"Match row: {args.match_row}")
    print(f"LiDAR timestamp: {match['lidar_timestamp']}")
    print(f"Radar timestamp: {match['radar_timestamp']}")
    print(f"Delta ms: {match['delta_ms']}")
    print(f"LiDAR points: {len(lidar_xyz)}")
    print(f"Radar points: {len(radar_xyz)}")
    print(f"LiDAR path: {lidar_path}")
    print(f"Radar path: {radar_path}")
    print(f"Calibration path: {calibration_path}")
    print(f"Radar XYZ in LiDAR frame min: {np.min(radar_xyz_lidar_frame, axis=0)}")
    print(f"Radar XYZ in LiDAR frame max: {np.max(radar_xyz_lidar_frame, axis=0)}")

    if args.aggregate_scans > 1:
        lidar_clouds, radar_clouds = load_aggregated_timestep_clouds(
            lidar_path,
            radar_path,
            scan_count=args.aggregate_scans,
        )

        print("\nAggregated timestep clouds:")
        for cloud in lidar_clouds:
            print(
                f"  LiDAR t+{cloud['offset']}: {len(cloud['xyz'])} points, "
                f"{cloud['path']}"
            )
        for cloud in radar_clouds:
            print(
                f"  Radar t+{cloud['offset']}: {len(cloud['xyz'])} points, "
                f"delta_ms={cloud['delta_ms']:.3f}, {cloud['path']}"
            )
        print("LiDAR colors: t=dark gray, t+1=medium gray, t+2=light gray")
        print("Radar colors: t=dark red, t+1=medium red, t+2=bright red")

        if not args.no_window:
            visualize_timestep_clouds(
                lidar_clouds,
                radar_clouds,
                lidar_voxel_size=args.lidar_voxel_size,
                radar_point_size=args.radar_point_size,
            )
        return

    if not args.no_window:
        visualize(
            lidar_xyz,
            radar_xyz_lidar_frame,
            lidar_voxel_size=args.lidar_voxel_size,
            radar_point_size=args.radar_point_size,
        )


if __name__ == "__main__":
    main()
