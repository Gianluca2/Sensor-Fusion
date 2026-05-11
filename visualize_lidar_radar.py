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
    r"\01_Day\Calibration\Continental_LiDAR.txt"
)


def load_match(match_index: Path, match_row: int):
    with open(match_index, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader):
            if index == match_row:
                return row

    raise IndexError(f"Match row {match_row} was not found in {match_index}")


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
        "--no-window",
        action="store_true",
        help="Load and transform frames without opening the Open3D viewer.",
    )
    args = parser.parse_args()

    match = load_match(Path(args.match_index), args.match_row)
    lidar_path = Path(match["lidar_path"])
    radar_path = Path(match["radar_path"])

    lidar_points = read_aeva_bin(lidar_path)
    radar_points = read_continental_bin(radar_path)

    lidar_xyz = lidar_points[:, :3]
    radar_xyz = radar_points[:, :3]

    lidar_to_radar = load_lidar_to_radar_transform(Path(args.calibration))
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
    print(f"Radar XYZ in LiDAR frame min: {np.min(radar_xyz_lidar_frame, axis=0)}")
    print(f"Radar XYZ in LiDAR frame max: {np.max(radar_xyz_lidar_frame, axis=0)}")

    if not args.no_window:
        visualize(
            lidar_xyz,
            radar_xyz_lidar_frame,
            lidar_voxel_size=args.lidar_voxel_size,
            radar_point_size=args.radar_point_size,
        )


if __name__ == "__main__":
    main()
