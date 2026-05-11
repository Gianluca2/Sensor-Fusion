from pathlib import Path
import argparse
import struct

import numpy as np
import open3d as o3d


AEVA_RECORD_SIZE_BYTES = 29
DEFAULT_AEVA_BIN = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\Data"
    r"\01_Day\LiDAR\LiDAR\Aeva\1750660177366429449.bin"
)


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

            points.append([
                x,
                y,
                z,
                reflectivity,
                velocity,
                time_offset_ns,
                line_index,
                intensity,
            ])

    return np.asarray(points, dtype=np.float32)


def color_by_height(xyz: np.ndarray) -> np.ndarray:
    z = xyz[:, 2]
    z_min = float(np.min(z))
    z_max = float(np.max(z))

    if z_max == z_min:
        normalized = np.zeros_like(z)
    else:
        normalized = (z - z_min) / (z_max - z_min)

    colors = np.zeros((len(xyz), 3), dtype=np.float64)
    colors[:, 0] = normalized
    colors[:, 1] = 0.4
    colors[:, 2] = 1.0 - normalized
    return colors


def visualize(points: np.ndarray, point_size: float):
    xyz = points[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(color_by_height(xyz))

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Aeva LiDAR Frame", width=1280, height=720)
    vis.add_geometry(pcd)
    vis.add_geometry(axes)

    render_options = vis.get_render_option()
    render_options.point_size = point_size
    render_options.background_color = np.asarray([0.02, 0.02, 0.02])

    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="Load and visualize one HeRCULES Aeva LiDAR .bin file."
    )
    parser.add_argument(
        "--bin-path",
        default=DEFAULT_AEVA_BIN,
        help="Path to one Aeva .bin file.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.5,
        help="Open3D point size.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Load the file and print statistics without opening the Open3D viewer.",
    )
    args = parser.parse_args()

    bin_path = Path(args.bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(f"Aeva .bin file not found: {bin_path}")

    points = read_aeva_bin(bin_path)
    if len(points) == 0:
        raise ValueError(f"No points found in {bin_path}")

    xyz = points[:, :3]
    print(f"Loaded {len(points)} Aeva points")
    print(f"File: {bin_path}")
    print(f"XYZ min: {np.min(xyz, axis=0)}")
    print(f"XYZ max: {np.max(xyz, axis=0)}")

    if not args.no_window:
        visualize(points, point_size=args.point_size)


if __name__ == "__main__":
    main()
