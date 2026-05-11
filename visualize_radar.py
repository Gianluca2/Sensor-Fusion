from pathlib import Path
import argparse
import struct

import numpy as np
import open3d as o3d


CONTINENTAL_RECORD_SIZE_BYTES = 29
DEFAULT_CONTINENTAL_BIN = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\Data"
    r"\01_Day\Radar\Continental\Continental\1750660177279495711.bin"
)


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

            points.append([
                x,
                y,
                z,
                velocity,
                radar_range,
                rcs,
                azimuth,
                elevation,
            ])

    return np.asarray(points, dtype=np.float32)


def color_by_velocity(velocity: np.ndarray) -> np.ndarray:
    max_abs = float(np.max(np.abs(velocity)))

    if max_abs == 0.0:
        normalized = np.zeros_like(velocity)
    else:
        normalized = velocity / max_abs

    colors = np.zeros((len(velocity), 3), dtype=np.float64)

    approaching = normalized < 0
    receding = normalized > 0

    colors[:, :] = [0.65, 0.65, 0.65]
    colors[approaching, 2] = np.abs(normalized[approaching])
    colors[approaching, 1] = 0.35
    colors[receding, 0] = normalized[receding]
    colors[receding, 1] = 0.35

    return np.clip(colors, 0.0, 1.0)


def visualize(points: np.ndarray, point_size: float):
    xyz = points[:, :3]
    velocity = points[:, 3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(color_by_velocity(velocity))

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Continental Radar Frame", width=1280, height=720)
    vis.add_geometry(pcd)
    vis.add_geometry(axes)

    render_options = vis.get_render_option()
    render_options.point_size = point_size
    render_options.background_color = np.asarray([0.02, 0.02, 0.02])

    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="Load and visualize one HeRCULES Continental radar .bin file."
    )
    parser.add_argument(
        "--bin-path",
        default=DEFAULT_CONTINENTAL_BIN,
        help="Path to one Continental radar .bin file.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=5.0,
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
        raise FileNotFoundError(f"Continental radar .bin file not found: {bin_path}")

    points = read_continental_bin(bin_path)
    if len(points) == 0:
        raise ValueError(f"No points found in {bin_path}")

    xyz = points[:, :3]
    velocity = points[:, 3]
    radar_range = points[:, 4]
    rcs = points[:, 5]

    print(f"Loaded {len(points)} Continental radar points")
    print(f"File: {bin_path}")
    print(f"XYZ min: {np.min(xyz, axis=0)}")
    print(f"XYZ max: {np.max(xyz, axis=0)}")
    print(f"Velocity min/max: {np.min(velocity):.3f}, {np.max(velocity):.3f}")
    print(f"Range min/max: {np.min(radar_range):.3f}, {np.max(radar_range):.3f}")
    print(f"RCS min/max: {np.min(rcs):.0f}, {np.max(rcs):.0f}")

    if not args.no_window:
        visualize(points, point_size=args.point_size)


if __name__ == "__main__":
    main()
