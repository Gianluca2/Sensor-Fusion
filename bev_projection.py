from pathlib import Path
import argparse
import json

import numpy as np

from visualize_lidar import read_aeva_bin
from visualize_radar import read_continental_bin
from visualize_lidar_radar import (
    DEFAULT_CALIBRATION,
    DEFAULT_MATCH_INDEX,
    invert_transform,
    load_lidar_to_radar_transform,
    load_match,
    transform_xyz,
)


DEFAULT_OUTPUT_DIR = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\bev"
)


def metric_to_grid(xyz: np.ndarray, x_range, y_range, resolution: float):
    x_min, x_max = x_range
    y_min, y_max = y_range

    valid = (
        (xyz[:, 0] >= x_min)
        & (xyz[:, 0] < x_max)
        & (xyz[:, 1] >= y_min)
        & (xyz[:, 1] < y_max)
    )

    xyz_valid = xyz[valid]
    cols = np.floor((xyz_valid[:, 1] - y_min) / resolution).astype(np.int32)
    rows_from_bottom = np.floor((xyz_valid[:, 0] - x_min) / resolution).astype(np.int32)

    height = int(np.ceil((x_max - x_min) / resolution))
    width = int(np.ceil((y_max - y_min) / resolution))
    rows = height - 1 - rows_from_bottom

    return xyz_valid, rows, cols, height, width, valid


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


def project_lidar_bev(lidar_xyz: np.ndarray, x_range, y_range, resolution: float):
    xyz, rows, cols, height, width, _ = metric_to_grid(
        lidar_xyz,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
    )

    density = np.zeros((height, width), dtype=np.float32)
    max_height = np.full((height, width), -np.inf, dtype=np.float32)

    np.add.at(density, (rows, cols), 1.0)
    np.maximum.at(max_height, (rows, cols), xyz[:, 2])

    occupied = density > 0
    height_normalized = normalize_occupied(max_height, occupied)
    density = normalize_by_max(np.log1p(density))

    return {
        "lidar_density": density,
        "lidar_height": height_normalized,
    }


def project_radar_bev(radar_points: np.ndarray, x_range, y_range, resolution: float):
    radar_xyz = radar_points[:, :3]
    velocity = radar_points[:, 3]
    radar_range = radar_points[:, 4]
    rcs = radar_points[:, 5]

    xyz, rows, cols, height, width, valid = metric_to_grid(
        radar_xyz,
        x_range=x_range,
        y_range=y_range,
        resolution=resolution,
    )

    density = np.zeros((height, width), dtype=np.float32)
    velocity_sum = np.zeros((height, width), dtype=np.float32)
    range_min = np.full((height, width), np.inf, dtype=np.float32)
    rcs_max = np.zeros((height, width), dtype=np.float32)

    valid_velocity = velocity[valid]
    valid_range = radar_range[valid]
    valid_rcs = rcs[valid]

    np.add.at(density, (rows, cols), 1.0)
    np.add.at(velocity_sum, (rows, cols), valid_velocity)
    np.minimum.at(range_min, (rows, cols), valid_range)
    np.maximum.at(rcs_max, (rows, cols), valid_rcs)

    mean_velocity = np.zeros_like(velocity_sum)
    np.divide(velocity_sum, density, out=mean_velocity, where=density > 0)
    occupied = density > 0
    range_normalized = normalize_occupied(range_min, occupied)
    range_min[~np.isfinite(range_min)] = 0.0

    return {
        "radar_density": normalize_by_max(np.log1p(density)),
        "radar_velocity": normalize_by_max(np.abs(mean_velocity)),
        "radar_range_min": range_normalized,
        "radar_rcs_max": normalize_by_max(rcs_max),
    }


def make_rgb_preview(bev_layers) -> np.ndarray:
    red = bev_layers["radar_density"]
    green = bev_layers["lidar_height"]
    blue = bev_layers["lidar_density"]

    rgb = np.stack([red, green, blue], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def write_ppm(path: Path, rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("PPM preview expects an RGB image")

    with open(path, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(rgb.tobytes())


def write_image(path: Path, rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".ppm":
        write_ppm(path, rgb)
        return

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Saving PNG/JPG images requires Pillow. Install it with: "
            "python -m pip install pillow"
        ) from exc

    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def save_bev(output_path: Path, bev_layers, rgb: np.ndarray, metadata: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        **bev_layers,
        rgb_preview=rgb,
        metadata_json=json.dumps(metadata, indent=2),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Project matched HeRCULES LiDAR and radar frames into a BEV grid."
    )
    parser.add_argument("--match-index", default=DEFAULT_MATCH_INDEX)
    parser.add_argument("--match-row", type=int, default=0)
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=80.0)
    parser.add_argument("--y-min", type=float, default=-40.0)
    parser.add_argument("--y-max", type=float, default=40.0)
    parser.add_argument("--resolution", type=float, default=0.2)
    args = parser.parse_args()

    match = load_match(Path(args.match_index), args.match_row)
    lidar_points = read_aeva_bin(Path(match["lidar_path"]))
    radar_points = read_continental_bin(Path(match["radar_path"]))

    lidar_xyz = lidar_points[:, :3]
    radar_xyz = radar_points[:, :3]

    lidar_to_radar = load_lidar_to_radar_transform(Path(args.calibration))
    radar_to_lidar = invert_transform(lidar_to_radar)
    radar_points_lidar = radar_points.copy()
    radar_points_lidar[:, :3] = transform_xyz(radar_xyz, radar_to_lidar)

    x_range = (args.x_min, args.x_max)
    y_range = (args.y_min, args.y_max)

    bev_layers = {}
    bev_layers.update(project_lidar_bev(lidar_xyz, x_range, y_range, args.resolution))
    bev_layers.update(project_radar_bev(radar_points_lidar, x_range, y_range, args.resolution))

    rgb = make_rgb_preview(bev_layers)

    output_dir = Path(args.output_dir)
    stem = f"bev_match_{args.match_row:06d}"
    npz_path = output_dir / f"{stem}.npz"
    image_path = output_dir / f"{stem}.png"

    metadata = {
        "match_row": args.match_row,
        "lidar_timestamp": match["lidar_timestamp"],
        "radar_timestamp": match["radar_timestamp"],
        "delta_ms": match["delta_ms"],
        "lidar_path": match["lidar_path"],
        "radar_path": match["radar_path"],
        "calibration": str(Path(args.calibration)),
        "x_range_m": list(x_range),
        "y_range_m": list(y_range),
        "resolution_m_per_cell": args.resolution,
        "grid_shape": list(rgb.shape[:2]),
        "preview_rgb_channels": {
            "red": "radar_density",
            "green": "lidar_height",
            "blue": "lidar_density",
        },
    }

    save_bev(npz_path, bev_layers, rgb, metadata)
    write_image(image_path, rgb)

    print(f"Wrote BEV arrays: {npz_path}")
    print(f"Wrote BEV preview: {image_path}")
    print(f"Grid shape: {rgb.shape[0]} rows x {rgb.shape[1]} cols")
    print(f"LiDAR occupied cells: {np.count_nonzero(bev_layers['lidar_density'])}")
    print(f"Radar occupied cells: {np.count_nonzero(bev_layers['radar_density'])}")


if __name__ == "__main__":
    main()
