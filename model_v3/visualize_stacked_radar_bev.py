from pathlib import Path
import argparse
import json
import sys

import numpy as np
from PIL import Image, ImageDraw

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
    transform_xyz,
)
from model_v3.rewrite_model_v3_samples import (
    find_first_dir_named,
    find_first_file_named,
    read_continental_bin,
)


WINDOWS_DATA_ROOT = Path(r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\Data")
LINUX_DATA_ROOT = Path("/mnt/3D10B36523559581/HeRCULES")
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\gianl\ThesisOutputs\HerculesFiles\outputs\radar_stack_bev")
DEFAULT_STACK_COUNTS = [10, 25, 50, 100, 200]


def default_data_root() -> Path:
    if LINUX_DATA_ROOT.exists():
        return LINUX_DATA_ROOT
    return WINDOWS_DATA_ROOT


def normalize_by_percentile(grid: np.ndarray, percentile: float) -> np.ndarray:
    if not np.any(grid > 0.0):
        return np.zeros_like(grid, dtype=np.float32)

    scale = float(np.percentile(grid[grid > 0.0], percentile))
    if scale <= 0.0:
        scale = float(np.max(grid))
    if scale <= 0.0:
        return np.zeros_like(grid, dtype=np.float32)

    return np.clip(grid / scale, 0.0, 1.0).astype(np.float32)


def density_to_rgb(density: np.ndarray, occupancy: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    density_norm = normalize_by_percentile(np.log1p(density), 99.0)
    velocity_norm = normalize_by_percentile(velocity, 99.0)
    occupancy_norm = np.clip(occupancy, 0.0, 1.0)

    red = density_norm
    green = np.maximum(density_norm * 0.65, velocity_norm)
    blue = occupancy_norm * 0.25
    rgb = np.stack([red, green, blue], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def add_title(rgb: np.ndarray, title: str) -> np.ndarray:
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 24), fill=(0, 0, 0))
    draw.text((5, 6), title, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def write_image(path: Path, rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def select_scene(data_root: Path, scene_name: str | None):
    scenes = discover_scene_roots(data_root)
    if scene_name:
        for scene in scenes:
            if scene["name"].lower() == scene_name.lower():
                return scene
        names = ", ".join(scene["name"] for scene in scenes)
        raise ValueError(f"Scene {scene_name!r} not found. Available scenes: {names}")

    for scene in scenes:
        radar_dir = find_first_dir_named(scene["root"], "continental")
        if radar_dir is not None:
            return scene

    raise FileNotFoundError(f"No scenes with Continental radar frames were found under {data_root}")


def load_scene_radar(scene):
    radar_dir = find_first_dir_named(scene["root"], "continental")
    if radar_dir is None:
        raise FileNotFoundError(f"No Continental radar folder found in {scene['root']}")

    radar_frames = load_frames(radar_dir)
    if not radar_frames:
        raise FileNotFoundError(f"No Continental .bin files found in {radar_dir}")

    radar_gt = find_first_file_named(scene["root"], "Continental_gt.txt")
    radar_poses = load_poses(radar_gt) if radar_gt else []
    return radar_dir, radar_frames, radar_gt, radar_poses


def frame_to_reference_transform(frame_timestamp: int, poses, reference_pose):
    if not poses or reference_pose is None:
        return None

    source_pose = nearest_by_timestamp(frame_timestamp, poses)
    if source_pose is None:
        return None

    reference_to_world = pose_to_transform(reference_pose)
    world_to_reference = invert_transform(reference_to_world)
    source_to_world = pose_to_transform(source_pose)
    return world_to_reference @ source_to_world


def project_stacked_radar(
    frames,
    poses,
    start_index: int,
    frame_count: int,
    x_range,
    y_range,
    resolution: float,
    motion_compensate: bool,
):
    x_min, x_max = x_range
    y_min, y_max = y_range
    height = int(np.ceil((x_max - x_min) / resolution))
    width = int(np.ceil((y_max - y_min) / resolution))

    density = np.zeros((height, width), dtype=np.float32)
    occupancy = np.zeros((height, width), dtype=np.float32)
    max_abs_velocity = np.zeros((height, width), dtype=np.float32)
    total_points = 0
    in_range_points = 0

    selected = frames[start_index:start_index + frame_count]
    if not selected:
        raise ValueError("No radar frames selected. Check --start-index and --stack-counts.")

    reference_pose = nearest_by_timestamp(selected[0]["timestamp"], poses) if poses else None

    for frame in selected:
        radar_points = read_continental_bin(frame["path"])
        if len(radar_points) == 0:
            continue

        xyz = radar_points[:, :3].astype(np.float32)
        velocity = np.abs(radar_points[:, 3].astype(np.float32))
        total_points += len(xyz)

        if motion_compensate:
            transform = frame_to_reference_transform(frame["timestamp"], poses, reference_pose)
            if transform is not None:
                xyz = transform_xyz(xyz, transform).astype(np.float32)

        valid = (
            (xyz[:, 0] >= x_min)
            & (xyz[:, 0] < x_max)
            & (xyz[:, 1] >= y_min)
            & (xyz[:, 1] < y_max)
        )
        xyz = xyz[valid]
        velocity = velocity[valid]
        in_range_points += len(xyz)

        if len(xyz) == 0:
            continue

        cols = np.floor((xyz[:, 1] - y_min) / resolution).astype(np.int32)
        rows_from_bottom = np.floor((xyz[:, 0] - x_min) / resolution).astype(np.int32)
        rows = height - 1 - rows_from_bottom
        in_grid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        rows = rows[in_grid]
        cols = cols[in_grid]
        velocity = velocity[in_grid]

        occupancy[rows, cols] = 1.0
        np.add.at(density, (rows, cols), 1.0)
        np.maximum.at(max_abs_velocity, (rows, cols), velocity)

    metadata = {
        "start_timestamp": int(selected[0]["timestamp"]),
        "end_timestamp": int(selected[-1]["timestamp"]),
        "frame_count": len(selected),
        "total_radar_points": int(total_points),
        "in_range_radar_points": int(in_range_points),
        "occupied_cells": int(np.count_nonzero(occupancy)),
        "max_cell_density": float(np.max(density)),
        "motion_compensated": bool(motion_compensate),
    }
    return density, occupancy, max_abs_velocity, metadata


def make_comparison_panel(images: list[np.ndarray]) -> np.ndarray:
    if not images:
        raise ValueError("No images to combine")

    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)
    padded = []
    for image in images:
        canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)
        canvas[:image.shape[0], :image.shape[1]] = image
        padded.append(canvas)

    top = np.concatenate(padded[:3], axis=1)
    if len(padded) <= 3:
        return top

    blank = np.zeros_like(padded[0])
    bottom_images = padded[3:] + [blank] * (3 - len(padded[3:]))
    bottom = np.concatenate(bottom_images, axis=1)
    return np.concatenate([top, bottom], axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Stack 10/25/50/100/200 Continental radar frames into BEV density PNGs."
    )
    parser.add_argument("--data-root", default=str(default_data_root()))
    parser.add_argument("--scene", default=None, help="Scene name, for example ParkingLot01_Day. Defaults to first scene with radar.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stack-counts", type=int, nargs="+", default=DEFAULT_STACK_COUNTS)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=80.0)
    parser.add_argument("--y-min", type=float, default=-40.0)
    parser.add_argument("--y-max", type=float, default=40.0)
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument(
        "--no-motion-compensation",
        action="store_true",
        help="Stack raw radar frames without ego-motion compensation.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    scene = select_scene(data_root, args.scene)
    radar_dir, radar_frames, radar_gt, radar_poses = load_scene_radar(scene)

    if args.start_index < 0 or args.start_index >= len(radar_frames):
        raise IndexError(f"--start-index must be between 0 and {len(radar_frames) - 1}")

    x_range = (args.x_min, args.x_max)
    y_range = (args.y_min, args.y_max)
    motion_compensate = not args.no_motion_compensation
    if motion_compensate and not radar_poses:
        print("No Continental_gt.txt found; falling back to raw stacking without motion compensation.")
        motion_compensate = False

    print(f"Scene: {scene['name']}")
    print(f"Radar dir: {radar_dir}")
    print(f"Radar GT: {radar_gt}")
    print(f"Radar frames available: {len(radar_frames)}")
    print(f"Start index: {args.start_index}")
    print(f"Stack counts: {args.stack_counts}")
    print(f"Motion compensation: {motion_compensate}")

    panels = []
    summary = {
        "scene": scene["name"],
        "radar_dir": str(radar_dir),
        "radar_gt": str(radar_gt) if radar_gt else None,
        "start_index": args.start_index,
        "x_range_m": list(x_range),
        "y_range_m": list(y_range),
        "resolution_m_per_cell": args.resolution,
        "motion_compensated": motion_compensate,
        "outputs": [],
    }

    for count in args.stack_counts:
        density, occupancy, velocity, metadata = project_stacked_radar(
            radar_frames,
            radar_poses,
            args.start_index,
            count,
            x_range,
            y_range,
            args.resolution,
            motion_compensate,
        )
        rgb = density_to_rgb(density, occupancy, velocity)
        title = (
            f"{count} radar frames | occupied={metadata['occupied_cells']} | "
            f"points={metadata['in_range_radar_points']}"
        )
        rgb = add_title(rgb, title)
        image_path = output_dir / f"{scene['name']}_radar_stack_{count:03d}.png"
        npz_path = output_dir / f"{scene['name']}_radar_stack_{count:03d}.npz"
        write_image(image_path, rgb)
        np.savez_compressed(
            npz_path,
            density=density.astype(np.float32),
            occupancy=occupancy.astype(np.float32),
            max_abs_velocity=velocity.astype(np.float32),
            metadata_json=json.dumps(metadata, indent=2),
        )
        panels.append(rgb)
        summary["outputs"].append({
            "stack_count": count,
            "image_path": str(image_path),
            "npz_path": str(npz_path),
            **metadata,
        })
        print(
            f"{count:>3} frames -> occupied cells={metadata['occupied_cells']}, "
            f"in-range points={metadata['in_range_radar_points']}, image={image_path}"
        )

    comparison = make_comparison_panel(panels)
    comparison_path = output_dir / f"{scene['name']}_radar_stack_comparison.png"
    metadata_path = output_dir / f"{scene['name']}_radar_stack_summary.json"
    write_image(comparison_path, comparison)
    metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote comparison: {comparison_path}")
    print(f"Wrote summary: {metadata_path}")


if __name__ == "__main__":
    main()
