from pathlib import Path
import argparse
from datetime import datetime
import json
import random

import numpy as np

from bev_projection import (
    make_rgb_preview,
    project_lidar_bev,
    project_radar_bev,
    write_image,
)
from build_bev_dataset import (
    find_aeva_dir,
    find_first_file,
    invert_transform,
    load_frames,
    load_poses,
    nearest_by_timestamp,
    pose_to_transform,
    transform_xyz,
)
from make_autoencoder_dataset import build_occupancy_map, make_sample
from bev_fault_visualization import bounding_box, draw_box
from visualize_lidar import read_aeva_bin
from visualize_radar import read_continental_bin


DEFAULT_DATA_ROOT = r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\Data"
DEFAULT_OUTPUT_DIR = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs"
    r"\radar_on_masked_lidar"
)
LIDAR_LAYERS = [
    "lidar_density",
    "lidar_height",
    "lidar_occupied_voxel_count",
    "lidar_height_spread",
    "lidar_height_bin_occupancy_ratio",
]


def find_continental_dir(root: Path):
    candidates = []
    for path in root.rglob("*"):
        if not path.is_dir() or not list(path.glob("*.bin")):
            continue

        lower_parts = [part.lower() for part in path.parts]
        if "continentalobject" in lower_parts or "continental_object" in lower_parts:
            continue
        if path.name.lower() == "continental" or "continental" in lower_parts:
            candidates.append(path)

    return sorted(candidates, key=lambda candidate: len(candidate.parts))[0] if candidates else None


def discover_scenes(data_root: Path):
    scenes = []
    for aeva_gt in data_root.rglob("Aeva_gt.txt"):
        scene_root = aeva_gt.parent.parent
        aeva_dir = find_aeva_dir(scene_root)
        radar_dir = find_continental_dir(scene_root)
        radar_gt = find_first_file(scene_root, "Continental_gt.txt")

        if aeva_dir and radar_dir and aeva_gt and radar_gt:
            scenes.append({
                "name": scene_root.name,
                "root": scene_root,
                "aeva_dir": aeva_dir,
                "radar_dir": radar_dir,
                "aeva_gt": aeva_gt,
                "radar_gt": radar_gt,
            })

    unique = {}
    for scene in scenes:
        unique[str(scene["root"]).lower()] = scene

    return sorted(unique.values(), key=lambda scene: str(scene["root"]).lower())


def choose_scene_and_index(data_root: Path, scene_name: str | None, frame_index: int):
    scenes = discover_scenes(data_root)
    if not scenes:
        raise FileNotFoundError(f"No scenes with Aeva and Continental data found under {data_root}")

    if scene_name is not None:
        scenes = [scene for scene in scenes if scene["name"].lower() == scene_name.lower()]
        if not scenes:
            raise FileNotFoundError(f"Scene '{scene_name}' was not found under {data_root}")

    scene = scenes[0]
    lidar_frames = load_frames(scene["aeva_dir"])
    if frame_index < 0 or frame_index >= len(lidar_frames):
        raise IndexError(f"Frame index {frame_index} is outside 0..{len(lidar_frames) - 1}")

    return scene, lidar_frames, frame_index


def aggregate_lidar(lidar_frames, lidar_poses, start_index, scan_count):
    reference = lidar_frames[start_index]
    reference_pose = nearest_by_timestamp(reference["timestamp"], lidar_poses)
    reference_to_world = pose_to_transform(reference_pose)
    world_to_reference = invert_transform(reference_to_world)
    aggregated = []

    for frame in lidar_frames[start_index:start_index + scan_count]:
        pose = nearest_by_timestamp(frame["timestamp"], lidar_poses)
        sensor_to_reference = world_to_reference @ pose_to_transform(pose)
        lidar_xyz = read_aeva_bin(frame["path"])[:, :3]
        aggregated.append(transform_xyz(lidar_xyz, sensor_to_reference).astype(np.float32))

    return np.vstack(aggregated), reference, reference_pose


def aggregate_radar_for_lidar_reference(
    lidar_frames,
    radar_frames,
    lidar_poses,
    radar_poses,
    start_index,
    scan_count,
):
    reference = lidar_frames[start_index]
    reference_pose = nearest_by_timestamp(reference["timestamp"], lidar_poses)
    reference_to_world = pose_to_transform(reference_pose)
    world_to_reference = invert_transform(reference_to_world)
    aggregated = []
    matched = []

    for lidar_frame in lidar_frames[start_index:start_index + scan_count]:
        radar_frame = nearest_by_timestamp(lidar_frame["timestamp"], radar_frames)
        radar_pose = nearest_by_timestamp(radar_frame["timestamp"], radar_poses)
        radar_to_reference = world_to_reference @ pose_to_transform(radar_pose)
        radar_points = read_continental_bin(radar_frame["path"])
        radar_points_lidar = radar_points.copy()
        radar_points_lidar[:, :3] = transform_xyz(radar_points[:, :3], radar_to_reference)
        aggregated.append(radar_points_lidar.astype(np.float32))
        matched.append({
            "lidar_timestamp": lidar_frame["timestamp"],
            "radar_timestamp": radar_frame["timestamp"],
            "delta_ms": (radar_frame["timestamp"] - lidar_frame["timestamp"]) / 1_000_000.0,
            "radar_path": str(radar_frame["path"]),
        })

    return np.vstack(aggregated), matched


def aggregate_radar_history_for_lidar_reference(
    lidar_frame,
    radar_frames,
    lidar_poses,
    radar_poses,
    history_scans,
):
    reference_pose = nearest_by_timestamp(lidar_frame["timestamp"], lidar_poses)
    reference_to_world = pose_to_transform(reference_pose)
    world_to_reference = invert_transform(reference_to_world)

    nearest_radar = nearest_by_timestamp(lidar_frame["timestamp"], radar_frames)
    nearest_index = next(
        index for index, frame in enumerate(radar_frames)
        if frame["path"] == nearest_radar["path"]
    )
    start_index = max(0, nearest_index - history_scans)
    selected_radar_frames = radar_frames[start_index:nearest_index + 1]

    aggregated = []
    matched = []
    for radar_frame in selected_radar_frames:
        radar_pose = nearest_by_timestamp(radar_frame["timestamp"], radar_poses)
        radar_to_reference = world_to_reference @ pose_to_transform(radar_pose)
        radar_points = read_continental_bin(radar_frame["path"])
        radar_points_lidar = radar_points.copy()
        radar_points_lidar[:, :3] = transform_xyz(radar_points[:, :3], radar_to_reference)
        aggregated.append(radar_points_lidar.astype(np.float32))
        matched.append({
            "radar_timestamp": radar_frame["timestamp"],
            "delta_ms": (radar_frame["timestamp"] - lidar_frame["timestamp"]) / 1_000_000.0,
            "radar_path": str(radar_frame["path"]),
        })

    return np.vstack(aggregated), matched


def masked_lidar_preview(lidar_layers: dict, fault_mask: np.ndarray):
    masked_layers = {}
    for key, value in lidar_layers.items():
        masked = np.array(value, copy=True)
        if key in LIDAR_LAYERS:
            masked[fault_mask] = 0.0
        masked_layers[key] = masked

    return make_rgb_preview(masked_layers), masked_layers


def overlay_radar(masked_lidar_rgb: np.ndarray, radar_density: np.ndarray, fault_mask: np.ndarray, threshold: float):
    output = np.array(masked_lidar_rgb, copy=True)
    radar_cells = radar_density > threshold
    radar_inside_mask = radar_cells & fault_mask
    radar_outside_mask = radar_cells & ~fault_mask

    radar_intensity = np.clip(radar_density * 255.0, 40, 255).astype(np.uint8)
    output[radar_outside_mask, 0] = np.maximum(output[radar_outside_mask, 0], radar_intensity[radar_outside_mask])
    output[radar_outside_mask, 1] = (output[radar_outside_mask, 1] * 0.35).astype(np.uint8)
    output[radar_outside_mask, 2] = (output[radar_outside_mask, 2] * 0.35).astype(np.uint8)

    output[radar_inside_mask] = [255, 0, 255]
    output = draw_box(output, bounding_box(fault_mask), [255, 255, 0])
    return output, radar_cells, radar_inside_mask


def radar_preview(radar_density: np.ndarray, fault_mask: np.ndarray):
    rgb = np.zeros((*radar_density.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(radar_density * 255.0, 0, 255).astype(np.uint8)
    rgb[fault_mask & (radar_density > 0.0)] = [255, 0, 255]
    return draw_box(rgb, bounding_box(fault_mask), [255, 255, 0])


def main():
    parser = argparse.ArgumentParser(
        description="Overlay unmasked radar BEV over a masked LiDAR BEV to inspect radar coverage."
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--scene", default=None, help="Optional scene folder name, e.g. Day_1_Parking.")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--aggregate-scans", type=int, default=1)
    parser.add_argument(
        "--radar-history-scans",
        type=int,
        default=20,
        help="Number of previous Continental radar scans to aggregate before the matched scan.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=80.0)
    parser.add_argument("--y-min", type=float, default=-40.0)
    parser.add_argument("--y-max", type=float, default=40.0)
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--z-min", type=float, default=-4.0)
    parser.add_argument("--z-max", type=float, default=6.0)
    parser.add_argument("--z-resolution", type=float, default=0.5)
    parser.add_argument("--radar-threshold", type=float, default=0.0)
    parser.add_argument("--min-mask-height", type=int, default=25)
    parser.add_argument("--max-mask-height", type=int, default=90)
    parser.add_argument("--min-mask-width", type=int, default=25)
    parser.add_argument("--max-mask-width", type=int, default=90)
    parser.add_argument("--min-mask-area-fraction", type=float, default=0.01)
    parser.add_argument("--max-mask-area-fraction", type=float, default=0.03)
    parser.add_argument("--min-mask-occupied-cells", type=int, default=50)
    parser.add_argument("--occupancy-threshold", type=float, default=0.0)
    parser.add_argument("--max-mask-attempts", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.SystemRandom().randint(0, 2**32 - 1)
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_root = Path(args.data_root)
    scene, lidar_frames, frame_index = choose_scene_and_index(data_root, args.scene, args.frame_index)
    radar_frames = load_frames(scene["radar_dir"])
    lidar_poses = load_poses(scene["aeva_gt"])
    radar_poses = load_poses(scene["radar_gt"])

    x_range = (args.x_min, args.x_max)
    y_range = (args.y_min, args.y_max)
    lidar_xyz, reference_lidar, _ = aggregate_lidar(
        lidar_frames,
        lidar_poses,
        frame_index,
        args.aggregate_scans,
    )
    reference_for_radar = lidar_frames[frame_index]
    radar_points, matched_radar = aggregate_radar_history_for_lidar_reference(
        reference_for_radar,
        radar_frames,
        lidar_poses,
        radar_poses,
        history_scans=args.radar_history_scans,
    )

    lidar_layers = project_lidar_bev(
        lidar_xyz,
        x_range,
        y_range,
        args.resolution,
        z_range=(args.z_min, args.z_max),
        z_resolution=args.z_resolution,
    )
    radar_layers = project_radar_bev(radar_points, x_range, y_range, args.resolution)
    clean_lidar_rgb = make_rgb_preview(lidar_layers)

    clean_tensor = np.stack([lidar_layers[layer] for layer in LIDAR_LAYERS], axis=0)
    occupancy = build_occupancy_map(clean_tensor, args.occupancy_threshold)
    _, fault_target, mask_metadata = make_sample(clean_tensor, occupancy, sample_index=0, args=args)
    fault_mask = fault_target.astype(bool)

    masked_rgb, _ = masked_lidar_preview(lidar_layers, fault_mask)
    overlay_rgb, radar_cells, radar_inside_mask = overlay_radar(
        masked_rgb,
        radar_layers["radar_density"],
        fault_mask,
        args.radar_threshold,
    )
    radar_rgb = radar_preview(radar_layers["radar_density"], fault_mask)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"radar_on_masked_lidar_{timestamp}_seed_{args.seed}"
    clean_path = output_dir / f"{stem}_clean_lidar.png"
    masked_path = output_dir / f"{stem}_masked_lidar.png"
    radar_path = output_dir / f"{stem}_radar_only.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    metadata_path = output_dir / f"{stem}_metadata.json"

    write_image(clean_path, clean_lidar_rgb)
    write_image(masked_path, draw_box(masked_rgb, bounding_box(fault_mask), [255, 255, 0]))
    write_image(radar_path, radar_rgb)
    write_image(overlay_path, overlay_rgb)

    radar_inside_count = int(np.count_nonzero(radar_inside_mask))
    radar_total_count = int(np.count_nonzero(radar_cells))
    mask_area = int(np.count_nonzero(fault_mask))
    metadata = {
        "scene": scene["name"],
        "reference_lidar_path": str(reference_lidar["path"]),
        "reference_lidar_timestamp": reference_lidar["timestamp"],
        "aggregate_scans": args.aggregate_scans,
        "radar_history_scans": args.radar_history_scans,
        "radar_aggregated_scan_count": len(matched_radar),
        "matched_radar": matched_radar,
        "mask": mask_metadata["mask"],
        "radar_cells_total": radar_total_count,
        "radar_cells_inside_mask": radar_inside_count,
        "mask_area_cells": mask_area,
        "radar_inside_mask_fraction_of_mask": radar_inside_count / mask_area if mask_area else 0.0,
        "outputs": {
            "clean_lidar": str(clean_path),
            "masked_lidar": str(masked_path),
            "radar_only": str(radar_path),
            "overlay": str(overlay_path),
        },
        "color_legend": {
            "yellow": "actual LiDAR mask outline",
            "red": "unmasked radar cells outside the LiDAR mask",
            "magenta": "unmasked radar cells inside the LiDAR mask",
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Scene: {scene['name']}")
    print(f"Reference LiDAR: {reference_lidar['path']}")
    print(f"Seed: {args.seed}")
    print(f"Clean LiDAR PNG: {clean_path}")
    print(f"Masked LiDAR PNG: {masked_path}")
    print(f"Radar-only PNG: {radar_path}")
    print(f"Overlay PNG: {overlay_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Radar cells total: {radar_total_count}")
    print(f"Radar cells inside mask: {radar_inside_count}")
    print(f"Radar inside mask / mask area: {metadata['radar_inside_mask_fraction_of_mask']:.4f}")
    print(f"Radar scans aggregated: {len(matched_radar)}")
    print("Colors: yellow=mask outline, red=radar outside mask, magenta=radar inside mask")


if __name__ == "__main__":
    main()
