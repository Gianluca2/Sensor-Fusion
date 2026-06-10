from pathlib import Path
import argparse
import json
import sys

import numpy as np
from PIL import Image, ImageDraw

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from bev_fault_visualization import make_input_preview, overlay_masks, probability_heatmap
from bev_projection import write_image
from rewrite_model_v3_samples import (
    FAULT_TYPES,
    SEVERITIES,
    V3_LAYERS,
    build_clean_faulty_and_radar_scan_lists,
    difference_target,
    prepare_scenes,
    project_lidar_bev_v3,
    project_radar_bev_v3,
    stack_layers,
)


DEFAULT_DATA_ROOT = r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\Data"
DEFAULT_OUTPUT_DIR = r"C:\Users\gianl\ThesisOutputs\HerculesFiles\outputs\model_v3_fault_diagnostics"


def add_title(rgb: np.ndarray, title: str) -> np.ndarray:
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 22), fill=(0, 0, 0))
    draw.text((5, 5), title, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def add_text_box(rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    padding = 4
    line_height = 13
    width = int(max(draw.textlength(line) for line in lines)) + padding * 2
    height = len(lines) * line_height + padding * 2
    draw.rectangle((0, 0, width, height), fill=(0, 0, 0))
    for index, line in enumerate(lines):
        draw.text((padding, padding + index * line_height), line, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def make_panel(clean, faulty, target, fault_type: str, severity: str, seed: int, scene_name: str):
    clean_rgb = add_title(make_input_preview(clean), "Clean V3 BEV")
    faulty_rgb = add_title(make_input_preview(faulty), f"Faulty: {fault_type}/{severity}")
    diff = np.max(np.abs(clean - faulty), axis=0)
    diff_rgb = add_title(probability_heatmap(np.clip(diff / max(float(diff.max()), 1e-6), 0.0, 1.0)), "Clean/Faulty Difference")
    overlay = overlay_masks(make_input_preview(faulty), target >= 0.5, np.zeros_like(target, dtype=bool))
    overlay = add_title(overlay, "Target Mask On Faulty BEV")
    overlay = add_text_box(
        overlay,
        [
            f"scene: {scene_name}",
            f"seed: {seed}",
            f"target cells: {int(np.count_nonzero(target))}",
            f"target frac: {np.count_nonzero(target) / target.size:.4f}",
        ],
    )
    top = np.concatenate([clean_rgb, faulty_rgb], axis=1)
    bottom = np.concatenate([diff_rgb, overlay], axis=1)
    return np.concatenate([top, bottom], axis=0)


def select_scene_and_start(scenes, sample_number: int, seed: int):
    rng = np.random.default_rng(seed + sample_number * 1009)
    scene = scenes[int(rng.integers(0, len(scenes)))]
    start_index = int(rng.integers(0, scene["max_start"] + 1))
    return scene, start_index


def main():
    parser = argparse.ArgumentParser(
        description="Write PNG diagnostics showing Model V3 point-level LiDAR fault injections."
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds-per-case", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--aggregate-scans", type=int, default=3)
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=80.0)
    parser.add_argument("--y-min", type=float, default=-40.0)
    parser.add_argument("--y-max", type=float, default=40.0)
    parser.add_argument("--resolution", type=float, default=0.2)
    parser.add_argument("--target-threshold", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes = prepare_scenes(Path(args.data_root), args.aggregate_scans)
    x_range = (args.x_min, args.x_max)
    y_range = (args.y_min, args.y_max)
    written = []
    sample_number = 0

    print(f"Found {len(scenes)} scenes")
    print(f"V3 layers: {', '.join(V3_LAYERS)}")

    for fault_type in FAULT_TYPES:
        for severity in SEVERITIES:
            for seed_offset in range(args.seeds_per_case):
                current_seed = args.seed + seed_offset
                scene, start_index = select_scene_and_start(scenes, sample_number, current_seed)
                rng = np.random.default_rng(current_seed + sample_number)
                clean_scans, faulty_scans, radar_scans, radar_velocities, _, _ = build_clean_faulty_and_radar_scan_lists(
                    scene,
                    start_index,
                    args.aggregate_scans,
                    fault_type,
                    severity,
                    rng,
                )
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
                target = difference_target(clean, faulty, args.target_threshold)
                panel = make_panel(
                    clean,
                    faulty,
                    target,
                    fault_type,
                    severity,
                    current_seed,
                    scene["name"],
                )
                file_name = f"{fault_type}_{severity}_seed_{current_seed}_sample_{sample_number:03d}.png"
                output_path = output_dir / file_name
                write_image(output_path, panel)
                written.append({
                    "path": str(output_path),
                    "fault_type": fault_type,
                    "severity": severity,
                    "seed": current_seed,
                    "scene": scene["name"],
                    "start_index": start_index,
                    "target_cells": int(np.count_nonzero(target)),
                    "target_fraction": float(np.count_nonzero(target) / target.size),
                })
                sample_number += 1
                print(f"Wrote {output_path}", flush=True)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"outputs": written}, indent=2), encoding="utf-8")
    print(f"Wrote {len(written)} diagnostic PNGs to {output_dir}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
