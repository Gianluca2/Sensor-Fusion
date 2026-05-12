from pathlib import Path
import argparse
import json
import random

import numpy as np


DEFAULT_BEV = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs"
    r"\bev\bev_match_000000.npz"
)
DEFAULT_OUTPUT_DIR = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\unet_dataset"
)
DEFAULT_BEV_DIR = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\bev_multi"
)
DEFAULT_LAYERS = [
    "lidar_density",
    "lidar_height",
    "radar_density",
    "radar_velocity",
    "radar_range_min",
    "radar_rcs_max",
]
OCCUPANCY_LAYER_INDICES = [0, 1, 2]


def load_clean_bev(path: Path, layers):
    with np.load(path) as data:
        missing = [layer for layer in layers if layer not in data.files]
        if missing:
            raise ValueError(f"Missing layers in {path}: {missing}")

        channels = [data[layer].astype(np.float32) for layer in layers]
        metadata = {}
        if "metadata_json" in data.files:
            metadata = json.loads(str(data["metadata_json"].item()))

    return np.stack(channels, axis=0), metadata


def find_bev_files(bev_dir: Path):
    return sorted(
        path for path in bev_dir.rglob("*.npz")
        if path.name != "manifest.npz"
    )


def random_rect(height, width, min_h, max_h, min_w, max_w, min_area, max_area):
    for _ in range(1000):
        rect_h = random.randint(min_h, max_h)
        rect_w = random.randint(min_w, max_w)
        area = rect_h * rect_w
        if min_area <= area <= max_area:
            break
    else:
        raise RuntimeError(
            "Could not sample a mask inside the requested area limits. "
            "Adjust --min-mask-area-fraction, --max-mask-area-fraction, or mask height/width limits."
        )

    row = random.randint(0, max(0, height - rect_h))
    col = random.randint(0, max(0, width - rect_w))
    return row, row + rect_h, col, col + rect_w


def build_occupancy_map(clean: np.ndarray, threshold: float) -> np.ndarray:
    occupancy_layers = clean[OCCUPANCY_LAYER_INDICES]
    return np.any(occupancy_layers > threshold, axis=0)


def choose_nonempty_rect(occupancy: np.ndarray, args):
    height, width = occupancy.shape
    min_area = int(height * width * args.min_mask_area_fraction)
    max_area = int(height * width * args.max_mask_area_fraction)

    for _ in range(args.max_mask_attempts):
        row_start, row_end, col_start, col_end = random_rect(
            height,
            width,
            min_h=args.min_mask_height,
            max_h=args.max_mask_height,
            min_w=args.min_mask_width,
            max_w=args.max_mask_width,
            min_area=min_area,
            max_area=max_area,
        )
        occupied_cells = int(np.count_nonzero(occupancy[row_start:row_end, col_start:col_end]))
        mask_area = (row_end - row_start) * (col_end - col_start)

        if occupied_cells >= args.min_mask_occupied_cells and min_area <= mask_area <= max_area:
            return row_start, row_end, col_start, col_end, occupied_cells, mask_area

    raise RuntimeError(
        "Could not find a mask region with enough occupied BEV cells. "
        "Try lowering --min-mask-occupied-cells or increasing --max-mask-attempts."
    )


def make_sample(clean: np.ndarray, occupancy: np.ndarray, sample_index: int, args):
    _, height, width = clean.shape
    row_start, row_end, col_start, col_end, occupied_cells, mask_area = choose_nonempty_rect(
        occupancy,
        args,
    )

    faulty = np.array(clean, copy=True)
    faulty[:, row_start:row_end, col_start:col_end] = 0.0

    target = np.zeros((height, width), dtype=np.float32)
    target[row_start:row_end, col_start:col_end] = 1.0

    metadata = {
        "sample_index": sample_index,
        "mask": {
            "row_start": row_start,
            "row_end": row_end,
            "col_start": col_start,
            "col_end": col_end,
            "area_cells": mask_area,
            "area_fraction": mask_area / (height * width),
            "occupied_cells_before_masking": occupied_cells,
        },
    }

    return faulty, target, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Create synthetic BEV mask samples for U-Net training."
    )
    parser.add_argument("--bev", default=DEFAULT_BEV)
    parser.add_argument(
        "--bev-dir",
        default=None,
        help="Directory tree of clean BEV .npz files. If provided, samples are drawn across all files.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-mask-height", type=int, default=25)
    parser.add_argument("--max-mask-height", type=int, default=90)
    parser.add_argument("--min-mask-width", type=int, default=25)
    parser.add_argument("--max-mask-width", type=int, default=90)
    parser.add_argument(
        "--min-mask-area-fraction",
        type=float,
        default=0.01,
        help="Minimum mask area as a fraction of the full BEV image area.",
    )
    parser.add_argument(
        "--max-mask-area-fraction",
        type=float,
        default=0.03,
        help="Maximum mask area as a fraction of the full BEV image area.",
    )
    parser.add_argument(
        "--min-mask-occupied-cells",
        type=int,
        default=50,
        help="Minimum occupied BEV cells required inside a sampled mask rectangle.",
    )
    parser.add_argument(
        "--occupancy-threshold",
        type=float,
        default=0.0,
        help="A BEV cell is occupied if any occupancy layer is greater than this value.",
    )
    parser.add_argument(
        "--max-mask-attempts",
        type=int,
        default=500,
        help="Maximum random rectangles to try per training sample.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.bev_dir is not None:
        bev_files = find_bev_files(Path(args.bev_dir))
        if not bev_files:
            raise FileNotFoundError(f"No BEV .npz files found under {args.bev_dir}")
    else:
        bev_files = [Path(args.bev)]

    manifest = {
        "source_bev": str(Path(args.bev)) if args.bev_dir is None else None,
        "source_bev_dir": str(Path(args.bev_dir)) if args.bev_dir is not None else None,
        "source_bev_count": len(bev_files),
        "num_samples": args.num_samples,
        "layers": DEFAULT_LAYERS,
        "occupancy_layers": [DEFAULT_LAYERS[index] for index in OCCUPANCY_LAYER_INDICES],
        "min_mask_occupied_cells": args.min_mask_occupied_cells,
        "min_mask_area_fraction": args.min_mask_area_fraction,
        "max_mask_area_fraction": args.max_mask_area_fraction,
        "samples": [],
    }

    for index in range(args.num_samples):
        source_bev = bev_files[index % len(bev_files)]
        clean, source_metadata = load_clean_bev(source_bev, DEFAULT_LAYERS)
        occupancy = build_occupancy_map(clean, args.occupancy_threshold)
        faulty, target, sample_metadata = make_sample(clean, occupancy, index, args)
        sample_path = output_dir / f"sample_{index:06d}.npz"
        sample_metadata["source_bev"] = str(source_bev)
        sample_metadata["source_metadata"] = source_metadata

        np.savez_compressed(
            sample_path,
            input=faulty.astype(np.float32),
            target=target.astype(np.float32),
            metadata_json=json.dumps(sample_metadata, indent=2),
        )
        manifest["samples"].append({
            "path": str(sample_path),
            **sample_metadata,
        })

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {args.num_samples} samples to {output_dir}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Source BEV files: {len(bev_files)}")
    print(f"Minimum occupied cells per mask: {args.min_mask_occupied_cells}")
    print(f"Minimum mask area fraction: {args.min_mask_area_fraction:.3f}")
    print(f"Maximum mask area fraction: {args.max_mask_area_fraction:.3f}")


if __name__ == "__main__":
    main()
