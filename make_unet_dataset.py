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
DEFAULT_LAYERS = [
    "lidar_density",
    "lidar_height",
    "radar_density",
    "radar_velocity",
    "radar_range_min",
    "radar_rcs_max",
]


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


def random_rect(height, width, min_h, max_h, min_w, max_w):
    rect_h = random.randint(min_h, max_h)
    rect_w = random.randint(min_w, max_w)
    row = random.randint(0, max(0, height - rect_h))
    col = random.randint(0, max(0, width - rect_w))
    return row, row + rect_h, col, col + rect_w


def make_sample(clean: np.ndarray, sample_index: int, args):
    _, height, width = clean.shape
    row_start, row_end, col_start, col_end = random_rect(
        height,
        width,
        min_h=args.min_mask_height,
        max_h=args.max_mask_height,
        min_w=args.min_mask_width,
        max_w=args.max_mask_width,
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
        },
    }

    return faulty, target, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Create synthetic BEV mask samples for U-Net training."
    )
    parser.add_argument("--bev", default=DEFAULT_BEV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-mask-height", type=int, default=25)
    parser.add_argument("--max-mask-height", type=int, default=90)
    parser.add_argument("--min-mask-width", type=int, default=25)
    parser.add_argument("--max-mask-width", type=int, default=90)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean, source_metadata = load_clean_bev(Path(args.bev), DEFAULT_LAYERS)

    manifest = {
        "source_bev": str(Path(args.bev)),
        "num_samples": args.num_samples,
        "layers": DEFAULT_LAYERS,
        "source_metadata": source_metadata,
        "samples": [],
    }

    for index in range(args.num_samples):
        faulty, target, sample_metadata = make_sample(clean, index, args)
        sample_path = output_dir / f"sample_{index:06d}.npz"

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
    print(f"Input shape per sample: {clean.shape}")


if __name__ == "__main__":
    main()
