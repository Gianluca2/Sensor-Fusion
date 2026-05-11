from pathlib import Path
import argparse
import json
from datetime import datetime

import numpy as np

from bev_projection import write_image


DEFAULT_INPUT = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs"
    r"\bev\bev_match_000000.npz"
)
DEFAULT_OUTPUT_DIR = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\bev_masked"
)


MASKABLE_LAYERS = [
    "lidar_density",
    "lidar_height",
    "radar_density",
    "radar_velocity",
    "radar_range_min",
    "radar_rcs_max",
]


def load_metadata(npz_file) -> dict:
    if "metadata_json" not in npz_file.files:
        return {}

    raw = npz_file["metadata_json"]
    if raw.shape == ():
        return json.loads(str(raw.item()))

    return json.loads(str(raw))


def metric_mask_to_cells(metadata: dict, x_min, x_max, y_min, y_max):
    grid_rows, grid_cols = metadata["grid_shape"]
    bev_x_min, bev_x_max = metadata["x_range_m"]
    bev_y_min, bev_y_max = metadata["y_range_m"]
    resolution = metadata["resolution_m_per_cell"]

    x_min = max(x_min, bev_x_min)
    x_max = min(x_max, bev_x_max)
    y_min = max(y_min, bev_y_min)
    y_max = min(y_max, bev_y_max)

    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Mask does not overlap the BEV grid")

    col_start = int(np.floor((y_min - bev_y_min) / resolution))
    col_end = int(np.ceil((y_max - bev_y_min) / resolution))

    row_from_bottom_start = int(np.floor((x_min - bev_x_min) / resolution))
    row_from_bottom_end = int(np.ceil((x_max - bev_x_min) / resolution))

    row_start = grid_rows - row_from_bottom_end
    row_end = grid_rows - row_from_bottom_start

    row_start = max(0, min(grid_rows, row_start))
    row_end = max(0, min(grid_rows, row_end))
    col_start = max(0, min(grid_cols, col_start))
    col_end = max(0, min(grid_cols, col_end))

    return row_start, row_end, col_start, col_end


def grid_mask_to_cells(row, col, height, width, grid_shape):
    grid_rows, grid_cols = grid_shape
    row_start = max(0, row)
    row_end = min(grid_rows, row + height)
    col_start = max(0, col)
    col_end = min(grid_cols, col + width)

    if row_start >= row_end or col_start >= col_end:
        raise ValueError("Grid mask does not overlap the BEV grid")

    return row_start, row_end, col_start, col_end


def apply_mask(arrays: dict, row_start, row_end, col_start, col_end):
    output = {}

    for key, value in arrays.items():
        masked = np.array(value, copy=True)

        if key in MASKABLE_LAYERS and masked.ndim == 2:
            masked[row_start:row_end, col_start:col_end] = 0
        elif key == "rgb_preview" and masked.ndim == 3:
            masked[row_start:row_end, col_start:col_end, :] = 0

        output[key] = masked

    mask = np.zeros_like(output["lidar_density"], dtype=np.uint8)
    mask[row_start:row_end, col_start:col_end] = 1
    output["fault_mask"] = mask

    return output


def save_masked_npz(output_path: Path, arrays: dict, metadata: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_arrays = {
        key: value
        for key, value in arrays.items()
        if key != "metadata_json"
    }
    save_arrays["metadata_json"] = json.dumps(metadata, indent=2)

    np.savez_compressed(output_path, **save_arrays)


def main():
    parser = argparse.ArgumentParser(
        description="Inject a rectangular black mask into a LiDAR/radar BEV grid."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input BEV .npz file.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--x-min", type=float, default=20.0)
    parser.add_argument("--x-max", type=float, default=35.0)
    parser.add_argument("--y-min", type=float, default=-5.0)
    parser.add_argument("--y-max", type=float, default=5.0)
    parser.add_argument("--row", type=int, default=None)
    parser.add_argument("--col", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    with np.load(input_path) as data:
        arrays = {key: data[key] for key in data.files}
        metadata = load_metadata(data)

    grid_shape = tuple(arrays["lidar_density"].shape)

    grid_args = [args.row, args.col, args.height, args.width]
    use_grid_mask = any(value is not None for value in grid_args)
    if use_grid_mask and not all(value is not None for value in grid_args):
        raise ValueError("Use all of --row, --col, --height, and --width for grid masks")

    if use_grid_mask:
        row_start, row_end, col_start, col_end = grid_mask_to_cells(
            args.row,
            args.col,
            args.height,
            args.width,
            grid_shape,
        )
        mask_description = {
            "coordinate_type": "grid",
            "row": args.row,
            "col": args.col,
            "height": args.height,
            "width": args.width,
        }
    else:
        row_start, row_end, col_start, col_end = metric_mask_to_cells(
            metadata,
            args.x_min,
            args.x_max,
            args.y_min,
            args.y_max,
        )
        mask_description = {
            "coordinate_type": "metric",
            "x_min_m": args.x_min,
            "x_max_m": args.x_max,
            "y_min_m": args.y_min,
            "y_max_m": args.y_max,
        }

    masked_arrays = apply_mask(arrays, row_start, row_end, col_start, col_end)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{input_path.stem}_masked_{timestamp}"
    npz_path = output_dir / f"{stem}.npz"
    image_path = output_dir / f"{stem}.png"
    manifest_path = output_dir / f"{stem}_manifest.json"

    fault_metadata = {
        "source_bev": str(input_path),
        "masked_bev": str(npz_path),
        "masked_preview": str(image_path),
        "fault_type": "bev_black_mask",
        "mask": {
            **mask_description,
            "row_start": row_start,
            "row_end": row_end,
            "col_start": col_start,
            "col_end": col_end,
        },
        "source_metadata": metadata,
    }

    save_masked_npz(npz_path, masked_arrays, fault_metadata)
    write_image(image_path, masked_arrays["rgb_preview"].astype(np.uint8))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(fault_metadata, indent=2), encoding="utf-8")

    print(f"Wrote masked BEV arrays: {npz_path}")
    print(f"Wrote masked BEV preview: {image_path}")
    print(f"Wrote fault manifest: {manifest_path}")
    print(
        "Masked grid cells: "
        f"rows {row_start}:{row_end}, cols {col_start}:{col_end}"
    )


if __name__ == "__main__":
    main()
