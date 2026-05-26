from pathlib import Path
import argparse
import json
import tempfile
import tkinter as tk

import numpy as np

from bev_projection import write_image, write_ppm


DEFAULT_BEV = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs"
    r"\bev\bev_match_000000.npz"
)
DEFAULT_MASKED_DIR = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\bev_masked"
)
DEFAULT_COMPARISON = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs"
    r"\bev\bev_mask_comparison.png"
)


def latest_masked_bev(masked_dir: Path) -> Path:
    candidates = sorted(
        masked_dir.glob("*_masked_*.npz"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No masked BEV .npz files found in {masked_dir}")

    return candidates[0]


def load_npz_rgb(path: Path):
    with np.load(path) as data:
        if "rgb_preview" not in data.files:
            raise ValueError(f"{path} does not contain rgb_preview")

        rgb = data["rgb_preview"].astype(np.uint8)
        metadata = {}
        fault_mask = None

        if "metadata_json" in data.files:
            metadata = json.loads(str(data["metadata_json"].item()))
        if "fault_mask" in data.files:
            fault_mask = data["fault_mask"].astype(bool)

    return rgb, metadata, fault_mask


def draw_mask_outline(rgb: np.ndarray, fault_mask: np.ndarray | None) -> np.ndarray:
    output = np.array(rgb, copy=True)
    if fault_mask is None or not np.any(fault_mask):
        return output

    rows, cols = np.where(fault_mask)
    row_start = int(np.min(rows))
    row_end = int(np.max(rows))
    col_start = int(np.min(cols))
    col_end = int(np.max(cols))

    output[row_start:row_end + 1, col_start, :] = [255, 0, 0]
    output[row_start:row_end + 1, col_end, :] = [255, 0, 0]
    output[row_start, col_start:col_end + 1, :] = [255, 0, 0]
    output[row_end, col_start:col_end + 1, :] = [255, 0, 0]

    return output


def add_separator(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError(f"BEV images must have same shape, got {left.shape} and {right.shape}")

    separator = np.full((left.shape[0], 8, 3), 255, dtype=np.uint8)
    return np.concatenate([left, separator, right], axis=1)


def save_comparison(original_rgb, masked_rgb, comparison_path: Path):
    comparison = add_separator(original_rgb, masked_rgb)
    write_image(comparison_path, comparison)
    return comparison


def show_image(rgb: np.ndarray, title: str):
    with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as temp:
        temp_path = Path(temp.name)

    write_ppm(temp_path, rgb)

    root = tk.Tk()
    root.title(title)

    image = tk.PhotoImage(file=str(temp_path))
    label = tk.Label(root, image=image)
    label.pack()

    root.mainloop()

    try:
        temp_path.unlink()
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Visualize original and masked LiDAR-only BEV projections."
    )
    parser.add_argument("--bev", default=DEFAULT_BEV, help="Original BEV .npz file.")
    parser.add_argument(
        "--masked-bev",
        default=None,
        help="Masked BEV .npz file. Defaults to the newest file in outputs/bev_masked.",
    )
    parser.add_argument(
        "--comparison-output",
        default=DEFAULT_COMPARISON,
        help="Output side-by-side comparison PPM path.",
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Write the comparison image without opening a viewer window.",
    )
    args = parser.parse_args()

    bev_path = Path(args.bev)
    masked_path = Path(args.masked_bev) if args.masked_bev else latest_masked_bev(Path(DEFAULT_MASKED_DIR))
    comparison_path = Path(args.comparison_output)

    original_rgb, original_metadata, _ = load_npz_rgb(bev_path)
    masked_rgb, masked_metadata, fault_mask = load_npz_rgb(masked_path)
    masked_with_outline = draw_mask_outline(masked_rgb, fault_mask)

    comparison = save_comparison(original_rgb, masked_with_outline, comparison_path)

    print(f"Original BEV: {bev_path}")
    print(f"Masked BEV: {masked_path}")
    print(f"Comparison image: {comparison_path}")
    print(f"Grid shape: {original_rgb.shape[0]} rows x {original_rgb.shape[1]} cols")
    print("Preview channels: red=LiDAR occupied voxel count, green=LiDAR height, blue=LiDAR density")

    mask_info = masked_metadata.get("mask")
    if mask_info:
        print(f"Mask: {mask_info}")

    if not args.save_only:
        show_image(comparison, title="Original BEV | Masked BEV")


if __name__ == "__main__":
    main()
