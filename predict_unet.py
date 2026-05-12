from pathlib import Path
import argparse

import numpy as np
import torch

from bev_projection import write_image
from unet_model import SmallUNet


DEFAULT_MODEL_PATH = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\models"
    r"\unet_bev_mask.pt"
)
DEFAULT_SAMPLE = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\unet_dataset"
    r"\sample_000000.npz"
)
DEFAULT_SAMPLE_DIR = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\unet_dataset"
)
DEFAULT_OUTPUT = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\unet_predictions"
    r"\sample_000000_overlay.png"
)
DEFAULT_OUTPUT_DIR = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\unet_predictions"
)


def load_sample(path: Path):
    with np.load(path) as data:
        x = data["input"].astype(np.float32)
        y = data["target"].astype(np.float32)
    return x, y


def make_input_preview(x: np.ndarray) -> np.ndarray:
    # red=radar density, green=LiDAR height, blue=LiDAR density
    radar_density = x[2] if x.shape[0] > 2 else np.zeros_like(x[0])
    lidar_height = x[1] if x.shape[0] > 1 else np.zeros_like(x[0])
    lidar_density = x[0]
    rgb = np.stack([radar_density, lidar_height, lidar_density], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def bounding_box(mask: np.ndarray):
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def draw_box(rgb: np.ndarray, box, color):
    output = np.array(rgb, copy=True)
    if box is None:
        return output

    row_start, row_end, col_start, col_end = box
    output[row_start:row_end + 1, col_start, :] = color
    output[row_start:row_end + 1, col_end, :] = color
    output[row_start, col_start:col_end + 1, :] = color
    output[row_end, col_start:col_end + 1, :] = color
    return output


def limit_prediction_area(probs: np.ndarray, threshold: float, max_area_fraction: float) -> np.ndarray:
    predicted = probs >= threshold
    max_cells = max(1, int(probs.size * max_area_fraction))

    if int(np.count_nonzero(predicted)) <= max_cells:
        return predicted

    flat_probs = probs.reshape(-1)
    top_indices = np.argpartition(flat_probs, -max_cells)[-max_cells:]
    limited = np.zeros_like(flat_probs, dtype=bool)
    limited[top_indices] = True
    return limited.reshape(probs.shape)


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not np.any(mask):
        return mask

    visited = np.zeros_like(mask, dtype=bool)
    best_component = []
    height, width = mask.shape

    starts = np.argwhere(mask)
    for start_row, start_col in starts:
        if visited[start_row, start_col]:
            continue

        stack = [(int(start_row), int(start_col))]
        visited[start_row, start_col] = True
        component = []

        while stack:
            row, col = stack.pop()
            component.append((row, col))

            for next_row, next_col in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if next_row < 0 or next_row >= height or next_col < 0 or next_col >= width:
                    continue
                if visited[next_row, next_col] or not mask[next_row, next_col]:
                    continue

                visited[next_row, next_col] = True
                stack.append((next_row, next_col))

        if len(component) > len(best_component):
            best_component = component

    output = np.zeros_like(mask, dtype=bool)
    if best_component:
        rows, cols = zip(*best_component)
        output[list(rows), list(cols)] = True

    return output


def postprocess_prediction(probs: np.ndarray, threshold: float, max_area_fraction: float) -> np.ndarray:
    if max_area_fraction >= 1.0:
        return probs >= threshold

    area_limited = limit_prediction_area(probs, threshold, max_area_fraction)
    return largest_connected_component(area_limited)


def overlay_masks(rgb: np.ndarray, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    output = np.array(rgb, copy=True)

    actual_bool = actual.astype(bool)
    predicted_bool = predicted.astype(bool)
    overlap = actual_bool & predicted_bool
    actual_only = actual_bool & ~predicted_bool
    predicted_only = predicted_bool & ~actual_bool

    output[actual_only] = [255, 0, 0]
    output[predicted_only] = [0, 0, 255]

    output = draw_box(output, bounding_box(actual_bool), [255, 0, 0])
    output = draw_box(output, bounding_box(predicted_bool), [0, 0, 255])
    output[overlap] = [255, 0, 255]
    return output


def mask_iou(predicted: np.ndarray, actual: np.ndarray):
    predicted = predicted.astype(bool)
    actual = actual.astype(bool)
    intersection = np.logical_and(predicted, actual).sum()
    union = np.logical_or(predicted, actual).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def load_model(model_path: Path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = SmallUNet(
        in_channels=checkpoint["in_channels"],
        base_channels=checkpoint["base_channels"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def predict_one(model, device, sample_path: Path, output_path: Path, threshold: float, max_prediction_area_fraction: float):
    x, actual = load_sample(sample_path)
    with torch.no_grad():
        tensor = torch.from_numpy(x[None, :, :, :]).to(device)
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    predicted = postprocess_prediction(probs, threshold, max_prediction_area_fraction)
    preview = make_input_preview(x)
    overlay = overlay_masks(preview, actual >= 0.5, predicted)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_image(output_path, overlay)

    iou = mask_iou(predicted, actual >= 0.5)
    print(f"Sample: {sample_path}")
    print(f"Output overlay: {output_path}")
    print(f"IoU: {iou:.4f}")
    print(f"Predicted area fraction: {np.count_nonzero(predicted) / predicted.size:.4f}")
    print(f"Actual box: {bounding_box(actual >= 0.5)}")
    print(f"Predicted box: {bounding_box(predicted)}")
    return iou


def main():
    parser = argparse.ArgumentParser(
        description="Predict a BEV fault mask with U-Net and save red/blue overlay."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sample", default=None)
    parser.add_argument("--sample-dir", default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-outputs", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-prediction-area-fraction", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(Path(args.model_path), device)
    threshold = args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5)

    print(f"Model: {args.model_path}")
    print(f"Threshold: {threshold:.3f}")
    print("Overlay colors: red=actual mask, blue=predicted mask, magenta=overlap")

    if args.sample is not None:
        predict_one(
            model,
            device,
            Path(args.sample),
            Path(args.output),
            threshold,
            args.max_prediction_area_fraction,
        )
        return

    sample_paths = sorted(Path(args.sample_dir).glob("sample_*.npz"))[:args.num_outputs]
    if not sample_paths:
        sample_paths = [Path(DEFAULT_SAMPLE)]

    ious = []
    output_dir = Path(args.output_dir)
    for sample_path in sample_paths:
        output_path = output_dir / f"{sample_path.stem}_overlay.png"
        ious.append(
            predict_one(
                model,
                device,
                sample_path,
                output_path,
                threshold,
                args.max_prediction_area_fraction,
            )
        )

    print(f"Wrote {len(ious)} overlay images to {output_dir}")
    print(f"Mean IoU over saved outputs: {sum(ious) / len(ious):.4f}")


if __name__ == "__main__":
    main()
