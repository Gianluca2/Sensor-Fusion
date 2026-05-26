from pathlib import Path
import argparse

import numpy as np
import torch

from autoencoder_model import BEVAutoEncoder
from bev_projection import write_image
from bev_fault_visualization import (
    bounding_box,
    make_input_preview,
    mask_iou,
    overlay_masks,
    probability_heatmap,
)


DEFAULT_MODEL_PATH = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\models"
    r"\bev_autoencoder.pt"
)
DEFAULT_SAMPLE_DIR = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\autoencoder_dataset"
)
DEFAULT_OUTPUT_DIR = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\autoencoder_predictions"
)


def load_sample(path: Path):
    with np.load(path) as data:
        x = data["input"].astype(np.float32)
        clean = data["clean"].astype(np.float32) if "clean" in data.files else x
        y = data["target"].astype(np.float32)
    return x, clean, y


def load_model(model_path: Path, device):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    model = BEVAutoEncoder(
        in_channels=checkpoint["in_channels"],
        base_channels=checkpoint["base_channels"],
        dropout=checkpoint.get("dropout", 0.0),
        depth=checkpoint.get("depth", 4),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def reconstruction_error(reconstruction: np.ndarray, corrupted: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(reconstruction - corrupted), axis=0)


def predict_from_error(error_map: np.ndarray, error_threshold: float | None, error_percentile: float):
    if error_threshold is not None:
        return error_map >= error_threshold

    keep_fraction = max(0.0, min(1.0, (100.0 - error_percentile) / 100.0))
    keep_cells = max(1, int(error_map.size * keep_fraction))
    flat = error_map.reshape(-1)
    top_indices = np.argpartition(flat, -keep_cells)[-keep_cells:]
    predicted = np.zeros_like(flat, dtype=bool)
    predicted[top_indices] = True
    return predicted.reshape(error_map.shape)


def normalized_error(error_map: np.ndarray) -> np.ndarray:
    error_min = float(np.min(error_map))
    error_max = float(np.max(error_map))
    if error_max <= error_min:
        return np.zeros_like(error_map, dtype=np.float32)
    return ((error_map - error_min) / (error_max - error_min)).astype(np.float32)


def predict_one(
    model,
    device,
    sample_path: Path,
    output_path: Path,
    error_threshold: float | None,
    error_percentile: float,
):
    x, clean, actual = load_sample(sample_path)
    with torch.no_grad():
        tensor = torch.from_numpy(x[None, :, :, :]).to(device)
        reconstruction = model(tensor)[0].cpu().numpy()

    error_map = reconstruction_error(reconstruction, x)
    predicted = predict_from_error(error_map, error_threshold, error_percentile)
    preview = make_input_preview(x)
    reconstruction_preview = make_input_preview(reconstruction)
    overlay = overlay_masks(preview, actual >= 0.5, predicted)
    error_preview = probability_heatmap(normalized_error(error_map))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_image(output_path, overlay)
    error_path = output_path.with_name(f"{output_path.stem}_error.png")
    reconstruction_path = output_path.with_name(f"{output_path.stem}_reconstruction.png")
    clean_path = output_path.with_name(f"{output_path.stem}_clean.png")
    write_image(error_path, error_preview)
    write_image(reconstruction_path, reconstruction_preview)
    write_image(clean_path, make_input_preview(clean))

    iou = mask_iou(predicted, actual >= 0.5)
    print(f"Sample: {sample_path}")
    print(f"Output overlay: {output_path}")
    print(f"Output error heatmap: {error_path}")
    print(f"Output reconstruction: {reconstruction_path}")
    print(f"IoU: {iou:.4f}")
    print(f"Max reconstruction error: {float(np.max(error_map)):.4f}")
    print(f"Predicted area fraction: {np.count_nonzero(predicted) / predicted.size:.4f}")
    print(f"Actual box: {bounding_box(actual >= 0.5)}")
    print(f"Predicted box: {bounding_box(predicted)}")
    return iou


def main():
    parser = argparse.ArgumentParser(
        description="Detect BEV faults from autoencoder reconstruction error and save red/blue overlays."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sample", default=None)
    parser.add_argument("--sample-dir", default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-outputs", type=int, default=10)
    parser.add_argument("--error-threshold", type=float, default=None)
    parser.add_argument("--error-percentile", type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(Path(args.model_path), device)
    error_threshold = (
        args.error_threshold
        if args.error_threshold is not None
        else checkpoint.get("error_threshold")
    )
    error_percentile = (
        args.error_percentile
        if args.error_percentile is not None
        else checkpoint.get("error_percentile", 98.0)
    )

    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    print(f"Error threshold: {error_threshold}")
    print(f"Error percentile: {error_percentile:.2f}")
    print("Overlay colors: red=actual mask, blue=predicted mask, magenta=overlap")

    output_dir = Path(args.output_dir)
    if args.sample is not None:
        predict_one(
            model,
            device,
            Path(args.sample),
            output_dir / f"{Path(args.sample).stem}_overlay.png",
            error_threshold,
            error_percentile,
        )
        return

    sample_paths = sorted(Path(args.sample_dir).glob("sample_*.npz"))[:args.num_outputs]
    if not sample_paths:
        raise FileNotFoundError(f"No sample_*.npz files found in {args.sample_dir}")

    ious = []
    for sample_path in sample_paths:
        output_path = output_dir / f"{sample_path.stem}_overlay.png"
        ious.append(
            predict_one(
                model,
                device,
                sample_path,
                output_path,
                error_threshold,
                error_percentile,
            )
        )

    print(f"Wrote {len(ious)} overlay images to {output_dir}")
    print(f"Mean IoU over saved outputs: {sum(ious) / len(ious):.4f}")


if __name__ == "__main__":
    main()
