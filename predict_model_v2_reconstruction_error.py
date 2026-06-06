from pathlib import Path
import argparse
import json

import numpy as np
import torch
from PIL import Image, ImageDraw

from Model_V2 import (
    BEVFaultRestorationModelV2,
    FAULT_CLASSES,
    SEVERITY_CLASSES,
    reconstruction_error_map,
)
from bev_fault_visualization import (
    bounding_box,
    make_input_preview,
    mask_iou,
    overlay_masks,
    probability_heatmap,
)
from bev_projection import write_image


FAST_OUTPUT_ROOT = r"C:\Users\gianl\ThesisOutputs\HerculesFiles\outputs"
DEFAULT_MODEL_PATH = str(Path(FAST_OUTPUT_ROOT) / "models" / "model_v2.pt")
DEFAULT_SAMPLE_DIR = str(Path(FAST_OUTPUT_ROOT) / "autoencoder_dataset")
DEFAULT_OUTPUT_DIR = str(Path(FAST_OUTPUT_ROOT) / "model_v2_reconstruction_error_predictions")
FAULT_TO_INDEX = {name: index for index, name in enumerate(FAULT_CLASSES)}
SEVERITY_TO_INDEX = {name: index for index, name in enumerate(SEVERITY_CLASSES)}


def load_sample(path: Path):
    with np.load(path) as data:
        faulty = data["input"].astype(np.float32)
        clean = data["clean"].astype(np.float32)
        actual_mask = data["target"].astype(np.float32)
        metadata = {}
        if "metadata_json" in data.files:
            metadata = json.loads(str(data["metadata_json"].item()))
    return faulty, clean, actual_mask, metadata


def load_model(model_path: Path, device):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    model = BEVFaultRestorationModelV2(
        in_channels=checkpoint["in_channels"],
        base_channels=checkpoint.get("base_channels", 48),
        depth=checkpoint.get("depth", 4),
        dropout=checkpoint.get("dropout", 0.0),
        num_fault_classes=len(checkpoint.get("fault_classes", FAULT_CLASSES)),
        num_severity_classes=len(checkpoint.get("severity_classes", SEVERITY_CLASSES)),
        fault_embedding_dim=checkpoint.get("fault_embedding_dim", 8),
        severity_embedding_dim=checkpoint.get("severity_embedding_dim", 4),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint


def add_panel_title(rgb: np.ndarray, title: str) -> np.ndarray:
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 20), fill=(0, 0, 0))
    draw.text((4, 4), title, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def add_overlay_text(rgb: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    padding = 4
    line_height = 13
    box_width = int(max(draw.textlength(line) for line in lines)) + padding * 2
    box_height = line_height * len(lines) + padding * 2
    draw.rectangle((0, 0, box_width, box_height), fill=(0, 0, 0))
    for index, line in enumerate(lines):
        draw.text((padding, padding + index * line_height), line, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def make_four_panel(clean, reconstruction, faulty, overlay, fault_type: str) -> np.ndarray:
    clean_panel = add_panel_title(make_input_preview(clean), "Clean BEV")
    reconstruction_panel = add_panel_title(make_input_preview(reconstruction), "Reconstructed clean BEV")
    faulty_panel = add_panel_title(make_input_preview(faulty), f"Faulty BEV: {fault_type}")
    overlay_panel = add_panel_title(overlay, "red=actual blue=error-mask magenta=overlap")
    top = np.concatenate([clean_panel, reconstruction_panel], axis=1)
    bottom = np.concatenate([faulty_panel, overlay_panel], axis=1)
    return np.concatenate([top, bottom], axis=0)


def predict_one(
    model,
    device,
    sample_path: Path,
    output_path: Path,
    threshold: float,
    fault_type_override: str | None = None,
    severity_override: str | None = None,
):
    faulty, clean, actual_mask, metadata = load_sample(sample_path)
    actual_fault_type = metadata.get("fault_type", "unknown")
    actual_severity = metadata.get("fault_severity", "unknown")
    condition_fault_type = fault_type_override or actual_fault_type
    condition_severity = severity_override or actual_severity
    fault_type_index = FAULT_TO_INDEX.get(condition_fault_type, 0)
    severity_index = SEVERITY_TO_INDEX.get(condition_severity, 1)

    with torch.no_grad():
        tensor = torch.from_numpy(faulty[None, :, :, :]).to(device)
        fault_tensor = torch.tensor([fault_type_index], dtype=torch.long, device=device)
        severity_tensor = torch.tensor([severity_index], dtype=torch.long, device=device)
        outputs = model(tensor, fault_tensor, severity_tensor)
        reconstruction = outputs["clean_reconstruction"][0].cpu().numpy()
        error = reconstruction_error_map(tensor, outputs["clean_reconstruction"])[0, 0].cpu().numpy()

    predicted_mask = error >= threshold
    actual_bool = actual_mask >= 0.5
    iou = mask_iou(predicted_mask, actual_bool)

    overlay = overlay_masks(make_input_preview(faulty), actual_bool, predicted_mask)
    overlay = add_overlay_text(
        overlay,
        [
            f"actual metadata: {actual_fault_type}/{actual_severity}",
            f"conditioning: {condition_fault_type}/{condition_severity}",
            f"error threshold: {threshold:.3f}",
            f"IoU: {iou:.3f}",
        ],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    composite = make_four_panel(clean, reconstruction, faulty, overlay, actual_fault_type)
    write_image(output_path, composite)

    error_path = output_path.with_name(f"{output_path.stem}_reconstruction_error.png")
    reconstruction_path = output_path.with_name(f"{output_path.stem}_reconstruction.png")
    overlay_path = output_path.with_name(f"{output_path.stem}_prediction_only.png")
    write_image(error_path, probability_heatmap(error))
    write_image(reconstruction_path, make_input_preview(reconstruction))
    write_image(overlay_path, overlay)

    print(f"Sample: {sample_path}")
    print(f"Output four-panel summary: {output_path}")
    print(f"Output reconstruction-error heatmap: {error_path}")
    print(f"IoU: {iou:.4f}")
    print(f"Conditioning used: {condition_fault_type}/{condition_severity}")
    print(f"Actual free-form mask extent: {bounding_box(actual_bool)}")
    print(f"Predicted free-form mask extent: {bounding_box(predicted_mask)}")
    return iou


def main():
    parser = argparse.ArgumentParser(
        description="Visualize conditioned Model_V2 restoration using reconstruction-error fault masks."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sample", default=None)
    parser.add_argument("--sample-dir", default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-outputs", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--fault-type", choices=FAULT_CLASSES, default=None)
    parser.add_argument("--severity", choices=SEVERITY_CLASSES, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(Path(args.model_path), device)
    threshold = args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.08)

    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    print(f"Reconstruction-error threshold: {threshold:.3f}")

    output_dir = Path(args.output_dir)
    if args.sample is not None:
        predict_one(
            model,
            device,
            Path(args.sample),
            output_dir / f"{Path(args.sample).stem}_model_v2_reconstruction_error_overlay.png",
            threshold,
            args.fault_type,
            args.severity,
        )
        return

    sample_paths = sorted(Path(args.sample_dir).glob("sample_*.npz"))[:args.num_outputs]
    if not sample_paths:
        raise FileNotFoundError(f"No sample_*.npz files found in {args.sample_dir}")

    ious = []
    for sample_path in sample_paths:
        output_path = output_dir / f"{sample_path.stem}_model_v2_reconstruction_error_overlay.png"
        ious.append(
            predict_one(
                model,
                device,
                sample_path,
                output_path,
                threshold,
                args.fault_type,
                args.severity,
            )
        )

    print(f"Wrote {len(ious)} reconstruction-error overlay images to {output_dir}")
    print(f"Mean IoU over saved outputs: {sum(ious) / len(ious):.4f}")


if __name__ == "__main__":
    main()
