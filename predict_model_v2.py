from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageDraw

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
DEFAULT_OUTPUT_DIR = str(Path(FAST_OUTPUT_ROOT) / "model_v2_predictions")
FAULT_CLASSES = ["laser", "photodetector", "scanning", "optical", "window", "mounting"]
SEVERITY_CLASSES = ["mild", "moderate", "severe"]


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        row_delta = skip.shape[-2] - x.shape[-2]
        col_delta = skip.shape[-1] - x.shape[-1]
        if row_delta != 0 or col_delta != 0:
            x = F.pad(
                x,
                [
                    col_delta // 2,
                    col_delta - col_delta // 2,
                    row_delta // 2,
                    row_delta - row_delta // 2,
                ],
            )
        return self.conv(torch.cat([skip, x], dim=1))


class LegacyBEVFaultRestorationModelV2(nn.Module):
    """
    Legacy Model_V2 architecture used by the old run:
    faulty BEV -> clean reconstruction + direct fault mask + fault/severity class heads.
    """

    def __init__(
        self,
        in_channels: int = 5,
        base_channels: int = 48,
        depth: int = 4,
        dropout: float = 0.12,
        num_fault_classes: int = len(FAULT_CLASSES),
        num_severity_classes: int = len(SEVERITY_CLASSES),
    ):
        super().__init__()
        channels = [base_channels * (2 ** level) for level in range(depth + 1)]
        self.input_block = DoubleConv(in_channels, channels[0], dropout)
        self.down_blocks = nn.ModuleList(
            DownBlock(channels[index], channels[index + 1], dropout)
            for index in range(depth)
        )
        self.up_blocks = nn.ModuleList(
            UpBlock(
                in_channels=channels[index + 1],
                skip_channels=channels[index],
                out_channels=channels[index],
                dropout=dropout,
            )
            for index in range(depth - 1, -1, -1)
        )
        self.clean_head = nn.Sequential(
            nn.Conv2d(channels[0], in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.mask_head = nn.Conv2d(channels[0], 1, kernel_size=1)

        bottleneck_channels = channels[-1]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fault_type_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(bottleneck_channels, bottleneck_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(bottleneck_channels // 2, num_fault_classes),
        )
        self.severity_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(bottleneck_channels, bottleneck_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(bottleneck_channels // 2, num_severity_classes),
        )

    def forward(self, faulty_bev):
        skips = []
        current = self.input_block(faulty_bev)
        skips.append(current)

        for down in self.down_blocks:
            current = down(current)
            skips.append(current)

        bottleneck = skips[-1]
        current = bottleneck
        for up, skip in zip(self.up_blocks, reversed(skips[:-1])):
            current = up(current, skip)

        return {
            "clean_reconstruction": self.clean_head(current),
            "fault_mask_logits": self.mask_head(current),
            "fault_type_logits": self.fault_type_head(self.global_pool(bottleneck)),
            "severity_logits": self.severity_head(self.global_pool(bottleneck)),
        }


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

    fault_classes = checkpoint.get("fault_classes", FAULT_CLASSES)
    severity_classes = checkpoint.get("severity_classes", SEVERITY_CLASSES)
    model = LegacyBEVFaultRestorationModelV2(
        in_channels=checkpoint["in_channels"],
        base_channels=checkpoint.get("base_channels", 48),
        depth=checkpoint.get("depth", 4),
        dropout=checkpoint.get("dropout", 0.0),
        num_fault_classes=len(fault_classes),
        num_severity_classes=len(severity_classes),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint, fault_classes, severity_classes


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
    reconstruction_panel = add_panel_title(make_input_preview(reconstruction), "Model_V2 reconstructed BEV")
    faulty_panel = add_panel_title(make_input_preview(faulty), f"Faulty BEV: {fault_type}")
    overlay_panel = add_panel_title(overlay, "red=actual blue=pred magenta=overlap")
    top = np.concatenate([clean_panel, reconstruction_panel], axis=1)
    bottom = np.concatenate([faulty_panel, overlay_panel], axis=1)
    return np.concatenate([top, bottom], axis=0)


def predict_one(model, device, sample_path: Path, output_path: Path, threshold: float, fault_classes, severity_classes):
    faulty, clean, actual_mask, metadata = load_sample(sample_path)
    with torch.no_grad():
        tensor = torch.from_numpy(faulty[None, :, :, :]).to(device)
        outputs = model(tensor)
        reconstruction = outputs["clean_reconstruction"][0].cpu().numpy()
        mask_probs = torch.sigmoid(outputs["fault_mask_logits"])[0, 0].cpu().numpy()
        fault_type_index = int(outputs["fault_type_logits"].argmax(dim=1).item())
        severity_index = int(outputs["severity_logits"].argmax(dim=1).item())

    predicted_mask = mask_probs >= threshold
    actual_bool = actual_mask >= 0.5
    iou = mask_iou(predicted_mask, actual_bool)
    actual_fault_type = metadata.get("fault_type", "unknown")
    actual_severity = metadata.get("fault_severity", "unknown")
    predicted_fault_type = fault_classes[fault_type_index]
    predicted_severity = severity_classes[severity_index]

    overlay = overlay_masks(make_input_preview(faulty), actual_bool, predicted_mask)
    overlay = add_overlay_text(
        overlay,
        [
            f"actual: {actual_fault_type}/{actual_severity}",
            f"pred: {predicted_fault_type}/{predicted_severity}",
            f"mask threshold: {threshold:.2f}",
            f"IoU: {iou:.3f}",
        ],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    composite = make_four_panel(clean, reconstruction, faulty, overlay, actual_fault_type)
    write_image(output_path, composite)

    probability_path = output_path.with_name(f"{output_path.stem}_mask_probability.png")
    reconstruction_path = output_path.with_name(f"{output_path.stem}_reconstruction.png")
    clean_path = output_path.with_name(f"{output_path.stem}_clean.png")
    faulty_path = output_path.with_name(f"{output_path.stem}_faulty.png")
    overlay_path = output_path.with_name(f"{output_path.stem}_prediction_only.png")
    write_image(probability_path, probability_heatmap(mask_probs))
    write_image(reconstruction_path, make_input_preview(reconstruction))
    write_image(clean_path, make_input_preview(clean))
    write_image(faulty_path, make_input_preview(faulty))
    write_image(overlay_path, overlay)

    print(f"Sample: {sample_path}")
    print(f"Output four-panel summary: {output_path}")
    print(f"Output prediction-only overlay: {overlay_path}")
    print(f"Output mask probability heatmap: {probability_path}")
    print(f"IoU: {iou:.4f}")
    print(f"Actual fault: {actual_fault_type}/{actual_severity}")
    print(f"Predicted fault: {predicted_fault_type}/{predicted_severity}")
    print(f"Actual free-form mask extent: {bounding_box(actual_bool)}")
    print(f"Predicted free-form mask extent: {bounding_box(predicted_mask)}")
    return iou


def main():
    parser = argparse.ArgumentParser(
        description="Visualize the legacy direct-mask Model_V2 checkpoint."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sample", default=None)
    parser.add_argument("--sample-dir", default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-outputs", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        quick_checkpoint = torch.load(Path(args.model_path), map_location="cpu", weights_only=True)
    except TypeError:
        quick_checkpoint = torch.load(Path(args.model_path), map_location="cpu")

    state_keys = quick_checkpoint.get("model_state", {}).keys()
    is_conditioned_reconstruction_model = (
        "fault_embedding.weight" in state_keys
        or quick_checkpoint.get("model_type") == "bev_fault_restoration_model_v2_reconstruction_error"
    )
    if is_conditioned_reconstruction_model:
        print(
            "This checkpoint is the conditioned reconstruction-error Model_V2, "
            "so I am using predict_model_v2_reconstruction_error.py instead."
        )
        from predict_model_v2_reconstruction_error import main as reconstruction_error_main

        reconstruction_error_main()
        return

    model, checkpoint, fault_classes, severity_classes = load_model(Path(args.model_path), device)
    threshold = args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.35)

    print(f"Model: {args.model_path}")
    print(f"Device: {device}")
    print(f"Mask threshold: {threshold:.2f}")
    print("Overlay colors: red=actual mask, blue=predicted mask, magenta=overlap")

    output_dir = Path(args.output_dir)
    if args.sample is not None:
        predict_one(
            model,
            device,
            Path(args.sample),
            output_dir / f"{Path(args.sample).stem}_model_v2_overlay.png",
            threshold,
            fault_classes,
            severity_classes,
        )
        return

    sample_paths = sorted(Path(args.sample_dir).glob("sample_*.npz"))[:args.num_outputs]
    if not sample_paths:
        raise FileNotFoundError(f"No sample_*.npz files found in {args.sample_dir}")

    ious = []
    for sample_path in sample_paths:
        output_path = output_dir / f"{sample_path.stem}_model_v2_overlay.png"
        ious.append(
            predict_one(
                model,
                device,
                sample_path,
                output_path,
                threshold,
                fault_classes,
                severity_classes,
            )
        )

    print(f"Wrote {len(ious)} Model_V2 overlay images to {output_dir}")
    print(f"Mean IoU over saved outputs: {sum(ious) / len(ious):.4f}")


if __name__ == "__main__":
    main()
