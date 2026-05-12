from pathlib import Path
import argparse
from datetime import datetime
import random

import numpy as np
import torch

from bev_projection import write_image
from make_unet_dataset import (
    DEFAULT_BEV,
    DEFAULT_BEV_DIR,
    DEFAULT_LAYERS,
    build_occupancy_map,
    find_bev_files,
    load_clean_bev,
    make_sample,
)
from predict_unet import (
    bounding_box,
    mask_iou,
    overlay_masks,
    make_input_preview,
    postprocess_prediction,
)
from unet_model import SmallUNet


DEFAULT_MODEL_PATH = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\models"
    r"\unet_bev_mask.pt"
)
DEFAULT_IOU_MODEL_PATH = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\models"
    r"\unet_bev_mask_bce_iou.pt"
)
DEFAULT_TVERSKY_MODEL_PATH = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\models"
    r"\unet_bev_mask_bce_tversky.pt"
)
DEFAULT_OUTPUT_DIR = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs"
    r"\unet_random_validation"
)


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


def predict_mask(model, device, x: np.ndarray, threshold: float, max_prediction_area_fraction: float):
    with torch.no_grad():
        tensor = torch.from_numpy(x[None, :, :, :].astype(np.float32)).to(device)
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    return postprocess_prediction(probs, threshold, max_prediction_area_fraction)


def stack_comparison(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError(f"Comparison images must have same shape, got {left.shape} and {right.shape}")

    separator = np.full((left.shape[0], 8, 3), 255, dtype=np.uint8)
    return np.concatenate([left, separator, right], axis=1)


def predict_and_overlay(model_path: Path, device, faulty, actual, threshold_override, max_prediction_area_fraction):
    model, checkpoint = load_model(model_path, device)
    threshold = threshold_override if threshold_override is not None else checkpoint.get("threshold", 0.5)
    predicted = predict_mask(model, device, faulty, threshold, max_prediction_area_fraction)
    overlay = overlay_masks(make_input_preview(faulty), actual >= 0.5, predicted)
    iou = mask_iou(predicted, actual >= 0.5)

    return {
        "model_path": model_path,
        "threshold": threshold,
        "predicted": predicted,
        "overlay": overlay,
        "iou": iou,
        "predicted_box": bounding_box(predicted),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create one fresh random BEV mask and compare IoU-loss vs Tversky-loss predictions."
    )
    parser.add_argument("--bev", default=DEFAULT_BEV)
    parser.add_argument(
        "--bev-dir",
        default=DEFAULT_BEV_DIR,
        help="Directory of clean BEV files. A random BEV is selected from here when available.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--iou-model-path", default=DEFAULT_IOU_MODEL_PATH)
    parser.add_argument("--tversky-model-path", default=DEFAULT_TVERSKY_MODEL_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-mask-height", type=int, default=25)
    parser.add_argument("--max-mask-height", type=int, default=90)
    parser.add_argument("--min-mask-width", type=int, default=25)
    parser.add_argument("--max-mask-width", type=int, default=90)
    parser.add_argument("--min-mask-area-fraction", type=float, default=0.01)
    parser.add_argument("--max-mask-area-fraction", type=float, default=0.03)
    parser.add_argument("--max-prediction-area-fraction", type=float, default=1.0)
    parser.add_argument("--min-mask-occupied-cells", type=int, default=50)
    parser.add_argument("--occupancy-threshold", type=float, default=0.0)
    parser.add_argument("--max-mask-attempts", type=int, default=500)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional fixed seed. Leave empty to generate a different mask every run.",
    )
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.SystemRandom().randint(0, 2**32 - 1)

    random.seed(args.seed)
    np.random.seed(args.seed)

    bev_files = find_bev_files(Path(args.bev_dir)) if args.bev_dir else []
    source_bev = random.choice(bev_files) if bev_files else Path(args.bev)
    clean, _ = load_clean_bev(source_bev, DEFAULT_LAYERS)
    occupancy = build_occupancy_map(clean, args.occupancy_threshold)
    faulty, actual, metadata = make_sample(clean, occupancy, sample_index=0, args=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iou_result = predict_and_overlay(
        Path(args.iou_model_path),
        device,
        faulty,
        actual,
        args.threshold,
        args.max_prediction_area_fraction,
    )
    tversky_result = predict_and_overlay(
        Path(args.tversky_model_path),
        device,
        faulty,
        actual,
        args.threshold,
        args.max_prediction_area_fraction,
    )
    comparison = stack_comparison(iou_result["overlay"], tversky_result["overlay"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = output_dir / f"random_validation_compare_{timestamp}_seed_{args.seed}.png"
    iou_path = output_dir / f"random_validation_bce_iou_{timestamp}_seed_{args.seed}.png"
    tversky_path = output_dir / f"random_validation_bce_tversky_{timestamp}_seed_{args.seed}.png"

    write_image(comparison_path, comparison)
    write_image(iou_path, iou_result["overlay"])
    write_image(tversky_path, tversky_result["overlay"])

    print(f"Saved comparison overlay: {comparison_path}")
    print(f"Saved IoU-loss overlay: {iou_path}")
    print(f"Saved Tversky-loss overlay: {tversky_path}")
    print(f"Seed: {args.seed}")
    print(f"Source BEV: {source_bev}")
    print(f"Actual box: {bounding_box(actual >= 0.5)}")
    print(f"IoU-loss model: {iou_result['model_path']}")
    print(f"  threshold: {iou_result['threshold']:.3f}")
    print(f"  mask IoU: {iou_result['iou']:.4f}")
    print(f"  predicted area fraction: {np.count_nonzero(iou_result['predicted']) / iou_result['predicted'].size:.4f}")
    print(f"  predicted box: {iou_result['predicted_box']}")
    print(f"Tversky-loss model: {tversky_result['model_path']}")
    print(f"  threshold: {tversky_result['threshold']:.3f}")
    print(f"  mask IoU: {tversky_result['iou']:.4f}")
    print(f"  predicted area fraction: {np.count_nonzero(tversky_result['predicted']) / tversky_result['predicted'].size:.4f}")
    print(f"  predicted box: {tversky_result['predicted_box']}")
    print(f"Mask metadata: {metadata['mask']}")
    print("Comparison image: left=BCE+IoU model, right=BCE+Tversky model")
    print("Overlay colors: red=actual mask, blue=predicted mask, magenta=overlap")


if __name__ == "__main__":
    main()
