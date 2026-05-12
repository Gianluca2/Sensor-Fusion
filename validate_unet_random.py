from pathlib import Path
import argparse
from datetime import datetime
import random

import numpy as np
import torch

from bev_projection import write_image
from make_unet_dataset import (
    DEFAULT_BEV,
    DEFAULT_LAYERS,
    build_occupancy_map,
    load_clean_bev,
    make_sample,
)
from predict_unet import bounding_box, mask_iou, overlay_masks, make_input_preview
from unet_model import SmallUNet


DEFAULT_MODEL_PATH = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\models"
    r"\unet_bev_mask.pt"
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


def predict_mask(model, device, x: np.ndarray, threshold: float):
    with torch.no_grad():
        tensor = torch.from_numpy(x[None, :, :, :].astype(np.float32)).to(device)
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    return probs >= threshold


def main():
    parser = argparse.ArgumentParser(
        description="Create one fresh random BEV mask, predict it, and save a PNG overlay."
    )
    parser.add_argument("--bev", default=DEFAULT_BEV)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-mask-height", type=int, default=25)
    parser.add_argument("--max-mask-height", type=int, default=90)
    parser.add_argument("--min-mask-width", type=int, default=25)
    parser.add_argument("--max-mask-width", type=int, default=90)
    parser.add_argument("--max-mask-area-fraction", type=float, default=0.08)
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

    clean, _ = load_clean_bev(Path(args.bev), DEFAULT_LAYERS)
    occupancy = build_occupancy_map(clean, args.occupancy_threshold)
    faulty, actual, metadata = make_sample(clean, occupancy, sample_index=0, args=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(Path(args.model_path), device)
    threshold = args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.5)

    predicted = predict_mask(model, device, faulty, threshold)
    overlay = overlay_masks(make_input_preview(faulty), actual >= 0.5, predicted)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"random_validation_{timestamp}_seed_{args.seed}.png"
    write_image(output_path, overlay)

    print(f"Saved validation overlay: {output_path}")
    print(f"Seed: {args.seed}")
    print(f"Threshold: {threshold:.3f}")
    print(f"IoU: {mask_iou(predicted, actual >= 0.5):.4f}")
    print(f"Actual box: {bounding_box(actual >= 0.5)}")
    print(f"Predicted box: {bounding_box(predicted)}")
    print(f"Mask metadata: {metadata['mask']}")
    print("Overlay colors: red=actual mask, blue=predicted mask, magenta=overlap")


if __name__ == "__main__":
    main()
