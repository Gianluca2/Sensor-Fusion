from pathlib import Path
import argparse
from datetime import datetime
import random

import numpy as np
import torch

from bev_projection import write_image
from make_autoencoder_dataset import (
    DEFAULT_LAYERS,
    build_occupancy_map,
    find_bev_files,
    load_clean_bev,
    make_sample,
)
from predict_autoencoder import (
    load_model,
    normalized_error,
    predict_from_error,
    reconstruction_error,
)
from bev_fault_visualization import (
    bounding_box,
    make_input_preview,
    mask_iou,
    overlay_masks,
    probability_heatmap,
)


OUTPUT_ROOT = Path(r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs")
DEFAULT_BEV_DIR = OUTPUT_ROOT / "bev_autoencoder_1500_even"
DEFAULT_MODEL_PATH = OUTPUT_ROOT / "models" / "bev_autoencoder_1500_even_30_epochs.pt"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "autoencoder_random_validation"


def main():
    parser = argparse.ArgumentParser(
        description="Create fresh random BEV masks and validate autoencoder reconstruction-error detection."
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--bev-dir", default=str(DEFAULT_BEV_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num-outputs", type=int, default=10)
    parser.add_argument("--error-threshold", type=float, default=None)
    parser.add_argument("--error-percentile", type=float, default=None)
    parser.add_argument("--min-mask-height", type=int, default=25)
    parser.add_argument("--max-mask-height", type=int, default=90)
    parser.add_argument("--min-mask-width", type=int, default=25)
    parser.add_argument("--max-mask-width", type=int, default=90)
    parser.add_argument("--min-mask-area-fraction", type=float, default=0.01)
    parser.add_argument("--max-mask-area-fraction", type=float, default=0.03)
    parser.add_argument("--min-mask-occupied-cells", type=int, default=50)
    parser.add_argument("--occupancy-threshold", type=float, default=0.0)
    parser.add_argument("--max-mask-attempts", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.SystemRandom().randint(0, 2**32 - 1)

    random.seed(args.seed)
    np.random.seed(args.seed)

    bev_files = find_bev_files(Path(args.bev_dir))
    if not bev_files:
        raise FileNotFoundError(f"No BEV .npz files found under {args.bev_dir}")

    first_clean, _ = load_clean_bev(bev_files[0], DEFAULT_LAYERS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(Path(args.model_path), device)
    if int(checkpoint["in_channels"]) != first_clean.shape[0]:
        raise ValueError(
            f"{Path(args.model_path).name} expects {checkpoint['in_channels']} channels, "
            f"but this BEV has {first_clean.shape[0]} channels. Retrain the autoencoder."
        )

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ious = []
    print(f"Model: {args.model_path}")
    print(f"BEV dir: {args.bev_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print(f"Error threshold: {error_threshold}")
    print(f"Error percentile: {error_percentile:.2f}")
    print(f"Seed: {args.seed}")
    print("Overlay colors: red=actual mask, blue=predicted error region, magenta=overlap")

    for index in range(args.num_outputs):
        source_bev = random.choice(bev_files)
        clean, _ = load_clean_bev(source_bev, DEFAULT_LAYERS)
        occupancy = build_occupancy_map(clean, args.occupancy_threshold)
        faulty, actual, metadata = make_sample(clean, occupancy, sample_index=index, args=args)

        with torch.no_grad():
            tensor = torch.from_numpy(faulty[None, :, :, :].astype(np.float32)).to(device)
            reconstruction = model(tensor)[0].cpu().numpy()

        error_map = reconstruction_error(reconstruction, faulty)
        predicted = predict_from_error(error_map, error_threshold, error_percentile)
        overlay = overlay_masks(make_input_preview(faulty), actual >= 0.5, predicted)
        heatmap = probability_heatmap(normalized_error(error_map))
        reconstruction_preview = make_input_preview(reconstruction)
        iou = mask_iou(predicted, actual >= 0.5)
        ious.append(iou)

        stem = f"autoencoder_validation_{timestamp}_sample_{index:03d}"
        overlay_path = output_dir / f"{stem}_overlay.png"
        heatmap_path = output_dir / f"{stem}_error.png"
        reconstruction_path = output_dir / f"{stem}_reconstruction.png"
        write_image(overlay_path, overlay)
        write_image(heatmap_path, heatmap)
        write_image(reconstruction_path, reconstruction_preview)

        predicted_area = np.count_nonzero(predicted) / predicted.size
        print(f"\nSample {index:03d}")
        print(f"  source BEV: {source_bev}")
        print(f"  overlay: {overlay_path}")
        print(f"  error heatmap: {heatmap_path}")
        print(f"  reconstruction: {reconstruction_path}")
        print(f"  IoU: {iou:.4f}")
        print(f"  max reconstruction error: {float(np.max(error_map)):.4f}")
        print(f"  predicted area fraction: {predicted_area:.4f}")
        print(f"  actual box: {bounding_box(actual >= 0.5)}")
        print(f"  predicted box: {bounding_box(predicted)}")
        print(f"  mask metadata: {metadata['mask']}")

    print(f"\nWrote {len(ious)} validation overlays to {output_dir}")
    print(f"Mean IoU: {sum(ious) / len(ious):.4f}")


if __name__ == "__main__":
    main()
