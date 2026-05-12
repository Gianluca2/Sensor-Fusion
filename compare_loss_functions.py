from pathlib import Path
import argparse
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\unet_dataset"
)
DEFAULT_OUTPUT_DIR = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\models"
)


def run_training(loss_name: str, args):
    output_dir = Path(args.output_dir)
    model_path = output_dir / f"unet_bev_mask_{loss_name}.pt"
    metrics_path = output_dir / f"metrics_{loss_name}.csv"

    command = [
        sys.executable,
        str(PROJECT_DIR / "train_unet.py"),
        "--dataset-dir",
        args.dataset_dir,
        "--model-path",
        str(model_path),
        "--metrics-path",
        str(metrics_path),
        "--loss",
        loss_name,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--base-channels",
        str(args.base_channels),
        "--lr",
        str(args.lr),
        "--threshold",
        str(args.threshold),
        "--seed",
        str(args.seed),
    ]

    if loss_name == "bce_tversky":
        command.extend([
            "--tversky-alpha",
            str(args.tversky_alpha),
            "--tversky-beta",
            str(args.tversky_beta),
        ])

    print(f"\n=== Training {loss_name} ===")
    print(" ".join(command))
    subprocess.run(command, cwd=PROJECT_DIR, check=True)

    return model_path, metrics_path


def main():
    parser = argparse.ArgumentParser(
        description="Train two U-Net models with different overlap-aware loss functions."
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tversky-alpha", type=float, default=0.6)
    parser.add_argument("--tversky-beta", type=float, default=0.6)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    results = [
        run_training("bce_iou", args),
        run_training("bce_tversky", args),
    ]

    print("\nDone. Compare these metrics files:")
    for model_path, metrics_path in results:
        print(f"  model: {model_path}")
        print(f"  metrics: {metrics_path}")

    print("\nHigher validation IoU/F1 is better.")
    print("Lower false_positive_cells means less extra blue prediction area.")
    print("Lower false_negative_cells means less missed red actual area.")


if __name__ == "__main__":
    main()
