from pathlib import Path
import argparse
import csv
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from unet_model import SmallUNet


DEFAULT_DATASET_DIR = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\unet_dataset"
)
DEFAULT_MODEL_PATH = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\models"
    r"\unet_bev_mask.pt"
)
DEFAULT_METRICS_PATH = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\models"
    r"\unet_training_metrics.csv"
)


class BEVMaskDataset(Dataset):
    def __init__(self, dataset_dir: Path):
        self.paths = sorted(dataset_dir.glob("sample_*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No sample_*.npz files found in {dataset_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with np.load(self.paths[index]) as data:
            x = data["input"].astype(np.float32)
            y = data["target"].astype(np.float32)[None, :, :]

        return torch.from_numpy(x), torch.from_numpy(y)


def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    intersection = torch.sum(probs * targets, dim=(1, 2, 3))
    union = torch.sum(probs, dim=(1, 2, 3)) + torch.sum(targets, dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def iou_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    intersection = torch.sum(probs * targets, dim=(1, 2, 3))
    union = torch.sum(probs + targets - probs * targets, dim=(1, 2, 3))
    iou = (intersection + smooth) / (union + smooth)
    return 1.0 - iou.mean()


def tversky_loss(logits, targets, alpha=0.6, beta=0.6, smooth=1.0):
    probs = torch.sigmoid(logits)
    tp = torch.sum(probs * targets, dim=(1, 2, 3))
    fp = torch.sum(probs * (1.0 - targets), dim=(1, 2, 3))
    fn = torch.sum((1.0 - probs) * targets, dim=(1, 2, 3))
    score = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - score.mean()


def segmentation_loss(logits, targets, loss_name: str, tversky_alpha: float, tversky_beta: float):
    bce = nn.BCEWithLogitsLoss()

    if loss_name == "bce_dice":
        return bce(logits, targets) + dice_loss(logits, targets)
    if loss_name == "bce_iou":
        return bce(logits, targets) + iou_loss(logits, targets)
    if loss_name == "bce_tversky":
        return bce(logits, targets) + tversky_loss(
            logits,
            targets,
            alpha=tversky_alpha,
            beta=tversky_beta,
        )

    raise ValueError(f"Unsupported loss: {loss_name}")


def mask_metrics(logits, targets, threshold: float):
    preds = torch.sigmoid(logits) >= threshold
    targets_bool = targets >= 0.5

    tp = torch.logical_and(preds, targets_bool).sum(dim=(1, 2, 3)).float()
    fp = torch.logical_and(preds, ~targets_bool).sum(dim=(1, 2, 3)).float()
    fn = torch.logical_and(~preds, targets_bool).sum(dim=(1, 2, 3)).float()

    iou = torch.where(tp + fp + fn > 0, tp / (tp + fp + fn), torch.ones_like(tp))
    precision = torch.where(tp + fp > 0, tp / (tp + fp), torch.ones_like(tp))
    recall = torch.where(tp + fn > 0, tp / (tp + fn), torch.ones_like(tp))
    f1 = torch.where(
        2 * tp + fp + fn > 0,
        (2 * tp) / (2 * tp + fp + fn),
        torch.ones_like(tp),
    )

    return {
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
        "false_positive_cells": fp.mean().item(),
        "false_negative_cells": fn.mean().item(),
    }


def run_epoch(model, loader, optimizer, device, threshold, args):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    totals = {
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "false_positive_cells": 0.0,
        "false_negative_cells": 0.0,
    }
    count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = segmentation_loss(
                logits,
                y,
                loss_name=args.loss,
                tversky_alpha=args.tversky_alpha,
                tversky_beta=args.tversky_beta,
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        metrics = mask_metrics(logits.detach(), y, threshold)
        for key in totals:
            totals[key] += metrics[key] * batch_size
        count += batch_size

    averaged = {key: value / count for key, value in totals.items()}
    averaged["loss"] = total_loss / count
    return averaged


def write_metrics_header(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch",
            "split",
            "loss_name",
            "loss",
            "iou",
            "precision",
            "recall",
            "f1",
            "false_positive_cells",
            "false_negative_cells",
        ])
        writer.writeheader()


def append_metrics(path: Path, epoch: int, split: str, loss_name: str, metrics: dict):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch",
            "split",
            "loss_name",
            "loss",
            "iou",
            "precision",
            "recall",
            "f1",
            "false_positive_cells",
            "false_negative_cells",
        ])
        row = {
            "epoch": epoch,
            "split": split,
            "loss_name": loss_name,
        }
        row.update({key: f"{value:.6f}" for key, value in metrics.items()})
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Train a small U-Net to detect BEV black masks.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", default=DEFAULT_METRICS_PATH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--loss",
        choices=["bce_dice", "bce_iou", "bce_tversky"],
        default="bce_dice",
    )
    parser.add_argument("--tversky-alpha", type=float, default=0.6)
    parser.add_argument("--tversky-beta", type=float, default=0.6)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BEVMaskDataset(Path(args.dataset_dir))

    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    first_x, _ = dataset[0]
    model = SmallUNet(in_channels=first_x.shape[0], base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_iou = -1.0
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_path)
    write_metrics_header(metrics_path)

    print(f"Device: {device}")
    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}")
    print(f"Loss: {args.loss}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.threshold, args)
        val_metrics = run_epoch(model, val_loader, None, device, args.threshold, args)
        append_metrics(metrics_path, epoch, "train", args.loss, train_metrics)
        append_metrics(metrics_path, epoch, "val", args.loss, val_metrics)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} "
            f"iou {train_metrics['iou']:.4f} f1 {train_metrics['f1']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} "
            f"iou {val_metrics['iou']:.4f} f1 {val_metrics['f1']:.4f} "
            f"fp {val_metrics['false_positive_cells']:.1f} "
            f"fn {val_metrics['false_negative_cells']:.1f}"
        )

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save({
                "model_state": model.state_dict(),
                "in_channels": first_x.shape[0],
                "base_channels": args.base_channels,
                "threshold": args.threshold,
                "loss": args.loss,
                "tversky_alpha": args.tversky_alpha,
                "tversky_beta": args.tversky_beta,
            }, model_path)

    print(f"Best val IoU: {best_iou:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
