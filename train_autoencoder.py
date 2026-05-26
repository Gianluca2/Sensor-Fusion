from pathlib import Path
import argparse
import csv
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from autoencoder_model import BEVAutoEncoder


DEFAULT_DATASET_DIR = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\autoencoder_dataset"
)
DEFAULT_MODEL_PATH = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\models"
    r"\bev_autoencoder.pt"
)
DEFAULT_METRICS_PATH = (
    r"C:\Users\gianl\OneDrive\Desktop\Thesis\HerculesFiles\outputs\models"
    r"\autoencoder_training_metrics.csv"
)


class BEVReconstructionDataset(Dataset):
    def __init__(self, dataset_dir: Path):
        self.paths = sorted(dataset_dir.glob("sample_*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No sample_*.npz files found in {dataset_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with np.load(self.paths[index]) as data:
            x = data["input"].astype(np.float32)
            clean = data["clean"].astype(np.float32) if "clean" in data.files else x
            mask = data["target"].astype(np.float32)[None, :, :]

        return torch.from_numpy(x), torch.from_numpy(clean), torch.from_numpy(mask)


def reconstruction_loss(reconstruction, clean, loss_name: str):
    if loss_name == "l1":
        return nn.functional.l1_loss(reconstruction, clean)
    if loss_name == "mse":
        return nn.functional.mse_loss(reconstruction, clean)
    if loss_name == "l1_mse":
        return nn.functional.l1_loss(reconstruction, clean) + 0.5 * nn.functional.mse_loss(
            reconstruction,
            clean,
        )
    raise ValueError(f"Unsupported loss: {loss_name}")


def error_maps(reconstruction, corrupted):
    return torch.mean(torch.abs(reconstruction - corrupted), dim=1, keepdim=True)


def predict_from_error(error, error_threshold: float | None, error_percentile: float):
    if error_threshold is not None:
        return error >= error_threshold

    flat = error.flatten(start_dim=1)
    keep_fraction = max(0.0, min(1.0, (100.0 - error_percentile) / 100.0))
    keep_cells = max(1, int(flat.shape[1] * keep_fraction))
    top_indices = torch.topk(flat, keep_cells, dim=1).indices
    predicted = torch.zeros_like(flat, dtype=torch.bool)
    predicted.scatter_(1, top_indices, True)
    return predicted.reshape_as(error)


def mask_metrics(reconstruction, corrupted, targets, error_threshold, error_percentile):
    error = error_maps(reconstruction, corrupted)
    preds = predict_from_error(error, error_threshold, error_percentile)
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
        "mean_error": error.mean().item(),
    }


def run_epoch(model, loader, optimizer, device, args):
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
        "mean_error": 0.0,
    }
    count = 0

    for x, clean, mask in loader:
        x = x.to(device)
        clean = clean.to(device)
        mask = mask.to(device)

        with torch.set_grad_enabled(training):
            reconstruction = model(x)
            loss = reconstruction_loss(reconstruction, clean, args.loss)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        metrics = mask_metrics(
            reconstruction.detach(),
            x,
            mask,
            args.error_threshold,
            args.error_percentile,
        )
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
            "mean_error",
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
            "mean_error",
        ])
        row = {
            "epoch": epoch,
            "split": split,
            "loss_name": loss_name,
        }
        row.update({key: f"{value:.6f}" for key, value in metrics.items()})
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Train a BEV autoencoder to reconstruct masked LiDAR BEV tensors."
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", default=DEFAULT_METRICS_PATH)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--loss", choices=["l1", "mse", "l1_mse"], default="l1_mse")
    parser.add_argument(
        "--error-threshold",
        type=float,
        default=None,
        help="Absolute reconstruction-error threshold. If omitted, percentile thresholding is used.",
    )
    parser.add_argument(
        "--error-percentile",
        type=float,
        default=98.0,
        help="Per-sample reconstruction-error percentile used to create the predicted fault mask.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BEVReconstructionDataset(Path(args.dataset_dir))

    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    first_x, _, _ = dataset[0]
    model = BEVAutoEncoder(
        in_channels=first_x.shape[0],
        base_channels=args.base_channels,
        dropout=args.dropout,
        depth=args.depth,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_val_loss = float("inf")
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_path)
    write_metrics_header(metrics_path)

    print(f"Device: {device}")
    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}")
    print(f"Loss: {args.loss}")
    print(f"Error percentile: {args.error_percentile:.2f}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args)
        val_metrics = run_epoch(model, val_loader, None, device, args)
        scheduler.step(val_metrics["loss"])
        append_metrics(metrics_path, epoch, "train", args.loss, train_metrics)
        append_metrics(metrics_path, epoch, "val", args.loss, val_metrics)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} "
            f"iou {train_metrics['iou']:.4f} f1 {train_metrics['f1']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} "
            f"iou {val_metrics['iou']:.4f} f1 {val_metrics['f1']:.4f} "
            f"fp {val_metrics['false_positive_cells']:.1f} "
            f"fn {val_metrics['false_negative_cells']:.1f} "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "model_state": model.state_dict(),
                "in_channels": first_x.shape[0],
                "base_channels": args.base_channels,
                "dropout": args.dropout,
                "depth": args.depth,
                "loss": args.loss,
                "error_threshold": args.error_threshold,
                "error_percentile": args.error_percentile,
                "model_type": "bev_autoencoder",
            }, model_path)

    print(f"Best val reconstruction loss: {best_val_loss:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()
