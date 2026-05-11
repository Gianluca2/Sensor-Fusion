from pathlib import Path
import argparse
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


def mask_iou(logits, targets, threshold: float):
    preds = torch.sigmoid(logits) >= threshold
    targets_bool = targets >= 0.5
    intersection = torch.logical_and(preds, targets_bool).sum(dim=(1, 2, 3)).float()
    union = torch.logical_or(preds, targets_bool).sum(dim=(1, 2, 3)).float()
    iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
    return iou.mean().item()


def run_epoch(model, loader, optimizer, device, threshold):
    training = optimizer is not None
    model.train(training)

    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_iou = 0.0
    count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(training):
            logits = model(x)
            loss = bce(logits, y) + dice_loss(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = x.shape[0]
        total_loss += loss.item() * batch_size
        total_iou += mask_iou(logits.detach(), y, threshold) * batch_size
        count += batch_size

    return total_loss / count, total_iou / count


def main():
    parser = argparse.ArgumentParser(description="Train a small U-Net to detect BEV black masks.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
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

    print(f"Device: {device}")
    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou = run_epoch(model, train_loader, optimizer, device, args.threshold)
        val_loss, val_iou = run_epoch(model, val_loader, None, device, args.threshold)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} iou {train_iou:.4f} | "
            f"val loss {val_loss:.4f} iou {val_iou:.4f}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                "model_state": model.state_dict(),
                "in_channels": first_x.shape[0],
                "base_channels": args.base_channels,
                "threshold": args.threshold,
            }, model_path)

    print(f"Best val IoU: {best_iou:.4f}")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
