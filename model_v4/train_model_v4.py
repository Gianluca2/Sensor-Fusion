from pathlib import Path
import argparse
import csv
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from Model_V4 import (
    BEVReliabilityModelV4,
    FAULT_CLASSES,
    SEVERITY_CLASSES,
    ModelV4LossWeights,
    model_v4_loss,
)

##1. predict fault mask
##2. dilate mask
#3. split BEV into 128x128 tiles
#4. keep tiles with enough faulty pixels
#5. run radar-conditioned diffusion only on those tiles
#6. paste repaired pixels back only inside the dilated mask

#IDeas
#Use BEV as the main grid, but add richer vertical channels:

#multiple height-bin occupancy channels
#intensity statistics
#vertical density histogram
#radar occupancy/doppler channels later
FAST_OUTPUT_ROOT = r"C:\Users\gianl\ThesisOutputs\HerculesFiles\outputs"
DEFAULT_DATASET_DIR = str(Path(FAST_OUTPUT_ROOT) / "autoencoder_dataset")
DEFAULT_MODEL_PATH = str(Path(FAST_OUTPUT_ROOT) / "models" / "Model_V4.pt")
DEFAULT_METRICS_PATH = str(Path(FAST_OUTPUT_ROOT) / "models" / "Model_V4_training_metrics.csv")


FAULT_TO_INDEX = {name: index for index, name in enumerate(FAULT_CLASSES)}
SEVERITY_TO_INDEX = {name: index for index, name in enumerate(SEVERITY_CLASSES)}


class PairedBEVDataset(Dataset):
    def __init__(self, dataset_dir: Path):
        self.paths = sorted(dataset_dir.glob("sample_*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No sample_*.npz files found in {dataset_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with np.load(self.paths[index]) as data:
            faulty = data["input"].astype(np.float32)
            clean = data["clean"].astype(np.float32)
            soft_target = data["target"].astype(np.float32)[None, :, :]
            if "binary_target" in data.files:
                binary_target = data["binary_target"].astype(np.float32)[None, :, :]
            else:
                binary_target = (soft_target >= 0.5).astype(np.float32)
            metadata = {}
            if "metadata_json" in data.files:
                metadata = json.loads(str(data["metadata_json"].item()))

        fault_type = metadata.get("fault_type", FAULT_CLASSES[0])
        severity = metadata.get("fault_severity", SEVERITY_CLASSES[1])
        fault_type_index = FAULT_TO_INDEX.get(fault_type, 0)
        severity_index = SEVERITY_TO_INDEX.get(severity, 1)

        return (
            torch.from_numpy(faulty),
            torch.from_numpy(clean),
            torch.from_numpy(soft_target),
            torch.from_numpy(binary_target),
            torch.tensor(fault_type_index, dtype=torch.long),
            torch.tensor(severity_index, dtype=torch.long),
            fault_type,
            severity,
        )


def load_faulty_array(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return data["input"].astype(np.float32)


def compute_channel_normalization(dataset: PairedBEVDataset, indices, max_samples: int, seed: int):
    selected_indices = list(indices)
    if max_samples > 0 and len(selected_indices) > max_samples:
        rng = random.Random(seed)
        selected_indices = rng.sample(selected_indices, max_samples)

    channel_sum = None
    channel_sq_sum = None
    cell_count = 0

    for index in selected_indices:
        array = load_faulty_array(dataset.paths[index])
        flattened = array.reshape(array.shape[0], -1).astype(np.float64)
        if channel_sum is None:
            channel_sum = flattened.sum(axis=1)
            channel_sq_sum = (flattened ** 2).sum(axis=1)
        else:
            channel_sum += flattened.sum(axis=1)
            channel_sq_sum += (flattened ** 2).sum(axis=1)
        cell_count += flattened.shape[1]

    mean = channel_sum / cell_count
    variance = np.maximum(channel_sq_sum / cell_count - mean ** 2, 1e-8)
    std = np.sqrt(variance)
    return mean.astype(np.float32), std.astype(np.float32), len(selected_indices)


def normalize_bev_tensor(tensor, channel_mean, channel_std):
    if channel_mean is None or channel_std is None:
        return tensor

    mean = torch.as_tensor(channel_mean, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    std = torch.as_tensor(channel_std, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    return (tensor - mean) / torch.clamp(std, min=1e-6)


def mask_metric_values(probabilities, targets, threshold: float):
    probs = probabilities
    preds = probabilities >= threshold
    targets_bool = targets >= 0.5
    batch_size = targets.shape[0]
    height = targets.shape[2]
    width = targets.shape[3]
    total_cells = torch.full(
        (batch_size,),
        targets[0].numel(),
        dtype=torch.float32,
        device=targets.device,
    )
    fault_cells = targets_bool.sum(dim=(1, 2, 3)).float()
    predicted_cells = preds.sum(dim=(1, 2, 3)).float()

    tp = torch.logical_and(preds, targets_bool).sum(dim=(1, 2, 3)).float()
    fp = torch.logical_and(preds, ~targets_bool).sum(dim=(1, 2, 3)).float()
    fn = torch.logical_and(~preds, targets_bool).sum(dim=(1, 2, 3)).float()

    iou = torch.where(tp + fp + fn > 0, tp / (tp + fp + fn), torch.ones_like(tp))
    precision = torch.where(tp + fp > 0, tp / (tp + fp), torch.ones_like(tp))
    recall = torch.where(tp + fn > 0, tp / (tp + fn), torch.ones_like(tp))
    f1 = torch.where(2 * tp + fp + fn > 0, (2 * tp) / (2 * tp + fp + fn), torch.ones_like(tp))
    f2 = torch.where(
        4 * precision + recall > 0,
        (5 * precision * recall) / (4 * precision + recall),
        torch.ones_like(precision),
    )
    area_error = torch.where(
        fault_cells > 0,
        torch.abs(predicted_cells - fault_cells) / fault_cells,
        predicted_cells / total_cells,
    )

    row_grid = torch.arange(height, device=targets.device, dtype=torch.float32).view(1, 1, height, 1)
    col_grid = torch.arange(width, device=targets.device, dtype=torch.float32).view(1, 1, 1, width)
    pred_denominator = torch.clamp(predicted_cells, min=1.0)
    target_denominator = torch.clamp(fault_cells, min=1.0)
    pred_row = (preds.float() * row_grid).sum(dim=(1, 2, 3)) / pred_denominator
    pred_col = (preds.float() * col_grid).sum(dim=(1, 2, 3)) / pred_denominator
    target_row = (targets_bool.float() * row_grid).sum(dim=(1, 2, 3)) / target_denominator
    target_col = (targets_bool.float() * col_grid).sum(dim=(1, 2, 3)) / target_denominator
    centroid_distance_cells = torch.sqrt((pred_row - target_row) ** 2 + (pred_col - target_col) ** 2)
    centroid_distance_norm = centroid_distance_cells / float(max(height, width))

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "false_positive_cells": fp,
        "false_negative_cells": fn,
        "fault_cells": fault_cells,
        "predicted_cells": predicted_cells,
        "area_error": area_error,
        "centroid_distance_norm": centroid_distance_norm,
        "mean_probability": probs.flatten(start_dim=1).mean(dim=1),
    }


METRIC_FIELDS = [
    "loss",
    "soft_reliability_loss",
    "mask_bce_loss",
    "mask_dice_loss",
    "learning_rate",
    "iou",
    "precision",
    "recall",
    "f1",
    "f2",
    "false_positive_cells",
    "false_negative_cells",
    "fault_cells",
    "predicted_cells",
    "area_error",
    "centroid_distance_norm",
    "mean_probability",
]


def run_epoch(model, loader, optimizer, device, args, weights, channel_mean, channel_std):
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in METRIC_FIELDS}
    count = 0

    for faulty, clean, soft_target, binary_target, fault_type_index, severity_index, _, _ in loader:
        faulty = faulty.to(device)
        clean = clean.to(device)
        soft_target = soft_target.to(device)
        binary_target = binary_target.to(device)
        fault_type_index = fault_type_index.to(device)
        severity_index = severity_index.to(device)

        with torch.set_grad_enabled(training):
            spatial_weight = None
            if args.range_loss_weight > 0.0:
                range_channel = faulty[:, args.range_channel_index:args.range_channel_index + 1, :, :]
                spatial_weight = 1.0 + args.range_loss_weight * torch.clamp(range_channel, 0.0, 1.0)
            normalized_faulty = normalize_bev_tensor(faulty, channel_mean, channel_std)
            outputs = model(normalized_faulty, fault_type_index, severity_index)
            loss, loss_parts = model_v4_loss(
                outputs,
                clean,
                soft_target,
                binary_target,
                weights,
                spatial_weight=spatial_weight,
            )
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        batch_size = faulty.shape[0]
        totals["loss"] += loss.detach().item() * batch_size
        totals["soft_reliability_loss"] += loss_parts["soft_reliability_loss"].detach().item() * batch_size
        totals["mask_bce_loss"] += loss_parts["mask_bce_loss"].detach().item() * batch_size
        totals["mask_dice_loss"] += loss_parts["mask_dice_loss"].detach().item() * batch_size

        probabilities = outputs["fault_probability"].detach()
        metric_values = mask_metric_values(probabilities, binary_target, args.threshold)
        for key, values in metric_values.items():
            totals[key] += values.mean().item() * batch_size
        count += batch_size

    metrics = {key: value / count for key, value in totals.items()}
    metrics["learning_rate"] = 0.0
    return metrics


def write_metrics_header(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "split"] + METRIC_FIELDS)
        writer.writeheader()


def append_metrics(path: Path, epoch: int, split: str, metrics: dict, learning_rate: float):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "split"] + METRIC_FIELDS)
        row = {"epoch": epoch, "split": split}
        row.update({key: f"{metrics[key]:.6f}" for key in METRIC_FIELDS})
        row["learning_rate"] = f"{learning_rate:.10f}"
        writer.writerow(row)


def make_checkpoint(
    model,
    optimizer,
    scheduler,
    first_faulty,
    args,
    weights,
    channel_mean,
    channel_std,
    epoch: int,
    best_epoch: int,
    best_val_loss: float,
    train_metrics: dict,
    val_metrics: dict,
):
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "in_channels": first_faulty.shape[0],
        "base_channels": args.base_channels,
        "depth": args.depth,
        "dropout": args.dropout,
        "fault_classes": FAULT_CLASSES,
        "severity_classes": SEVERITY_CLASSES,
        "fault_embedding_dim": 8,
        "severity_embedding_dim": 4,
        "threshold": args.threshold,
        "loss_weights": weights.__dict__,
        "channel_normalization": args.channel_normalization,
        "channel_mean": channel_mean.tolist() if channel_mean is not None else None,
        "channel_std": channel_std.tolist() if channel_std is not None else None,
        "model_type": "bev_soft_reliability_model_v4",
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Model_V4 as a soft LiDAR reliability/unreliability estimator."
    )
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--latest-model-path",
        default=None,
        help="Optional checkpoint path overwritten after every epoch with the latest model state.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Optional checkpoint path to resume training from.",
    )
    parser.add_argument("--metrics-path", default=DEFAULT_METRICS_PATH)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=0,
        help="DataLoader workers. Use 0 on Windows to avoid shared-memory file mapping errors.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-reduce-patience", type=int, default=2)
    parser.add_argument("--lr-reduce-factor", type=float, default=0.5)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument(
        "--channel-normalization",
        choices=["dataset", "none"],
        default="dataset",
        help="Normalize BEV input channels using train-split mean/std, or disable normalization.",
    )
    parser.add_argument(
        "--normalization-samples",
        type=int,
        default=2048,
        help="Number of train samples used to estimate channel normalization. Use 0 for all train samples.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Fault-probability threshold used to convert the direct mask output into a binary mask.",
    )
    parser.add_argument(
        "--soft-weight",
        type=float,
        default=1.0,
        help="Weight for SmoothL1 regression against the soft damage/unreliability target.",
    )
    parser.add_argument(
        "--bce-weight",
        type=float,
        default=0.5,
        help="Weight for BCE against the binary damage target.",
    )
    parser.add_argument(
        "--dice-weight",
        type=float,
        default=1.0,
        help="Weight applied to Dice loss against the binary damage target.",
    )
    parser.add_argument(
        "--range-loss-weight",
        type=float,
        default=1.0,
        help="Extra spatial loss weight applied by normalized range: 1 + weight * range_channel.",
    )
    parser.add_argument(
        "--range-channel-index",
        type=int,
        default=4,
        help="Input channel index containing range_from_sensor before channel normalization.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PairedBEVDataset(Path(args.dataset_dir))
    test_size = max(1, int(len(dataset) * args.test_fraction)) if args.test_fraction > 0 else 0
    train_val_size = len(dataset) - test_size
    val_size = max(1, int(train_val_size * args.val_fraction))
    train_size = train_val_size - val_size
    split_sets = random_split(
        dataset,
        [train_size, val_size] + ([test_size] if test_size else []),
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_set = split_sets[0]
    val_set = split_sets[1]
    test_set = split_sets[2] if test_size else None

    if args.channel_normalization == "dataset":
        channel_mean, channel_std, normalization_count = compute_channel_normalization(
            dataset,
            train_set.indices,
            args.normalization_samples,
            args.seed,
        )
    else:
        channel_mean = None
        channel_std = None
        normalization_count = 0

    pin_memory = device.type == "cuda"
    loader_kwargs = {
        "num_workers": args.loader_workers,
        "pin_memory": pin_memory,
    }
    if args.loader_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = (
        DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        if test_set is not None
        else None
    )

    first_faulty, _, _, _, _, _, _, _ = dataset[0]
    model = BEVReliabilityModelV4(
        in_channels=first_faulty.shape[0],
        base_channels=args.base_channels,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.lr_reduce_patience,
        factor=args.lr_reduce_factor,
    )
    weights = ModelV4LossWeights(
        soft=args.soft_weight,
        bce=args.bce_weight,
        dice=args.dice_weight,
        range=args.range_loss_weight,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    resume_path = Path(args.resume_from) if args.resume_from else None
    if resume_path is not None and resume_path.exists():
        try:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(resume_path, map_location=device)

        if int(checkpoint.get("in_channels", first_faulty.shape[0])) != first_faulty.shape[0]:
            raise ValueError(
                f"Checkpoint expects {checkpoint.get('in_channels')} channels, "
                f"but dataset has {first_faulty.shape[0]} channels."
            )

        model.load_state_dict(checkpoint["model_state"])
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        scheduler_state = checkpoint.get("scheduler_state")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        print(f"Resuming from checkpoint: {resume_path}")
        print(f"Resume start epoch: {start_epoch}")
        print(f"Best epoch so far: {best_epoch}, best val loss: {best_val_loss:.6f}")
    elif resume_path is not None:
        print(f"Resume checkpoint not found, starting fresh: {resume_path}")

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    latest_model_path = Path(args.latest_model_path) if args.latest_model_path else None
    if latest_model_path is not None:
        latest_model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_path)
    if start_epoch == 1 or not metrics_path.exists():
        write_metrics_header(metrics_path)
    else:
        print(f"Appending metrics to existing file: {metrics_path}")

    print(f"Device: {device}")
    print(f"Dataset dir: {args.dataset_dir}")
    print(f"Samples: train={train_size}, val={val_size}, test={test_size}")
    print(f"Input: faulty BEV plus provided fault/severity conditioning")
    print("Targets: soft clean-vs-faulty BEV damage map plus binary damage target for metrics")
    print("Output: soft LiDAR unreliability probability map.")
    print("Fault/severity labels are provided to the reliability model; it does not classify them.")
    print(f"Model: base_channels={args.base_channels}, depth={args.depth}, dropout={args.dropout}")
    if channel_mean is not None:
        print(
            "Channel normalization: train-split mean/std, "
            f"samples={normalization_count}, "
            f"mean={np.array2string(channel_mean, precision=4)}, "
            f"std={np.array2string(channel_std, precision=4)}"
        )
    else:
        print("Channel normalization: disabled")
    print(
        "Reliability loss: SmoothL1 soft target + BCE + Dice, "
        f"soft_weight={args.soft_weight}, "
        f"bce_weight={args.bce_weight}, "
        f"dice_weight={args.dice_weight}, "
        f"range_loss_weight={args.range_loss_weight}"
    )
    print(
        "Adaptive LR: ReduceLROnPlateau monitors validation loss, "
        f"patience={args.lr_reduce_patience}, factor={args.lr_reduce_factor}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args,
            weights,
            channel_mean,
            channel_std,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            device,
            args,
            weights,
            channel_mean,
            channel_std,
        )
        previous_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        append_metrics(metrics_path, epoch, "train", train_metrics, current_lr)
        append_metrics(metrics_path, epoch, "val", val_metrics, current_lr)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} "
            f"soft {train_metrics['soft_reliability_loss']:.4f} "
            f"bce {train_metrics['mask_bce_loss']:.4f} "
            f"dice {train_metrics['mask_dice_loss']:.4f} "
            f"iou {train_metrics['iou']:.4f} "
            f"recall {train_metrics['recall']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} "
            f"soft {val_metrics['soft_reliability_loss']:.4f} "
            f"bce {val_metrics['mask_bce_loss']:.4f} "
            f"dice {val_metrics['mask_dice_loss']:.4f} "
            f"iou {val_metrics['iou']:.4f} "
            f"recall {val_metrics['recall']:.4f} "
            f"precision {val_metrics['precision']:.4f} "
            f"mean_prob {val_metrics['mean_probability']:.4f} "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )
        if current_lr < previous_lr:
            print(f"Learning rate reduced: {previous_lr:.2e} -> {current_lr:.2e}")

        should_stop = False
        if val_metrics["loss"] < best_val_loss - args.min_delta:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_checkpoint = make_checkpoint(
                model,
                optimizer,
                scheduler,
                first_faulty,
                args,
                weights,
                channel_mean,
                channel_std,
                epoch,
                best_epoch,
                best_val_loss,
                train_metrics,
                val_metrics,
            )
            torch.save(best_checkpoint, model_path)
        else:
            epochs_without_improvement += 1
            print(f"No val-loss improvement for {epochs_without_improvement}/{args.early_stop_patience} epochs")
            if epochs_without_improvement >= args.early_stop_patience:
                should_stop = True

        if latest_model_path is not None:
            latest_checkpoint = make_checkpoint(
                model,
                optimizer,
                scheduler,
                first_faulty,
                args,
                weights,
                channel_mean,
                channel_std,
                epoch,
                best_epoch,
                best_val_loss,
                train_metrics,
                val_metrics,
            )
            torch.save(latest_checkpoint, latest_model_path)
            print(f"Saved latest epoch checkpoint: {latest_model_path}")

        if should_stop:
            break

    if test_loader is not None and model_path.exists():
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        test_metrics = run_epoch(
            model,
            test_loader,
            None,
            device,
            args,
            weights,
            channel_mean,
            channel_std,
        )
        append_metrics(metrics_path, best_epoch, "test", test_metrics, optimizer.param_groups[0]["lr"])
        print(
            f"Test | loss {test_metrics['loss']:.4f} "
            f"soft {test_metrics['soft_reliability_loss']:.4f} "
            f"bce {test_metrics['mask_bce_loss']:.4f} "
            f"dice {test_metrics['mask_dice_loss']:.4f} "
            f"iou {test_metrics['iou']:.4f} "
            f"recall {test_metrics['recall']:.4f} "
            f"precision {test_metrics['precision']:.4f} "
            f"mean_prob {test_metrics['mean_probability']:.4f}"
        )

    print(f"Saved best model: {model_path}")
    print(f"Wrote metrics: {metrics_path}")


if __name__ == "__main__":
    main()


