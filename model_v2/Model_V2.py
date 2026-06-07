from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


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


@dataclass(frozen=True)
class ModelV2LossWeights:
    positive: float = 3.0
    dice: float = 1.0


class BEVFaultRestorationModelV2(nn.Module):
    """
    Supervised conditioned BEV fault-segmentation model.

    Training target:
        faulty BEV + known fault/severity -> fault mask

    Inference input:
        faulty BEV + known fault/severity
    """

    def __init__(
        self,
        in_channels: int = 5,
        base_channels: int = 48,
        depth: int = 4,
        dropout: float = 0.12,
        num_fault_classes: int = len(FAULT_CLASSES),
        num_severity_classes: int = len(SEVERITY_CLASSES),
        fault_embedding_dim: int = 8,
        severity_embedding_dim: int = 4,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("Model_V2 depth must be at least 2")

        self.in_channels = in_channels
        self.fault_embedding = nn.Embedding(num_fault_classes, fault_embedding_dim)
        self.severity_embedding = nn.Embedding(num_severity_classes, severity_embedding_dim)
        conditioned_in_channels = in_channels + fault_embedding_dim + severity_embedding_dim

        channels = [base_channels * (2 ** level) for level in range(depth + 1)]
        self.input_block = DoubleConv(conditioned_in_channels, channels[0], dropout)
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

        self.mask_head = nn.Conv2d(channels[0], 1, kernel_size=1)

    def make_condition_maps(self, faulty_bev, fault_type_index=None, severity_index=None):
        batch_size, _, height, width = faulty_bev.shape
        device = faulty_bev.device

        if fault_type_index is None:
            fault_type_index = torch.zeros(batch_size, dtype=torch.long, device=device)
        if severity_index is None:
            severity_index = torch.ones(batch_size, dtype=torch.long, device=device)

        fault_features = self.fault_embedding(fault_type_index).view(batch_size, -1, 1, 1)
        severity_features = self.severity_embedding(severity_index).view(batch_size, -1, 1, 1)
        condition = torch.cat([fault_features, severity_features], dim=1)
        return condition.expand(-1, -1, height, width)

    def forward(self, faulty_bev, fault_type_index=None, severity_index=None):
        condition_maps = self.make_condition_maps(faulty_bev, fault_type_index, severity_index)
        faulty_bev = torch.cat([faulty_bev, condition_maps], dim=1)
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

        fault_logits = self.mask_head(current)
        return {
            "fault_logits": fault_logits,
            "fault_probability": torch.sigmoid(fault_logits),
        }


def reconstruction_error_map(faulty_bev, reconstructed_clean):
    return torch.mean(torch.abs(faulty_bev - reconstructed_clean), dim=1, keepdim=True)


def reconstruction_error_mask(faulty_bev, reconstructed_clean, threshold: float):
    error = reconstruction_error_map(faulty_bev, reconstructed_clean)
    return error >= threshold


def bce_dice_mask_loss(logits, mask_target, positive_weight: float = 3.0, dice_weight: float = 1.0):
    positive_weight_tensor = torch.as_tensor(
        positive_weight,
        dtype=logits.dtype,
        device=logits.device,
    )
    bce = F.binary_cross_entropy_with_logits(
        logits,
        mask_target,
        pos_weight=positive_weight_tensor,
    )

    probs = torch.sigmoid(logits)
    smooth = 1.0
    intersection = (probs * mask_target).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + mask_target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + smooth) / (denominator + smooth))
    dice = dice.mean()

    return bce + dice_weight * dice, bce, dice


def model_v2_loss(
    outputs: dict,
    clean_target,
    mask_target,
    weights: ModelV2LossWeights | None = None,
):
    if weights is None:
        weights = ModelV2LossWeights()

    total, bce, dice = bce_dice_mask_loss(
        outputs["fault_logits"],
        mask_target,
        positive_weight=weights.positive,
        dice_weight=weights.dice,
    )
    parts = {
        "mask_bce_loss": bce,
        "mask_dice_loss": dice,
        "total_loss": total,
    }

    return total, parts
