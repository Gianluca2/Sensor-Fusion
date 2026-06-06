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
    clean_area: float = 1.0
    fault_area: float = 10.0


class BEVFaultRestorationModelV2(nn.Module):
    """
    Supervised paired BEV restoration model.

    Training target:
        faulty BEV -> clean BEV reconstruction

    Inference input:
        faulty BEV only

    Fault localization is derived after reconstruction by comparing the faulty
    BEV with the reconstructed clean BEV.
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

        self.clean_head = nn.Sequential(
            nn.Conv2d(channels[0], in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

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

        return {
            "clean_reconstruction": self.clean_head(current),
        }


def reconstruction_error_map(faulty_bev, reconstructed_clean):
    return torch.mean(torch.abs(faulty_bev - reconstructed_clean), dim=1, keepdim=True)


def reconstruction_error_mask(faulty_bev, reconstructed_clean, threshold: float):
    error = reconstruction_error_map(faulty_bev, reconstructed_clean)
    return error >= threshold


def weighted_reconstruction_l1_mse_loss(
    predicted_clean,
    clean_target,
    mask_target,
    clean_area_weight: float = 1.0,
    fault_area_weight: float = 10.0,
):
    pixel_weights = torch.where(
        mask_target >= 0.5,
        torch.as_tensor(fault_area_weight, device=predicted_clean.device),
        torch.as_tensor(clean_area_weight, device=predicted_clean.device),
    )
    l1 = torch.abs(predicted_clean - clean_target)
    mse = (predicted_clean - clean_target) ** 2
    weighted = pixel_weights * (l1 + 0.5 * mse)
    return weighted.mean()


def model_v2_loss(
    outputs: dict,
    clean_target,
    mask_target,
    weights: ModelV2LossWeights | None = None,
):
    if weights is None:
        weights = ModelV2LossWeights()

    reconstruction = weighted_reconstruction_l1_mse_loss(
        outputs["clean_reconstruction"],
        clean_target,
        mask_target,
        clean_area_weight=weights.clean_area,
        fault_area_weight=weights.fault_area,
    )
    parts = {
        "reconstruction_loss": reconstruction,
        "total_loss": reconstruction,
    }

    return reconstruction, parts
