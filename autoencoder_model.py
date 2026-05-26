import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class BEVAutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        base_channels: int = 16,
        dropout: float = 0.0,
        depth: int = 4,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("Autoencoder depth must be at least 2")

        channels = [base_channels * (2 ** level) for level in range(depth + 1)]
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        previous_channels = in_channels
        for out_channels in channels:
            self.encoders.append(DoubleConv(previous_channels, out_channels, dropout=dropout))
            previous_channels = out_channels

        for _ in range(depth):
            self.pools.append(nn.MaxPool2d(2))

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(depth - 1, -1, -1):
            self.ups.append(
                nn.ConvTranspose2d(
                    channels[level + 1],
                    channels[level],
                    kernel_size=2,
                    stride=2,
                )
            )
            self.decoders.append(DoubleConv(channels[level], channels[level], dropout=dropout))

        self.out = nn.Sequential(
            nn.Conv2d(channels[0], in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        current = x

        for index, encoder in enumerate(self.encoders):
            current = encoder(current)
            if index < len(self.pools):
                current = self.pools[index](current)

        for up, decoder in zip(self.ups, self.decoders):
            current = up(current)
            current = decoder(current)

        return self.out(current)
