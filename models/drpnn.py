"""
DRPNN - Deep Residual Pan-sharpening Neural Network

Deeper architecture with skip connections for better feature learning.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block with two conv layers."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class DRPNN(nn.Module):
    """
    Deep Residual Pan-sharpening Neural Network.

    Features:
    - Deeper architecture (8 residual blocks by default)
    - Local and global residual learning
    - Better feature representation
    """

    def __init__(self, ms_bands: int = 4, n_filters: int = 64, n_resblocks: int = 8):
        """
        Args:
            ms_bands: Number of multispectral bands
            n_filters: Number of filters in hidden layers
            n_resblocks: Number of residual blocks
        """
        super().__init__()

        self.ms_bands = ms_bands
        input_channels = ms_bands + 1

        # Shallow feature extraction
        self.shallow = nn.Sequential(
            nn.Conv2d(input_channels, n_filters, 9, padding=4),
            nn.ReLU(inplace=True)
        )

        # Deep feature extraction with residual blocks
        self.deep = nn.ModuleList([
            ResidualBlock(n_filters) for _ in range(n_resblocks)
        ])

        # Global residual learning
        self.global_residual = nn.Conv2d(n_filters, n_filters, 3, padding=1)

        # Reconstruction
        self.reconstruct = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters, ms_bands, 3, padding=1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, ms: torch.Tensor, pan: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            ms: Upsampled MS (B, bands, H, W)
            pan: PAN image (B, 1, H, W)

        Returns:
            Fused image (B, bands, H, W)
        """
        x = torch.cat([ms, pan], dim=1)

        # Shallow features
        shallow_feat = self.shallow(x)

        # Deep features with local residual learning
        feat = shallow_feat
        for res_block in self.deep:
            feat = res_block(feat)

        # Global residual
        feat = self.global_residual(feat) + shallow_feat

        # Reconstruction (residual output)
        residual = self.reconstruct(feat)

        # Add to input MS
        return ms + residual
