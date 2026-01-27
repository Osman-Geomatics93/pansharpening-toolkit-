"""
PanNet - A Deep Network Architecture for Pan-Sharpening

Based on Yang et al., "PanNet: A Deep Network Architecture for
Pan-Sharpening" (ICCV 2017)

Key idea: Learn high-frequency residuals, not the full image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class PanNet(nn.Module):
    """
    PanNet architecture.

    Features:
    - High-pass filter for detail extraction
    - Residual learning
    - Domain-specific design for pansharpening
    """

    def __init__(self, ms_bands: int = 4, n_filters: int = 64, n_resblocks: int = 4):
        """
        Args:
            ms_bands: Number of multispectral bands
            n_filters: Number of filters in hidden layers
            n_resblocks: Number of residual blocks
        """
        super().__init__()

        self.ms_bands = ms_bands

        # High-pass filter (Laplacian)
        hp_filter = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('hp_filter', hp_filter)

        # Initial feature extraction (MS + PAN + PAN_highpass = ms_bands + 2)
        self.initial = nn.Sequential(
            nn.Conv2d(ms_bands + 2, n_filters, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(n_filters) for _ in range(n_resblocks)]
        )

        # Output reconstruction
        self.output = nn.Sequential(
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

        # Initialize output layer to produce small residuals initially
        # This helps with residual learning - start near identity
        for m in self.output.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
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
        # Extract high-frequency from PAN
        hp_filter = self.hp_filter.to(pan.device)
        pan_hp = F.conv2d(pan, hp_filter, padding=1)

        # Concatenate MS + PAN + PAN high-pass for better feature extraction
        x = torch.cat([ms, pan, pan_hp], dim=1)

        # Feature extraction
        feat = self.initial(x)
        feat = self.res_blocks(feat)

        # Output residual
        residual = self.output(feat)

        # Final: MS + learned residual
        return ms + residual
