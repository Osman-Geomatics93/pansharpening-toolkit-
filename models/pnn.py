"""
PNN (Pansharpening Neural Network)

Based on Masi et al., "Pansharpening by Convolutional Neural Networks"
Simple 3-layer CNN architecture.
"""

import torch
import torch.nn as nn


class PNN(nn.Module):
    """
    Pansharpening Neural Network (PNN).

    Architecture:
    - Conv1: 9x9 kernel, 64 filters, ReLU
    - Conv2: 5x5 kernel, 32 filters, ReLU
    - Conv3: 5x5 kernel, output_bands filters
    - Residual connection from input MS
    """

    def __init__(self, ms_bands: int = 4):
        """
        Args:
            ms_bands: Number of multispectral bands
        """
        super().__init__()

        self.ms_bands = ms_bands
        input_channels = ms_bands + 1  # MS + PAN

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, ms_bands, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, ms: torch.Tensor, pan: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            ms: Upsampled MS image (B, bands, H, W)
            pan: PAN image (B, 1, H, W)

        Returns:
            Fused image (B, bands, H, W)
        """
        x = torch.cat([ms, pan], dim=1)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        # Residual connection
        out = x + ms

        return out
