"""
Multi-Scale PanNet (MSPanNet)

Feature pyramid style architecture for pansharpening.
Extracts features at multiple scales and fuses them progressively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CBAM


class ConvBlock(nn.Module):
    """Basic convolution block: Conv -> BN -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualBlock(nn.Module):
    """Residual block with optional attention."""

    def __init__(self, channels: int, use_attention: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.attention = CBAM(channels) if use_attention else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        return self.relu(out + residual)


class MultiScaleEncoder(nn.Module):
    """
    Multi-scale encoder that extracts features at 3 scales.

    Scale 1: 1x (original resolution)
    Scale 2: 1/2x
    Scale 3: 1/4x
    """

    def __init__(self, in_channels: int, base_channels: int = 64):
        super().__init__()

        # Scale 1: 1x
        self.scale1_conv = ConvBlock(in_channels, base_channels)
        self.scale1_res = ResidualBlock(base_channels)

        # Scale 2: 1/2x (downsample then process)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.scale2_res = ResidualBlock(base_channels * 2)

        # Scale 3: 1/4x
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.scale3_res = ResidualBlock(base_channels * 4)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Tuple of features at 3 scales
        """
        # Scale 1
        feat1 = self.scale1_conv(x)
        feat1 = self.scale1_res(feat1)

        # Scale 2
        feat2 = self.down1(feat1)
        feat2 = self.scale2_res(feat2)

        # Scale 3
        feat3 = self.down2(feat2)
        feat3 = self.scale3_res(feat3)

        return feat1, feat2, feat3


class FusionModule(nn.Module):
    """
    Fusion module for combining MS and PAN features at a single scale.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = ConvBlock(channels * 2, channels)
        self.attention = CBAM(channels)

    def forward(self, ms_feat: torch.Tensor, pan_feat: torch.Tensor) -> torch.Tensor:
        """Fuse MS and PAN features."""
        combined = torch.cat([ms_feat, pan_feat], dim=1)
        fused = self.conv(combined)
        return self.attention(fused)


class MultiScaleDecoder(nn.Module):
    """
    Progressive decoder that upsamples and merges multi-scale features.
    """

    def __init__(self, base_channels: int = 64, out_channels: int = 4):
        super().__init__()

        # Upsample from scale 3 to scale 2
        self.up3_to_2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.merge2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.res2 = ResidualBlock(base_channels * 2)

        # Upsample from scale 2 to scale 1
        self.up2_to_1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.merge1 = ConvBlock(base_channels * 2, base_channels)
        self.res1 = ResidualBlock(base_channels)

        # Final output
        self.output = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor,
                feat3: torch.Tensor) -> torch.Tensor:
        """
        Progressive decoding with skip connections.

        Args:
            feat1: Scale 1 features (1x resolution)
            feat2: Scale 2 features (1/2x resolution)
            feat3: Scale 3 features (1/4x resolution)

        Returns:
            Decoded output at 1x resolution
        """
        # Scale 3 -> Scale 2
        up3 = self.up3_to_2(feat3)
        merged2 = torch.cat([up3, feat2], dim=1)
        merged2 = self.merge2(merged2)
        merged2 = self.res2(merged2)

        # Scale 2 -> Scale 1
        up2 = self.up2_to_1(merged2)
        merged1 = torch.cat([up2, feat1], dim=1)
        merged1 = self.merge1(merged1)
        merged1 = self.res1(merged1)

        return self.output(merged1)


class MultiScalePanNet(nn.Module):
    """
    Multi-Scale PanNet (MSPanNet).

    Architecture:
    - Dual multi-scale encoders (MS and PAN)
    - Multi-scale fusion at each level
    - Progressive decoder with skip connections
    - Residual output

    Features:
    - Captures both local and global context
    - Multi-scale feature fusion
    - CBAM attention at each scale
    """

    def __init__(self, ms_bands: int = 4, base_channels: int = 64):
        """
        Args:
            ms_bands: Number of multispectral bands
            base_channels: Base number of channels (doubles at each scale)
        """
        super().__init__()

        self.ms_bands = ms_bands

        # High-pass filter
        hp_filter = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('hp_filter', hp_filter)

        # Dual encoders
        self.ms_encoder = MultiScaleEncoder(ms_bands, base_channels)
        self.pan_encoder = MultiScaleEncoder(2, base_channels)  # PAN + HP

        # Multi-scale fusion modules
        self.fusion1 = FusionModule(base_channels)
        self.fusion2 = FusionModule(base_channels * 2)
        self.fusion3 = FusionModule(base_channels * 4)

        # Decoder
        self.decoder = MultiScaleDecoder(base_channels, ms_bands)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
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
        # Extract high-frequency
        hp_filter = self.hp_filter.to(pan.device)
        pan_hp = F.conv2d(pan, hp_filter, padding=1)
        pan_input = torch.cat([pan, pan_hp], dim=1)

        # Multi-scale encoding
        ms_feat1, ms_feat2, ms_feat3 = self.ms_encoder(ms)
        pan_feat1, pan_feat2, pan_feat3 = self.pan_encoder(pan_input)

        # Multi-scale fusion
        fused1 = self.fusion1(ms_feat1, pan_feat1)
        fused2 = self.fusion2(ms_feat2, pan_feat2)
        fused3 = self.fusion3(ms_feat3, pan_feat3)

        # Progressive decoding
        residual = self.decoder(fused1, fused2, fused3)

        # Residual learning
        return ms + residual
