"""
PanNet with CBAM Attention

Enhanced PanNet architecture with Convolutional Block Attention Modules (CBAM)
integrated into residual blocks for improved spatial and spectral feature learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CBAM, CrossAttention


class AttentionResidualBlock(nn.Module):
    """
    Residual block enhanced with CBAM attention.

    Structure: Conv -> BN -> ReLU -> Conv -> BN -> CBAM -> Residual Add -> ReLU
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: Number of channels
            reduction: Reduction ratio for CBAM channel attention
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.cbam = CBAM(channels, reduction)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out = out + residual
        return self.relu(out)


class PanNetCBAM(nn.Module):
    """
    PanNet with CBAM Attention.

    Enhancements over PanNet:
    - CBAM attention in residual blocks for adaptive feature refinement
    - Optional cross-attention between MS and PAN feature streams
    - High-pass filter for detail extraction (like original PanNet)

    Architecture:
        Input: MS (B,C,H,W) + PAN (B,1,H,W)
        -> Concat with high-pass PAN
        -> Initial Conv (n_filters)
        -> 4x AttentionResidualBlock (Conv + BN + ReLU + Conv + BN + CBAM)
        -> Output Conv
        -> Add to input MS (residual)
        Output: Fused (B,C,H,W)
    """

    def __init__(
        self,
        ms_bands: int = 4,
        n_filters: int = 64,
        n_resblocks: int = 4,
        reduction: int = 16,
        use_cross_attention: bool = False
    ):
        """
        Args:
            ms_bands: Number of multispectral bands
            n_filters: Number of filters in hidden layers
            n_resblocks: Number of attention residual blocks
            reduction: Reduction ratio for CBAM
            use_cross_attention: Whether to use cross-attention between streams
        """
        super().__init__()

        self.ms_bands = ms_bands
        self.use_cross_attention = use_cross_attention

        # High-pass filter (Laplacian)
        hp_filter = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('hp_filter', hp_filter)

        # Initial feature extraction
        # Input: MS (ms_bands) + PAN (1) + PAN_highpass (1)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(ms_bands + 2, n_filters, 3, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True)
        )

        # Optional separate PAN encoder for cross-attention
        if use_cross_attention:
            self.pan_encoder = nn.Sequential(
                nn.Conv2d(2, n_filters, 3, padding=1),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_filters, n_filters, 3, padding=1),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(inplace=True)
            )
            self.cross_attention = CrossAttention(n_filters, num_heads=4)

        # Attention residual blocks
        self.res_blocks = nn.ModuleList([
            AttentionResidualBlock(n_filters, reduction)
            for _ in range(n_resblocks)
        ])

        # Output reconstruction
        self.output = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters, ms_bands, 3, padding=1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Initialize output layer to produce small residuals
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

        # Concatenate inputs
        x = torch.cat([ms, pan, pan_hp], dim=1)

        # Initial features
        feat = self.initial_conv(x)

        # Optional cross-attention with PAN features
        if self.use_cross_attention:
            pan_feat = self.pan_encoder(torch.cat([pan, pan_hp], dim=1))
            feat = feat + self.cross_attention(feat, pan_feat)

        # Apply attention residual blocks
        for res_block in self.res_blocks:
            feat = res_block(feat)

        # Output residual
        residual = self.output(feat)

        # Final: MS + learned residual
        return ms + residual
