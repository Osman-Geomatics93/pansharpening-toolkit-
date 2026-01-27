"""
Attention Modules for Deep Learning Pansharpening

Implements various attention mechanisms:
- ChannelAttention: Squeeze-and-Excitation style channel attention
- SpatialAttention: CBAM spatial attention
- CBAM: Convolutional Block Attention Module (channel + spatial)
- SEBlock: Lightweight Squeeze-and-Excitation block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (Squeeze-and-Excitation style).

    Learns to emphasize informative channels and suppress less useful ones.
    Uses both average-pooling and max-pooling for robust feature aggregation.
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for the bottleneck
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Channel-attended tensor (B, C, H, W)
        """
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attention = torch.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.

    Learns to emphasize important spatial locations.
    Aggregates channel information using average and max pooling.
    """

    def __init__(self, kernel_size: int = 7):
        """
        Args:
            kernel_size: Convolution kernel size (default 7 as in CBAM paper)
        """
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Spatially-attended tensor (B, C, H, W)
        """
        # Aggregate channel information
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]

        # Concatenate and apply convolution
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(combined))

        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Sequentially applies channel and spatial attention.
    Reference: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        """
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for channel attention
            kernel_size: Kernel size for spatial attention
        """
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Attended tensor (B, C, H, W)
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (lightweight version).

    Only uses average pooling for efficiency.
    Reference: "Squeeze-and-Excitation Networks" (Hu et al., CVPR 2018)
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for the bottleneck
        """
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            SE-attended tensor (B, C, H, W)
        """
        b, c, _, _ = x.size()
        # Squeeze
        y = self.squeeze(x).view(b, c)
        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale
        return x * y


class CrossAttention(nn.Module):
    """
    Cross-stream Attention Module.

    Enables attention between two feature streams (e.g., MS and PAN).
    Useful for learning cross-modal relationships.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        """
        Args:
            channels: Number of input channels
            num_heads: Number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: Query tensor (B, C, H, W)
            context: Context tensor for key/value (B, C, H, W)

        Returns:
            Cross-attended tensor (B, C, H, W)
        """
        b, c, h, w = query.shape

        # Project
        q = self.q_proj(query)
        k = self.k_proj(context)
        v = self.v_proj(context)

        # Reshape for multi-head attention
        q = q.view(b, self.num_heads, self.head_dim, h * w).transpose(-2, -1)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w).transpose(-2, -1)

        # Attention
        attn = torch.matmul(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Apply attention
        out = torch.matmul(attn, v)
        out = out.transpose(-2, -1).contiguous().view(b, c, h, w)

        return self.out_proj(out)
