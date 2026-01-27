"""
PanFormer - Transformer-based Pansharpening

Two-stream transformer architecture with cross-attention for MS-PAN fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbed(nn.Module):
    """
    Patch Embedding layer.

    Converts images to patch sequences for transformer processing.
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 4):
        """
        Args:
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            patch_size: Size of each patch
        """
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Patch embeddings (B, N, embed_dim), (H_patches, W_patches)
        """
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        H_p, W_p = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        x = self.norm(x)
        return x, (H_p, W_p)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention module.
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, N, C)

        Returns:
            Attended tensor (B, N, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """
    Cross-Attention module for attending to another sequence.
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: Query tensor (B, N_q, C)
            context: Context tensor for key/value (B, N_kv, C)

        Returns:
            Cross-attended tensor (B, N_q, C)
        """
        B, N_q, C = query.shape
        N_kv = context.shape[1]

        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """
    Feed-forward network with GELU activation.
    """

    def __init__(self, dim: int, hidden_dim: int = None, drop: float = 0.):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and MLP.
    """

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.,
                 qkv_bias: bool = True, drop: float = 0., attn_drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for MS-PAN fusion.
    """

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.,
                 qkv_bias: bool = True, drop: float = 0., attn_drop: float = 0.):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = query + self.cross_attn(self.norm_q(query), self.norm_kv(context))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchUnEmbed(nn.Module):
    """
    Converts patch embeddings back to image.
    """

    def __init__(self, embed_dim: int, out_channels: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(embed_dim, out_channels * patch_size * patch_size)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, hw: tuple) -> torch.Tensor:
        """
        Args:
            x: Patch embeddings (B, N, embed_dim)
            hw: (H_patches, W_patches)

        Returns:
            Image tensor (B, out_channels, H, W)
        """
        H_p, W_p = hw
        B, N, _ = x.shape
        x = self.proj(x)  # (B, N, C*p*p)
        x = x.reshape(B, H_p, W_p, self.out_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, C, H_p, p, W_p, p)
        x = x.reshape(B, self.out_channels, H_p * self.patch_size, W_p * self.patch_size)
        return x


class PanFormer(nn.Module):
    """
    PanFormer - Transformer-based Pansharpening Network.

    Architecture:
    - Patch embedding for MS and PAN
    - Two-stream transformer with self-attention
    - Cross-attention fusion at each depth
    - Patch un-embedding and residual output

    Features:
    - Global receptive field through self-attention
    - Cross-modal fusion through cross-attention
    - Position embeddings for spatial awareness
    """

    def __init__(
        self,
        ms_bands: int = 4,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.,
        patch_size: int = 4,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.
    ):
        """
        Args:
            ms_bands: Number of multispectral bands
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            patch_size: Patch size for embedding
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
        """
        super().__init__()

        self.ms_bands = ms_bands
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # High-pass filter
        hp_filter = torch.tensor([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('hp_filter', hp_filter)

        # Patch embeddings
        self.ms_embed = PatchEmbed(ms_bands, embed_dim, patch_size)
        self.pan_embed = PatchEmbed(2, embed_dim, patch_size)  # PAN + HP

        # Positional embeddings (will be resized based on input)
        self.pos_embed_ms = nn.Parameter(torch.zeros(1, 256, embed_dim))
        self.pos_embed_pan = nn.Parameter(torch.zeros(1, 256, embed_dim))
        nn.init.trunc_normal_(self.pos_embed_ms, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_pan, std=0.02)

        # MS stream transformer blocks
        self.ms_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])

        # PAN stream transformer blocks
        self.pan_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])

        # Cross-attention fusion
        self.cross_attn_ms = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])

        # Fusion and output
        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.unembed = PatchUnEmbed(embed_dim, ms_bands, patch_size)

        # Refinement conv
        self.refine = nn.Sequential(
            nn.Conv2d(ms_bands, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, ms_bands, 3, padding=1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _resize_pos_embed(self, pos_embed: torch.Tensor, num_patches: int) -> torch.Tensor:
        """Resize positional embeddings if needed."""
        if pos_embed.shape[1] == num_patches:
            return pos_embed

        # Interpolate positional embeddings
        pos_embed = pos_embed.transpose(1, 2)  # (1, embed_dim, N)
        pos_embed = F.interpolate(pos_embed, size=num_patches, mode='linear', align_corners=False)
        pos_embed = pos_embed.transpose(1, 2)  # (1, N, embed_dim)
        return pos_embed

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

        # Patch embedding
        ms_tokens, ms_hw = self.ms_embed(ms)
        pan_tokens, pan_hw = self.pan_embed(pan_input)

        # Add positional embeddings
        num_patches = ms_tokens.shape[1]
        pos_ms = self._resize_pos_embed(self.pos_embed_ms, num_patches)
        pos_pan = self._resize_pos_embed(self.pos_embed_pan, num_patches)

        ms_tokens = ms_tokens + pos_ms
        pan_tokens = pan_tokens + pos_pan

        # Two-stream processing with cross-attention
        for ms_block, pan_block, cross_block in zip(
            self.ms_blocks, self.pan_blocks, self.cross_attn_ms
        ):
            # Self-attention in each stream
            ms_tokens = ms_block(ms_tokens)
            pan_tokens = pan_block(pan_tokens)

            # Cross-attention: MS attends to PAN
            ms_tokens = cross_block(ms_tokens, pan_tokens)

        # Final fusion
        fused_tokens = self.fusion_norm(ms_tokens)

        # Convert back to image
        residual = self.unembed(fused_tokens, ms_hw)

        # Refinement
        residual = self.refine(residual)

        # Handle size mismatch (if input size is not divisible by patch_size)
        if residual.shape[2:] != ms.shape[2:]:
            residual = F.interpolate(residual, size=ms.shape[2:], mode='bilinear', align_corners=False)

        # Residual learning
        return ms + residual
