"""
PanFormer Lite - Lightweight Transformer for Pansharpening

Efficient version of PanFormer using:
- Smaller embedding dimension
- Fewer transformer blocks
- Window-based attention for memory efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition image into non-overlapping windows.

    Args:
        x: Input (B, H, W, C)
        window_size: Window size

    Returns:
        Windows (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse window partition.

    Args:
        windows: Windows (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H, W: Original image dimensions

    Returns:
        Reconstructed image (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention.

    More memory efficient than global attention.
    """

    def __init__(self, dim: int, window_size: int, num_heads: int = 4,
                 qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (num_windows*B, window_size*window_size, C)

        Returns:
            Attended tensor (num_windows*B, window_size*window_size, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowCrossAttention(nn.Module):
    """
    Window-based cross-attention.
    """

    def __init__(self, dim: int, window_size: int, num_heads: int = 4,
                 qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
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
            query: Query (num_windows*B, N, C)
            context: Context for key/value (num_windows*B, N, C)

        Returns:
            Cross-attended tensor (num_windows*B, N, C)
        """
        B_, N, C = query.shape

        q = self.q_proj(query).reshape(B_, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B_, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feed-forward network."""

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


class WindowTransformerBlock(nn.Module):
    """
    Transformer block with window attention.
    """

    def __init__(self, dim: int, window_size: int, num_heads: int = 4,
                 mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop: float = 0., attn_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: Input (B, H*W, C)
            H, W: Spatial dimensions

        Returns:
            Output (B, H*W, C)
        """
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad if needed
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = x.shape[1], x.shape[2]

        # Window partition
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window attention
        attn_windows = self.attn(x_windows)

        # Reverse window partition
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class WindowCrossAttentionBlock(nn.Module):
    """
    Cross-attention block with window-based attention.
    """

    def __init__(self, dim: int, window_size: int, num_heads: int = 4,
                 mlp_ratio: float = 4., qkv_bias: bool = True,
                 drop: float = 0., attn_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = WindowCrossAttention(dim, window_size, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, query: torch.Tensor, context: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            query: Query (B, H*W, C)
            context: Context (B, H*W, C)
            H, W: Spatial dimensions

        Returns:
            Output (B, H*W, C)
        """
        B, L, C = query.shape

        shortcut = query
        query = self.norm_q(query).view(B, H, W, C)
        context = self.norm_kv(context).view(B, H, W, C)

        # Pad if needed
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        context = F.pad(context, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = query.shape[1], query.shape[2]

        # Window partition
        q_windows = window_partition(query, self.window_size).view(-1, self.window_size * self.window_size, C)
        c_windows = window_partition(context, self.window_size).view(-1, self.window_size * self.window_size, C)

        # Window cross-attention
        attn_windows = self.cross_attn(q_windows, c_windows)

        # Reverse window partition
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class PatchEmbed(nn.Module):
    """Patch Embedding with convolution."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        B, C, H, W = x.shape
        x = self.proj(x)
        H_p, W_p = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (H_p, W_p)


class PanFormerLite(nn.Module):
    """
    PanFormer Lite - Lightweight Transformer for Pansharpening.

    Features:
    - Smaller model size (embed_dim=64, depth=2)
    - Window-based attention for efficiency
    - Cross-stream attention between MS and PAN
    - Residual learning
    """

    def __init__(
        self,
        ms_bands: int = 4,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.,
        window_size: int = 8,
        patch_size: int = 2,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.
    ):
        """
        Args:
            ms_bands: Number of multispectral bands
            embed_dim: Embedding dimension (smaller than full PanFormer)
            depth: Number of transformer blocks (fewer than full PanFormer)
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio (smaller for efficiency)
            window_size: Window size for local attention
            patch_size: Patch size for embedding
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
        """
        super().__init__()

        self.ms_bands = ms_bands
        self.patch_size = patch_size
        self.window_size = window_size
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
        self.pan_embed = PatchEmbed(2, embed_dim, patch_size)

        # MS stream window transformer blocks
        self.ms_blocks = nn.ModuleList([
            WindowTransformerBlock(embed_dim, window_size, num_heads, mlp_ratio,
                                   drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])

        # PAN stream window transformer blocks
        self.pan_blocks = nn.ModuleList([
            WindowTransformerBlock(embed_dim, window_size, num_heads, mlp_ratio,
                                   drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])

        # Cross-attention fusion
        self.cross_attn = nn.ModuleList([
            WindowCrossAttentionBlock(embed_dim, window_size, num_heads, mlp_ratio,
                                      drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])

        # Output
        self.norm = nn.LayerNorm(embed_dim)

        # Decoder: upsample from patches to full resolution
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * patch_size * patch_size, 3, padding=1),
            nn.PixelShuffle(patch_size),
            nn.Conv2d(embed_dim, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, ms_bands, 3, padding=1)
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

    def forward(self, ms: torch.Tensor, pan: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            ms: Upsampled MS (B, bands, H, W)
            pan: PAN image (B, 1, H, W)

        Returns:
            Fused image (B, bands, H, W)
        """
        B, C, H_in, W_in = ms.shape

        # Extract high-frequency
        hp_filter = self.hp_filter.to(pan.device)
        pan_hp = F.conv2d(pan, hp_filter, padding=1)
        pan_input = torch.cat([pan, pan_hp], dim=1)

        # Patch embedding
        ms_tokens, (H_p, W_p) = self.ms_embed(ms)
        pan_tokens, _ = self.pan_embed(pan_input)

        # Two-stream processing with window attention and cross-attention
        for ms_block, pan_block, cross_block in zip(
            self.ms_blocks, self.pan_blocks, self.cross_attn
        ):
            # Self-attention in each stream
            ms_tokens = ms_block(ms_tokens, H_p, W_p)
            pan_tokens = pan_block(pan_tokens, H_p, W_p)

            # Cross-attention: MS attends to PAN
            ms_tokens = cross_block(ms_tokens, pan_tokens, H_p, W_p)

        # Final normalization
        fused_tokens = self.norm(ms_tokens)

        # Reshape to spatial
        fused = fused_tokens.transpose(1, 2).view(B, self.embed_dim, H_p, W_p)

        # Decode to output
        residual = self.decoder(fused)

        # Handle size mismatch
        if residual.shape[2:] != ms.shape[2:]:
            residual = F.interpolate(residual, size=(H_in, W_in), mode='bilinear', align_corners=False)

        # Residual learning
        return ms + residual
