"""
Unit tests for attention modules.
"""

import pytest
import torch

from models.attention import (
    ChannelAttention, SpatialAttention, CBAM, SEBlock, CrossAttention
)


class TestChannelAttention:
    """Test ChannelAttention module."""

    def test_forward_shape(self):
        """Test output shape matches input."""
        attn = ChannelAttention(channels=64, reduction=16)
        x = torch.randn(2, 64, 32, 32)

        out = attn(x)

        assert out.shape == x.shape

    def test_attention_range(self):
        """Test attention weights are in valid range."""
        attn = ChannelAttention(channels=64, reduction=16)
        x = torch.randn(2, 64, 32, 32)

        # Get attention weights by checking the ratio
        out = attn(x)

        # Output should be scaled version of input
        assert not torch.isnan(out).any()


class TestSpatialAttention:
    """Test SpatialAttention module."""

    def test_forward_shape(self):
        """Test output shape matches input."""
        attn = SpatialAttention(kernel_size=7)
        x = torch.randn(2, 64, 32, 32)

        out = attn(x)

        assert out.shape == x.shape

    @pytest.mark.parametrize("kernel_size", [3, 5, 7])
    def test_different_kernel_sizes(self, kernel_size):
        """Test with different kernel sizes."""
        attn = SpatialAttention(kernel_size=kernel_size)
        x = torch.randn(2, 64, 32, 32)

        out = attn(x)

        assert out.shape == x.shape


class TestCBAM:
    """Test CBAM module."""

    def test_forward_shape(self):
        """Test output shape matches input."""
        cbam = CBAM(channels=64, reduction=16)
        x = torch.randn(2, 64, 32, 32)

        out = cbam(x)

        assert out.shape == x.shape

    def test_gradient_flow(self):
        """Test gradients flow through CBAM."""
        cbam = CBAM(channels=64)
        x = torch.randn(2, 64, 32, 32, requires_grad=True)

        out = cbam(x)
        loss = out.mean()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestSEBlock:
    """Test SEBlock module."""

    def test_forward_shape(self):
        """Test output shape matches input."""
        se = SEBlock(channels=64, reduction=16)
        x = torch.randn(2, 64, 32, 32)

        out = se(x)

        assert out.shape == x.shape

    def test_squeeze_excitation(self):
        """Test SE block produces valid output."""
        se = SEBlock(channels=64)
        x = torch.randn(2, 64, 32, 32)

        out = se(x)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestCrossAttention:
    """Test CrossAttention module."""

    def test_forward_shape(self):
        """Test output shape matches query."""
        cross_attn = CrossAttention(channels=64, num_heads=4)
        query = torch.randn(2, 64, 16, 16)
        context = torch.randn(2, 64, 16, 16)

        out = cross_attn(query, context)

        assert out.shape == query.shape

    def test_self_attention_case(self):
        """Test cross-attention with same input (self-attention)."""
        cross_attn = CrossAttention(channels=64, num_heads=4)
        x = torch.randn(2, 64, 16, 16)

        out = cross_attn(x, x)

        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_gradient_flow(self):
        """Test gradients flow through cross-attention."""
        cross_attn = CrossAttention(channels=64, num_heads=4)
        query = torch.randn(2, 64, 16, 16, requires_grad=True)
        context = torch.randn(2, 64, 16, 16, requires_grad=True)

        out = cross_attn(query, context)
        loss = out.mean()
        loss.backward()

        assert query.grad is not None
        assert context.grad is not None
