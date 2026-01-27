"""
Unit tests for loss functions.
"""

import pytest
import torch

from models import (
    GradientLoss, CombinedLoss, SpectralAngleLoss, SSIMLoss,
    AdvancedCombinedLoss, create_loss, AVAILABLE_LOSSES
)


class TestLossCreation:
    """Test loss function instantiation."""

    @pytest.mark.parametrize("loss_name", ["combined", "advanced", "spectral_focus", "spatial_focus"])
    def test_create_loss(self, loss_name):
        """Test that losses can be created via factory function."""
        loss_fn = create_loss(loss_name)
        assert loss_fn is not None

    def test_create_loss_invalid(self):
        """Test that invalid loss name raises error."""
        with pytest.raises(ValueError):
            create_loss("invalid_loss")


class TestGradientLoss:
    """Test GradientLoss."""

    def test_gradient_loss_computation(self):
        """Test gradient loss computes correctly."""
        loss_fn = GradientLoss()
        pred = torch.randn(2, 4, 64, 64)
        target = torch.randn(2, 4, 64, 64)

        loss = loss_fn(pred, target)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0
        assert not torch.isnan(loss)

    def test_gradient_loss_identical(self):
        """Test gradient loss is zero for identical inputs."""
        loss_fn = GradientLoss()
        x = torch.randn(1, 4, 32, 32)

        loss = loss_fn(x, x)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestSpectralAngleLoss:
    """Test SpectralAngleLoss."""

    def test_sam_loss_computation(self):
        """Test SAM loss computes correctly."""
        loss_fn = SpectralAngleLoss()
        pred = torch.randn(2, 4, 64, 64)
        target = torch.randn(2, 4, 64, 64)

        loss = loss_fn(pred, target)

        assert loss.ndim == 0
        assert loss >= 0
        assert not torch.isnan(loss)

    def test_sam_loss_identical(self):
        """Test SAM loss is zero for identical inputs."""
        loss_fn = SpectralAngleLoss()
        x = torch.rand(1, 4, 32, 32) + 0.1  # Positive values

        loss = loss_fn(x, x)

        assert loss.item() == pytest.approx(0.0, abs=1e-5)


class TestSSIMLoss:
    """Test SSIMLoss."""

    def test_ssim_loss_computation(self):
        """Test SSIM loss computes correctly."""
        loss_fn = SSIMLoss()
        pred = torch.randn(2, 4, 64, 64)
        target = torch.randn(2, 4, 64, 64)

        loss = loss_fn(pred, target)

        assert loss.ndim == 0
        assert 0 <= loss <= 2  # SSIM loss = 1 - SSIM, range [0, 2]
        assert not torch.isnan(loss)

    def test_ssim_loss_identical(self):
        """Test SSIM loss is zero for identical inputs."""
        loss_fn = SSIMLoss()
        x = torch.randn(1, 4, 64, 64)

        loss = loss_fn(x, x)

        assert loss.item() == pytest.approx(0.0, abs=1e-5)


class TestCombinedLoss:
    """Test CombinedLoss."""

    def test_combined_loss_computation(self):
        """Test combined loss returns loss and dict."""
        loss_fn = CombinedLoss()
        pred = torch.randn(2, 4, 64, 64)
        target = torch.randn(2, 4, 64, 64)

        total_loss, loss_dict = loss_fn(pred, target)

        assert total_loss.ndim == 0
        assert not torch.isnan(total_loss)
        assert 'l1' in loss_dict
        assert 'mse' in loss_dict
        assert 'gradient' in loss_dict

    def test_combined_loss_weights(self):
        """Test combined loss respects weights."""
        loss_fn = CombinedLoss(l1_weight=1.0, mse_weight=0.0, gradient_weight=0.0)
        pred = torch.randn(2, 4, 64, 64)
        target = torch.randn(2, 4, 64, 64)

        total_loss, loss_dict = loss_fn(pred, target)

        # Total should equal L1 since other weights are 0
        assert total_loss.item() == pytest.approx(loss_dict['l1'], rel=1e-5)


class TestAdvancedCombinedLoss:
    """Test AdvancedCombinedLoss."""

    def test_advanced_loss_all_components(self):
        """Test advanced loss with all components."""
        loss_fn = AdvancedCombinedLoss(
            l1_weight=1.0,
            mse_weight=1.0,
            gradient_weight=0.1,
            ssim_weight=0.1,
            sam_weight=0.1,
            perceptual_weight=0.0  # Skip perceptual for speed
        )
        pred = torch.randn(2, 4, 64, 64)
        target = torch.randn(2, 4, 64, 64)

        total_loss, loss_dict = loss_fn(pred, target)

        assert total_loss.ndim == 0
        assert not torch.isnan(total_loss)
        assert 'l1' in loss_dict
        assert 'ssim' in loss_dict
        assert 'sam' in loss_dict

    def test_advanced_loss_selective(self):
        """Test advanced loss with selective components."""
        loss_fn = AdvancedCombinedLoss(
            l1_weight=1.0,
            mse_weight=0.0,
            gradient_weight=0.0,
            ssim_weight=0.0,
            sam_weight=0.0,
            perceptual_weight=0.0
        )
        pred = torch.randn(2, 4, 64, 64)
        target = torch.randn(2, 4, 64, 64)

        total_loss, loss_dict = loss_fn(pred, target)

        assert 'l1' in loss_dict
        assert 'mse' not in loss_dict  # Weight is 0, so not computed
