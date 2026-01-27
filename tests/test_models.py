"""
Unit tests for pansharpening models.
"""

import pytest
import torch

from models import (
    PNN, PanNet, DRPNN, PanNetCBAM, MultiScalePanNet,
    PanFormer, PanFormerLite, create_model, AVAILABLE_MODELS
)


class TestModelCreation:
    """Test model instantiation."""

    @pytest.mark.parametrize("model_name", AVAILABLE_MODELS)
    def test_create_model(self, model_name):
        """Test that all models can be created via factory function."""
        model = create_model(model_name, ms_bands=4)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_create_model_invalid(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError):
            create_model("invalid_model")

    @pytest.mark.parametrize("ms_bands", [4, 8, 3])
    def test_model_different_bands(self, ms_bands):
        """Test models with different number of MS bands."""
        model = create_model("pannet", ms_bands=ms_bands)
        assert model.ms_bands == ms_bands


class TestModelForward:
    """Test model forward passes."""

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensors."""
        batch_size = 2
        ms_bands = 4
        height, width = 64, 64
        ms = torch.randn(batch_size, ms_bands, height, width)
        pan = torch.randn(batch_size, 1, height, width)
        return ms, pan

    @pytest.mark.parametrize("model_class", [PNN, PanNet, DRPNN, PanNetCBAM])
    def test_cnn_forward(self, model_class, sample_input):
        """Test CNN-based models forward pass."""
        ms, pan = sample_input
        model = model_class(ms_bands=4)
        model.eval()

        with torch.no_grad():
            output = model(ms, pan)

        assert output.shape == ms.shape
        assert not torch.isnan(output).any()

    def test_mspannet_forward(self, sample_input):
        """Test MultiScalePanNet forward pass."""
        ms, pan = sample_input
        model = MultiScalePanNet(ms_bands=4)
        model.eval()

        with torch.no_grad():
            output = model(ms, pan)

        assert output.shape == ms.shape
        assert not torch.isnan(output).any()

    def test_panformer_forward(self, sample_input):
        """Test PanFormer forward pass."""
        ms, pan = sample_input
        model = PanFormer(ms_bands=4, embed_dim=64, depth=2, num_heads=4)
        model.eval()

        with torch.no_grad():
            output = model(ms, pan)

        assert output.shape == ms.shape
        assert not torch.isnan(output).any()

    def test_panformer_lite_forward(self, sample_input):
        """Test PanFormerLite forward pass."""
        ms, pan = sample_input
        model = PanFormerLite(ms_bands=4, embed_dim=32, depth=1, num_heads=2)
        model.eval()

        with torch.no_grad():
            output = model(ms, pan)

        assert output.shape == ms.shape
        assert not torch.isnan(output).any()


class TestModelGradients:
    """Test model gradient computation."""

    @pytest.mark.parametrize("model_name", ["pnn", "pannet", "pannet_cbam"])
    def test_gradients_flow(self, model_name):
        """Test that gradients flow through the model."""
        model = create_model(model_name, ms_bands=4)
        model.train()

        ms = torch.randn(1, 4, 32, 32, requires_grad=True)
        pan = torch.randn(1, 1, 32, 32, requires_grad=True)

        output = model(ms, pan)
        loss = output.mean()
        loss.backward()

        # Check gradients exist
        assert ms.grad is not None
        assert pan.grad is not None

        # Check model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestModelParameters:
    """Test model parameter counts."""

    def test_pnn_parameters(self):
        """Test PNN has expected parameter count."""
        model = PNN(ms_bands=4)
        n_params = sum(p.numel() for p in model.parameters())
        assert 10000 < n_params < 100000  # ~50K expected

    def test_pannet_parameters(self):
        """Test PanNet has expected parameter count."""
        model = PanNet(ms_bands=4)
        n_params = sum(p.numel() for p in model.parameters())
        assert 100000 < n_params < 500000  # ~340K expected

    def test_panformer_lite_parameters(self):
        """Test PanFormerLite has expected parameter count."""
        model = PanFormerLite(ms_bands=4)
        n_params = sum(p.numel() for p in model.parameters())
        assert 100000 < n_params < 500000  # ~370K expected
