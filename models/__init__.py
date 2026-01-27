"""
Deep Learning Models for Pansharpening

Available Models:
- PNN: Simple 3-layer CNN (baseline)
- PanNet: ResNet-style with high-pass filtering
- DRPNN: Deep Residual PanNet
- PanNetCBAM: PanNet with CBAM attention
- MultiScalePanNet: Multi-scale feature pyramid
- PanFormer: Transformer-based with cross-attention
- PanFormerLite: Lightweight transformer with window attention

Available Losses:
- GradientLoss: Edge/spatial fidelity loss
- CombinedLoss: L1 + MSE + Gradient
- SpectralAngleLoss: SAM-based spectral loss
- SSIMLoss: Structural similarity loss
- PerceptualLoss: VGG feature-based loss
- AdvancedCombinedLoss: Configurable multi-loss
"""

from .pnn import PNN
from .pannet import PanNet
from .drpnn import DRPNN
from .pannet_cbam import PanNetCBAM
from .mspannet import MultiScalePanNet
from .panformer import PanFormer
from .panformer_lite import PanFormerLite
from .losses import (
    GradientLoss,
    CombinedLoss,
    SpectralAngleLoss,
    SSIMLoss,
    PerceptualLoss,
    AdvancedCombinedLoss
)
from .attention import (
    ChannelAttention,
    SpatialAttention,
    CBAM,
    SEBlock,
    CrossAttention
)

__all__ = [
    # Models
    'PNN',
    'PanNet',
    'DRPNN',
    'PanNetCBAM',
    'MultiScalePanNet',
    'PanFormer',
    'PanFormerLite',
    # Losses
    'GradientLoss',
    'CombinedLoss',
    'SpectralAngleLoss',
    'SSIMLoss',
    'PerceptualLoss',
    'AdvancedCombinedLoss',
    # Attention
    'ChannelAttention',
    'SpatialAttention',
    'CBAM',
    'SEBlock',
    'CrossAttention',
    # Factory functions
    'create_model',
    'create_loss',
    'AVAILABLE_MODELS',
    'AVAILABLE_LOSSES'
]

# Model registry
AVAILABLE_MODELS = [
    'pnn',
    'pannet',
    'drpnn',
    'pannet_cbam',
    'mspannet',
    'panformer',
    'panformer_lite'
]

# Loss registry
AVAILABLE_LOSSES = [
    'combined',
    'advanced',
    'l1',
    'mse',
    'gradient',
    'ssim',
    'sam',
    'perceptual'
]


def create_model(model_name: str, ms_bands: int = 4, **kwargs):
    """
    Factory function to create a pansharpening model.

    Args:
        model_name: Name of the model ('pnn', 'pannet', 'drpnn', 'pannet_cbam',
                    'mspannet', 'panformer', 'panformer_lite')
        ms_bands: Number of multispectral bands
        **kwargs: Additional model-specific arguments

    Returns:
        Instantiated model

    Example:
        >>> model = create_model('pannet_cbam', ms_bands=4, use_cross_attention=True)
    """
    model_name = model_name.lower()

    if model_name == 'pnn':
        return PNN(ms_bands=ms_bands, **kwargs)

    elif model_name == 'pannet':
        return PanNet(ms_bands=ms_bands, **kwargs)

    elif model_name == 'drpnn':
        return DRPNN(ms_bands=ms_bands, **kwargs)

    elif model_name == 'pannet_cbam':
        return PanNetCBAM(ms_bands=ms_bands, **kwargs)

    elif model_name == 'mspannet':
        return MultiScalePanNet(ms_bands=ms_bands, **kwargs)

    elif model_name == 'panformer':
        return PanFormer(ms_bands=ms_bands, **kwargs)

    elif model_name == 'panformer_lite':
        return PanFormerLite(ms_bands=ms_bands, **kwargs)

    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {AVAILABLE_MODELS}"
        )


def create_loss(loss_name: str = 'combined', **kwargs):
    """
    Factory function to create a loss function.

    Args:
        loss_name: Name of the loss function or preset
            Presets:
            - 'combined': CombinedLoss (L1 + MSE + Gradient)
            - 'advanced': AdvancedCombinedLoss with all components
            - 'default': Same as 'combined'
            - 'spectral_focus': Higher SAM weight for spectral fidelity
            - 'spatial_focus': Higher gradient and SSIM for spatial quality

            Individual losses:
            - 'l1': L1 loss only
            - 'mse': MSE loss only
            - 'gradient': Gradient loss only
            - 'ssim': SSIM loss only
            - 'sam': Spectral Angle loss only
            - 'perceptual': Perceptual (VGG) loss only

        **kwargs: Override default weights for AdvancedCombinedLoss

    Returns:
        Instantiated loss function

    Example:
        >>> loss_fn = create_loss('spectral_focus')
        >>> loss_fn = create_loss('advanced', ssim_weight=0.5, sam_weight=0.3)
    """
    import torch.nn as nn

    loss_name = loss_name.lower()

    # Presets
    if loss_name in ['combined', 'default']:
        return CombinedLoss(**kwargs)

    elif loss_name == 'advanced':
        # Default advanced loss with all components
        defaults = {
            'l1_weight': 1.0,
            'mse_weight': 0.5,
            'gradient_weight': 0.1,
            'ssim_weight': 0.1,
            'sam_weight': 0.1,
            'perceptual_weight': 0.0  # Off by default (slow)
        }
        defaults.update(kwargs)
        return AdvancedCombinedLoss(**defaults)

    elif loss_name == 'spectral_focus':
        # Higher spectral fidelity emphasis
        defaults = {
            'l1_weight': 1.0,
            'mse_weight': 0.3,
            'gradient_weight': 0.05,
            'ssim_weight': 0.05,
            'sam_weight': 0.5,
            'perceptual_weight': 0.0
        }
        defaults.update(kwargs)
        return AdvancedCombinedLoss(**defaults)

    elif loss_name == 'spatial_focus':
        # Higher spatial quality emphasis
        defaults = {
            'l1_weight': 1.0,
            'mse_weight': 0.3,
            'gradient_weight': 0.3,
            'ssim_weight': 0.3,
            'sam_weight': 0.05,
            'perceptual_weight': 0.0
        }
        defaults.update(kwargs)
        return AdvancedCombinedLoss(**defaults)

    # Individual losses
    elif loss_name == 'l1':
        return nn.L1Loss()

    elif loss_name == 'mse':
        return nn.MSELoss()

    elif loss_name == 'gradient':
        return GradientLoss()

    elif loss_name == 'ssim':
        return SSIMLoss(**kwargs)

    elif loss_name == 'sam':
        return SpectralAngleLoss(**kwargs)

    elif loss_name == 'perceptual':
        return PerceptualLoss(**kwargs)

    else:
        raise ValueError(
            f"Unknown loss: {loss_name}. "
            f"Available: {AVAILABLE_LOSSES}"
        )
