"""
Configuration for Pansharpening Project
"""

from pathlib import Path

# Project paths - handle both script and Jupyter notebook execution
try:
    PROJECT_ROOT = Path(__file__).parent.parent
except NameError:
    # Running in Jupyter notebook
    PROJECT_ROOT = Path(r"D:\Udemy_Cour\Pancharping\pansharpening_project")

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Default data paths (now inside project)
DEFAULT_PAN_PATH = str(DATA_DIR / "pan.tif")
DEFAULT_MS_PATH = str(DATA_DIR / "ms.tif")

# Training configuration
TRAINING_CONFIG = {
    'epochs': 200,
    'batch_size': 16,
    'learning_rate': 1e-4,  # Lower LR for stable training
    'patch_size': 64,
    'stride': 32,  # Overlap patches for more training data
    'val_split': 0.2,
}

# Loss weights - increased gradient weight for spatial details
LOSS_CONFIG = {
    'l1_weight': 1.0,
    'mse_weight': 0.5,
    'gradient_weight': 0.5  # Higher weight for spatial fidelity
}

# Model options
AVAILABLE_MODELS = [
    'pnn',           # Simple 3-layer CNN (baseline)
    'pannet',        # ResNet style with high-pass filter
    'drpnn',         # Deep Residual PanNet
    'pannet_cbam',   # PanNet with CBAM attention
    'mspannet',      # Multi-scale feature pyramid
    'panformer',     # Transformer-based with cross-attention
    'panformer_lite' # Lightweight transformer with window attention
]

# Model-specific configurations
MODEL_CONFIGS = {
    'pnn': {
        'n_filters': 64,
    },
    'pannet': {
        'n_filters': 64,
        'n_resblocks': 4,
    },
    'drpnn': {
        'n_filters': 64,
        'n_resblocks': 8,
    },
    'pannet_cbam': {
        'n_filters': 64,
        'n_resblocks': 4,
        'reduction': 16,
        'use_cross_attention': False,
    },
    'mspannet': {
        'base_channels': 64,
    },
    'panformer': {
        'embed_dim': 128,
        'depth': 4,
        'num_heads': 8,
        'mlp_ratio': 4.0,
        'patch_size': 4,
    },
    'panformer_lite': {
        'embed_dim': 64,
        'depth': 2,
        'num_heads': 4,
        'mlp_ratio': 2.0,
        'window_size': 8,
        'patch_size': 2,
    },
}

# Loss configurations
AVAILABLE_LOSSES = [
    'combined',       # Default: L1 + MSE + Gradient
    'advanced',       # Advanced: L1 + MSE + Gradient + SSIM + SAM
    'spectral_focus', # Higher SAM weight for spectral fidelity
    'spatial_focus',  # Higher gradient and SSIM for spatial quality
]

LOSS_CONFIGS = {
    'combined': {
        'l1_weight': 1.0,
        'mse_weight': 0.5,
        'gradient_weight': 0.1,
    },
    'advanced': {
        'l1_weight': 1.0,
        'mse_weight': 0.5,
        'gradient_weight': 0.1,
        'ssim_weight': 0.1,
        'sam_weight': 0.1,
        'perceptual_weight': 0.0,
    },
    'spectral_focus': {
        'l1_weight': 1.0,
        'mse_weight': 0.3,
        'gradient_weight': 0.05,
        'ssim_weight': 0.05,
        'sam_weight': 0.5,
        'perceptual_weight': 0.0,
    },
    'spatial_focus': {
        'l1_weight': 1.0,
        'mse_weight': 0.3,
        'gradient_weight': 0.3,
        'ssim_weight': 0.3,
        'sam_weight': 0.05,
        'perceptual_weight': 0.0,
    },
}

# Classic methods
AVAILABLE_CLASSIC_METHODS = ['brovey', 'ihs', 'sfim', 'gram_schmidt', 'hpf']
