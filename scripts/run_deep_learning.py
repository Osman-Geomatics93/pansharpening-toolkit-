"""
Run Deep Learning Pansharpening
===============================

This script trains and evaluates deep learning pansharpening models.

Available Models:
- pnn: Simple 3-layer CNN (baseline)
- pannet: ResNet style with high-pass filter
- drpnn: Deep Residual PanNet
- pannet_cbam: PanNet with CBAM attention
- mspannet: Multi-scale feature pyramid
- panformer: Transformer-based with cross-attention
- panformer_lite: Lightweight transformer with window attention

Available Losses:
- combined: L1 + MSE + Gradient (default)
- advanced: L1 + MSE + Gradient + SSIM + SAM
- spectral_focus: Higher SAM weight for spectral fidelity
- spatial_focus: Higher gradient and SSIM for spatial quality

Usage:
    python run_deep_learning.py --pan path/to/pan.tif --ms path/to/ms.tif --model pannet_cbam
    python run_deep_learning.py --model panformer_lite --loss spectral_focus --epochs 100
"""

import argparse
from pathlib import Path
import sys

# Add project to path - handle both script and Jupyter notebook execution
try:
    PROJECT_DIR = Path(__file__).parent
except NameError:
    # Running in Jupyter notebook
    PROJECT_DIR = Path(r"D:\Udemy_Cour\Pancharping\pansharpening_project")

sys.path.insert(0, str(PROJECT_DIR))

from utils import (
    load_image, normalize_image, upsample_image, save_geotiff,
    create_patches, create_wald_patches, compute_all_metrics, print_metrics,
    plot_comparison, plot_training_history
)
from methods.classic import sfim_fusion
from methods.deep_learning import PansharpeningTrainer
from models import create_model, create_loss, AVAILABLE_MODELS
from configs.config import (
    DEFAULT_PAN_PATH, DEFAULT_MS_PATH, RESULTS_DIR, CHECKPOINT_DIR,
    TRAINING_CONFIG, LOSS_CONFIG, AVAILABLE_LOSSES, MODEL_CONFIGS, LOSS_CONFIGS
)


def run_deep_learning(
    pan_path: str,
    ms_path: str,
    model_type: str = 'pannet',
    loss_type: str = 'combined',
    output_dir: str = None,
    checkpoint_dir: str = None,
    epochs: int = None
):
    """Train and evaluate deep learning pansharpening."""

    output_dir = Path(output_dir or RESULTS_DIR / 'deep_learning')
    checkpoint_dir = Path(checkpoint_dir or CHECKPOINT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    epochs = epochs or TRAINING_CONFIG['epochs']

    # Load images
    print("Loading images...")
    pan_data, pan_meta = load_image(pan_path)
    ms_data, ms_meta = load_image(ms_path)

    print(f"PAN shape: {pan_data.shape}")
    print(f"MS shape: {ms_data.shape}")

    # Normalize
    print("Normalizing...")
    pan_norm, pan_params = normalize_image(pan_data)
    ms_norm, ms_params = normalize_image(ms_data)

    # Upsample MS
    print("Upsampling MS to PAN resolution...")
    target_size = (pan_meta['height'], pan_meta['width'])
    ms_up = upsample_image(ms_norm, target_size, method='bicubic')

    # Generate SFIM reference for visualization and metrics (always needed)
    print("Generating SFIM reference for comparison...")
    target = sfim_fusion(ms_up, pan_norm)

    # Create training patches
    # Use SFIM as target to provide a sharp reference for the network to learn from
    # This is more effective than Wald's protocol when we don't have true HR reference
    print("\nCreating training patches with SFIM target...")
    print("  - Using SFIM fusion as ground truth target")
    print("  - This provides sharp spatial details for the network to learn")
    ms_patches, pan_patches, target_patches = create_patches(
        ms_up, pan_norm, target,
        patch_size=TRAINING_CONFIG['patch_size'],
        stride=TRAINING_CONFIG['stride']
    )
    print(f"Created {len(ms_patches)} training patches")

    # Create model using factory function
    ms_bands = ms_data.shape[0]
    print(f"\n{'='*60}")
    print(f"Creating {model_type.upper()} model")
    print(f"{'='*60}")

    # Get model-specific config if available
    model_kwargs = MODEL_CONFIGS.get(model_type, {})
    model = create_model(model_type, ms_bands=ms_bands, **model_kwargs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_type}")
    print(f"Parameters: {n_params:,}")

    # Create loss using factory function
    print(f"\nLoss function: {loss_type}")
    if loss_type in LOSS_CONFIGS:
        loss_kwargs = LOSS_CONFIGS[loss_type]
        if loss_type == 'combined':
            from models import CombinedLoss
            criterion = CombinedLoss(**loss_kwargs)
        else:
            from models import AdvancedCombinedLoss
            criterion = AdvancedCombinedLoss(**loss_kwargs)
    else:
        criterion = create_loss(loss_type)

    # Create trainer
    trainer = PansharpeningTrainer(
        model=model,
        criterion=criterion,
        lr=TRAINING_CONFIG['learning_rate'],
        checkpoint_dir=str(checkpoint_dir)
    )

    # Create dataloaders
    train_loader, val_loader = trainer.create_dataloaders(
        ms_patches, pan_patches, target_patches,
        batch_size=TRAINING_CONFIG['batch_size'],
        val_split=TRAINING_CONFIG['val_split']
    )

    # Train
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()}")
    print(f"{'='*60}")

    trainer.train(train_loader, val_loader, epochs=epochs)

    # Load best model and run inference
    print("\n" + "="*60)
    print("Running Inference")
    print("="*60)

    trainer.load_checkpoint(str(checkpoint_dir / 'best.pth'))
    fused_dl = trainer.predict(ms_up, pan_norm)

    # Compute metrics
    print("\nQuality Metrics:")
    metrics_dl = compute_all_metrics(ms_up, fused_dl)
    print_metrics(metrics_dl, f"{model_type.upper()} (Deep Learning)")

    metrics_sfim = compute_all_metrics(ms_up, target)
    print_metrics(metrics_sfim, "SFIM (Reference)")

    # Save results
    save_geotiff(fused_dl, str(output_dir / f'fused_{model_type}.tif'), pan_meta)

    # Visualization
    print("\nGenerating visualization...")
    images = {
        'MS (Upsampled)': ms_up,
        'PAN': pan_norm,
        'SFIM (Reference)': target,
        f'{model_type.upper()} (DL)': fused_dl
    }
    plot_comparison(images, str(output_dir / f'{model_type}_comparison.png'))

    # Plot training history
    plot_training_history(trainer.history, str(output_dir / f'{model_type}_training.png'))

    print(f"\nResults saved to: {output_dir}")

    return fused_dl, metrics_dl


def is_jupyter():
    """Check if running in a Jupyter notebook."""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Run deep learning pansharpening',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default PanNet model
    python run_deep_learning.py --model pannet

    # Run with CBAM attention model
    python run_deep_learning.py --model pannet_cbam

    # Run transformer with spectral focus loss
    python run_deep_learning.py --model panformer_lite --loss spectral_focus

    # Full example with all options
    python run_deep_learning.py --pan data/pan.tif --ms data/ms.tif --model mspannet --loss advanced --epochs 100
        """
    )
    parser.add_argument('--pan', type=str, default=DEFAULT_PAN_PATH,
                        help='Path to PAN image')
    parser.add_argument('--ms', type=str, default=DEFAULT_MS_PATH,
                        help='Path to MS image')
    parser.add_argument('--model', type=str, default='pannet',
                        choices=AVAILABLE_MODELS,
                        help='Model architecture (default: pannet)')
    parser.add_argument('--loss', type=str, default='combined',
                        choices=AVAILABLE_LOSSES,
                        help='Loss function (default: combined)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')

    # Handle Jupyter notebook environment
    if is_jupyter():
        # Use defaults when running in Jupyter
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    run_deep_learning(
        args.pan, args.ms, args.model,
        loss_type=args.loss,
        output_dir=args.output,
        epochs=args.epochs
    )


if __name__ == '__main__':
    main()
