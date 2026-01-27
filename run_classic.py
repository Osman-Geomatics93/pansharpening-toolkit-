"""
Run Classic Pansharpening Methods
================================

This script applies all classic pansharpening methods to your images.

Usage:
    python run_classic.py --pan path/to/pan.tif --ms path/to/ms.tif
"""

import argparse
from pathlib import Path
import sys

# Add project to path - handle both script and Jupyter notebook
try:
    PROJECT_DIR = Path(__file__).parent
except NameError:
    PROJECT_DIR = Path(r"D:\Udemy_Cour\Pancharping\pansharpening_project")

sys.path.insert(0, str(PROJECT_DIR))

from utils import (
    load_image, normalize_image, upsample_image, save_geotiff,
    compute_all_metrics, print_metrics, plot_comparison
)
from methods.classic import (
    brovey_fusion, ihs_fusion, sfim_fusion,
    gram_schmidt_fusion, hpf_fusion
)
from configs.config import DEFAULT_PAN_PATH, DEFAULT_MS_PATH, RESULTS_DIR


def run_classic_methods(pan_path: str, ms_path: str, output_dir: str):
    """Run all classic pansharpening methods."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Upsample MS to PAN resolution
    print("Upsampling MS to PAN resolution...")
    target_size = (pan_meta['height'], pan_meta['width'])
    ms_up = upsample_image(ms_norm, target_size, method='bicubic')
    print(f"Upsampled MS shape: {ms_up.shape}")

    # Apply all methods
    print("\n" + "="*60)
    print("Applying Classic Pansharpening Methods")
    print("="*60)

    methods = {
        'Brovey': brovey_fusion,
        'IHS': ihs_fusion,
        'SFIM': sfim_fusion,
        'Gram-Schmidt': gram_schmidt_fusion,
        'HPF': hpf_fusion
    }

    results = {}
    all_metrics = {}

    for name, method in methods.items():
        print(f"\n{name}...")
        fused = method(ms_up, pan_norm)
        results[name] = fused

        # Compute metrics
        metrics = compute_all_metrics(ms_up, fused)
        all_metrics[name] = metrics
        print_metrics(metrics, name)

        # Save result
        output_path = output_dir / f"fused_{name.lower().replace('-', '_')}.tif"
        save_geotiff(fused, str(output_path), pan_meta)

    # Visualization
    print("\n" + "="*60)
    print("Generating Visualization")
    print("="*60)

    images_to_plot = {'MS (Upsampled)': ms_up, 'PAN': pan_norm}
    images_to_plot.update(results)

    plot_comparison(
        images_to_plot,
        output_path=str(output_dir / 'classic_comparison.png')
    )

    # Summary
    print("\n" + "="*60)
    print("Summary - Best Methods:")
    print("="*60)

    best_psnr = max(all_metrics.items(), key=lambda x: x[1]['PSNR'])
    best_ssim = max(all_metrics.items(), key=lambda x: x[1]['SSIM'])
    best_sam = min(all_metrics.items(), key=lambda x: x[1]['SAM'])

    print(f"Best PSNR: {best_psnr[0]} ({best_psnr[1]['PSNR']:.2f} dB)")
    print(f"Best SSIM: {best_ssim[0]} ({best_ssim[1]['SSIM']:.4f})")
    print(f"Best SAM:  {best_sam[0]} ({best_sam[1]['SAM']:.4f}°)")

    print(f"\nResults saved to: {output_dir}")

    return results, all_metrics


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
    parser = argparse.ArgumentParser(description='Run classic pansharpening methods')
    parser.add_argument('--pan', type=str, default=DEFAULT_PAN_PATH,
                        help='Path to PAN image')
    parser.add_argument('--ms', type=str, default=DEFAULT_MS_PATH,
                        help='Path to MS image')
    parser.add_argument('--output', type=str, default=str(RESULTS_DIR / 'classic'),
                        help='Output directory')

    # Handle Jupyter notebook environment
    if is_jupyter():
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    run_classic_methods(args.pan, args.ms, args.output)


if __name__ == '__main__':
    main()
