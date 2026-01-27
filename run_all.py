"""
Run All Pansharpening Methods
=============================

This script runs both classic and deep learning methods for comparison.

Usage:
    python run_all.py --pan path/to/pan.tif --ms path/to/ms.tif
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

from run_classic import run_classic_methods
from run_deep_learning import run_deep_learning
from utils import plot_metrics_comparison
from configs.config import DEFAULT_PAN_PATH, DEFAULT_MS_PATH, RESULTS_DIR


def run_all(pan_path: str, ms_path: str, output_dir: str):
    """Run all pansharpening methods."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("PANSHARPENING - COMPLETE COMPARISON")
    print("="*70)

    # Run classic methods
    print("\n" + "="*70)
    print("PART 1: CLASSIC METHODS")
    print("="*70)
    classic_results, classic_metrics = run_classic_methods(
        pan_path, ms_path, str(output_dir / 'classic')
    )

    # Run deep learning methods
    print("\n" + "="*70)
    print("PART 2: DEEP LEARNING METHODS")
    print("="*70)

    dl_metrics = {}
    for model in ['pnn', 'pannet', 'drpnn']:
        print(f"\n--- {model.upper()} ---")
        try:
            _, metrics = run_deep_learning(
                pan_path, ms_path, model,
                output_dir=str(output_dir / 'deep_learning' / model),
                epochs=200  # Reduced for full comparison
            )
            dl_metrics[model.upper()] = metrics
        except Exception as e:
            print(f"Error with {model}: {e}")

    # Combined comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)

    all_metrics = {**classic_metrics, **dl_metrics}

    # Print table
    print(f"\n{'Method':<15} {'PSNR':>10} {'SSIM':>10} {'SAM':>10} {'ERGAS':>10}")
    print("-" * 60)
    for method, metrics in all_metrics.items():
        print(f"{method:<15} {metrics['PSNR']:>10.2f} {metrics['SSIM']:>10.4f} "
              f"{metrics['SAM']:>10.4f} {metrics['ERGAS']:>10.4f}")

    # Plot comparison
    plot_metrics_comparison(all_metrics, str(output_dir / 'all_methods_comparison.png'))

    # Find best methods
    best_psnr = max(all_metrics.items(), key=lambda x: x[1]['PSNR'])
    best_ssim = max(all_metrics.items(), key=lambda x: x[1]['SSIM'])
    best_sam = min(all_metrics.items(), key=lambda x: x[1]['SAM'])

    print("\n" + "="*70)
    print("BEST METHODS:")
    print("="*70)
    print(f"Best PSNR:  {best_psnr[0]} ({best_psnr[1]['PSNR']:.2f} dB)")
    print(f"Best SSIM:  {best_ssim[0]} ({best_ssim[1]['SSIM']:.4f})")
    print(f"Best SAM:   {best_sam[0]} ({best_sam[1]['SAM']:.4f}°)")

    print(f"\nAll results saved to: {output_dir}")


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
    parser = argparse.ArgumentParser(description='Run all pansharpening methods')
    parser.add_argument('--pan', type=str, default=DEFAULT_PAN_PATH,
                        help='Path to PAN image')
    parser.add_argument('--ms', type=str, default=DEFAULT_MS_PATH,
                        help='Path to MS image')
    parser.add_argument('--output', type=str, default=str(RESULTS_DIR),
                        help='Output directory')

    # Handle Jupyter notebook environment
    if is_jupyter():
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    run_all(args.pan, args.ms, args.output)


if __name__ == '__main__':
    main()
