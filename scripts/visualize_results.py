#!/usr/bin/env python3
"""
Modern Pansharpening Visualization Tool
=======================================
A beautiful, interactive visualization for comparing pansharpening results.

Features:
- Side-by-side comparison (MS, PAN, Fused)
- Interactive before/after slider
- Zoom and pan capabilities
- Quality metrics display
- Multiple visualization modes
- Export to high-resolution images

Usage:
    python scripts/visualize_results.py
    python scripts/visualize_results.py --interactive
    python scripts/visualize_results.py --mode slider
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# Add project root to path
try:
    PROJECT_ROOT = Path(__file__).parent.parent
except NameError:
    # Running in Jupyter notebook or interactive environment
    PROJECT_ROOT = Path.cwd()
    if PROJECT_ROOT.name == 'notebooks':
        PROJECT_ROOT = PROJECT_ROOT.parent
    elif PROJECT_ROOT.name != 'pansharpening_project':
        # Try to find the project root
        for parent in [PROJECT_ROOT] + list(PROJECT_ROOT.parents):
            if (parent / 'data').exists() and (parent / 'models').exists():
                PROJECT_ROOT = parent
                break

sys.path.insert(0, str(PROJECT_ROOT))
print(f"Project root: {PROJECT_ROOT}")

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    from PIL import Image

# Modern dark theme colors
COLORS = {
    'bg': '#1a1a2e',
    'panel': '#16213e',
    'accent': '#0f3460',
    'highlight': '#e94560',
    'text': '#eaeaea',
    'text_dim': '#a0a0a0',
    'success': '#00d9ff',
    'warning': '#ffd700',
}


def load_image(path):
    """Load image from file (supports GeoTIFF and common formats)."""
    path = Path(path)

    if HAS_RASTERIO and path.suffix.lower() in ['.tif', '.tiff']:
        with rasterio.open(path) as src:
            img = src.read()
            if img.ndim == 3:
                img = np.transpose(img, (1, 2, 0))
            return img.astype(np.float32)
    else:
        img = np.array(Image.open(path))
        return img.astype(np.float32)


def normalize_for_display(img, percentile=(2, 98)):
    """Normalize image for display using percentile stretching."""
    img = img.copy()

    if img.ndim == 2:
        p_low, p_high = np.percentile(img, percentile)
        img = np.clip((img - p_low) / (p_high - p_low + 1e-8), 0, 1)
    else:
        for i in range(img.shape[-1]):
            band = img[..., i]
            p_low, p_high = np.percentile(band, percentile)
            img[..., i] = np.clip((band - p_low) / (p_high - p_low + 1e-8), 0, 1)

    return img


def create_rgb_composite(ms_img, bands=(2, 1, 0)):
    """Create RGB composite from multispectral image."""
    if ms_img.ndim == 2:
        return np.stack([ms_img] * 3, axis=-1)

    if ms_img.shape[-1] >= 3:
        rgb = ms_img[..., list(bands)]
    else:
        rgb = np.stack([ms_img[..., 0]] * 3, axis=-1)

    return normalize_for_display(rgb)


def calculate_metrics(fused, reference):
    """Calculate quality metrics between fused and reference images."""
    metrics = {}

    # Ensure same shape
    if fused.shape != reference.shape:
        return {'error': 'Shape mismatch'}

    # Normalize
    fused_norm = fused.astype(np.float64)
    ref_norm = reference.astype(np.float64)

    # PSNR
    mse = np.mean((fused_norm - ref_norm) ** 2)
    if mse > 0:
        max_val = max(fused_norm.max(), ref_norm.max())
        metrics['PSNR'] = 10 * np.log10(max_val ** 2 / mse)
    else:
        metrics['PSNR'] = float('inf')

    # SSIM (simplified)
    def ssim_single(a, b):
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        mu_a = np.mean(a)
        mu_b = np.mean(b)
        sigma_a = np.std(a)
        sigma_b = np.std(b)
        sigma_ab = np.mean((a - mu_a) * (b - mu_b))

        ssim = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / \
               ((mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a ** 2 + sigma_b ** 2 + c2))
        return ssim

    if fused_norm.ndim == 2:
        metrics['SSIM'] = ssim_single(fused_norm, ref_norm)
    else:
        ssim_vals = [ssim_single(fused_norm[..., i], ref_norm[..., i])
                     for i in range(fused_norm.shape[-1])]
        metrics['SSIM'] = np.mean(ssim_vals)

    # SAM (Spectral Angle Mapper)
    if fused_norm.ndim == 3:
        fused_flat = fused_norm.reshape(-1, fused_norm.shape[-1])
        ref_flat = ref_norm.reshape(-1, ref_norm.shape[-1])

        dot_product = np.sum(fused_flat * ref_flat, axis=1)
        norm_fused = np.linalg.norm(fused_flat, axis=1)
        norm_ref = np.linalg.norm(ref_flat, axis=1)

        cos_angle = dot_product / (norm_fused * norm_ref + 1e-8)
        cos_angle = np.clip(cos_angle, -1, 1)
        sam = np.mean(np.arccos(cos_angle)) * 180 / np.pi
        metrics['SAM'] = sam

    return metrics


def apply_modern_style():
    """Apply modern dark theme to matplotlib."""
    plt.style.use('dark_background')

    plt.rcParams.update({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor': COLORS['panel'],
        'axes.edgecolor': COLORS['accent'],
        'axes.labelcolor': COLORS['text'],
        'axes.titlecolor': COLORS['text'],
        'xtick.color': COLORS['text_dim'],
        'ytick.color': COLORS['text_dim'],
        'text.color': COLORS['text'],
        'grid.color': COLORS['accent'],
        'grid.alpha': 0.3,
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 11,
        'figure.titlesize': 16,
    })


class PansharpeningVisualizer:
    """Modern interactive visualizer for pansharpening results."""

    def __init__(self, ms_path, pan_path, fused_path=None):
        self.ms_path = Path(ms_path)
        self.pan_path = Path(pan_path)
        self.fused_path = Path(fused_path) if fused_path else None

        # Load images
        print("Loading images...")
        self.ms_raw = load_image(ms_path)
        self.pan_raw = load_image(pan_path)

        if self.fused_path and self.fused_path.exists():
            self.fused_raw = load_image(fused_path)
        else:
            # Generate simple fused result for demo
            self.fused_raw = self._simple_fusion()

        # Prepare display images
        self.ms_display = create_rgb_composite(self.ms_raw)
        self.pan_display = normalize_for_display(self.pan_raw)
        self.fused_display = create_rgb_composite(self.fused_raw)

        # Resize MS for comparison if needed
        if self.ms_display.shape[:2] != self.pan_display.shape[:2]:
            from scipy.ndimage import zoom
            scale_h = self.pan_display.shape[0] / self.ms_display.shape[0]
            scale_w = self.pan_display.shape[1] / self.ms_display.shape[1]
            self.ms_display_upscaled = zoom(self.ms_display, (scale_h, scale_w, 1), order=1)
        else:
            self.ms_display_upscaled = self.ms_display

        print("Images loaded successfully!")
        self._print_info()

    def _simple_fusion(self):
        """Create a simple fusion for demo purposes."""
        from scipy.ndimage import zoom

        # Upsample MS to PAN resolution
        scale_h = self.pan_raw.shape[0] / self.ms_raw.shape[0]
        scale_w = self.pan_raw.shape[1] / self.ms_raw.shape[1]

        if self.ms_raw.ndim == 2:
            ms_up = zoom(self.ms_raw, (scale_h, scale_w), order=1)
            return ms_up
        else:
            ms_up = zoom(self.ms_raw, (scale_h, scale_w, 1), order=1)

        # Simple Brovey transform
        pan = self.pan_raw if self.pan_raw.ndim == 2 else self.pan_raw[..., 0]
        intensity = np.mean(ms_up, axis=-1)
        ratio = pan / (intensity + 1e-8)

        fused = ms_up * ratio[..., np.newaxis]
        return fused

    def _print_info(self):
        """Print image information."""
        print(f"\n{'='*50}")
        print("IMAGE INFORMATION")
        print(f"{'='*50}")
        print(f"MS Image:    {self.ms_raw.shape} - {self.ms_path.name}")
        print(f"PAN Image:   {self.pan_raw.shape} - {self.pan_path.name}")
        print(f"Fused Image: {self.fused_raw.shape}")
        print(f"{'='*50}\n")

    def show_comparison(self):
        """Display side-by-side comparison with metrics."""
        apply_modern_style()

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('🛰️ Pansharpening Results Comparison', fontsize=18, fontweight='bold',
                     color=COLORS['success'])

        gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 0.05, 0.15],
                      hspace=0.3, wspace=0.2)

        # Images
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        # Display images
        ax1.imshow(self.ms_display_upscaled)
        ax1.set_title('📊 Multispectral (MS)', fontsize=12, pad=10)
        ax1.axis('off')

        ax2.imshow(self.pan_display, cmap='gray')
        ax2.set_title('📷 Panchromatic (PAN)', fontsize=12, pad=10)
        ax2.axis('off')

        ax3.imshow(self.fused_display)
        ax3.set_title('✨ Pansharpened Result', fontsize=12, pad=10,
                      color=COLORS['success'])
        ax3.axis('off')

        # Add labels
        for ax, label in zip([ax1, ax2, ax3], ['Low Resolution\nHigh Spectral',
                                                 'High Resolution\nSingle Band',
                                                 'High Resolution\nHigh Spectral']):
            ax.text(0.5, -0.05, label, transform=ax.transAxes,
                    ha='center', va='top', fontsize=9, color=COLORS['text_dim'])

        # Metrics panel
        ax_metrics = fig.add_subplot(gs[2, :])
        ax_metrics.axis('off')

        # Calculate and display metrics
        metrics = calculate_metrics(self.fused_display, self.ms_display_upscaled)

        metrics_text = "  |  ".join([
            f"📈 PSNR: {metrics.get('PSNR', 0):.2f} dB",
            f"🎯 SSIM: {metrics.get('SSIM', 0):.4f}",
            f"🌈 SAM: {metrics.get('SAM', 0):.2f}°" if 'SAM' in metrics else ""
        ])

        ax_metrics.text(0.5, 0.5, metrics_text, transform=ax_metrics.transAxes,
                        ha='center', va='center', fontsize=14,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['accent'],
                                  edgecolor=COLORS['success'], linewidth=2))

        # Add info box
        info_text = f"MS: {self.ms_raw.shape}  →  Fused: {self.fused_raw.shape}"
        fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, color=COLORS['text_dim'])

        plt.tight_layout()
        plt.savefig(PROJECT_ROOT / 'results' / 'comparison_modern.png',
                    dpi=150, facecolor=COLORS['bg'], edgecolor='none',
                    bbox_inches='tight')
        print(f"Saved comparison to: results/comparison_modern.png")
        plt.show()

    def show_slider_comparison(self):
        """Interactive before/after slider comparison."""
        apply_modern_style()

        fig, ax = plt.subplots(figsize=(12, 10))
        fig.suptitle('🔄 Before / After Comparison (Drag Slider)',
                     fontsize=16, fontweight='bold', color=COLORS['success'])

        # Ensure same size
        before = self.ms_display_upscaled
        after = self.fused_display

        # Initial display
        combined = after.copy()
        split_pos = 0.5
        split_x = int(combined.shape[1] * split_pos)
        combined[:, :split_x] = before[:, :split_x]

        img_display = ax.imshow(combined)
        line = ax.axvline(x=split_x, color=COLORS['highlight'], linewidth=3)
        ax.axis('off')

        # Labels
        ax.text(0.15, 0.95, '← BEFORE (MS)', transform=ax.transAxes,
                fontsize=12, color=COLORS['warning'], fontweight='bold',
                ha='center', va='top')
        ax.text(0.85, 0.95, 'AFTER (Fused) →', transform=ax.transAxes,
                fontsize=12, color=COLORS['success'], fontweight='bold',
                ha='center', va='top')

        # Slider
        ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03], facecolor=COLORS['accent'])
        slider = Slider(ax_slider, 'Position', 0.0, 1.0, valinit=0.5,
                        color=COLORS['highlight'])

        def update(val):
            split_pos = slider.val
            split_x = int(combined.shape[1] * split_pos)
            new_combined = after.copy()
            new_combined[:, :split_x] = before[:, :split_x]
            img_display.set_data(new_combined)
            line.set_xdata([split_x, split_x])
            fig.canvas.draw_idle()

        slider.on_changed(update)

        plt.tight_layout()
        plt.show()

    def show_detailed_analysis(self):
        """Show detailed analysis with histograms and profiles."""
        apply_modern_style()

        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('📊 Detailed Pansharpening Analysis', fontsize=18,
                     fontweight='bold', color=COLORS['success'])

        gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

        # Row 1: Images
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[0, 3])

        ax1.imshow(self.ms_display_upscaled)
        ax1.set_title('MS (Upscaled)', fontsize=11)
        ax1.axis('off')

        ax2.imshow(self.pan_display, cmap='gray')
        ax2.set_title('PAN', fontsize=11)
        ax2.axis('off')

        ax3.imshow(self.fused_display)
        ax3.set_title('Fused', fontsize=11, color=COLORS['success'])
        ax3.axis('off')

        # Difference map
        diff = np.abs(self.fused_display - self.ms_display_upscaled)
        diff_display = np.mean(diff, axis=-1) if diff.ndim == 3 else diff
        im4 = ax4.imshow(diff_display, cmap='hot')
        ax4.set_title('Difference Map', fontsize=11)
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        # Row 2: Histograms
        ax5 = fig.add_subplot(gs[1, :2])
        ax6 = fig.add_subplot(gs[1, 2:])

        # MS histogram
        if self.ms_display_upscaled.ndim == 3:
            colors = ['red', 'green', 'blue']
            for i, c in enumerate(colors):
                ax5.hist(self.ms_display_upscaled[..., i].ravel(), bins=100,
                         alpha=0.5, color=c, label=f'Band {i+1}')
        ax5.set_title('MS Histogram', fontsize=11)
        ax5.set_xlabel('Pixel Value')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Fused histogram
        if self.fused_display.ndim == 3:
            colors = ['red', 'green', 'blue']
            for i, c in enumerate(colors):
                ax6.hist(self.fused_display[..., i].ravel(), bins=100,
                         alpha=0.5, color=c, label=f'Band {i+1}')
        ax6.set_title('Fused Histogram', fontsize=11, color=COLORS['success'])
        ax6.set_xlabel('Pixel Value')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # Row 3: Profiles and metrics
        ax7 = fig.add_subplot(gs[2, :2])
        ax8 = fig.add_subplot(gs[2, 2:])

        # Horizontal profile
        mid_row = self.fused_display.shape[0] // 2
        if self.fused_display.ndim == 3:
            profile_ms = np.mean(self.ms_display_upscaled[mid_row, :, :], axis=-1)
            profile_fused = np.mean(self.fused_display[mid_row, :, :], axis=-1)
        else:
            profile_ms = self.ms_display_upscaled[mid_row, :]
            profile_fused = self.fused_display[mid_row, :]

        ax7.plot(profile_ms, label='MS', alpha=0.7, linewidth=2, color=COLORS['warning'])
        ax7.plot(profile_fused, label='Fused', alpha=0.7, linewidth=2, color=COLORS['success'])
        ax7.set_title(f'Horizontal Profile (Row {mid_row})', fontsize=11)
        ax7.set_xlabel('Column')
        ax7.set_ylabel('Mean Intensity')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # Metrics summary
        ax8.axis('off')
        metrics = calculate_metrics(self.fused_display, self.ms_display_upscaled)

        metrics_lines = [
            "╔══════════════════════════════════╗",
            "║     QUALITY METRICS SUMMARY      ║",
            "╠══════════════════════════════════╣",
            f"║  📈 PSNR:  {metrics.get('PSNR', 0):>8.2f} dB          ║",
            f"║  🎯 SSIM:  {metrics.get('SSIM', 0):>8.4f}             ║",
            f"║  🌈 SAM:   {metrics.get('SAM', 0):>8.2f}°             ║" if 'SAM' in metrics else "",
            "╠══════════════════════════════════╣",
            f"║  Input:  {str(self.ms_raw.shape):>20}   ║",
            f"║  Output: {str(self.fused_raw.shape):>20}   ║",
            "╚══════════════════════════════════╝",
        ]

        ax8.text(0.5, 0.5, '\n'.join(metrics_lines), transform=ax8.transAxes,
                 ha='center', va='center', fontsize=11, family='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['panel'],
                           edgecolor=COLORS['success'], linewidth=2))

        plt.tight_layout()
        plt.savefig(PROJECT_ROOT / 'results' / 'detailed_analysis.png',
                    dpi=150, facecolor=COLORS['bg'], edgecolor='none',
                    bbox_inches='tight')
        print(f"Saved detailed analysis to: results/detailed_analysis.png")
        plt.show()

    def show_zoom_comparison(self, zoom_factor=2):
        """Show zoomed comparison of a region."""
        apply_modern_style()

        fig = plt.figure(figsize=(16, 8))
        fig.suptitle('🔍 Zoom Comparison', fontsize=16, fontweight='bold',
                     color=COLORS['success'])

        # Get center region
        h, w = self.fused_display.shape[:2]
        crop_h, crop_w = h // (zoom_factor * 2), w // (zoom_factor * 2)
        y1, y2 = h // 2 - crop_h, h // 2 + crop_h
        x1, x2 = w // 2 - crop_w, w // 2 + crop_w

        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2)

        # Full images (top row)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        # Zoomed regions (bottom row)
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])

        # Full images with rectangle
        for ax, img, title in [
            (ax1, self.ms_display_upscaled, 'MS'),
            (ax2, self.pan_display, 'PAN'),
            (ax3, self.fused_display, 'Fused')
        ]:
            if img.ndim == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax.set_title(title, fontsize=11)
            ax.axis('off')

            # Draw rectangle
            rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1,
                                       linewidth=2, edgecolor=COLORS['highlight'],
                                       facecolor='none')
            ax.add_patch(rect)

        # Zoomed regions
        for ax, img, title in [
            (ax4, self.ms_display_upscaled[y1:y2, x1:x2], 'MS (Zoomed)'),
            (ax5, self.pan_display[y1:y2, x1:x2], 'PAN (Zoomed)'),
            (ax6, self.fused_display[y1:y2, x1:x2], 'Fused (Zoomed)')
        ]:
            if img.ndim == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax.set_title(title, fontsize=11, color=COLORS['success'] if 'Fused' in title else COLORS['text'])
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(PROJECT_ROOT / 'results' / 'zoom_comparison.png',
                    dpi=150, facecolor=COLORS['bg'], edgecolor='none',
                    bbox_inches='tight')
        print(f"Saved zoom comparison to: results/zoom_comparison.png")
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Modern Pansharpening Visualization Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/visualize_results.py                    # Basic comparison
  python scripts/visualize_results.py --mode slider      # Interactive slider
  python scripts/visualize_results.py --mode detailed    # Detailed analysis
  python scripts/visualize_results.py --mode zoom        # Zoom comparison
  python scripts/visualize_results.py --mode all         # All visualizations
        """
    )

    parser.add_argument('--ms', type=str, default='data/ms.tif',
                        help='Path to multispectral image')
    parser.add_argument('--pan', type=str, default='data/pan.tif',
                        help='Path to panchromatic image')
    parser.add_argument('--fused', type=str, default=None,
                        help='Path to fused image (optional)')
    parser.add_argument('--mode', type=str, default='comparison',
                        choices=['comparison', 'slider', 'detailed', 'zoom', 'all'],
                        help='Visualization mode')

    args = parser.parse_args()

    # Resolve paths
    ms_path = PROJECT_ROOT / args.ms
    pan_path = PROJECT_ROOT / args.pan
    fused_path = PROJECT_ROOT / args.fused if args.fused else None

    # Check for existing fused results
    if fused_path is None:
        possible_fused = [
            PROJECT_ROOT / 'results' / 'deep_learning' / 'fused_pannet.tif',
            PROJECT_ROOT / 'results' / 'classic' / 'fused_sfim.tif',
            PROJECT_ROOT / 'results' / 'classic' / 'fused_brovey.tif',
        ]
        for p in possible_fused:
            if p.exists():
                fused_path = p
                print(f"Found fused image: {p}")
                break

    print("\n" + "="*60)
    print("   🛰️  PANSHARPENING VISUALIZATION TOOL  🛰️")
    print("="*60 + "\n")

    # Create visualizer
    viz = PansharpeningVisualizer(ms_path, pan_path, fused_path)

    # Run selected visualization
    if args.mode == 'comparison' or args.mode == 'all':
        print("\n📊 Generating comparison view...")
        viz.show_comparison()

    if args.mode == 'slider' or args.mode == 'all':
        print("\n🔄 Opening slider comparison...")
        viz.show_slider_comparison()

    if args.mode == 'detailed' or args.mode == 'all':
        print("\n📈 Generating detailed analysis...")
        viz.show_detailed_analysis()

    if args.mode == 'zoom' or args.mode == 'all':
        print("\n🔍 Generating zoom comparison...")
        viz.show_zoom_comparison()

    print("\n✅ Visualization complete!")


if __name__ == '__main__':
    main()
