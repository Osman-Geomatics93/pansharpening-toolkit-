"""
Visualization Utilities for Pansharpening
- Image display
- Comparison plots
- Training curves
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
from pathlib import Path


def to_rgb(image: np.ndarray, bands: Tuple[int, int, int] = (2, 1, 0),
           percentile_stretch: bool = True) -> np.ndarray:
    """
    Convert multi-band image to RGB for display.

    Args:
        image: Image array (bands, H, W)
        bands: Band indices for RGB (R, G, B)
        percentile_stretch: Apply percentile stretch for better contrast

    Returns:
        RGB image (H, W, 3)
    """
    if image.shape[0] >= 3:
        rgb = np.stack([image[bands[0]], image[bands[1]], image[bands[2]]], axis=-1)
    else:
        rgb = np.stack([image[0]] * 3, axis=-1)

    if percentile_stretch:
        p2, p98 = np.percentile(rgb, (2, 98))
        rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-10), 0, 1)
    else:
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-10)

    return rgb


def plot_comparison(images: Dict[str, np.ndarray],
                    output_path: Optional[str] = None,
                    rgb_bands: Tuple[int, int, int] = (2, 1, 0),
                    figsize: Tuple[int, int] = (20, 10)):
    """
    Plot comparison of multiple images.

    Args:
        images: Dictionary of {name: image_array}
        output_path: Path to save figure
        rgb_bands: Band indices for RGB display
        figsize: Figure size
    """
    n_images = len(images)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes = np.array(axes).flatten()

    for idx, (name, img) in enumerate(images.items()):
        if img.shape[0] == 1:  # Grayscale
            display = (img[0] - img[0].min()) / (img[0].max() - img[0].min() + 1e-10)
            axes[idx].imshow(display, cmap='gray')
        else:
            axes[idx].imshow(to_rgb(img, rgb_bands))

        axes[idx].set_title(name)
        axes[idx].axis('off')

    # Hide unused axes
    for idx in range(len(images), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()


def plot_detailed_comparison(ms: np.ndarray, pan: np.ndarray,
                             fused: np.ndarray, method_name: str,
                             output_path: Optional[str] = None,
                             rgb_bands: Tuple[int, int, int] = (2, 1, 0)):
    """
    Plot detailed comparison with zoomed regions.

    Args:
        ms: MS image (upsampled)
        pan: PAN image
        fused: Fused result
        method_name: Name of fusion method
        output_path: Path to save figure
        rgb_bands: Band indices for RGB display
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: Full images
    axes[0, 0].imshow(to_rgb(ms, rgb_bands))
    axes[0, 0].set_title('MS (Upsampled)')
    axes[0, 0].axis('off')

    pan_disp = (pan[0] - pan[0].min()) / (pan[0].max() - pan[0].min() + 1e-10)
    axes[0, 1].imshow(pan_disp, cmap='gray')
    axes[0, 1].set_title('PAN')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(to_rgb(fused, rgb_bands))
    axes[0, 2].set_title(f'Fused ({method_name})')
    axes[0, 2].axis('off')

    # Difference map
    diff = np.abs(fused - ms).mean(axis=0)
    im = axes[0, 3].imshow(diff, cmap='hot')
    axes[0, 3].set_title('Difference')
    axes[0, 3].axis('off')
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

    # Row 2: Zoomed details
    h, w = pan.shape[1], pan.shape[2]
    crop_size = min(200, h // 3, w // 3)
    y_start = h // 2 - crop_size // 2
    x_start = w // 2 - crop_size // 2
    y_end, x_end = y_start + crop_size, x_start + crop_size

    axes[1, 0].imshow(to_rgb(ms[:, y_start:y_end, x_start:x_end], rgb_bands))
    axes[1, 0].set_title('MS Detail')
    axes[1, 0].axis('off')

    pan_crop = pan[0, y_start:y_end, x_start:x_end]
    pan_crop_disp = (pan_crop - pan_crop.min()) / (pan_crop.max() - pan_crop.min() + 1e-10)
    axes[1, 1].imshow(pan_crop_disp, cmap='gray')
    axes[1, 1].set_title('PAN Detail')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(to_rgb(fused[:, y_start:y_end, x_start:x_end], rgb_bands))
    axes[1, 2].set_title('Fused Detail')
    axes[1, 2].axis('off')

    # Edge comparison
    from scipy import ndimage
    fused_gray = fused.mean(axis=0)
    ms_gray = ms.mean(axis=0)

    edges_fused = np.abs(ndimage.sobel(fused_gray))
    edges_ms = np.abs(ndimage.sobel(ms_gray))
    edges_pan = np.abs(ndimage.sobel(pan[0]))

    edges_fused = edges_fused / (edges_fused.max() + 1e-10)
    edges_ms = edges_ms / (edges_ms.max() + 1e-10)
    edges_pan = edges_pan / (edges_pan.max() + 1e-10)

    edge_compare = np.stack([edges_pan, edges_fused, edges_ms], axis=-1)
    axes[1, 3].imshow(edge_compare)
    axes[1, 3].set_title('Edges: R=PAN, G=Fused, B=MS')
    axes[1, 3].axis('off')

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()


def plot_training_history(history: Dict[str, List[float]],
                          output_path: Optional[str] = None):
    """
    Plot training history curves.

    Args:
        history: Dictionary with 'train_loss' and 'val_loss'
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]],
                            output_path: Optional[str] = None):
    """
    Plot bar chart comparing metrics across methods.

    Args:
        metrics_dict: {method_name: {metric_name: value}}
        output_path: Path to save figure
    """
    methods = list(metrics_dict.keys())
    metric_names = ['PSNR', 'SSIM', 'SAM', 'ERGAS']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    for idx, metric in enumerate(metric_names):
        values = [metrics_dict[m][metric] for m in methods]
        bars = axes[idx].bar(methods, values, color=colors)
        axes[idx].set_title(metric)
        axes[idx].set_ylabel(metric)

        # Add value labels
        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                          f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        axes[idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()
