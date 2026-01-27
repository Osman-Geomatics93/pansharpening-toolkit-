"""
Quality Metrics for Pansharpening Evaluation
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- SAM (Spectral Angle Mapper)
- ERGAS (Relative Global Error)
"""

import numpy as np
from typing import Dict
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func


def compute_psnr(reference: np.ndarray, test: np.ndarray) -> float:
    """
    Compute PSNR (average across bands).

    Args:
        reference: Reference image (bands, H, W)
        test: Test image (bands, H, W)

    Returns:
        Mean PSNR in dB
    """
    psnr_values = []
    for i in range(reference.shape[0]):
        ref_band = reference[i].astype(np.float64)
        test_band = test[i].astype(np.float64)
        data_range = max(ref_band.max() - ref_band.min(),
                         test_band.max() - test_band.min())
        if data_range > 0:
            p = psnr_func(ref_band, test_band, data_range=data_range)
            psnr_values.append(p)

    return float(np.mean(psnr_values)) if psnr_values else 0.0


def compute_ssim(reference: np.ndarray, test: np.ndarray) -> float:
    """
    Compute SSIM (average across bands).

    Args:
        reference: Reference image (bands, H, W)
        test: Test image (bands, H, W)

    Returns:
        Mean SSIM value
    """
    ssim_values = []
    for i in range(reference.shape[0]):
        ref_band = reference[i].astype(np.float64)
        test_band = test[i].astype(np.float64)
        data_range = max(ref_band.max() - ref_band.min(),
                         test_band.max() - test_band.min())
        if data_range > 0:
            s = ssim_func(ref_band, test_band, data_range=data_range)
            ssim_values.append(s)

    return float(np.mean(ssim_values)) if ssim_values else 0.0


def compute_sam(reference: np.ndarray, test: np.ndarray) -> float:
    """
    Compute Spectral Angle Mapper (SAM).

    Args:
        reference: Reference image (bands, H, W)
        test: Test image (bands, H, W)

    Returns:
        Mean SAM value in degrees
    """
    # Reshape to (H*W, bands)
    ref_flat = reference.reshape(reference.shape[0], -1).T
    test_flat = test.reshape(test.shape[0], -1).T

    # Compute dot product and norms
    dot_product = np.sum(ref_flat * test_flat, axis=1)
    ref_norm = np.linalg.norm(ref_flat, axis=1)
    test_norm = np.linalg.norm(test_flat, axis=1)

    # Compute angle
    cos_angle = dot_product / (ref_norm * test_norm + 1e-10)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)

    return float(np.degrees(np.mean(angle)))


def compute_ergas(reference: np.ndarray, test: np.ndarray,
                  ratio: float = 4.0) -> float:
    """
    Compute ERGAS (Relative Global Dimensional Synthesis Error).

    Args:
        reference: Reference image (bands, H, W)
        test: Test image (bands, H, W)
        ratio: Scale ratio between PAN and MS

    Returns:
        ERGAS value
    """
    n_bands = reference.shape[0]
    sum_term = 0.0

    for i in range(n_bands):
        ref_band = reference[i]
        test_band = test[i]

        rmse = np.sqrt(np.mean((ref_band - test_band) ** 2))
        mean_ref = np.mean(ref_band)

        if mean_ref > 0:
            sum_term += (rmse / mean_ref) ** 2

    ergas = 100 / ratio * np.sqrt(sum_term / n_bands)
    return float(ergas)


def compute_all_metrics(reference: np.ndarray, test: np.ndarray,
                        ratio: float = 4.0) -> Dict[str, float]:
    """
    Compute all quality metrics.

    Args:
        reference: Reference image (bands, H, W)
        test: Test image (bands, H, W)
        ratio: Scale ratio for ERGAS

    Returns:
        Dictionary of metrics
    """
    return {
        'PSNR': compute_psnr(reference, test),
        'SSIM': compute_ssim(reference, test),
        'SAM': compute_sam(reference, test),
        'ERGAS': compute_ergas(reference, test, ratio)
    }


def print_metrics(metrics: Dict[str, float], name: str = ""):
    """Print metrics in a formatted way."""
    print(f"\n{'='*40}")
    if name:
        print(f"Metrics for: {name}")
    print(f"{'='*40}")
    print(f"PSNR:  {metrics['PSNR']:.2f} dB")
    print(f"SSIM:  {metrics['SSIM']:.4f}")
    print(f"SAM:   {metrics['SAM']:.4f} degrees")
    print(f"ERGAS: {metrics['ERGAS']:.4f}")
