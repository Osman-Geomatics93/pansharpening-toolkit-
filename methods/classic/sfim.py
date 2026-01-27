"""
SFIM (Smoothing Filter-based Intensity Modulation) Pansharpening

Preserves spectral information better than Brovey/IHS.
Formula: Fused_i = MS_i * PAN / PAN_lowpass
"""

import numpy as np
from scipy.ndimage import uniform_filter


def sfim_fusion(ms: np.ndarray, pan: np.ndarray,
                window_size: int = 7) -> np.ndarray:
    """
    SFIM (Smoothing Filter-based Intensity Modulation) fusion.

    Args:
        ms: Multispectral image (bands, H, W), normalized [0, 1]
        pan: Panchromatic image (1, H, W), normalized [0, 1]
        window_size: Size of smoothing filter

    Returns:
        Fused image (bands, H, W)
    """
    # Create lowpass version of PAN
    pan_lowpass = uniform_filter(pan[0], size=window_size)
    pan_lowpass = np.maximum(pan_lowpass, 1e-10)

    # Calculate ratio (high-frequency modulation)
    ratio = pan[0] / pan_lowpass

    # Apply to each band
    fused = np.zeros_like(ms)
    for i in range(ms.shape[0]):
        fused[i] = ms[i] * ratio

    return np.clip(fused, 0, 1)
