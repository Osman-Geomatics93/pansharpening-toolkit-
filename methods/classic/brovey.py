"""
Brovey Transform Pansharpening

A simple and effective method for visual quality.
Formula: Fused_i = MS_i * PAN / mean(MS)
"""

import numpy as np


def brovey_fusion(ms: np.ndarray, pan: np.ndarray) -> np.ndarray:
    """
    Brovey Transform fusion.

    Args:
        ms: Multispectral image (bands, H, W), normalized [0, 1]
        pan: Panchromatic image (1, H, W), normalized [0, 1]

    Returns:
        Fused image (bands, H, W)
    """
    # Calculate intensity as mean of MS bands
    intensity = np.mean(ms, axis=0, keepdims=True)

    # Avoid division by zero
    intensity = np.maximum(intensity, 1e-10)

    # Apply Brovey transform
    fused = ms * (pan / intensity)

    # Clip to valid range
    return np.clip(fused, 0, 1)
