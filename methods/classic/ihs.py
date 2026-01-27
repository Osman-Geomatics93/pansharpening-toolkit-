"""
IHS (Intensity-Hue-Saturation) Pansharpening

Classic method that replaces the intensity component with PAN.
"""

import numpy as np
from skimage import exposure


def ihs_fusion(ms: np.ndarray, pan: np.ndarray) -> np.ndarray:
    """
    IHS (Intensity-Hue-Saturation) fusion.

    Steps:
    1. Calculate intensity from MS
    2. Match PAN histogram to intensity
    3. Apply ratio to each band

    Args:
        ms: Multispectral image (bands, H, W), normalized [0, 1]
        pan: Panchromatic image (1, H, W), normalized [0, 1]

    Returns:
        Fused image (bands, H, W)
    """
    n_bands = ms.shape[0]

    # Calculate intensity from MS
    intensity = np.mean(ms, axis=0)

    # Match PAN histogram to intensity
    pan_matched = exposure.match_histograms(pan[0], intensity)

    # Calculate the ratio
    ratio = np.where(intensity > 1e-10, pan_matched / intensity, 1.0)

    # Apply ratio to each band
    fused = np.zeros_like(ms)
    for i in range(n_bands):
        fused[i] = ms[i] * ratio

    return np.clip(fused, 0, 1)
