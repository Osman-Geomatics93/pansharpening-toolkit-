"""
Gram-Schmidt Pansharpening

Good spectral preservation through orthogonal transformation.
"""

import numpy as np
from skimage import exposure


def gram_schmidt_fusion(ms: np.ndarray, pan: np.ndarray,
                        weights: np.ndarray = None) -> np.ndarray:
    """
    Gram-Schmidt fusion (simplified version).

    Args:
        ms: Multispectral image (bands, H, W), normalized [0, 1]
        pan: Panchromatic image (1, H, W), normalized [0, 1]
        weights: Optional weights for intensity calculation

    Returns:
        Fused image (bands, H, W)
    """
    n_bands = ms.shape[0]
    h, w = pan.shape[1], pan.shape[2]

    # Default weights (can be adjusted based on sensor)
    if weights is None:
        weights = np.array([0.25, 0.35, 0.25, 0.15])[:n_bands]
        weights = weights / weights.sum()

    # Simulate intensity from MS (weighted average)
    intensity = np.zeros((h, w), dtype=np.float32)
    for i in range(n_bands):
        intensity += weights[i] * ms[i]

    # Match PAN histogram to intensity
    pan_matched = exposure.match_histograms(pan[0], intensity)

    # Calculate adjustment
    adjustment = pan_matched - intensity

    # Add adjustment to each band (scaled by correlation)
    fused = np.zeros_like(ms)
    for i in range(n_bands):
        # Calculate gain based on band correlation
        corr = np.corrcoef(ms[i].flatten(), intensity.flatten())[0, 1]
        gain = max(0.5, min(1.5, corr))  # Limit gain
        fused[i] = ms[i] + gain * adjustment

    return np.clip(fused, 0, 1)
