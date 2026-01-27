"""
HPF (High-Pass Filter) Pansharpening

Injects high-frequency details from PAN into MS.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def hpf_fusion(ms: np.ndarray, pan: np.ndarray,
               sigma: float = 2.0, gain_factor: float = 0.5) -> np.ndarray:
    """
    High-Pass Filter (HPF) fusion.

    Args:
        ms: Multispectral image (bands, H, W), normalized [0, 1]
        pan: Panchromatic image (1, H, W), normalized [0, 1]
        sigma: Gaussian filter sigma for lowpass
        gain_factor: Injection gain factor

    Returns:
        Fused image (bands, H, W)
    """
    # Extract high-frequency details from PAN
    pan_lowpass = gaussian_filter(pan[0], sigma=sigma)
    pan_highpass = pan[0] - pan_lowpass

    # Calculate statistics for adaptive gain
    pan_std = np.std(pan[0])

    # Inject high-frequency details into each band
    fused = np.zeros_like(ms)
    for i in range(ms.shape[0]):
        # Band-specific gain based on local statistics
        band_std = np.std(ms[i])
        band_gain = band_std / (pan_std + 1e-10) * gain_factor
        fused[i] = ms[i] + band_gain * pan_highpass

    return np.clip(fused, 0, 1)
