"""
Data Utilities for Pansharpening
- Image loading and saving
- Normalization
- Upsampling
- Patch extraction
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from pathlib import Path


def load_image(filepath: str) -> Tuple[np.ndarray, dict]:
    """
    Load a GeoTIFF image.

    Args:
        filepath: Path to the TIF file

    Returns:
        data: Image array (bands, H, W)
        meta: Metadata dictionary
    """
    with rasterio.open(filepath) as src:
        data = src.read().astype(np.float32)
        meta = {
            'transform': src.transform,
            'crs': src.crs,
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'dtype': str(src.dtypes[0]),
            'bounds': src.bounds,
            'nodata': src.nodata
        }
    return data, meta


def normalize_image(image: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, dict]:
    """
    Normalize image to [0, 1] range.

    Args:
        image: Input image (bands, H, W)
        method: 'minmax' or 'percentile'

    Returns:
        normalized: Normalized image
        params: Normalization parameters for inverse transform
    """
    params = {'method': method}
    normalized = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        band = image[i]

        if method == 'minmax':
            min_val = np.min(band)
            max_val = np.max(band)
        else:  # percentile
            min_val = np.percentile(band, 2)
            max_val = np.percentile(band, 98)

        params[f'band_{i}_min'] = min_val
        params[f'band_{i}_max'] = max_val

        if max_val > min_val:
            normalized[i] = np.clip((band - min_val) / (max_val - min_val), 0, 1)

    return normalized, params


def denormalize_image(image: np.ndarray, params: dict) -> np.ndarray:
    """Inverse normalization to original value range."""
    denormalized = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        min_val = params[f'band_{i}_min']
        max_val = params[f'band_{i}_max']
        denormalized[i] = image[i] * (max_val - min_val) + min_val

    return denormalized


def upsample_image(image: np.ndarray, target_size: Tuple[int, int],
                   method: str = 'bicubic') -> np.ndarray:
    """
    Upsample image to target size.

    Args:
        image: Input image (bands, H, W)
        target_size: Target size (H, W)
        method: 'bicubic', 'bilinear', or 'nearest'

    Returns:
        Upsampled image
    """
    tensor = torch.from_numpy(image).unsqueeze(0).float()

    align_corners = False if method != 'nearest' else None
    upsampled = F.interpolate(tensor, size=target_size, mode=method,
                              align_corners=align_corners)

    return upsampled.squeeze(0).numpy()


def create_patches(ms: np.ndarray, pan: np.ndarray, target: np.ndarray,
                   patch_size: int = 64, stride: int = 32,
                   min_std: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create training patches from images.

    Args:
        ms: MS image (bands, H, W)
        pan: PAN image (1, H, W)
        target: Target image (bands, H, W)
        patch_size: Size of patches
        stride: Stride between patches
        min_std: Minimum std to include patch

    Returns:
        ms_patches, pan_patches, target_patches
    """
    _, h, w = pan.shape

    ms_patches, pan_patches, target_patches = [], [], []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            ms_p = ms[:, y:y+patch_size, x:x+patch_size]
            pan_p = pan[:, y:y+patch_size, x:x+patch_size]
            target_p = target[:, y:y+patch_size, x:x+patch_size]

            if np.std(pan_p) > min_std:
                ms_patches.append(ms_p)
                pan_patches.append(pan_p)
                target_patches.append(target_p)

    return np.array(ms_patches), np.array(pan_patches), np.array(target_patches)


def downsample_image(image: np.ndarray, scale_factor: int = 4,
                     method: str = 'bilinear') -> np.ndarray:
    """
    Downsample image by a scale factor.

    Args:
        image: Input image (bands, H, W)
        scale_factor: Downsampling factor
        method: 'bilinear', 'bicubic', or 'area'

    Returns:
        Downsampled image
    """
    _, h, w = image.shape
    target_size = (h // scale_factor, w // scale_factor)

    tensor = torch.from_numpy(image).unsqueeze(0).float()

    if method == 'area':
        downsampled = F.avg_pool2d(tensor, kernel_size=scale_factor, stride=scale_factor)
    else:
        align_corners = False if method != 'nearest' else None
        downsampled = F.interpolate(tensor, size=target_size, mode=method,
                                    align_corners=align_corners)

    return downsampled.squeeze(0).numpy()


def create_wald_patches(ms_hr: np.ndarray, pan_hr: np.ndarray,
                        patch_size: int = 64, stride: int = 32,
                        scale_factor: int = 4, min_std: float = 0.01
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create training patches using Wald's protocol.

    Wald's protocol: Downsample the images and use the original MS as target.
    This gives the network a proper signal to learn spatial enhancement.

    Args:
        ms_hr: High-resolution MS (already upsampled to PAN size) (bands, H, W)
        pan_hr: High-resolution PAN (1, H, W)
        patch_size: Size of patches at HR scale
        stride: Stride between patches
        scale_factor: Downsampling factor (typically 4)
        min_std: Minimum std to include patch

    Returns:
        ms_lr_patches: Low-res MS patches (input)
        pan_lr_patches: Low-res PAN patches (input)
        ms_hr_patches: High-res MS patches (target/ground truth)
    """
    _, h, w = pan_hr.shape

    # Ensure dimensions are divisible by scale_factor
    h_valid = (h // scale_factor) * scale_factor
    w_valid = (w // scale_factor) * scale_factor
    ms_hr = ms_hr[:, :h_valid, :w_valid]
    pan_hr = pan_hr[:, :h_valid, :w_valid]

    # Downsample for Wald protocol (simulate lower resolution)
    ms_lr = downsample_image(ms_hr, scale_factor, method='area')
    pan_lr = downsample_image(pan_hr, scale_factor, method='area')

    # Upsample back to HR size (simulating the input we'll have at test time)
    lr_h, lr_w = ms_lr.shape[1], ms_lr.shape[2]
    ms_lr_up = upsample_image(ms_lr, (h_valid, w_valid), method='bicubic')
    pan_lr_up = upsample_image(pan_lr, (h_valid, w_valid), method='bicubic')

    # Now extract patches:
    # Input: upsampled low-res MS + upsampled low-res PAN
    # Target: original high-res MS
    ms_lr_patches, pan_lr_patches, ms_hr_patches = [], [], []

    for y in range(0, h_valid - patch_size + 1, stride):
        for x in range(0, w_valid - patch_size + 1, stride):
            ms_lr_p = ms_lr_up[:, y:y+patch_size, x:x+patch_size]
            pan_lr_p = pan_lr_up[:, y:y+patch_size, x:x+patch_size]
            ms_hr_p = ms_hr[:, y:y+patch_size, x:x+patch_size]

            # Filter out flat patches
            if np.std(pan_lr_p) > min_std and np.std(ms_hr_p) > min_std:
                ms_lr_patches.append(ms_lr_p)
                pan_lr_patches.append(pan_lr_p)
                ms_hr_patches.append(ms_hr_p)

    print(f"Created {len(ms_lr_patches)} Wald patches")
    print(f"  Input MS range: [{ms_lr_up.min():.3f}, {ms_lr_up.max():.3f}]")
    print(f"  Target MS range: [{ms_hr.min():.3f}, {ms_hr.max():.3f}]")

    # Compute and print residual statistics (what the network needs to learn)
    if len(ms_lr_patches) > 0:
        ms_lr_arr = np.array(ms_lr_patches)
        ms_hr_arr = np.array(ms_hr_patches)
        residual = ms_hr_arr - ms_lr_arr
        print(f"  Residual (target-input) range: [{residual.min():.3f}, {residual.max():.3f}]")
        print(f"  Residual std: {residual.std():.4f}")

    return np.array(ms_lr_patches), np.array(pan_lr_patches), np.array(ms_hr_patches)


def save_geotiff(data: np.ndarray, output_path: str, reference_meta: dict,
                 denorm_params: Optional[dict] = None):
    """
    Save result as GeoTIFF.

    Args:
        data: Image data (bands, H, W)
        output_path: Output file path
        reference_meta: Reference metadata for CRS/transform
        denorm_params: Optional denormalization parameters
    """
    if denorm_params:
        data = denormalize_image(data, denorm_params)

    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': data.shape[2],
        'height': data.shape[1],
        'count': data.shape[0],
        'crs': reference_meta['crs'],
        'transform': reference_meta['transform']
    }

    # Adjust transform if sizes differ
    if data.shape[1] != reference_meta['height'] or data.shape[2] != reference_meta['width']:
        bounds = reference_meta['bounds']
        profile['transform'] = from_bounds(
            bounds.left, bounds.bottom, bounds.right, bounds.top,
            data.shape[2], data.shape[1]
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data.astype(np.float32))

    print(f"Saved: {output_path}")
