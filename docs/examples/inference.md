# Inference Examples

## Basic Inference

```python
import torch
from models import create_model

# Load model
model = create_model('panformer_lite', ms_bands=4)
model.load_state_dict(torch.load('checkpoints/panformer_lite_best.pth'))
model.eval()

# Prepare inputs
ms = torch.randn(1, 4, 256, 256)
pan = torch.randn(1, 1, 256, 256)

# Run inference
with torch.no_grad():
    fused = model(ms, pan)

print(f"Output shape: {fused.shape}")
```

## GPU Inference

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
ms = ms.to(device)
pan = pan.to(device)

with torch.no_grad():
    fused = model(ms, pan)
```

## Inference with GeoTIFF

```python
import rasterio
from utils.data_utils import load_geotiff, save_geotiff

# Load images
ms, ms_profile = load_geotiff('data/ms.tif')
pan, pan_profile = load_geotiff('data/pan.tif')

# Convert to tensors
ms_tensor = torch.from_numpy(ms).unsqueeze(0).float()
pan_tensor = torch.from_numpy(pan).unsqueeze(0).float()

# Run model
with torch.no_grad():
    fused = model(ms_tensor, pan_tensor)

# Save with geospatial metadata
fused_np = fused.squeeze(0).numpy()
save_geotiff('results/fused.tif', fused_np, pan_profile)
```

## Batch Inference

```python
# Process multiple patches
batch_size = 8
results = []

for i in range(0, len(patches), batch_size):
    batch_ms = patches_ms[i:i+batch_size]
    batch_pan = patches_pan[i:i+batch_size]

    with torch.no_grad():
        batch_fused = model(batch_ms, batch_pan)

    results.append(batch_fused)

fused = torch.cat(results, dim=0)
```

## Classic Methods Inference

No training required for classic methods:

```python
from methods.classic import brovey, ihs, sfim, gram_schmidt, hpf

# Run classic methods
fused_brovey = brovey(pan, ms)
fused_ihs = ihs(pan, ms)
fused_sfim = sfim(pan, ms)
fused_gs = gram_schmidt(pan, ms)
fused_hpf = hpf(pan, ms)
```

## Evaluate Results

```python
from utils.metrics import calculate_metrics

# Calculate quality metrics
metrics = calculate_metrics(fused, ground_truth)

print(f"PSNR: {metrics['psnr']:.2f} dB")
print(f"SSIM: {metrics['ssim']:.4f}")
print(f"SAM: {metrics['sam']:.4f}")
print(f"ERGAS: {metrics['ergas']:.4f}")
```

## Visualization

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(ms[:3].permute(1, 2, 0))
axes[0].set_title('MS (RGB)')

axes[1].imshow(pan[0], cmap='gray')
axes[1].set_title('PAN')

axes[2].imshow(fused[:3].permute(1, 2, 0))
axes[2].set_title('Fused')

axes[3].imshow(ground_truth[:3].permute(1, 2, 0))
axes[3].set_title('Ground Truth')

plt.tight_layout()
plt.savefig('results/comparison.png', dpi=150)
plt.show()
```
