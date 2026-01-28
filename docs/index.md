# Pansharpening Toolkit

A comprehensive toolkit implementing both **classic** and **state-of-the-art deep learning** methods for fusing multispectral (MS) and panchromatic (PAN) satellite images.

![Pansharpening Comparison](assets/comparison.png)

## Features

- **5 Classic Methods**: Brovey, IHS, SFIM, Gram-Schmidt, HPF
- **7 Deep Learning Models**: From simple CNNs to Transformers
- **Advanced Loss Functions**: L1, MSE, SSIM, SAM, Perceptual
- **Attention Mechanisms**: CBAM, SE blocks, Cross-attention
- **Multi-scale Architectures**: Feature pyramid networks
- **Transformer Models**: PanFormer with window attention
- **Quality Metrics**: PSNR, SSIM, SAM, ERGAS
- **GeoTIFF Support**: Preserves geospatial metadata

## Quick Example

```python
from models import create_model, create_loss
import torch

# Create model
model = create_model('panformer_lite', ms_bands=4)

# Create loss function
criterion = create_loss('spectral_focus')

# Run inference
ms = torch.randn(1, 4, 256, 256)
pan = torch.randn(1, 1, 256, 256)
fused = model(ms, pan)
```

## Benchmark Results

Results on test dataset (100 epochs):

| Model | PSNR (dB) | SSIM | SAM | ERGAS |
|-------|-----------|------|-----|-------|
| SFIM (classic) | 30.30 | 0.828 | 0.02 | 5.50 |
| PanNet | 30.79 | 0.839 | 0.04 | 2.41 |
| PanNetCBAM | 30.35 | 0.828 | 2.13 | 5.47 |
| **PanFormerLite** | **34.62** | **0.908** | 8.48 | **3.37** |

## License

This project is licensed under the MIT License.
