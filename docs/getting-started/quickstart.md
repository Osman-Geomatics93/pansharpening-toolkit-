# Quick Start

## 1. Prepare Your Data

Place your PAN and MS images in the `data/` directory:

```
data/
  pan.tif    # Panchromatic image (1 band, high resolution)
  ms.tif     # Multispectral image (N bands, low resolution)
```

## 2. Run Pansharpening

### Using Deep Learning Models

```bash
# Run with default PanNet model
python run_deep_learning.py --model pannet

# Run with transformer model
python run_deep_learning.py --model panformer_lite --epochs 100

# With spectral-focused loss
python run_deep_learning.py --model panformer_lite --loss spectral_focus --epochs 200
```

### Using Classic Methods

```bash
# Run all classic methods
python run_classic.py
```

## 3. Python API

```python
from models import create_model, create_loss
import torch

# Create model
model = create_model('panformer_lite', ms_bands=4)

# Create loss function
criterion = create_loss('spectral_focus')

# Prepare inputs
ms = torch.randn(1, 4, 256, 256)   # Multispectral
pan = torch.randn(1, 1, 256, 256)  # Panchromatic

# Run inference
with torch.no_grad():
    fused = model(ms, pan)

print(f"Output shape: {fused.shape}")  # [1, 4, 256, 256]
```

## 4. Load Pre-trained Weights

```python
import torch
from models import create_model

# Create model
model = create_model('panformer_lite', ms_bands=4)

# Load pre-trained weights
model.load_state_dict(torch.load('checkpoints/panformer_lite_best.pth'))
model.eval()
```

## 5. Evaluate Results

```python
from utils.metrics import calculate_metrics

metrics = calculate_metrics(fused, ground_truth)
print(f"PSNR: {metrics['psnr']:.2f} dB")
print(f"SSIM: {metrics['ssim']:.4f}")
print(f"SAM: {metrics['sam']:.4f}")
```
