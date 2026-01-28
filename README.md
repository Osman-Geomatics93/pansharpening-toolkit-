# Pansharpening Toolkit

[![CI](https://github.com/Osman-Geomatics93/pansharpening-toolkit-/actions/workflows/ci.yml/badge.svg)](https://github.com/Osman-Geomatics93/pansharpening-toolkit-/actions/workflows/ci.yml)
[![Documentation](https://github.com/Osman-Geomatics93/pansharpening-toolkit-/actions/workflows/docs.yml/badge.svg)](https://osman-geomatics93.github.io/pansharpening-toolkit-/)
[![codecov](https://codecov.io/gh/Osman-Geomatics93/pansharpening-toolkit-/branch/main/graph/badge.svg)](https://codecov.io/gh/Osman-Geomatics93/pansharpening-toolkit-/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Osman-Geomatics93/pansharpening-toolkit-/blob/main/notebooks/01_quick_start.ipynb)
[![Docker](https://img.shields.io/badge/Docker-Available-2496ED.svg?logo=docker)](https://github.com/Osman-Geomatics93/pansharpening-toolkit-/pkgs/container/pansharpening-toolkit)

A comprehensive pansharpening toolkit implementing both **classic** and **state-of-the-art deep learning** methods for fusing multispectral (MS) and panchromatic (PAN) satellite images.

<p align="center">
  <img src="docs/comparison.png" alt="Pansharpening Comparison" width="800">
</p>

## Features

- **5 Classic Methods**: Brovey, IHS, SFIM, Gram-Schmidt, HPF
- **7 Deep Learning Models**: From simple CNNs to Transformers
- **Advanced Loss Functions**: L1, MSE, SSIM, SAM, Perceptual
- **Attention Mechanisms**: CBAM, SE blocks, Cross-attention
- **Multi-scale Architectures**: Feature pyramid networks
- **Transformer Models**: PanFormer with window attention
- **Quality Metrics**: PSNR, SSIM, SAM, ERGAS
- **GeoTIFF Support**: Preserves geospatial metadata

## Installation

### Using pip

```bash
git clone https://github.com/Osman-Geomatics93/pansharpening-toolkit-.git
cd pansharpening-toolkit
pip install -e .
```

### Using conda

```bash
git clone https://github.com/Osman-Geomatics93/pansharpening-toolkit-.git
cd pansharpening-toolkit
conda env create -f environment.yml
conda activate pansharpening
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.9
- CUDA (optional, for GPU acceleration)

## Quick Start

### 1. Prepare Your Data

Place your PAN and MS images in the `data/` directory:
```
data/
  pan.tif    # Panchromatic image (1 band, high resolution)
  ms.tif     # Multispectral image (N bands, low resolution)
```

### 2. Run Pansharpening

```bash
# Run with default PanNet model
python run_deep_learning.py --model pannet

# Run with transformer model
python run_deep_learning.py --model panformer_lite --epochs 100

# Run all classic methods
python run_classic.py
```

## Available Models

### Deep Learning Models

| Model | Architecture | Parameters | Description |
|-------|-------------|------------|-------------|
| `pnn` | 3-layer CNN | ~50K | Basic baseline |
| `pannet` | ResNet + High-pass | ~80K | Residual learning |
| `drpnn` | Deep ResNet | ~300K | Deeper network |
| `pannet_cbam` | PanNet + CBAM | ~340K | Attention-enhanced |
| `mspannet` | Multi-scale FPN | ~500K | Feature pyramid |
| `panformer` | Transformer | ~1M | Cross-attention |
| `panformer_lite` | Window Transformer | ~370K | Efficient transformer |

### Classic Methods

| Method | Description |
|--------|-------------|
| `brovey` | Component substitution with band ratios |
| `ihs` | Intensity-Hue-Saturation transformation |
| `sfim` | Smoothing Filter-based Intensity Modulation |
| `gram_schmidt` | Gram-Schmidt spectral sharpening |
| `hpf` | High-Pass Filter injection |

## Usage Examples

### Training a Model

```bash
# Basic training
python run_deep_learning.py --model pannet_cbam --epochs 100

# With spectral-focused loss
python run_deep_learning.py --model panformer_lite --loss spectral_focus --epochs 200

# Custom data paths
python run_deep_learning.py \
    --pan path/to/pan.tif \
    --ms path/to/ms.tif \
    --model mspannet \
    --epochs 150
```

### Loss Functions

| Loss | Description | Use Case |
|------|-------------|----------|
| `combined` | L1 + MSE + Gradient | Default, balanced |
| `advanced` | + SSIM + SAM | Better quality |
| `spectral_focus` | Higher SAM weight | Spectral preservation |
| `spatial_focus` | Higher Gradient/SSIM | Spatial details |

### Python API

```python
from models import create_model, create_loss

# Create model
model = create_model('panformer_lite', ms_bands=4)

# Create loss function
criterion = create_loss('spectral_focus')

# Run inference
import torch
ms = torch.randn(1, 4, 256, 256)
pan = torch.randn(1, 1, 256, 256)
fused = model(ms, pan)
```

## Project Structure

```
pansharpening_project/
├── configs/
│   └── config.py              # Configuration and hyperparameters
├── models/
│   ├── attention.py           # CBAM, SE, Cross-attention modules
│   ├── pnn.py                 # PNN model
│   ├── pannet.py              # PanNet model
│   ├── drpnn.py               # DRPNN model
│   ├── pannet_cbam.py         # PanNet with CBAM attention
│   ├── mspannet.py            # Multi-scale PanNet
│   ├── panformer.py           # Transformer model
│   ├── panformer_lite.py      # Lightweight transformer
│   └── losses.py              # Loss functions
├── methods/
│   ├── classic/               # Classic pansharpening methods
│   └── deep_learning/         # Training pipeline
├── utils/
│   ├── data_utils.py          # Data loading and preprocessing
│   ├── metrics.py             # Quality metrics
│   └── visualization.py       # Plotting utilities
├── data/                      # Input images
├── results/                   # Output results
├── checkpoints/               # Model weights
├── run_classic.py             # Run classic methods
├── run_deep_learning.py       # Train DL models
└── run_all.py                 # Complete comparison
```

## Quality Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| **PSNR** | Peak Signal-to-Noise Ratio | Higher is better |
| **SSIM** | Structural Similarity Index | 1.0 (identical) |
| **SAM** | Spectral Angle Mapper | 0° (identical) |
| **ERGAS** | Relative Global Error | Lower is better |

## Benchmark Results

Results on test dataset (100 epochs):

| Model | PSNR (dB) | SSIM | SAM (°) | ERGAS |
|-------|-----------|------|---------|-------|
| SFIM (classic) | 30.30 | 0.828 | 0.02 | 5.50 |
| PanNet | 30.79 | 0.839 | 0.04 | 2.41 |
| PanNetCBAM | 30.35 | 0.828 | 2.13 | 5.47 |
| **PanFormerLite** | **34.62** | **0.908** | 8.48 | **3.37** |

## Architecture Details

### PanFormer

The transformer-based model uses:
- Patch embedding (4x4 patches)
- Two-stream architecture (MS + PAN)
- Self-attention in each stream
- Cross-attention for fusion
- Progressive upsampling decoder

### Attention Mechanisms

```
CBAM (Convolutional Block Attention Module):
  Input -> Channel Attention -> Spatial Attention -> Output

Cross-Attention:
  MS features (Query) + PAN features (Key, Value) -> Fused features
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{pansharpening_toolkit,
  title = {Pansharpening Toolkit: Classic and Deep Learning Methods},
  author = {Ibrahim, Osman O.A.},
  year = {2026},
  url = {https://github.com/Osman-Geomatics93/pansharpening-toolkit-}
}
```

## References

- [PNN] Masi et al., "Pansharpening by Convolutional Neural Networks" (2016)
- [PanNet] Yang et al., "PanNet: A Deep Network Architecture for Pan-Sharpening" (ICCV 2017)
- [CBAM] Woo et al., "CBAM: Convolutional Block Attention Module" (ECCV 2018)
- [Transformers] Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Satellite imagery processing community
- PyTorch team for the deep learning framework
- Open-source contributors
