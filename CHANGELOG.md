# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-28

### Added

#### Deep Learning Models
- **PNN**: Basic 3-layer CNN baseline
- **PanNet**: ResNet-style architecture with high-pass filtering
- **DRPNN**: Deep Residual PanNet
- **PanNetCBAM**: PanNet with CBAM attention modules
- **MultiScalePanNet**: Feature pyramid architecture with multi-scale fusion
- **PanFormer**: Transformer-based model with cross-attention
- **PanFormerLite**: Lightweight transformer with window attention

#### Attention Modules
- Channel Attention (Squeeze-and-Excitation style)
- Spatial Attention
- CBAM (Convolutional Block Attention Module)
- SE Block
- Cross-Attention for multi-stream fusion

#### Loss Functions
- Gradient Loss (edge preservation)
- Combined Loss (L1 + MSE + Gradient)
- Spectral Angle Loss (SAM)
- SSIM Loss (structural similarity)
- Perceptual Loss (VGG features)
- Advanced Combined Loss (configurable multi-loss)

#### Classic Methods
- Brovey Transform
- IHS Fusion
- SFIM (Smoothing Filter-based Intensity Modulation)
- Gram-Schmidt Spectral Sharpening
- High-Pass Filter

#### Infrastructure
- Factory functions: `create_model()`, `create_loss()`
- Training pipeline with warmup and cosine annealing
- Quality metrics: PSNR, SSIM, SAM, ERGAS
- GeoTIFF support with metadata preservation
- Jupyter notebooks for tutorials
- GitHub Actions CI/CD
- Comprehensive test suite

### Technical Details
- Python 3.8+ support
- PyTorch 1.9+ compatibility
- CUDA support for GPU acceleration
- Modern packaging with pyproject.toml

---

## [Unreleased]

### Planned
- Pre-trained model weights
- Additional transformer variants
- Distributed training support
- ONNX export for deployment
