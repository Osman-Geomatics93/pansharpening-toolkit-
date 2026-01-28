# Models Overview

The Pansharpening Toolkit provides a comprehensive set of models ranging from classic signal processing methods to state-of-the-art deep learning architectures.

## Model Categories

### Classic Methods

Traditional pansharpening algorithms based on signal processing:

| Method | Description |
|--------|-------------|
| Brovey | Component substitution with band ratios |
| IHS | Intensity-Hue-Saturation transformation |
| SFIM | Smoothing Filter-based Intensity Modulation |
| Gram-Schmidt | Gram-Schmidt spectral sharpening |
| HPF | High-Pass Filter injection |

### Deep Learning Models

Neural network-based approaches:

| Model | Architecture | Parameters | Description |
|-------|-------------|------------|-------------|
| `pnn` | 3-layer CNN | ~50K | Basic baseline |
| `pannet` | ResNet + High-pass | ~340K | Residual learning |
| `drpnn` | Deep ResNet | ~300K | Deeper network |
| `pannet_cbam` | PanNet + CBAM | ~340K | Attention-enhanced |
| `mspannet` | Multi-scale FPN | ~500K | Feature pyramid |
| `panformer` | Transformer | ~1M | Cross-attention |
| `panformer_lite` | Window Transformer | ~370K | Efficient transformer |

## Model Selection Guide

!!! tip "Recommendations"
    - **Quick experiments**: Use `pnn` or `pannet`
    - **Best quality**: Use `panformer_lite` with 100+ epochs
    - **Spectral preservation**: Use `pannet_cbam` with `spectral_focus` loss
    - **Limited compute**: Use classic methods (no training needed)

## Factory Function

All models can be created using the factory function:

```python
from models import create_model, AVAILABLE_MODELS

# List available models
print(AVAILABLE_MODELS)
# ['pnn', 'pannet', 'drpnn', 'pannet_cbam', 'mspannet', 'panformer', 'panformer_lite']

# Create a model
model = create_model('panformer_lite', ms_bands=4)
```
