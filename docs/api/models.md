# Models API Reference

## Factory Function

### `create_model`

```python
def create_model(model_name: str, ms_bands: int = 4, **kwargs) -> nn.Module
```

Create a pansharpening model by name.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | required | Model identifier |
| `ms_bands` | int | 4 | Number of multispectral bands |
| `**kwargs` | dict | {} | Additional model-specific arguments |

**Returns:** PyTorch nn.Module

**Example:**
```python
from models import create_model

model = create_model('panformer_lite', ms_bands=4)
```

---

## CNN Models

### PNN

```python
class PNN(nn.Module):
    def __init__(self, ms_bands: int = 4)
```

Basic 3-layer CNN for pansharpening.

### PanNet

```python
class PanNet(nn.Module):
    def __init__(
        self,
        ms_bands: int = 4,
        n_blocks: int = 4,
        n_features: int = 64
    )
```

ResNet-style pansharpening with high-pass filtering.

### DRPNN

```python
class DRPNN(nn.Module):
    def __init__(
        self,
        ms_bands: int = 4,
        n_blocks: int = 8,
        n_features: int = 64
    )
```

Deep residual pansharpening network.

### PanNetCBAM

```python
class PanNetCBAM(nn.Module):
    def __init__(
        self,
        ms_bands: int = 4,
        n_blocks: int = 4,
        n_features: int = 64,
        reduction: int = 16
    )
```

PanNet with CBAM attention modules.

### MultiScalePanNet

```python
class MultiScalePanNet(nn.Module):
    def __init__(
        self,
        ms_bands: int = 4,
        base_features: int = 64
    )
```

Multi-scale feature pyramid network.

---

## Transformer Models

### PanFormer

```python
class PanFormer(nn.Module):
    def __init__(
        self,
        ms_bands: int = 4,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        patch_size: int = 4,
        mlp_ratio: float = 4.0
    )
```

Full transformer with cross-attention fusion.

### PanFormerLite

```python
class PanFormerLite(nn.Module):
    def __init__(
        self,
        ms_bands: int = 4,
        embed_dim: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        window_size: int = 8,
        patch_size: int = 4
    )
```

Lightweight transformer with window attention.

---

## Forward Pass

All models have the same forward signature:

```python
def forward(self, ms: Tensor, pan: Tensor) -> Tensor
```

**Parameters:**

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `ms` | (B, C, H, W) | Multispectral image |
| `pan` | (B, 1, H, W) | Panchromatic image |

**Returns:** Tensor of shape (B, C, H, W)

**Example:**
```python
import torch
from models import create_model

model = create_model('pannet', ms_bands=4)

ms = torch.randn(1, 4, 256, 256)
pan = torch.randn(1, 1, 256, 256)

with torch.no_grad():
    fused = model(ms, pan)

print(fused.shape)  # torch.Size([1, 4, 256, 256])
```
