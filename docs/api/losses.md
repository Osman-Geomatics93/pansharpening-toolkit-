# Losses API Reference

## Factory Function

### `create_loss`

```python
def create_loss(loss_name: str, **kwargs) -> nn.Module
```

Create a loss function by name.

**Available Losses:**

| Name | Description |
|------|-------------|
| `combined` | L1 + MSE + Gradient (default) |
| `advanced` | + SSIM + SAM |
| `spectral_focus` | Higher SAM weight |
| `spatial_focus` | Higher Gradient/SSIM weight |

**Example:**
```python
from models import create_loss

criterion = create_loss('spectral_focus')
loss, loss_dict = criterion(pred, target)
```

---

## Individual Loss Functions

### GradientLoss

```python
class GradientLoss(nn.Module):
    def __init__(self)
```

Computes L1 loss on image gradients for edge preservation.

```python
loss_fn = GradientLoss()
loss = loss_fn(pred, target)  # Returns scalar
```

### SpectralAngleLoss

```python
class SpectralAngleLoss(nn.Module):
    def __init__(self, eps: float = 1e-8)
```

Spectral Angle Mapper (SAM) loss for spectral fidelity.

```python
loss_fn = SpectralAngleLoss()
loss = loss_fn(pred, target)  # Returns scalar (radians)
```

### SSIMLoss

```python
class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11)
```

Structural Similarity Index loss.

```python
loss_fn = SSIMLoss()
loss = loss_fn(pred, target)  # Returns 1 - SSIM
```

### PerceptualLoss

```python
class PerceptualLoss(nn.Module):
    def __init__(self, layers: List[str] = None)
```

VGG-based perceptual loss using feature maps.

---

## Combined Loss Functions

### CombinedLoss

```python
class CombinedLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 1.0,
        mse_weight: float = 1.0,
        gradient_weight: float = 0.1
    )
```

Basic combined loss with L1, MSE, and gradient terms.

**Returns:** `(total_loss, loss_dict)`

```python
loss_fn = CombinedLoss()
total_loss, loss_dict = loss_fn(pred, target)
# loss_dict = {'l1': ..., 'mse': ..., 'gradient': ..., 'total': ...}
```

### AdvancedCombinedLoss

```python
class AdvancedCombinedLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 1.0,
        mse_weight: float = 1.0,
        gradient_weight: float = 0.1,
        ssim_weight: float = 0.1,
        sam_weight: float = 0.1,
        perceptual_weight: float = 0.0
    )
```

Advanced loss with all components configurable.

---

## Loss Configurations

### Default (combined)
```python
{'l1': 1.0, 'mse': 1.0, 'gradient': 0.1}
```

### Advanced
```python
{'l1': 1.0, 'mse': 1.0, 'gradient': 0.1, 'ssim': 0.1, 'sam': 0.1}
```

### Spectral Focus
```python
{'l1': 1.0, 'mse': 0.5, 'gradient': 0.05, 'ssim': 0.1, 'sam': 0.5}
```

### Spatial Focus
```python
{'l1': 1.0, 'mse': 0.5, 'gradient': 0.2, 'ssim': 0.3, 'sam': 0.05}
```
