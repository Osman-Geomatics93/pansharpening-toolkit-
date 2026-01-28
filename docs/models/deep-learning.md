# Deep Learning Models

## PNN (Pansharpening Neural Network)

The simplest baseline model with 3 convolutional layers.

```python
from models import PNN

model = PNN(ms_bands=4)
```

**Architecture:**
```
Input: MS (B,4,H,W) + PAN (B,1,H,W)
  -> Concat -> (B,5,H,W)
  -> Conv2d(5, 64, 9) + ReLU
  -> Conv2d(64, 32, 5) + ReLU
  -> Conv2d(32, 4, 5)
Output: Fused (B,4,H,W)
```

## PanNet

ResNet-style architecture with high-pass filtering for residual learning.

```python
from models import PanNet

model = PanNet(ms_bands=4, n_blocks=4)
```

**Key Features:**
- High-pass filtered PAN input
- Residual blocks with skip connections
- Learns residual details, not full reconstruction

## DRPNN (Deep Residual PanNet)

Deeper version of PanNet with more residual blocks.

```python
from models import DRPNN

model = DRPNN(ms_bands=4, n_blocks=8)
```

## PanNetCBAM

PanNet enhanced with CBAM (Convolutional Block Attention Module).

```python
from models import PanNetCBAM

model = PanNetCBAM(ms_bands=4, n_blocks=4)
```

**CBAM Attention:**
```
Input -> Channel Attention -> Spatial Attention -> Output
```

- **Channel Attention**: "What" to focus on
- **Spatial Attention**: "Where" to focus

## MultiScalePanNet

Feature pyramid architecture with multi-scale fusion.

```python
from models import MultiScalePanNet

model = MultiScalePanNet(ms_bands=4)
```

**Architecture:**
```
Input: MS + PAN
  -> Encoder (3 scales: 1x, 1/2x, 1/4x)
  -> Multi-scale fusion
  -> Decoder with skip connections
Output: Fused image
```

## Training Example

```python
from models import create_model, create_loss
import torch.optim as optim

# Create model and loss
model = create_model('pannet_cbam', ms_bands=4)
criterion = create_loss('advanced')

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    output = model(ms, pan)
    loss, loss_dict = criterion(output, target)
    loss.backward()
    optimizer.step()
```
