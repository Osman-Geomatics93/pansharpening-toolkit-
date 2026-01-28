# Training Examples

## Basic Training

Train PanNet with default settings:

```bash
python scripts/run_deep_learning.py --model pannet --epochs 100
```

## Advanced Training

### With Custom Loss

```bash
# Spectral-focused training
python scripts/run_deep_learning.py \
    --model panformer_lite \
    --loss spectral_focus \
    --epochs 200

# Spatial-focused training
python scripts/run_deep_learning.py \
    --model pannet_cbam \
    --loss spatial_focus \
    --epochs 150
```

### With Custom Data

```bash
python scripts/run_deep_learning.py \
    --pan path/to/pan.tif \
    --ms path/to/ms.tif \
    --model mspannet \
    --epochs 150
```

## Python Training Script

```python
import torch
import torch.optim as optim
from models import create_model, create_loss
from utils.data_utils import load_data

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
ms, pan, target = load_data('data/ms.tif', 'data/pan.tif')
ms = ms.to(device)
pan = pan.to(device)
target = target.to(device)

# Create model
model = create_model('panformer_lite', ms_bands=4).to(device)

# Create loss and optimizer
criterion = create_loss('advanced')
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Training loop
for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    output = model(ms, pan)
    loss, loss_dict = criterion(output, target)

    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), 'checkpoints/model_final.pth')
```

## Multi-GPU Training

```python
import torch.nn as nn

# Wrap model for multi-GPU
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)
```

## Training Tips

!!! tip "Best Practices"
    1. **Start Small**: Test with PNN first, then move to complex models
    2. **Monitor Metrics**: Track PSNR, SSIM, and SAM during training
    3. **Use Warmup**: Essential for transformers
    4. **Save Checkpoints**: Save best model based on validation metrics
    5. **Visualize**: Check outputs periodically
