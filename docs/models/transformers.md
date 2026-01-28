# Transformer Models

## PanFormer

Full transformer architecture with cross-attention fusion.

```python
from models import PanFormer

model = PanFormer(
    ms_bands=4,
    embed_dim=128,
    depth=4,
    num_heads=8,
    patch_size=4
)
```

**Architecture:**
```
Input: MS + PAN
  |
  v
Patch Embedding (4x4 patches)
  |
  +---> MS Stream: Self-Attention Blocks
  |
  +---> PAN Stream: Self-Attention Blocks
  |
  v
Cross-Attention Fusion (MS queries, PAN keys/values)
  |
  v
Decoder + Reconstruction Head
  |
  v
Output + Residual Connection
```

**Key Components:**

1. **Patch Embedding**: Converts images to patch tokens
2. **Self-Attention**: Models long-range dependencies within each stream
3. **Cross-Attention**: Fuses information between MS and PAN streams
4. **Progressive Decoder**: Reconstructs high-resolution output

## PanFormerLite

Lightweight transformer optimized for efficiency.

```python
from models import PanFormerLite

model = PanFormerLite(
    ms_bands=4,
    embed_dim=64,
    depth=2,
    num_heads=4,
    window_size=8
)
```

**Optimizations:**
- Smaller embedding dimension (64 vs 128)
- Fewer transformer blocks (2 vs 4)
- Window attention instead of global attention
- ~370K parameters vs ~1M for full PanFormer

## Window Attention

PanFormerLite uses window-based attention for efficiency:

```
Image (H, W)
  -> Split into windows (window_size x window_size)
  -> Self-attention within each window
  -> Merge windows
```

This reduces complexity from O(N^2) to O(N * window_size^2).

## Training Tips

!!! tip "Best Practices"
    1. **Learning Rate**: Use lower LR (1e-4 to 5e-5) for transformers
    2. **Warmup**: Always use LR warmup (5-10 epochs)
    3. **Epochs**: Train for 100+ epochs for best results
    4. **Loss**: Use `spectral_focus` or `advanced` loss
    5. **Batch Size**: Larger batches help transformer training

## Example Training

```bash
python scripts/run_deep_learning.py \
    --model panformer_lite \
    --loss spectral_focus \
    --epochs 100 \
    --lr 5e-5
```

## Benchmark Results

| Model | PSNR | SSIM | Parameters | Training Time |
|-------|------|------|------------|---------------|
| PanNet | 30.79 | 0.839 | 340K | 1x |
| PanFormer | 35.0+ | 0.92+ | 1M | 3x |
| PanFormerLite | 34.62 | 0.908 | 370K | 1.5x |
