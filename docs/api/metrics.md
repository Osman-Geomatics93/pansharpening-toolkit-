# Metrics API Reference

## Quality Metrics

### PSNR (Peak Signal-to-Noise Ratio)

```python
def calculate_psnr(pred: Tensor, target: Tensor, max_val: float = 1.0) -> float
```

Higher is better. Measures reconstruction quality in dB.

**Typical Values:**
- 20-25 dB: Poor
- 25-30 dB: Fair
- 30-35 dB: Good
- 35+ dB: Excellent

### SSIM (Structural Similarity Index)

```python
def calculate_ssim(pred: Tensor, target: Tensor) -> float
```

Range: [0, 1], where 1 = identical.

Measures structural similarity considering luminance, contrast, and structure.

### SAM (Spectral Angle Mapper)

```python
def calculate_sam(pred: Tensor, target: Tensor) -> float
```

Lower is better (degrees). Measures spectral distortion.

**Typical Values:**
- < 3: Excellent spectral preservation
- 3-5: Good
- 5-10: Fair
- > 10: Poor

### ERGAS (Relative Global Error)

```python
def calculate_ergas(pred: Tensor, target: Tensor, ratio: int = 4) -> float
```

Lower is better. Comprehensive quality metric.

---

## Usage

### Calculate Individual Metrics

```python
from utils.metrics import calculate_psnr, calculate_ssim, calculate_sam, calculate_ergas

psnr = calculate_psnr(pred, target)
ssim = calculate_ssim(pred, target)
sam = calculate_sam(pred, target)
ergas = calculate_ergas(pred, target)

print(f"PSNR: {psnr:.2f} dB")
print(f"SSIM: {ssim:.4f}")
print(f"SAM: {sam:.4f}")
print(f"ERGAS: {ergas:.4f}")
```

### Calculate All Metrics

```python
from utils.metrics import calculate_metrics

metrics = calculate_metrics(pred, target)
# Returns: {'psnr': ..., 'ssim': ..., 'sam': ..., 'ergas': ...}
```

---

## Metric Interpretation

| Metric | Measures | Ideal | Good Range |
|--------|----------|-------|------------|
| PSNR | Reconstruction | Higher | > 30 dB |
| SSIM | Structure | 1.0 | > 0.85 |
| SAM | Spectral | 0 | < 5 |
| ERGAS | Overall | 0 | < 5 |

!!! note "Trade-offs"
    Improving spatial quality (PSNR, SSIM) may degrade spectral quality (SAM), and vice versa. Use appropriate loss functions to balance these trade-offs.
