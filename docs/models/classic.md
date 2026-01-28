# Classic Methods

Classic pansharpening methods use signal processing techniques without requiring training data.

## Brovey Transform

The Brovey transform is a component substitution method that uses band ratios:

```python
from methods.classic import brovey

fused = brovey(pan, ms)
```

**Algorithm:**
$$MS_{sharp}^i = MS^i \times \frac{PAN}{\sum_j MS^j}$$

## IHS Fusion

Intensity-Hue-Saturation transformation:

```python
from methods.classic import ihs

fused = ihs(pan, ms)
```

**Process:**
1. Convert MS to IHS color space
2. Replace Intensity with PAN
3. Convert back to RGB

## SFIM

Smoothing Filter-based Intensity Modulation:

```python
from methods.classic import sfim

fused = sfim(pan, ms)
```

**Algorithm:**
$$MS_{sharp}^i = MS^i \times \frac{PAN}{PAN_{lowpass}}$$

## Gram-Schmidt

Gram-Schmidt spectral sharpening:

```python
from methods.classic import gram_schmidt

fused = gram_schmidt(pan, ms)
```

## High-Pass Filter (HPF)

Injects high-frequency details from PAN into MS:

```python
from methods.classic import hpf

fused = hpf(pan, ms)
```

**Algorithm:**
$$MS_{sharp}^i = MS^i + w \times (PAN - PAN_{lowpass})$$

## Comparison

| Method | Spectral Quality | Spatial Quality | Speed |
|--------|-----------------|-----------------|-------|
| Brovey | Medium | High | Fast |
| IHS | Medium | High | Fast |
| SFIM | High | Medium | Fast |
| Gram-Schmidt | High | High | Medium |
| HPF | Medium | High | Fast |
