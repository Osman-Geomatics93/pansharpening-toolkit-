# Installation

## Requirements

- Python >= 3.8
- PyTorch >= 1.9
- CUDA (optional, for GPU acceleration)

## Using pip

```bash
git clone https://github.com/Osman-Geomatics93/pansharpening-toolkit-.git
cd pansharpening-toolkit-
pip install -e .
```

## Using conda

```bash
git clone https://github.com/Osman-Geomatics93/pansharpening-toolkit-.git
cd pansharpening-toolkit-
conda env create -f environment.yml
conda activate pansharpening
```

## Using Docker

```bash
# Pull the image
docker pull ghcr.io/osman-geomatics93/pansharpening-toolkit:latest

# Or build locally
docker build -t pansharpening-toolkit .

# Run container
docker run -it --gpus all pansharpening-toolkit
```

## Verify Installation

```bash
python -c "from models import create_model; print('Success!')"
```

Or run the test suite:

```bash
pytest tests/ -v
```
