# Contributing

We welcome contributions! This guide will help you get started.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/pansharpening-toolkit-.git
cd pansharpening-toolkit-
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style

We use the following tools for code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **Flake8**: Linting

Run before committing:
```bash
pre-commit run --all-files
```

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ -v --cov=models --cov=utils --cov-report=html
```

## Pull Request Process

1. Create a feature branch:
```bash
git checkout -b feature/amazing-feature
```

2. Make your changes and commit:
```bash
git add .
git commit -m "Add amazing feature"
```

3. Push to your fork:
```bash
git push origin feature/amazing-feature
```

4. Open a Pull Request

## Adding a New Model

1. Create `models/your_model.py`:
```python
import torch.nn as nn

class YourModel(nn.Module):
    def __init__(self, ms_bands=4):
        super().__init__()
        self.ms_bands = ms_bands
        # Define layers

    def forward(self, ms, pan):
        # Implementation
        return fused
```

2. Register in `models/__init__.py`:
```python
from .your_model import YourModel

AVAILABLE_MODELS.append('your_model')

# Add to create_model function
```

3. Add tests in `tests/test_models.py`

4. Add documentation in `docs/models/`

## Adding a New Loss

1. Add to `models/losses.py`:
```python
class YourLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Implementation
        return loss
```

2. Register in `create_loss()` factory

3. Add tests in `tests/test_losses.py`

## Reporting Issues

Please use the issue templates:
- Bug Report
- Feature Request

Include:
- Python version
- PyTorch version
- OS
- Steps to reproduce
- Expected vs actual behavior
