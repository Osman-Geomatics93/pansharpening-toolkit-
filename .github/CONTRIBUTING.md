# Contributing to Pansharpening Toolkit

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a virtual environment
4. Install dependencies: `pip install -e ".[dev]"`
5. Install pre-commit hooks: `pre-commit install`

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run tests: `pytest tests/ -v`
4. Run formatters: `pre-commit run --all-files`
5. Commit and push
6. Open a Pull Request

## Code Style

- We use Black for formatting
- We use isort for import sorting
- We use flake8 for linting
- Maximum line length: 100 characters

## Adding New Models

1. Create `models/your_model.py`
2. Register in `models/__init__.py`
3. Add tests in `tests/test_models.py`
4. Update documentation

## Questions?

Open an issue for discussion.
