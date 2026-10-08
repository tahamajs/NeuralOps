# Contributing to NeuralOps

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/tahamajs/SVD_linear_Attention.git
cd SVD_linear_Attention
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest neuralops/tests/ -v
pytest neuralops/tests/ -m "not slow and not gpu"
```

## Code Style

We use:
- **Ruff** for linting
- **Black** for formatting
- **isort** for import sorting
- **mypy** for type checking

```bash
ruff check neuralops/
black neuralops/
isort neuralops/
mypy neuralops/
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write code + tests
4. Run tests: `pytest neuralops/tests/`
5. Commit: `git commit -m "feat: add my feature"`
6. Push: `git push origin feature/my-feature`
7. Open a Pull Request

## Adding a New Module

1. Create the module file in the appropriate subpackage
2. Register it with `@ATTENTION_REGISTRY.register("name")` or `@COMPRESSOR_REGISTRY.register("name")`
3. Add type hints and docstrings
4. Add a test in `neuralops/tests/`
5. Update `docs/COMPLETE_MODULE_REFERENCE.md`

## Reporting Bugs

Open an issue with:
- Python version
- PyTorch version
- Minimal reproduction script
- Full traceback

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
