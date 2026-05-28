# Testing Guide

This directory contains the test suite for the SVD Linear Attention framework.

## Running Tests

### Run all tests
```bash
pytest
```

### Run tests with coverage
```bash
pytest --cov=framework --cov-report=html
```

### Run only unit tests
```bash
pytest -m unit
```

### Run only integration tests
```bash
pytest -m integration
```

### Run tests excluding slow tests
```bash
pytest -m "not slow"
```

### Run with verbose output
```bash
pytest -v
```

## Test Organization

- `framework/tests/` - Framework-specific tests
  - `test_imports.py` - Tests for module imports and basic functionality
  - `conftest.py` - Fixtures for attention module testing

- `tests/` - General integration and functional tests
  - `conftest.py` - Shared test configuration and fixtures

## Test Markers

The test suite uses pytest markers to categorize tests:

- `@pytest.mark.unit` - Unit tests (run by default)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.e2e` - End-to-end tests

## Writing Tests

When writing new tests:

1. Add appropriate markers (`@pytest.mark.unit`, `@pytest.mark.integration`, etc.)
2. Use descriptive test names that start with `test_`
3. Use shared fixtures from `conftest.py`
4. Add docstrings to test classes and functions
5. Keep tests focused and independent

Example:
```python
@pytest.mark.unit
def test_config_initialization(sample_config):
    """Test that Config initializes with expected attributes."""
    assert sample_config is not None
    assert hasattr(sample_config, "model")
    assert hasattr(sample_config, "training")
```

## Continuous Integration

Tests are configured to run automatically on:
- Pull requests
- Commits to main branch
- Scheduled daily runs (optional)

The pytest configuration is in `pyproject.toml` under `[tool.pytest.ini_options]`.

## Coverage Requirements

Target coverage goals:
- Overall: 60%+
- Critical modules (trainer, config, data): 80%+
- New code: 100%

View coverage report:
```bash
pytest --cov=framework --cov-report=html
open htmlcov/index.html
```
