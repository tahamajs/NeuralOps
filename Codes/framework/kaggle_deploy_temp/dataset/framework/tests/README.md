# Testing the Efficient Transformer Toolkit

This directory contains a **comprehensive, automated test suite** for every component of the framework.  
It validates **all registered attention modules** (80+ variants), the injection pipeline, registry, and serialisation — and integrates with GitHub Actions for continuous integration.

---

## Quick Start

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-xdist

# Run all tests
pytest

# Run only attention module tests
pytest tests/ -m attention

# Run with coverage report
pytest --cov=framework --cov-report=html

# Skip slow / GPU tests
pytest -m "not slow and not gpu"

# Run a specific test file
pytest tests/test_attention_modules.py -v
```

---

## Test Structure

| File | Purpose |
|------|---------|
| `conftest.py` | Shared fixtures (test inputs, masks, pretrained weights, registry) |
| `test_attention_modules.py` | Core tests for **every** registered attention variant |
| `test_attention_registry.py` | Validates the ATTENTION_REGISTRY itself |
| `test_attention_injection.py` | Tests the injection pipeline on a real DistilBERT model |
| `test_hub_serialization.py` | Tests save/load round‑trip via `UnifiedTransformerModel` |
| `README.md` | This file |

---

## What Is Tested

### 1. Shape Correctness  
Every attention module must produce an output of shape `(B, L, D)` for a standard forward pass.

### 2. Output Finiteness  
No NaN or Inf values in the output tensor.

### 3. Zero‑Shot Fidelity  
When pretrained DistilBERT weights are copied into a custom attention module, the output must match the original within a generous tolerance (mean absolute difference < 0.5).

### 4. Gradient Flow  
Gradients must flow back to at least one trainable parameter in the module.

### 5. Padding Mask  
Positions masked by a 2D binary padding mask (1 = keep, 0 = pad) must produce near‑zero output at those positions.

### 6. Multi‑Head Compatibility  
Reshaping operations must not raise errors; the module must correctly handle the split into heads.

### 7. Return Format  
Every module must return a tuple `(output, attention_weights_or_None)`.

### 8. Registry Integrity  
All registry keys are strings, all values are callable, and the registry is not empty.

### 9. Injection Pipeline  
Injecting a known attention type into a real DistilBERT model must succeed and produce correct shapes.

### 10. HuggingFace Hub Serialization  
A model with custom attention can be saved with `save_pretrained`, loaded back, and the custom attention is correctly restored.

---

## Fixtures (defined in `conftest.py`)

| Fixture | Description |
|---------|-------------|
| `hidden_states` | Random (2, 32, 768) tensor, fixed seed |
| `small_hidden_states` | Random (2, 8, 768) tensor (for quick tests) |
| `attention_mask` | Binary 2D mask (2, 32) – last 8 tokens are padding |
| `small_attention_mask` | Binary 2D mask (2, 8) – last 2 tokens are padding |
| `model_config` | Standard `DistilBertConfig` (768 hidden, 12 heads) |
| `distilbert_attention` | The first layer’s attention module from a real DistilBERT |
| `registered_attention_keys` | List of all keys in `ATTENTION_REGISTRY` |

Helper functions `build_attention(key, config, pretrained_attn)` and `get_testable_attention_keys(config, pretrained_attn)` are also available for writing custom tests.

---

## Markers

Use markers to filter tests:

| Marker | Description |
|--------|-------------|
| `attention` | All attention module tests |
| `zero_shot` | Zero‑shot fidelity checks |
| `grad` | Gradient flow tests |
| `slow` | Tests that take a long time |
| `gpu` | Tests that require a GPU |
| `flash_attn` | Tests requiring `flash_attn` |
| `integration` | Tests that load real models |

**Examples**:

```bash
pytest -m "attention and not slow"   # fast attention tests only
pytest -m grad                        # gradient tests only
pytest -m gpu                         # GPU tests only (needs CUDA)
```

---

## Writing New Tests

1. Import the fixtures from `conftest.py` (they are automatically available to all test files).
2. Use `build_attention(key, config)` to instantiate any attention module by its registry key.
3. Add your test function to an existing test file (or create a new one).
4. Mark it with the appropriate marker(s).
5. Run the test to verify.

**Example**:

```python
@pytest.mark.attention
def test_my_custom_check(key, small_hidden_states, model_config):
    attn = build_attention(key, model_config)
    out = attn(small_hidden_states)
    if isinstance(out, tuple):
        out = out[0]
    # your assertion here
    assert out.shape == small_hidden_states.shape
```

---

## Continuous Integration

GitHub Actions workflows automatically run on every push and pull request to `main` / `develop`.

### `ci.yml` (CPU)

- Matrix build for Python 3.10, 3.11, 3.12
- Installs all dependencies, caches HuggingFace models, runs full test suite with coverage

### `ci-gpu.yml` (GPU – self‑hosted runner)

- Runs GPU‑marked tests on a self‑hosted machine with CUDA
- Only triggers when GPU runner is available

Coverage is uploaded to Codecov (or available locally via `htmlcov/index.html`).

---

## Configuration Files

- `pytest.ini` – test discovery, markers, coverage settings
- `.coveragerc` – coverage exclusions, report precision
- `.github/workflows/ci.yml` – CPU CI workflow
- `.github/workflows/ci-gpu.yml` – GPU CI workflow

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- Transformers (HuggingFace)
- pytest, pytest‑cov, pytest‑xdist (optional for parallel execution)

Install them with:

```bash
pip install -e ".[test]"
```

or manually:

```bash
pip install pytest pytest-cov pytest-xdist torch transformers datasets
```

---

## Coverage Report

After running `pytest --cov`, you can view a line‑by‑line report:

```bash
open htmlcov/index.html   # macOS
xdg-open htmlcov/index.html  # Linux
```

Coverage thresholds are set in `pytest.ini` (`cov-fail-under`) — currently 0, but can be raised as modules mature.

---
