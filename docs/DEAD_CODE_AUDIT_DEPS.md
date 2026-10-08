# Dead Dependencies Audit (Round 2)

## Core Dependencies

Total defined core dependencies: 1

| Package | Import Key | Status |
| :--- | :--- | :---: |
| `packaging` | `packaging` | ✓ Used directly |

## Optional Dependencies Extras

### Group `[cuda]` (12 packages)
- `torch>=2.2.0` (✓ Used in code/tests)
- `triton>=2.3.0; sys_platform == 'linux'` (✓ Used in code/tests)
- `transformers>=4.40.0` (✓ Used in code/tests)
- `datasets>=2.18.0` (✓ Used in code/tests)
- `einops>=0.7.0` (Optional runtime/accelerator)
- `tqdm>=4.66.0` (✓ Used in code/tests)
- `pyyaml>=6.0` (Optional runtime/accelerator)
- `safetensors>=0.4.0` (Optional runtime/accelerator)
- `pandas>=2.0.0` (✓ Used in code/tests)
- `scipy>=1.11.0` (✓ Used in code/tests)
- `numpy>=1.24.0` (✓ Used in code/tests)
- `packaging>=23.0` (✓ Used in code/tests)

### Group `[rocm]` (11 packages)
- `torch>=2.2.0` (✓ Used in code/tests)
- `transformers>=4.40.0` (✓ Used in code/tests)
- `datasets>=2.18.0` (✓ Used in code/tests)
- `einops>=0.7.0` (Optional runtime/accelerator)
- `tqdm>=4.66.0` (✓ Used in code/tests)
- `pyyaml>=6.0` (Optional runtime/accelerator)
- `safetensors>=0.4.0` (Optional runtime/accelerator)
- `pandas>=2.0.0` (✓ Used in code/tests)
- `scipy>=1.11.0` (✓ Used in code/tests)
- `numpy>=1.24.0` (✓ Used in code/tests)
- `packaging>=23.0` (✓ Used in code/tests)

### Group `[mps]` (11 packages)
- `torch>=2.2.0` (✓ Used in code/tests)
- `transformers>=4.40.0` (✓ Used in code/tests)
- `datasets>=2.18.0` (✓ Used in code/tests)
- `einops>=0.7.0` (Optional runtime/accelerator)
- `tqdm>=4.66.0` (✓ Used in code/tests)
- `pyyaml>=6.0` (Optional runtime/accelerator)
- `safetensors>=0.4.0` (Optional runtime/accelerator)
- `pandas>=2.0.0` (✓ Used in code/tests)
- `scipy>=1.11.0` (✓ Used in code/tests)
- `numpy>=1.24.0` (✓ Used in code/tests)
- `packaging>=23.0` (✓ Used in code/tests)

### Group `[cpu]` (11 packages)
- `torch>=2.2.0` (✓ Used in code/tests)
- `transformers>=4.40.0` (✓ Used in code/tests)
- `datasets>=2.18.0` (✓ Used in code/tests)
- `einops>=0.7.0` (Optional runtime/accelerator)
- `tqdm>=4.66.0` (✓ Used in code/tests)
- `pyyaml>=6.0` (Optional runtime/accelerator)
- `safetensors>=0.4.0` (Optional runtime/accelerator)
- `pandas>=2.0.0` (✓ Used in code/tests)
- `scipy>=1.11.0` (✓ Used in code/tests)
- `numpy>=1.24.0` (✓ Used in code/tests)
- `packaging>=23.0` (✓ Used in code/tests)

### Group `[eval]` (3 packages)
- `scikit-learn>=1.3.0` (Optional runtime/accelerator)
- `evaluate>=0.4.0` (Optional runtime/accelerator)
- `lm-eval>=0.4.0` (Optional runtime/accelerator)

### Group `[benchmark]` (3 packages)
- `matplotlib>=3.7.0` (✓ Used in code/tests)
- `seaborn>=0.12.0` (✓ Used in code/tests)
- `pandas>=2.0.0` (✓ Used in code/tests)

### Group `[dev]` (11 packages)
- `pytest>=7.4.0` (✓ Used in code/tests)
- `pytest-cov>=4.1.0` (Optional runtime/accelerator)
- `pytest-xdist>=3.5.0` (Optional runtime/accelerator)
- `ruff>=0.4.0` (Optional runtime/accelerator)
- `black>=24.0.0` (Optional runtime/accelerator)
- `isort>=5.13.0` (Optional runtime/accelerator)
- `mypy>=1.10.0` (Optional runtime/accelerator)
- `pre-commit>=3.7.0` (Optional runtime/accelerator)
- `build>=1.2.0` (Optional runtime/accelerator)
- `twine>=5.0.0` (Optional runtime/accelerator)
- `pdoc3>=0.10.0` (Optional runtime/accelerator)

### Group `[docs]` (4 packages)
- `mkdocs>=1.5.0` (Optional runtime/accelerator)
- `mkdocs-material>=9.5.0` (Optional runtime/accelerator)
- `mkdocstrings[python]>=0.24.0` (Optional runtime/accelerator)
- `pdoc3>=0.10.0` (Optional runtime/accelerator)

### Group `[all]` (1 packages)
- `neuralops[cuda,eval,benchmark,dev,docs]` (✓ Used in code/tests)

## Recommendations

- Keep `packaging` in core dependencies as it is required for semantic version checking across kernel dispatchers.
