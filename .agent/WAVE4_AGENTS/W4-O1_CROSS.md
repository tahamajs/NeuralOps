# Wave 4 Agent W4-O1 Cross-Path Findings

### CROSS-01: tests/conftest.py static config test whitelist
- **File:** `tests/conftest.py` (lines 231-271)
- **Issue:** The pytest collection modifier automatically adds `pytest.mark.gpu` and `pytest.mark.triton` to any test file matching `*triton*` in its path, skipping it unless it matches the hardcoded `is_static_config_test` list in `tests/conftest.py`. AST validation tests and CPU reference tests under `tests/ops/triton/` were skipped on non-CUDA runners until handled via `tests/ops/triton/conftest.py`.
- **Recommendation for Wave 5:** Add `"tests/ops/triton"` (or general unit test paths that run on CPU) to `is_static_config_test` in `tests/conftest.py`.
