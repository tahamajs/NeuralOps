# AGT-05 Completion Report: Dispatch Auto-Mode and Missing PyTorch Fallback Guard

**Task:** AGT-05 (P0 — Findings F-27, F-28)
**Date:** 2026-11-07
**Environment:** Existing project venv (`/Users/tahamajs/dev/Project/.venv`, Python 3.14.3)
**Status:** Complete & Verified

---

## 1. Executive Summary

This task resolved two critical dispatch and fallback bugs in `neuralops/ops/cuda/_fallback.py`:
1. **Auto-Mode Resolution Bug (F-27):** Previously, `if backend == "auto": backend = _FORCED_BACKEND.get()` left `backend = "auto"` because the default ContextVar value was `"auto"`. Consequently, `if backend == "cuda"` was `False`, CUDA was never attempted in `auto` mode, and execution jumped straight to Triton, causing an unexpected failure on platforms without Triton. We introduced a centralized `_resolve_backend()` function that properly resolves `"auto"` to `"cuda"` (or the forced backend if non-default) to guarantee the 3-tier Level 1 (CUDA) -> Level 2 (Triton) -> Level 3 (PyTorch) fallback order.
2. **Missing PyTorch Fallback Level and Guards (F-28):** All 18 `dispatch_*` helpers (`dispatch_direct_svd`, `dispatch_asvd`, `dispatch_fwsvd`, `dispatch_hnc_svd`, `dispatch_awq`, `dispatch_fp8_compressor`, `dispatch_kmeans_quantize`, `dispatch_wanda`, `dispatch_sparsegpt`, `dispatch_admm`, `dispatch_routing_attention`, `dispatch_lsh_attention`, `dispatch_cross_attention`, `dispatch_adaptive_sparse`, `dispatch_dynamic_conv`, `dispatch_token_compressed`, `dispatch_chela_slay`, `dispatch_rala`) only implemented CUDA -> Triton and crashed when executed on CPU without Triton. Furthermore, `dispatch()` and the helpers had no guard for `pytorch_fn is None`, resulting in raw `TypeError: 'NoneType' object is not callable` crashes. We added `pytorch_fn: Any = None` to all 18 helpers, wired Level 3 PyTorch fallback execution, and added explicit `RuntimeError(f"No PyTorch fallback supplied for kernel '{name}'; CUDA and Triton both unavailable")` guards across `dispatch()` and all 18 helpers.

---

## 2. Files Changed and Created

### Files Modified:
- `/Users/tahamajs/dev/Project/neuralops/ops/cuda/_fallback.py`:
  - Added `_resolve_backend(backend: str = "auto") -> str`.
  - Added `if pytorch_fn is None` guard in `dispatch()` immediately after the argument aliasing block.
  - Updated all 18 `dispatch_*` helpers with `pytorch_fn: Any = None`, `_resolve_backend()`, Triton availability guards, and fail-closed PyTorch fallback logic.
- `/Users/tahamajs/dev/Project/CHANGELOG.md`:
  - Added entry under `[2.0.6] -> Fixed` documenting the dispatch auto-mode fix and the backward-compatible `pytorch_fn` parameter addition to all 18 dispatch helpers.

### Files Created:
- `/Users/tahamajs/dev/Project/tests/test_ops_dispatch.py`:
  - Contains all 4 required unit tests verifying auto-mode CUDA-first dispatch, CPU PyTorch fallback, helper missing-fallback RuntimeError, and `dispatch()` missing-fallback RuntimeError.

---

## 3. Auto-Resolution Code: Before vs. After

### Before:
```python
# In each of the 18 helpers:
def dispatch_<name>(*args: Any, backend: str = "auto", **kwargs: Any) -> Any:
    if backend == "auto":
        backend = _FORCED_BACKEND.get()
    if backend == "cuda" and _<name>_cuda_available():
        try:
            from neuralops.ops.cuda.<mod> import <cuda_fn>
            return <cuda_fn>(*args, **kwargs)
        except Exception as e:
            logger.warning(f"<name> CUDA failed: {e}. Falling back to Triton.")
            # fall through to Triton
    from neuralops.ops.triton.<mod> import <triton_fn> as _triton_fwd
    return _triton_fwd(*args, **kwargs)
```
*Bug:* Because `_FORCED_BACKEND.get()` returns `"auto"`, `backend == "cuda"` evaluates to `False`. CUDA is never tried. Furthermore, there is no PyTorch fallback level or `pytorch_fn` parameter.

### After:
```python
# Shared Resolver:
def _resolve_backend(backend: str = "auto") -> str:
    """Resolve backend selection respecting _FORCED_BACKEND context.

    If backend is 'auto', checks _FORCED_BACKEND. If forced is not 'auto',
    returns the forced backend. Otherwise resolves 'auto' to 'cuda' so that
    the Level 1 CUDA implementation is attempted first in the fallback chain.
    """
    if backend == "auto":
        forced = _FORCED_BACKEND.get()
        if forced != "auto":
            return forced
        return "cuda"
    return backend

# In each of the 18 helpers:
def dispatch_<name>(*args: Any, pytorch_fn: Any = None, backend: str = "auto", **kwargs: Any) -> Any:
    target_backend = _resolve_backend(backend)
    is_auto = (backend == "auto" and _FORCED_BACKEND.get() == "auto")
    if target_backend == "cuda":
        if _<name>_cuda_available():
            try:
                from neuralops.ops.cuda.<mod> import <cuda_fn>
                return <cuda_fn>(*args, **kwargs)
            except (ImportError, RuntimeError) as e:
                logger.warning(f"[<name>] CUDA failed: {e}. Falling back to Triton.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[<name>] Backend forced to 'cuda', but no CUDA implementation is available.")
        if is_auto:
            target_backend = "triton"
    if target_backend == "triton":
        if is_kernel_available("<name>", backend="triton"):
            try:
                from neuralops.ops.triton.<mod> import <triton_fn> as _triton_fwd
                return _triton_fwd(*args, **kwargs)
            except (ImportError, AttributeError) as e:
                logger.warning(f"[<name>] Triton failed: {e}. Falling back to PyTorch.")
                if not is_auto:
                    raise
        elif not is_auto:
            raise RuntimeError("[<name>] Backend forced to 'triton', but Triton is not available.")
        if is_auto:
            target_backend = "pytorch"
    if pytorch_fn is None:
        raise RuntimeError(
            "No PyTorch fallback supplied for kernel '<name>'; "
            "CUDA and Triton both unavailable"
        )
    return pytorch_fn(*args, **kwargs)
```

---

## 4. Verification Checklist

- [x] `auto` mode tries CUDA first.
- [x] All 18 helpers accept `pytorch_fn` (with default `None`).
- [x] All 18 helpers fall back to PyTorch on CPU when `pytorch_fn` is supplied.
- [x] No `pytorch_fn` call site in `_fallback.py` is unguarded.
- [x] Missing fallback raises `RuntimeError` explicitly naming the kernel (never `TypeError` or raw `ImportError`).
- [x] All new tests in `tests/test_ops_dispatch.py` pass.

---

## 5. Verification Evidence

### 5.1 Pytest Execution (`tests/test_ops_dispatch.py`)
```bash
.venv/bin/python -m pytest tests/test_ops_dispatch.py -v
```
**Output:**
```text
============================= test session starts ==============================
platform darwin -- Python 3.14.3, pytest-9.1.1, pluggy-1.6.0 -- /Users/tahamajs/dev/Project/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/tahamajs/dev/Project
configfile: pytest.ini (WARNING: ignoring pytest config in pyproject.toml!)
plugins: anyio-4.11.0, cov-7.1.0, xdist-3.8.0, django-4.12.0, hydra-core-1.3.2, langsmith-0.8.5
collecting ... collected 4 items

tests/test_ops_dispatch.py::test_dispatch_helpers_auto_mode_tries_cuda_first PASSED [ 25%]
tests/test_ops_dispatch.py::test_dispatch_helpers_fall_back_to_pytorch_on_cpu PASSED [ 50%]
tests/test_ops_dispatch.py::test_dispatch_helpers_without_pytorch_fn_raise_runtime_error PASSED [ 75%]
tests/test_ops_dispatch.py::test_dispatch_without_pytorch_fn_raises_runtime_error PASSED [100%]

============================== 4 passed in 0.25s ===============================
```

### 5.2 Ruff Linter Check
```bash
.venv/bin/python -m ruff check neuralops/ops/cuda/_fallback.py tests/test_ops_dispatch.py
```
**Output:**
```text
All checks passed!
```

### 5.3 Fallback Guard Count (`pytorch_fn is None`)
```bash
grep -n "pytorch_fn is None" neuralops/ops/cuda/_fallback.py
```
**Output:**
```text
252:    if pytorch_fn is None:
483:    if pytorch_fn is None:
523:    if pytorch_fn is None:
563:    if pytorch_fn is None:
603:    if pytorch_fn is None:
643:    if pytorch_fn is None:
683:    if pytorch_fn is None:
723:    if pytorch_fn is None:
763:    if pytorch_fn is None:
803:    if pytorch_fn is None:
843:    if pytorch_fn is None:
883:    if pytorch_fn is None:
923:    if pytorch_fn is None:
963:    if pytorch_fn is None:
1003:    if pytorch_fn is None:
1043:    if pytorch_fn is None:
1083:    if pytorch_fn is None:
1123:    if pytorch_fn is None:
1163:    if pytorch_fn is None:
```
*Count:* Exactly 19 guard sites (1 in `dispatch()` and 18 in the 18 helpers).

### 5.4 Import Smoke Check
```bash
.venv/bin/python -c "from neuralops.ops.cuda._fallback import dispatch, dispatch_direct_svd, dispatch_wanda; print('imports ok')"
```
**Output:**
```text
imports ok
```

### 5.5 CPU PyTorch Fallback One-Liner Execution
```bash
.venv/bin/python -c "from neuralops.ops.cuda._fallback import dispatch_wanda; print(dispatch_wanda(pytorch_fn=lambda *a, **k: 'pytorch-result'))"
```
**Output:**
```text
pytorch-result
```

### 5.6 Missing PyTorch Fallback Guard Validation
```bash
.venv/bin/python -c "from neuralops.ops.cuda._fallback import dispatch_wanda; dispatch_wanda()"
```
**Output:**
```text
Traceback (most recent call last):
  ...
RuntimeError: No PyTorch fallback supplied for kernel 'wanda'; CUDA and Triton both unavailable
```

---

## 6. Git Status and Diff Statistics

### `git diff --stat` (Task Scope)
```text
 CHANGELOG.md                    |   4 +
 neuralops/ops/cuda/_fallback.py | 872 ++++++++++++++++++++++++++++++----------
 2 files changed, 660 insertions(+), 216 deletions(-)
```

### `git diff --stat` (Working Tree)
```text
 CHANGELOG.md                      |   4 +
 README.md                         |   4 +-
 neuralops/api.py                  |  17 +-
 neuralops/benchmarks/cli.py       |  23 +-
 neuralops/benchmarks/config.py    |  24 +-
 neuralops/cli/serve_cmd.py        |  22 +-
 neuralops/integrations/serving.py |  58 ++-
 neuralops/ops/cuda/_fallback.py   | 872 ++++++++++++++++++++++++++++----------
 neuralops/ui/cli.py               |  46 +-
 neuralops/ui/server.py            |   2 +-
 10 files changed, 821 insertions(+), 251 deletions(-)
```

### `git status --short`
```text
 M CHANGELOG.md
 M README.md
 M neuralops/api.py
 M neuralops/benchmarks/cli.py
 M neuralops/benchmarks/config.py
 M neuralops/cli/serve_cmd.py
 M neuralops/integrations/serving.py
 M neuralops/ops/cuda/_fallback.py
 M neuralops/ui/cli.py
 M neuralops/ui/server.py
?? .agents/DECISIONS.md
?? .agents/agt-prompts/
?? .agents/agt-reports/
?? tests/benchmarks/test_hf_token_secret.py
?? tests/test_ops_dispatch.py
?? tests/test_serve_security.py
?? tests/ui/test_ui_cors.py
```
