# AGT-02 Final Report: Removing Silent Random-Model Fallback in Training CLI (P0 / Finding F-47)

**Date**: 2026-11-07  
**Commit Baseline**: `5e84792c670461277b8687716d63b848fec34151`  
**Target File**: `neuralops/cli/train_cmd.py`  
**Test File**: `tests/test_train_cmd_fallback.py`  
**Environment**: `/Users/tahamajs/dev/Project/.venv` (Python 3.14.3)  

---

## 1. Executive Summary

In `neuralops/cli/train_cmd.py`, when a requested pretrained model failed to load via HuggingFace `AutoModelForCausalLM` or `AutoModel` (e.g., due to an invalid model name, missing local checkpoint path, or network/hub offline issues), the `_load_model()` function previously issued a logger warning and silently instantiated `_FallbackLanguageModel` — a random 4-layer `nn.TransformerEncoder` with hidden dimension 512 and vocabulary size 32,000. `cmd_train()` then executed the entire training schedule on this synthetic random model and saved it under the requested model's name.

This behavior violated repository operational rules (`AGENTS.md` §0.2: zero fake artifacts and metrics, §0.3: fail closed). We have:
1. Completely deleted `_FallbackLanguageModel` and all references to it.
2. Updated `_load_model()` to fail closed by re-raising any loader exception as an explicit `RuntimeError` (with chained causes via `from last_exc`) containing the requested model identifier, root cause details, remediation steps, and the refusal statement: `"Training will NOT continue on a random model."`
3. Verified that no `try/except` in `cmd_train()` catches or swallows this error, ensuring fast failure before any output directory or checkpoint is created.
4. Created comprehensive unit tests in `tests/test_train_cmd_fallback.py` (3 passed).
5. Documented this intentional breaking change under version `2.0.6` in `CHANGELOG.md`.

---

## 2. Checklist

- [x] `_FallbackLanguageModel` class deleted from codebase (`grep` returns 0 matches)
- [x] Model load failure raises `RuntimeError` naming model, cause, remedy, and refusal statement
- [x] No output directory or checkpoint files created upon load failure
- [x] `neuralops/cli/serve_cmd.py` fail-closed refusal pattern matched
- [x] All 3 new unit tests in `tests/test_train_cmd_fallback.py` pass (and would have failed before the fix)
- [x] No other files modified outside authorized scope (`neuralops/cli/train_cmd.py`, `tests/test_train_cmd_fallback.py`, `CHANGELOG.md`)

---

## 3. Reference Refusal Pattern in `serve_cmd.py`

In `neuralops/cli/serve_cmd.py` (lines 60–70), the existing canonical fail-closed pattern was:
```python
def _load_serving_model(model_name_or_path: str, device: str) -> Any:
    """Load the requested pretrained causal LM without substituting random weights."""
    try:
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load pretrained causal language model {model_name_or_path!r}; "
            "the serving command will not start with random placeholder weights."
        ) from exc
```

Our updated `_load_model()` in `neuralops/cli/train_cmd.py` mirrors this tone and contract:
```python
def _load_model(model_name_or_path: str, device: str) -> nn.Module:
    """Load pretrained model architecture; fails closed without random fallbacks."""
    errors: list[str] = []
    try:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        model.to(device)
        return model
    except Exception as exc_causal:
        errors.append(f"AutoModelForCausalLM: {exc_causal}")
        last_exc: Exception = exc_causal

    try:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_name_or_path)
        model.to(device)
        return model
    except Exception as exc_auto:
        errors.append(f"AutoModel: {exc_auto}")
        last_exc = exc_auto

    detail = "; ".join(errors)
    raise RuntimeError(
        f"Could not load pretrained model {model_name_or_path!r}. "
        f"Root cause: {detail}. "
        "Remedy: Check the model identifier for typos, verify the local checkpoint path exists, "
        "or verify network/HuggingFace Hub connectivity (and check HF_HUB_OFFLINE settings). "
        "Training will NOT continue on a random model."
    ) from last_exc
```

---

## 4. Verification Evidence

### 4.1 Zero `_FallbackLanguageModel` References
```bash
grep -rn "_FallbackLanguageModel" neuralops/
```
**Output**: `0 matches` (Exited with code 0).

### 4.2 Exact Refusal Sentence Presence
```bash
grep -n "will NOT continue on a random model" neuralops/cli/train_cmd.py
```
**Output**:
```text
239:        "Training will NOT continue on a random model."
```

### 4.3 Clean Module Import Check
```bash
/Users/tahamajs/dev/Project/.venv/bin/python -c "from neuralops.cli.train_cmd import _load_model; print(_load_model)"
```
**Output**:
```text
<function _load_model at 0x131625b10>
```

### 4.4 Unit Tests Execution (`tests/test_train_cmd_fallback.py`)
```bash
/Users/tahamajs/dev/Project/.venv/bin/python -m pytest tests/test_train_cmd_fallback.py -v
```
**Output**:
```text
============================== test session starts ==============================
platform darwin -- Python 3.14.3, pytest-9.1.1, pluggy-1.6.0 -- /Users/tahamajs/dev/Project/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/tahamajs/dev/Project
configfile: pytest.ini (WARNING: ignoring pytest config in pyproject.toml!)
plugins: anyio-4.11.0, cov-7.1.0, xdist-3.8.0, django-4.12.0, hydra-core-1.3.2, langsmith-0.8.5
collecting ... collecting 3 items                                                             collected 3 items                                                              

tests/test_train_cmd_fallback.py::test_load_model_failure_raises_not_random PASSED [ 33%]
tests/test_train_cmd_fallback.py::test_cmd_train_fails_fast_on_missing_checkpoint PASSED [ 66%]
tests/test_train_cmd_fallback.py::test_error_message_names_model_and_refuses_random PASSED [100%]

============================== 3 passed in 17.05s ==============================
```

### 4.5 Combined Regression Test Suite
```bash
/Users/tahamajs/dev/Project/.venv/bin/python -m pytest tests/test_packaging_wheel_cli.py tests/test_train_cmd_fallback.py -v
```
**Output**:
```text
============================== 5 passed in 53.65s ==============================
```

### 4.6 Ruff Linter
```bash
/Users/tahamajs/dev/Project/.venv/bin/ruff check neuralops/cli/train_cmd.py tests/test_train_cmd_fallback.py
```
**Output**:
```text
All checks passed!
```

---

## 5. Why Each Test Would Have Failed Before the Fix

1. **`test_load_model_failure_raises_not_random`**:
   - *Pre-fix behavior*: When `AutoModelForCausalLM` and `AutoModel` raised `OSError`, `_load_model()` caught the exceptions, logged warnings, and returned an instance of `_FallbackLanguageModel`.
   - *Why it would have failed*: The test asserts that `_load_model()` raises `RuntimeError`. Pre-fix, no exception was raised and a fallback model was returned.
   - *Post-fix behavior*: Raises `RuntimeError` immediately with `from last_exc`. **PASS**.

2. **`test_cmd_train_fails_fast_on_missing_checkpoint`**:
   - *Pre-fix behavior*: Calling `cmd_train()` with a missing checkpoint would load `_FallbackLanguageModel`, create the output directory at line 405 (`os.makedirs(save_dir, exist_ok=True)`), proceed through the training loop, and save a final checkpoint to `save_dir/final_model/pytorch_model.bin`.
   - *Why it would have failed*: The test asserts `cmd_train` raises `RuntimeError` and that `save_dir` does not exist on disk. Pre-fix, `cmd_train` would have succeeded and populated `save_dir` with checkpoints.
   - *Post-fix behavior*: `cmd_train` halts at line 356 during model load; `os.makedirs` at line 405 is never reached, leaving disk completely untouched. **PASS**.

3. **`test_error_message_names_model_and_refuses_random`**:
   - *Pre-fix behavior*: Calling `_load_model("definitely-not-a-real-model-xyz")` never raised an exception.
   - *Why it would have failed*: The test requires a `RuntimeError` containing the exact model name and `"will NOT continue on a random model"`. Pre-fix, neither the exception nor the message existed.
   - *Post-fix behavior*: The raised error explicitly formats the requested identifier and includes the refusal guarantee. **PASS**.

---

## 6. Audit of `neuralops/main.py:228,246` (Task 5 Findings)

As required by Task 5, we audited `neuralops/main.py` lines 215–260 to determine whether model loading in `run_train()` contains a similar random-model fallback:

```python
    logger.info("Loading tokeniser: %s", cfg.model_name)
    try:
        tokenizer = fw["AutoTokenizer"].from_pretrained(cfg.model_name, use_fast=True)
    except Exception:
        tokenizer = fw["AutoTokenizer"].from_pretrained(
            cfg.model_name, use_fast=True, local_files_only=True
        )
...
    logger.info("Loading model: %s  (num_labels=%d)", cfg.model_name, num_labels)
    try:
        model = fw["AutoModelForSequenceClassification"].from_pretrained(
            cfg.model_name, num_labels=num_labels
        )
    except Exception:
        model = fw["AutoModelForSequenceClassification"].from_pretrained(
            cfg.model_name, num_labels=num_labels, local_files_only=True
        )
```

**Finding**:
- `neuralops/main.py` does **NOT** instantiate a random fallback model or synthetic architecture.
- However, it performs a silent fallback from online loading to `local_files_only=True`, completely swallowing the initial exception without logging or notification. If local files are also missing, Hugging Face raises an `OSError` out of the second `from_pretrained` call.
- Per prompt instructions, `neuralops/main.py` was left untouched and reserved for AGT-06.

---

## 7. Documentation Audit (Task 7 Findings)

We searched `README.md` and all documentation in `docs/` for any language implying that training on random fallback models is supported or acceptable:
- No mentions of random fallback models or placeholder architectures for training were found.
- The only occurrences of "fallback" in documentation refer to:
  1. Kernel execution hierarchy (Level 1 CUDA -> Level 2 Triton -> Level 3 PyTorch eager).
  2. Attention mechanism causal fallbacks for decoder architectures (e.g. FlashAttention SDPA fallback).
  3. Hardware unsupported warnings (e.g. Mamba on Windows).
- Consequently, no documentation edits were necessary.

---

## 8. Git Status & Diff Summary

```text
 M CHANGELOG.md
 M neuralops/cli/train_cmd.py
?? tests/test_train_cmd_fallback.py
```
*(Combined with previous AGT-01 work, `git status --short` shows modifications to `.github/workflows/quality_gate.yml`, `CHANGELOG.md`, `neuralops/cli/train_cmd.py`, `pyproject.toml`, and untracked files `BASELINE.md`, `tests/test_packaging_wheel_cli.py`, `tests/test_train_cmd_fallback.py`).*

No commits have been made.

---

## 9. Verification Limitations

- **Physical Training Execution**: We verified `_load_model()` and `cmd_train()` via mocked unit tests and synthetic inputs. We did not download a multi-gigabyte foundation model from HuggingFace Hub or run full end-to-end epochs on GPU hardware in this static testing tier.
