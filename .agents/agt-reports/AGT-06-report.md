# AGT-06 Completion Report: Elimination of Silent Error Swallowing in Data-Producing Paths

**Task:** AGT-06 (P0 — Finding F-84)  
**Date:** 2026-11-07  
**Environment:** Existing project venv (`/Users/tahamajs/dev/Project/.venv`, Python 3.14.3)  
**Status:** Complete & Verified  

---

## 1. Executive Summary

This task eliminated silent error swallowing from the P0 data-producing paths in NeuralOps:
1. **`run_eval` (in `neuralops/main.py:378-381`):** Previously, a checkpoint-load failure only issued a warning and silently evaluated the base model instead, reporting metrics for the wrong model. Now, it fails closed by logging an error and raising a `RuntimeError` containing the checkpoint path and root cause, never substituting the base model.
2. **`run_compress` (in `neuralops/main.py:500-501`):** Previously, an evaluation failure on a compressed model only issued a warning and continued. Now, it logs an error and records an explicit `status="failed"` (along with `"error": str(e)`) in the returned stats and persisted results manifest.
3. **Metadata Registration (in `neuralops/api.py:163-169`):** Previously, failures to register `_neuralops_metadata` and `neuralops_attention` on model config were silently swallowed with `except Exception: pass`. Now, any failure logs an error naming the model and re-raises to prevent loss of surgery history.
4. **`load_exported_model` (in `neuralops/export/exporter.py`):** Previously, a failure to replay a surgery manifest warned and fell back to a legacy replay, while a safetensors weight-loading failure warned and returned a model with unloaded weights. Now, both failure modes log errors and fail closed by raising `RuntimeError`.

---

## 2. Files Changed and Created

### Files Modified:
- `/Users/tahamajs/dev/Project/neuralops/main.py`:
  - `run_eval` (lines 376-383): Replaced base-model fallback with logging + `raise RuntimeError(...)`.
  - `run_compress` (lines 474-526): Added explicit `status="failed"` and `"error"` recording on eval failure.
- `/Users/tahamajs/dev/Project/neuralops/api.py`:
  - `replace_attention` (lines 146-177): Removed `except Exception: pass`, replaced with logging + re-raise on metadata/config registration failure.
- `/Users/tahamajs/dev/Project/neuralops/export/exporter.py`:
  - `load_exported_model`: Manifest replay failure raises `RuntimeError`; safetensors loading failure raises `RuntimeError`.
- `/Users/tahamajs/dev/Project/.github/workflows/quality_gate.yml`:
  - Added workflow step `Verify No Silent Swallowing in Fixed Paths` executing `python scripts/utils/check_no_silent_swallow.py`.
- `/Users/tahamajs/dev/Project/CHANGELOG.md`:
  - Documented breaking change under `[2.0.6] -> Changed`.

### Files Created:
- `/Users/tahamajs/dev/Project/tests/test_no_silent_swallow.py`:
  - Contains regression tests: `test_no_bare_except_pass_in_fixed_files`, `test_run_eval_fails_on_checkpoint_failure`, and `test_load_exported_model_fails_on_weight_failure`.
- `/Users/tahamajs/dev/Project/scripts/utils/check_no_silent_swallow.py`:
  - Static audit script enforcing no `except Exception: pass` or bare `except:` in fixed paths.

---

## 3. Before vs. After Code Shape (Fixed Sites)

1. **`neuralops/main.py:378-381` (`run_eval`):**
   - *Was:* swallow checkpoint failure and fall back to base model:
     ```python
     except Exception:
         logger.warning("Could not load from %s, falling back to base model", model_path)
         model = fw["AutoModelForSequenceClassification"].from_pretrained(cfg.model_name, num_labels=num_labels)
     ```
   - *Now:* fail closed and raise:
     ```python
     except Exception as e:
         logger.error("Could not load checkpoint from %s: %s", model_path, e)
         raise RuntimeError(f"Failed to load checkpoint from {model_path}: {e}") from e
     ```
2. **`neuralops/main.py:500-501` (`run_compress`):**
   - *Was:* warn and continue with zeroed metrics:
     ```python
     except Exception as e:
         logger.warning("Could not evaluate compressed model on validation set: %s", e)
     ```
   - *Now:* record explicit `status="failed"` in stats and results manifest:
     ```python
     except Exception as e:
         logger.error("Could not evaluate compressed model on validation set: %s", e)
         status = "failed"
         eval_error = str(e)
     ```
3. **`neuralops/api.py:163-170` (`replace_attention`):**
   - *Was:* silent swallow with `pass`:
     ```python
     try:
         out_model.config.neuralops_attention = attn_name
     except Exception:
         pass
     ```
   - *Now:* log error naming model and re-raise:
     ```python
     try:
         out_model.config.neuralops_attention = attn_name
     except Exception as e:
         logger.error("Failed to set neuralops_attention on model config for %s: %s", type(out_model).__name__, e)
         raise
     ```
4. **`neuralops/export/exporter.py:392-393` (Manifest Replay in `load_exported_model`):**
   - *Was:* warn and fall back to legacy replay:
     ```python
     except Exception as e:
         logger.warning("Failed to replay surgery manifest, falling back to legacy replay: %s", e)
     ```
   - *Now:* log error and raise:
     ```python
     except Exception as e:
         logger.error("Failed to replay surgery manifest from '%s': %s", surgery_manifest_file, e)
         raise RuntimeError(f"Failed to replay surgery manifest from {surgery_manifest_file}: {e}") from e
     ```
5. **`neuralops/export/exporter.py:424-425` (Safetensors Load in `load_exported_model`):**
   - *Was:* warn and continue with unloaded weights:
     ```python
     except Exception as err:
         logger.warning("Could not load safetensors file: %s", err)
     ```
   - *Now:* log error and raise:
     ```python
     except Exception as err:
         logger.error("Could not load safetensors file '%s': %s", safetensors_path, err)
         raise RuntimeError(f"Failed to load weights from {safetensors_path}: {err}") from err
     ```

---

## 4. Deferred Swallow Sites (Later Phase)

The following non-P0 sites were inspected per Task 3 instructions and left unmodified for a later phase:
- `neuralops/api.py:459,495,508,540,659,663,669,672` (decode/tokenize failures in `TextGenerationPipeline` falling back to `str(seq.tolist())` as "decoded text"):
  - *Swallows:* Decoding errors in text generation pipeline fallback.
  - *Correct Fix:* Propagate tokenizer decode exceptions or flag token decode failure explicitly in output dictionary.
- `neuralops/api.py:753-755` (`list_attentions(include_extensions=True)` dropping FLA listing errors):
  - *Swallows:* `ImportError` / extension loading errors when listing FLA modules.
  - *Correct Fix:* Narrow exception to `ImportError` and log diagnostic warning detailing missing optional FLA dependencies.
- `neuralops/api.py:1094-1097` (parallel-healing failure falling back to single-device healing):
  - *Swallows:* Parallel accelerator initialization failure.
  - *Correct Fix:* Log warning and record fallback telemetry in healing manifest or fail closed if multi-GPU execution was explicitly requested.
- `neuralops/main.py:228,246` (tokenizer/model load failure silently retried with `local_files_only=True`):
  - *Swallows:* Online huggingface hub connection/authentication failures.
  - *Correct Fix:* Catch specific network/offline errors (`OSError`, `requests.exceptions.RequestException`) rather than generic `Exception`.
- `neuralops/__init__.py:292,301,315,441,484,513` (alias-finder and `_patch_torch_compat` swallowing everything):
  - *Swallows:* Module discovery errors and monkeypatch errors.
  - *Correct Fix:* Replace bare `except Exception: pass` with targeted `AttributeError` / `ImportError` guards.
- `neuralops/_optional.py:50` (`_require_torch` swallowing around `_patch_torch_compat()`):
  - *Swallows:* Compatibility patching errors when importing torch.
  - *Correct Fix:* Narrow exception or emit debug log.
- `neuralops/compression/__init__.py:67` (`__dir__` swallowing):
  - *Swallows:* Registry introspection errors during attribute autocomplete.
  - *Correct Fix:* Narrow exception to `AttributeError` or log debug message.
- `neuralops/cli/main.py:51,58,65,72,79,86,94,102,128,140` (subcommand-registration `try/except` blocks):
  - *Swallows:* Missing CLI module dependencies.
  - *Correct Fix:* Narrow to `ImportError` and log debug info when optional command dependencies are absent.

---

## 5. Test Results & Before/After Proof

### Test Execution:
```bash
.venv/bin/python -m pytest tests/test_no_silent_swallow.py -v
```
**Output:**
```text
============================= test session starts ==============================
platform darwin -- Python 3.14.3, pytest-9.1.1, pluggy-1.6.0 -- /Users/tahamajs/dev/Project/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/tahamajs/dev/Project
configfile: pytest.ini (WARNING: ignoring pytest config in pyproject.toml!)
plugins: anyio-4.11.0, cov-7.1.0, xdist-3.8.0, django-4.12.0, hydra-core-1.3.2, langsmith-0.8.5
collecting ... collected 3 items

tests/test_no_silent_swallow.py::test_no_bare_except_pass_in_fixed_files PASSED [ 33%]
tests/test_no_silent_swallow.py::test_run_eval_fails_on_checkpoint_failure PASSED [ 66%]
tests/test_no_silent_swallow.py::test_load_exported_model_fails_on_weight_failure PASSED [100%]

============================== 3 passed in 4.23s ===============================
```

### Why Each Test Failed Before the Fix:
1. `test_no_bare_except_pass_in_fixed_files`: Failed with `Failed: Silent 'except Exception: pass' found in replace_attention at line 164`. The original code used `try: ... except Exception: pass` to ignore metadata/config attribute setting errors.
2. `test_run_eval_fails_on_checkpoint_failure`: Failed with `Failed: DID NOT RAISE RuntimeError`. When `AutoModelForSequenceClassification.from_pretrained` raised on a corrupted checkpoint, `run_eval` caught it and silently loaded `cfg.model_name`, proceeding to evaluate the base model.
3. `test_load_exported_model_fails_on_weight_failure`: Failed with `Failed: DID NOT RAISE RuntimeError`. When `safetensors.torch.load_file` raised a `RuntimeError("Header truncated")`, `load_exported_model` only issued a warning and returned the model without loaded weights.

---

## 6. Verification Evidence

### 6.1 Check Script Execution
```bash
.venv/bin/python scripts/utils/check_no_silent_swallow.py; echo "exit=$?"
```
**Output:**
```text
PASS: No silent swallowing detected in fixed data-producing paths.
exit=0
```

### 6.2 Grep for `except Exception: pass` in Fixed Files
```bash
grep -rn "except Exception: pass" neuralops/api.py neuralops/main.py neuralops/export/exporter.py
```
**Output:**
```text
(0 results, exit code 1)
```

### 6.3 Ruff Linter Check
```bash
.venv/bin/python -m ruff check neuralops/api.py neuralops/main.py neuralops/export/exporter.py tests/test_no_silent_swallow.py scripts/utils/check_no_silent_swallow.py
```
**Output:**
```text
All checks passed!
```

### 6.4 Import Smoke Check
```bash
.venv/bin/python -c "import neuralops.api, neuralops.export.exporter; print('imports ok')"
```
**Output:**
```text
imports ok
```

---

## 7. What Could NOT Be Verified

- **Real Checkpoint Eval with Weights:** Verification was performed using mock checkpoints and unit tests. Physical evaluation against a live Hugging Face checkpoint on GPU was not executed because the local macOS environment lacks GPU acceleration and network model downloads were avoided per test constraints.

---

## 8. Verification Checklist

- [x] `run_eval` raises on checkpoint failure and never evaluates the base model.
- [x] Compressed-model eval failure returns explicit `status="failed"` and records `"error"`.
- [x] Metadata-registration failure logs error naming the model and re-raises.
- [x] `load_exported_model` raises on safetensors weight failure and manifest replay failure.
- [x] No new `except Exception: pass` added.
- [x] All new tests in `tests/test_no_silent_swallow.py` pass.
- [x] Deferred swallow sites listed verbatim and left unmodified.

---

## 9. Git Status and Diff Statistics

### `git diff --stat` (Task Scope)
```text
 .github/workflows/quality_gate.yml |  4 ++++
 CHANGELOG.md                       |  5 +++++
 neuralops/api.py                   | 37 ++++++++++++++++++++++++++++---------
 neuralops/export/exporter.py       |  6 ++++--
 neuralops/main.py                  | 17 +++++++++++------
 5 files changed, 52 insertions(+), 17 deletions(-)
```

### `git status --short`
```text
 M .github/workflows/quality_gate.yml
 M CHANGELOG.md
 M README.md
 M neuralops/api.py
 M neuralops/benchmarks/cli.py
 M neuralops/benchmarks/config.py
 M neuralops/cli/serve_cmd.py
 M neuralops/export/exporter.py
 M neuralops/integrations/serving.py
 M neuralops/main.py
 M neuralops/ops/cuda/_fallback.py
 M neuralops/ui/cli.py
 M neuralops/ui/server.py
?? .agents/DECISIONS.md
?? .agents/agt-prompts/
?? .agents/agt-reports/
?? scripts/utils/check_no_silent_swallow.py
?? tests/benchmarks/test_hf_token_secret.py
?? tests/test_no_silent_swallow.py
?? tests/test_ops_dispatch.py
?? tests/test_serve_security.py
?? tests/ui/test_ui_cors.py
```
