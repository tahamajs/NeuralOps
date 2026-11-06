# AGT-03 Final Report: Making `trust_remote_code` Opt-In and Logged (P0 / Finding F-87)

**Date**: 2026-11-07  
**Commit Baseline**: `5e84792c670461277b8687716d63b848fec34151`  
**Target Files**: `neuralops/benchmarks/config.py`, `neuralops/benchmarks/core/model_loader.py`, `neuralops/benchmarks/cli.py`, `neuralops/benchmarks/core/runner.py`, `neuralops/export/exporter.py`  
**Test File**: `tests/benchmarks/test_trust_remote_code.py`  
**Environment**: `/Users/tahamajs/dev/Project/.venv` (Python 3.14.3)  

---

## 1. Executive Summary

Previously, NeuralOps defaulted `trust_remote_code=True` across benchmark configuration and model loading routines. This allowed arbitrary executable code inside HuggingFace repository files to execute by default during benchmark runs without explicit user consent. Furthermore, the CLI only provided an opt-out flag (`--no-trust-remote-code`), and execution manifests failed to record whether remote code execution was permitted.

This violated repository operational standards (`AGENTS.md` §15: `trust_remote_code` must be explicit and recorded).

In this task, we:
1. Changed `trust_remote_code` default from `True` to `False` in `BenchmarkConfig`, `load_model_and_tokenizer`, and across all other loader entry points in `neuralops/`.
2. Changed the CLI from opt-out to opt-in: added `--trust-remote-code` (default: False) to `neuralops run`, and retained `--no-trust-remote-code` only as a backward-compatible deprecated alias emitting a `DeprecationWarning`.
3. Injected a load-time security warning via module logger (`LOG.warning`) whenever `trust_remote_code=True` is explicitly requested, naming the model and stating `"remote code WILL be executed"`.
4. Recorded `trust_remote_code: true/false` and `model_name` directly in `manifest.json` via `BenchmarkRunner`.
5. Added explicit `trust_remote_code=False` to `load_exported_model` in `neuralops/export/exporter.py`.
6. Authored comprehensive unit and AST regression tests in `tests/benchmarks/test_trust_remote_code.py` (3 passed).
7. Documented this breaking change under version `2.0.6` in `CHANGELOG.md`.

---

## 2. Checklist

- [x] Both defaults (`BenchmarkConfig.trust_remote_code` and `load_model_and_tokenizer`) are `False`
- [x] All `from_pretrained` calls in `model_loader.py` pass `trust_remote_code` explicitly
- [x] CLI `neuralops run` is opt-in (`--trust-remote-code` added, `--no-trust-remote-code` deprecated)
- [x] Explicit `trust_remote_code=True` logs a warning naming the model and stating remote code will be executed
- [x] Run manifest (`manifest.json`) records `trust_remote_code` and `model_name`
- [x] Package-wide sweep completed; no `= True` default remains in `neuralops/`
- [x] All 3 new tests in `tests/benchmarks/test_trust_remote_code.py` pass (and would have failed before the fix)
- [x] All previous regression tests continue to pass (8 total tests across AGT-01, AGT-02, AGT-03)

---

## 3. Pre-Fix State & Location Verification

1. **Second `model_loader.py` Check**:
   - Checked `ls neuralops/benchmarks/core/model_loader.py neuralops/core/model_loader.py`.
   - `neuralops/core/model_loader.py` does not exist (`ls: neuralops/core/model_loader.py: No such file or directory`).
   - Only `neuralops/benchmarks/core/model_loader.py` exists in the codebase.
2. **Pre-fix Insecure Defaults**:
   - `neuralops/benchmarks/config.py:69`: `trust_remote_code: bool = True`
   - `neuralops/benchmarks/core/model_loader.py:77`: `trust_remote_code: bool = True,`
   - `neuralops/benchmarks/cli.py:187`: `run_p.add_argument("--no-trust-remote-code", action="store_true")`
   - `neuralops/benchmarks/cli.py:804`: `trust_remote_code=not args.no_trust_remote_code`

---

## 4. Exact Changes Made

### 4.1 `neuralops/benchmarks/config.py`
Changed line 69 from `True` to `False`:
```python
    mixed_precision: str = "no"
    trust_remote_code: bool = False
    hf_token: Optional[str] = None
```

### 4.2 `neuralops/benchmarks/core/model_loader.py`
1. Changed function signature parameter default (line 77) from `True` to `False`:
   ```python
   def load_model_and_tokenizer(
       model_name: str,
       device: str | torch.device = "cpu",
       dtype: str = "float32",
       num_labels: int | None = None,
       is_causal: bool = False,
       task: str | None = None,
       dataset: str | None = None,
       trust_remote_code: bool = False,
   ) -> tuple[nn.Module, Any, Any]:
   ```
2. Added security warning logging when `trust_remote_code=True` is explicitly passed:
   ```python
   if trust_remote_code:
       LOG.warning(
           "Explicit trust_remote_code=True specified for model '%s'; remote code WILL be executed.",
           model_name,
       )
   ```
3. Verified all 8 `from_pretrained` calls (AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel, and their CPU fallback variants) explicitly pass `trust_remote_code=trust_remote_code`.

### 4.3 `neuralops/benchmarks/cli.py`
1. Replaced opt-out argument with opt-in `--trust-remote-code` and deprecated `--no-trust-remote-code`:
   ```python
   run_p.add_argument(
       "--trust-remote-code",
       action="store_true",
       default=False,
       help="Enable execution of untrusted remote code from HuggingFace model repositories (default: False)",
   )
   run_p.add_argument(
       "--no-trust-remote-code",
       action="store_true",
       default=False,
       help="[Deprecated] Redundant alias ensuring trust_remote_code is disabled (default: False)",
   )
   ```
2. In `handle_run`, mapped flags with deprecation warning:
   ```python
   if getattr(args, "no_trust_remote_code", False):
       import warnings

       warnings.warn(
           "--no-trust-remote-code is deprecated and redundant (trust_remote_code now defaults to False).",
           DeprecationWarning,
           stacklevel=2,
       )
   trust_remote_code_val = bool(getattr(args, "trust_remote_code", False)) and not bool(getattr(args, "no_trust_remote_code", False))
   ```
   And passed `trust_remote_code=trust_remote_code_val` to `BenchmarkConfig`.

### 4.4 `neuralops/benchmarks/core/runner.py`
Recorded `trust_remote_code` and `model_name` in `manifest_data`:
```python
manifest_data: Dict[str, Any] = {
    "": "https://raw.githubusercontent.com/tahamajs/NeuralOps/main/schemas/benchmark_manifest_v1.json",
    "run_id": self.paths.run_id,
    "benchmark_name": self.config.benchmark_suite,
    "timestamp_start_utc": start_iso,
    "timestamp_end_utc": None,
    "duration_seconds": None,
    "git_provenance": git_prov,
    "host_environment": host_env,
    "model_name": getattr(self.config, "models", None) or getattr(self.config, "model", None),
    "trust_remote_code": bool(getattr(self.config, "trust_remote_code", False)),
    "execution_outcome": { ... },
}
```

### 4.5 Package-Wide Sweep
Updated all remaining loaders and defaults across the package:
- `neuralops/export/exporter.py:374`: `model = AutoModel.from_pretrained(base_name, trust_remote_code=False)`
- `neuralops/benchmarks/tasks/adapter.py:109`: `trust_remote_code=self.extra_kwargs.get("trust_remote_code", False)`
- `neuralops/benchmarks/models/loader.py:178`: `trust_remote_code: bool = False`
- `neuralops/evaluation/harness.py:77`: `trust_remote_code: bool = False`
- `neuralops/cli/core/commands/benchmark.py:226`: `trust_remote_code=bool(cfg.get("trust_remote_code", False))`
- `neuralops/benchmarks/terminal_ui.py:263` & `neuralops/cli/terminal_ui.py:268`: `trust_remote_code=False`

---

## 5. Verification Evidence

### 5.1 Verification Commands
1. **Config & Loader Defaults**:
   ```bash
   python -c "from neuralops.benchmarks.config import BenchmarkConfig; c = BenchmarkConfig(); print(c.trust_remote_code)"
   # Output: False

   python -c "import inspect; from neuralops.benchmarks.core.model_loader import load_model_and_tokenizer; print(inspect.signature(load_model_and_tokenizer).parameters['trust_remote_code'].default)"
   # Output: False
   ```

2. **No `trust_remote_code.*=.*True` Defaults in Config & Loader**:
   ```bash
   grep -rn "trust_remote_code.*=.*True" neuralops/benchmarks/config.py neuralops/benchmarks/core/model_loader.py
   # Output: Only the warning string at line 114: "Explicit trust_remote_code=True specified for model '%s'; remote code WILL be executed."
   ```

3. **CLI Opt-In Presentation**:
   ```bash
   python -m neuralops.benchmarks run --help
   ```
   Output snippet:
   ```text
     --trust-remote-code   Enable execution of untrusted remote code from
                           HuggingFace model repositories (default: False)
     --no-trust-remote-code
                           [Deprecated] Redundant alias ensuring
                           trust_remote_code is disabled (default: False)
   ```

### 5.2 Unit Tests Execution (`tests/benchmarks/test_trust_remote_code.py`)
```bash
/Users/tahamajs/dev/Project/.venv/bin/python -m pytest tests/benchmarks/test_trust_remote_code.py -v
```
Output:
```text
============================== test session starts ==============================
collected 3 items

tests/benchmarks/test_trust_remote_code.py::test_default_is_false PASSED [ 33%]
tests/benchmarks/test_trust_remote_code.py::test_true_is_logged_and_recorded PASSED [ 66%]
tests/benchmarks/test_trust_remote_code.py::test_no_from_pretrained_without_explicit_flag PASSED [100%]

============================== 3 passed in 27.17s ==============================
```

### 5.3 Combined Regression Suites (AGT-01, AGT-02, AGT-03)
```bash
/Users/tahamajs/dev/Project/.venv/bin/python -m pytest tests/test_packaging_wheel_cli.py tests/test_train_cmd_fallback.py tests/benchmarks/test_trust_remote_code.py -v
```
Output:
```text
============================== 8 passed in 52.89s ==============================
```

### 5.4 Linter Validation
```bash
/Users/tahamajs/dev/Project/.venv/bin/ruff check neuralops/benchmarks/config.py neuralops/benchmarks/core/model_loader.py neuralops/benchmarks/cli.py tests/benchmarks/test_trust_remote_code.py neuralops/benchmarks/core/runner.py neuralops/export/exporter.py
# Output: All checks passed!
```

---

## 6. Why Each Test Would Have Failed Before the Fix

1. **`test_default_is_false`**:
   - *Pre-fix*: `BenchmarkConfig().trust_remote_code` evaluated to `True`, and `load_model_and_tokenizer` signature default parameter was `True`.
   - *Failure*: Both assertions (`assert config.trust_remote_code is False` and `assert default_val is False`) would have failed immediately.
   - *Post-fix*: Both default to `False`. **PASS**.

2. **`test_true_is_logged_and_recorded`**:
   - *Pre-fix*: Pre-fix `load_model_and_tokenizer` contained no warning check or log message for `trust_remote_code=True`. Furthermore, `manifest_data` in `Runner` did not record `trust_remote_code`.
   - *Failure*: The test asserts that `LOG.warning` logs `"remote code WILL be executed"` and that `manifest.json` contains `"trust_remote_code": true`. Pre-fix, neither existed.
   - *Post-fix*: Warning logged and manifest contains `"trust_remote_code": true`. **PASS**.

3. **`test_no_from_pretrained_without_explicit_flag`**:
   - *Pre-fix*: In `neuralops/export/exporter.py`, `AutoModel.from_pretrained(base_name)` was invoked without an explicit `trust_remote_code` keyword argument.
   - *Post-fix*: All calls in `model_loader.py` and `exporter.py` now explicitly specify `trust_remote_code`. **PASS**.

---

## 7. Post-Fix Package-Wide Grep Audit

Running `grep -rn "trust_remote_code" neuralops/` post-fix confirms that every configuration and loader default is `False`:
- `neuralops/config/model_config.py:23`: `trust_remote_code: bool = False`
- `neuralops/recipe/schema.py:108`: `trust_remote_code: Optional[bool] = None`
- `neuralops/cli/terminal_ui.py:268`: `trust_remote_code=False`
- `neuralops/cli/core/commands/benchmark.py:226`: `trust_remote_code=bool(cfg.get("trust_remote_code", False))`
- `neuralops/benchmarks/terminal_ui.py:263`: `trust_remote_code=False`
- `neuralops/benchmarks/tasks/adapter.py:109`: `trust_remote_code=self.extra_kwargs.get("trust_remote_code", False)`
- `neuralops/benchmarks/config.py:69`: `trust_remote_code: bool = False`
- `neuralops/benchmarks/core/runner.py:150`: `"trust_remote_code": bool(getattr(self.config, "trust_remote_code", False))`
- `neuralops/benchmarks/core/model_loader.py:77`: `trust_remote_code: bool = False`
- `neuralops/benchmarks/models/loader.py:178`: `trust_remote_code: bool = False`
- `neuralops/evaluation/harness.py:77`: `trust_remote_code: bool = False`
- `neuralops/export/exporter.py:374`: `AutoModel.from_pretrained(base_name, trust_remote_code=False)`

*(Remaining references passing `True` in test harnesses or specialized scripts, such as `evaluation/niah.py` or `export/runtime_compat.py`, are explicit argument call sites, not library defaults).*

---

## 8. Documentation Audit

A search across `README.md` and `docs/` found no sentences presenting `trust_remote_code=True` as the default. References in `docs/recipe-format.md` state it is an optional boolean flag, and examples in `docs/roadmap/` show explicit opt-in parameters. No documentation edits were required.

---

## 9. Verification Limitations

- **Live Remote Repository Loading**: Verification utilized unit mocks and AST inspection. We did not download an untrusted model containing custom execution code from Hugging Face Hub during this test run.
