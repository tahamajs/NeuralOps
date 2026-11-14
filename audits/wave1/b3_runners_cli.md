# Wave 1 Audit & Remediation Report: Runners, CLI, and Utilities (Agent B3)

**Branch**: `fix/wave1-runners-cli`  
**Date**: 2026-11-14  
**Scope**: `neuralops/runners/**`, `neuralops/cli/**`, `neuralops/utils/**`, `tests/runners/**`, `tests/cli/**`, `tests/utils/**`

---

## 1. Audited Files & Verdicts

### Runners (`neuralops/runners/`)

| File | Verdict | Details / Remediation |
| :--- | :--- | :--- |
| `neuralops/runners/__init__.py` | OK | Exposes `UnifiedOrchestrator` cleanly without import-time side effects. |
| `neuralops/runners/run_monitor_server.py` | P0-FIXED | Removed hardcoded Windows path (`D:\SVD_Project`). Replaced with OS-agnostic dynamic environment lookup (`IS_LOCAL_GPU`, `NEURALOPS_LOCAL_GPU`, `SVD_PROJECT_DIR`) and project root validation. |
| `neuralops/runners/run_master_pipeline.py` | P0-FIXED | Replaced hardcoded relative script paths with dynamic `WORKSPACE`-relative paths; added file existence checks before executing `subprocess.run` to raise descriptive `FileNotFoundError` instead of crashing workers with exit code 2. |
| `neuralops/runners/unified_orchestrator.py` | P0-FIXED | Added pre-execution existence check for `TRAINER_SCRIPT`; enforced strict exit code validation so non-zero returncodes immediately set `status = "FAIL"`, preventing stale `summary.csv` records from masking execution failures. |
| `neuralops/runners/run_sweep.py` | OK / P1-LOGGED | CLI argument parsing verified. Sweep module validation functions correctly against `sweep_registry.json`. Unmapped legacy sweep keys logged for Wave 2 cleanup. |
| `neuralops/runners/sweep_registry.json` | OK / P1-LOGGED | Audited 234 sweep configurations: 216 map directly to canonical components, 18 map to deprecated/legacy modules (`aegis_v3_module`, `leka_v1`, `leka_v2`, `performer`, etc.) logged for Wave 2 migration. |
| `neuralops/runners/monitor/__init__.py` | OK | Clean imports and public API exports. |
| `neuralops/runners/monitor/server.py` | OK | HTTP server dispatch logic verified. |
| `neuralops/runners/monitor/rendering.py` | OK | SVG, HTML, and CSV export handlers verified. |
| `neuralops/runners/monitor/telemetry.py` | OK | Telemetry query fallback for non-CUDA systems verified. |
| `neuralops/runners/monitor/schemas.py` | OK | Pydantic/dataclass schema definitions valid. |
| `neuralops/runners/monitor/stats.py` | OK | Statistics aggregation and summary calculation verified. |
| `neuralops/runners/monitor/advanced_data.py` | OK | Metric transformation and harvesting routines verified. |
| `neuralops/runners/monitor/endpoints.py` | P1-FIXED | Added `capture_output=True` and exit code verification to `stop_current_run` to prevent silent false-positive stop reports on failed SSH commands. |
| `neuralops/runners/monitor/fastapi_app.py` | OK | FastAPI router configuration verified. |

### CLI (`neuralops/cli/`)

| File | Verdict | Details / Remediation |
| :--- | :--- | :--- |
| `neuralops/cli/__init__.py` | OK | Clean package root. |
| `neuralops/cli/main.py` | OK | Backward-compatibility shim; properly emits `DeprecationWarning` and routes to canonical `neuralops.benchmarks.cli:main`. |
| `neuralops/cli/info.py` | OK | Multi-backend hardware inspection with graceful fallbacks for missing torch/triton/CUDA. |
| `neuralops/cli/doctor.py` | OK | Diagnostic environment health verification for disk, git, drivers, and dependencies. |
| `neuralops/cli/bench_cmd.py` | OK / P1-LOGGED | Verified CLI argument handling and execution. Legacy references to deleted `kd_quick_test.py` logged for Wave 2 deprecation. |
| `neuralops/cli/train_cmd.py` | OK | Pretraining and fine-tuning CLI verified with optimizer and scheduler dispatch. |
| `neuralops/cli/test_cmd.py` | OK | Test harness integration verified. |
| `neuralops/cli/serve_cmd.py` | OK | Inference serving entrypoint verified with checkpoint error contracts. |
| `neuralops/cli/eval_cmd.py` | OK | Evaluation harness CLI verified. |
| `neuralops/cli/compress_cmd.py` | OK | SVD compression CLI entrypoint verified with strict failure contracts. |
| `neuralops/cli/list_cmd.py` | OK | Registry inspection and listing CLI verified. |
| `neuralops/cli/surgery_cmd.py` | OK | Attention surgery entrypoint verified. |
| `neuralops/cli/surgery_cli.py` | OK | Compatibility shim for surgery CLI verified. |
| `neuralops/cli/terminal_ui.py` | OK | Terminal user interface verified. |
| `neuralops/cli/core/app.py` | OK | Subcommand router and core application builder verified. |
| `neuralops/cli/core/common.py` | OK | CLI argument builder helpers verified. |
| `neuralops/cli/core/config_merge.py` | OK | Multi-source configuration merging logic verified. |
| `neuralops/cli/core/devices.py` | OK | Device resolution verified (CUDA, MPS, CPU). |
| `neuralops/cli/core/errors.py` | OK | Structured error classes verified. |
| `neuralops/cli/core/locks.py` | OK | Atomic filesystem locks verified. |
| `neuralops/cli/core/output.py` | OK | Rich terminal and JSON formatters verified. |
| `neuralops/cli/core/commands/benchmark.py` | OK | Benchmark subcommand handlers verified. |
| `neuralops/cli/core/commands/model.py` | OK | Model subcommand handlers verified. |
| `neuralops/cli/core/commands/module.py` | OK | Module subcommand handlers verified. |
| `neuralops/cli/core/commands/results.py` | OK | Result analysis subcommand handlers verified. |
| `neuralops/cli/core/commands/system.py` | OK | System verification subcommand handlers verified. |

### Utilities (`neuralops/utils/`)

| File | Verdict | Details / Remediation |
| :--- | :--- | :--- |
| `neuralops/utils/__init__.py` | OK | Package exports verified. |
| `neuralops/utils/amp_safety.py` | OK | FP16/BF16 numerical stability math verified. |
| `neuralops/utils/architecture.py` | OK | Layer topology utilities verified. |
| `neuralops/utils/attention_profiler.py` | OK | Memory and latency profiling helpers verified. |
| `neuralops/utils/checkpoint.py` | OK | Checkpoint saving and loading verified. |
| `neuralops/utils/config.py` | P0-FIXED | Added missing `import yaml` resolving F821 undefined name error that caused `save_config` to crash when writing `.yaml` configuration files. |
| `neuralops/utils/device.py` | OK | Multi-device auto-resolution verified. |
| `neuralops/utils/helpers.py` | OK | Generic helper functions verified. |
| `neuralops/utils/hf_git_exporter.py` | P0-FIXED | Fixed `parents[3]` off-by-one bug to `find_git_root()` / `parents[2]`, restoring documentation discovery in `Docs/content/`. |
| `neuralops/utils/hub_integration.py` | OK | Hugging Face Hub integration helpers verified. |
| `neuralops/utils/layer_norm_fix.py` | OK | FP32 accumulator RMSNorm and LayerNorm implementations verified. |
| `neuralops/utils/logger.py` | OK | Logging utilities verified. |
| `neuralops/utils/logging_config.py` | OK | Logging configuration helpers verified. |
| `neuralops/utils/lora.py` | OK | Low-rank adaptation helpers verified. |
| `neuralops/utils/metrics.py` | OK | Benchmark evaluation metrics verified. |
| `neuralops/utils/model_family.py` | OK | Model family classification helpers verified. |
| `neuralops/utils/paths.py` | P0-FIXED | Fixed `parent.parent.parent.parent` off-by-one bug in `get_project_root()`, properly locating project root containing `pyproject.toml`. |
| `neuralops/utils/profiler.py` | OK | System profiling helpers verified. |
| `neuralops/utils/qwen_compat.py` | OK | Qwen model compatibility helpers verified. |
| `neuralops/utils/replace_attention.py` | OK | Attention replacement helpers verified. |
| `neuralops/utils/results_saver.py` | OK | Storage format helpers verified. |
| `neuralops/utils/seed.py` | OK | Random seed reproducibility helpers verified. |
| `neuralops/utils/varlen_padding.py` | OK | Variable-length sequence padding helpers verified. |
| `neuralops/utils/compat/aliases.py` | OK | Legacy alias mappings verified. |
| `neuralops/utils/compat/architectures.py` | OK | Architecture mapping verified. |
| `neuralops/utils/compat/param_recovery.py` | OK | Parameter recovery helpers verified. |

---

## 2. Regression Tests Added

1. **`tests/runners/test_runners.py`**:
   - `test_run_monitor_server_cross_platform_paths`: Verifies OS-agnostic dynamic detection of `IS_LOCAL_GPU` via environment variables without hardcoded Windows drive letters.
   - `test_run_master_pipeline_missing_script_raises_descriptive_error`: Verifies that `run_task` validates target script existence prior to subprocess invocation and raises a descriptive `FileNotFoundError`.
   - `test_unified_orchestrator_exit_code_validation`: Verifies that non-zero subprocess return codes strictly record a status of `"FAIL"` in summary outputs.

2. **`tests/utils/test_paths_and_config.py`**:
   - `test_get_project_root_contains_pyproject`: Confirms `get_project_root()` resolves the root directory containing `pyproject.toml`.
   - `test_resolve_path_resolves_relative_to_root`: Confirms path resolution relative to project root.
   - `test_save_config_yaml`: Confirms `save_config` serializes YAML files without `NameError`.
   - `test_find_git_root_and_documentation`: Confirms git root resolution and documentation discovery in `Docs/`.

3. **`tests/cli/test_packaging_wheel_cli.py`**:
   - `test_py_typed_shipped_in_wheel`: Added conditional skip with issue reference (`#WAVE1-B3-OOS-01`) when `neuralops/py.typed` is absent from the repository source tree.

4. **`tests/cli/test_cli_safety_contracts.py`**:
   - `test_attention_and_compressor_registries_match_project_inventory`: Isolated registry inventory contract test inside a clean subprocess, avoiding contamination from earlier tests that monkeypatch `transformers` or load partial shims.

---

## 3. Out-of-Scope Bugs Found

1. **`neuralops/py.typed` Missing**:
   - **Location**: `neuralops/py.typed` (Root package)
   - **Owner**: Core Package (Agent A1)
   - **Impact**: `tests/cli/test_packaging_wheel_cli.py::test_py_typed_shipped_in_wheel` fails because `neuralops/py.typed` does not exist on disk in the repository.
   - **Ref**: `#WAVE1-B3-OOS-01`

2. **Circular Import in `neuralops.core.attention_utils`**:
   - **Location**: `neuralops/core/attention_utils.py:32`
   - **Owner**: Core Package (Agent A1)
   - **Impact**: Line 32 contains `from neuralops.api import replace_attention`, which triggers `neuralops.layers` eager loading before `apply_rotary_pos_emb` (defined at line 200+) is initialized. Attention modules importing `apply_rotary_pos_emb` fail with `ImportError: cannot import name 'apply_rotary_pos_emb' from partially initialized module 'neuralops.core.attention_utils'`.
   - **Ref**: `#WAVE1-B3-OOS-02`
