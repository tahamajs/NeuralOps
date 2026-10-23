# NeuralOps v1.6.0 Multi-Agent Ownership Matrix

This document defines the strict, non-overlapping ownership matrix for the NeuralOps v1.6.0 overhaul.
No agent may edit files outside its declared ownership boundary.

## Wave 1 — Foundation (Completed)
- **Agent A (Foundation Architect)**
  - Branch: `refactor/v1.6.0-cpu-stability` (commits `633924c713`, `a3d0c433a2`, `8095c5da0d`)
  - Owned Files:
    - `.gitignore`
    - `neuralops/core/cuda_kernels/__init__.py`, `neuralops/core/cuda_kernels/_map.py`
    - `tests/test_cuda_kernels_lazy.py`
    - `neuralops/_lazy.py`, `neuralops/_optional.py`, `neuralops/core/errors.py`
    - `neuralops/core/triton_kernels/_base.py`, `neuralops/core/registry.py`
    - `neuralops/ops/`, `neuralops/layers/`, `neuralops/compression/`
    - `neuralops/backward_compat/*`
    - `tests/test_lazy_architecture.py`
    - `CITATION.cff`, `neuralops/__init__.py`
  - Audit IDs: P001, P002, P003, P004, P005, P011, P019

---

## Wave 2 — Parallel Independent Work (5 Concurrent Agents)

### Agent B — Test Infrastructure
- **Branch**: `agent/b-wave2-test-infra`
- **Audit IDs**: P023
- **Owned Files**:
  - `tests/conftest.py`
  - `tests/cuda_kernels/test_*.py` (add module-level `pytest.skip` guards)
  - All test files referencing `.cuda()` or `torch.cuda` or triton without skip guards
  - `scripts/add_gpu_markers.py`
- **Gate**:
  - `pytest tests/ -m "not gpu and not triton" --co -q` completes without SIGABRT

### Agent C — Silent Failure Cleanup
- **Branch**: `agent/c-wave2-silent-failures`
- **Audit IDs**: P024, P030, P031, P032, P044, P053, P054
- **Owned Files**:
  - `neuralops/core/roofline.py`
  - `neuralops/core/composable.py`
  - `neuralops/core/inspector.py`
  - `neuralops/core/hardware.py`
  - `neuralops/core/snapshot_restore.py`
  - `neuralops/layers/hybrid/spectron_fp8_attention.py`
  - `neuralops/export/exporter.py`
  - `tests/test_silent_failures.py`
- **Gate**:
  - `grep -rn "except Exception:\s*$" neuralops/ | grep -A1 "^\s*pass"` returns 0 results
  - `pytest tests/test_silent_failures.py -v` passes

### Agent D — Version and Dependencies
- **Branch**: `agent/d-wave2-version-deps`
- **Audit IDs**: P010, P018, P041, P042
- **Owned Files**:
  - `pyproject.toml` (version, dependencies, extras)
  - `neuralops/__init__.py` (version validation)
  - `CITATION.cff` (version validation)
  - Files importing `vllm` or `tensorrt_llm` at module level (wrap in try/except and set `HAS_VLLM` / `HAS_TENSORRT_LLM`)
- **Gate**:
  - `pip install -e ".[dev,cpu]"` succeeds
  - Version `1.6.0` matches across all three files
  - `import rich, psutil, sklearn, huggingface_hub` succeeds after install

### Agent E — Integration Honesty
- **Branch**: `agent/e-wave2-integrations`
- **Audit IDs**: P006, P007, P008, P009
- **Owned Files**:
  - `neuralops/integrations/vllm.py`
  - `neuralops/integrations/sglang.py` (renamed to `neuralops/integrations/radix_prefix_cache.py`, create stub `sglang.py`)
  - `neuralops/integrations/tensorrt_llm.py`
  - `neuralops/integrations/tgi.py` (moved to `neuralops/serving/request_batcher.py`, create stub `tgi.py`)
  - `README.md` (Integration status table only)
  - Note: Does not touch `pyproject.toml`. Requests entry-point removal via doc.
- **Gate**:
  - `from neuralops.integrations.vllm import register_neuralops_vllm_plugin` returns False with warning
  - `from neuralops.integrations.sglang import register_neuralops_sglang_plugin` raises `IntegrationNotImplementedError`
  - README table reflects reality
  - `import neuralops` still works

### Agent F — Surgery Bug Fixes
- **Branch**: `agent/f-wave2-surgery`
- **Audit IDs**: P040, P053
- **Owned Files**:
  - `neuralops/surgery/calibration.py`
  - `neuralops/surgery/universal_calibrator.py`
  - `neuralops/surgery/layer_replacement.py`
  - `tests/surgery/test_calibration_grad.py`
  - `tests/surgery/test_sequential_traversal.py`
- **Gate**:
  - `.backward()` outside `torch.no_grad()`
  - `replace_attention_recursive()` supports nested `nn.Sequential`
  - `pytest tests/test_surgery_*.py` (13 existing + 2 new pass)

---

## Wave 3 — Structural Consolidation & Phase 2 Completion

### Agent A2 — Phase 2 Completion Architect (Wave 3a — Blocking)
- **Branch**: `agent/a2-wave3-phase2-completion`
- **Audit IDs**: P002, P003, P004, P005
- **Owned Files**:
  - `neuralops/attentions/` -> `neuralops/layers/` (physical migration via `git mv`)
  - `neuralops/core/cuda_kernels/` -> `neuralops/ops/cuda/` (physical migration)
  - `neuralops/core/triton_kernels/` -> `neuralops/ops/triton/` (physical migration)
  - `neuralops/compression_engine/` -> `neuralops/compression/` (physical migration)
  - Shims: `neuralops/backward_compat/{attentions,cuda_kernels,triton_kernels,compression_engine}.py`, legacy directory shims
  - Internal import callsite rewrites to new paths
  - `tests/test_phase2_migration.py`
- **Gate**:
  - `pytest tests/test_phase2_migration.py -v` (8 tests pass)
  - `pytest tests/test_cuda_kernels_lazy.py tests/test_lazy_architecture.py -v` (16 tests pass)
  - `python3 -c "from neuralops import replace_attention, compress; print('OK')"`
  - `git ls-files neuralops/attentions/ | wc -l` (<20)
  - `git ls-files neuralops/layers/ | wc -l` (>=130)
  - `time python3 -c "import neuralops"` (<100ms)

### Agent G — Analysis Module Relocation (Wave 3b — Parallel with A2)
- **Branch**: `agent/g-wave3-analysis-relocation`
- **Audit IDs**: P015, P045, P046, P047
- **Owned Files**:
  - `neuralops/core/run_long_context_sweep.py` -> `neuralops/benchmarks/run_long_context_sweep.py`
  - `neuralops/benchmarks/multi_mode_benchmark.py`
  - `neuralops/core/paper_export.py` -> `neuralops/analysis/paper_export.py`
  - `neuralops/core/results_ledger.py` -> `neuralops/analysis/results_ledger.py`
  - `neuralops/core/roofline.py` -> `neuralops/analysis/roofline.py`
  - `neuralops/analysis/inspector.py`, `neuralops/core/inspector.py`
  - Compatibility shims in `neuralops/core/`
- **Gate**:
  - `import neuralops.core` does not import `matplotlib` or `pandas`
  - `from neuralops.analysis import paper_export, results_ledger, roofline, inspector` works
  - `from neuralops import paper_export, profile_roofline` works

### Agent H — Duplicate Consolidation (Wave 3c — After A2 Merge)
- **Branch**: `agent/h-wave3-duplicate-consolidation`
- **Audit IDs**: P014, P016, P017, P025, P026, P027, P028, P029, P033, P034, P035, P049
- **Owned Files**:
  - Canonical definitions: `neuralops/core/peft/lora.py`, `neuralops/core/low_rank/svd_linear.py`, `neuralops/core/low_rank/factorized.py`, `neuralops/core/config.py`
  - Update duplicate files to import from canonical
  - Delete obsolete shims / dead stubs: `neuralops/attention/`, `neuralops/compression/` old duplicates, `neuralops/monitor/`, `neuralops/attentions/_base_deprecated.py`
  - Consolidate `_compute_flops_estimate` into `analysis/flops_utils.py`
  - `tests/test_no_duplicates.py`
- **Gate**:
  - Each class appears exactly once in the codebase
  - `import neuralops.monitor` raises `ModuleNotFoundError`
  - `pytest tests/test_no_duplicates.py` passes

---

## Wave 4 — Verification and Release

### Agent I — Pre-Release Verification Lead (Wave 4a)
- **Branch**: `agent/i-wave4-verification`
- **Audit IDs**: Verification of P001-P054
- **Owned Files**:
  - `RELEASE_SIGNOFF_v1.6.0.md` (read-only across codebase)
- **Gate**:
  - All eleven Phase 9 verification categories pass with raw evidence

### Agent J — Release Manager (Wave 4b)
- **Branch**: `agent/j-wave4-release`
- **Owned Files**:
  - `CHANGELOG.md`
  - `pyproject.toml` (version)
  - `neuralops/__init__.py` (version)
  - `CITATION.cff`
  - Git tag `v1.6.0`
- **Gate**:
  - Version parity, clean tag, push to origin
