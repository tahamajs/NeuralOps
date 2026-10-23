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

## Wave 3 — Structural Consolidation (2 Concurrent Agents)

### Agent G — Analysis Module Relocation
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

### Agent H — Duplicate Consolidation
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

## Wave 4 — Verification and Release (Orchestrator)
- Phase 9: Eleven-category exhaustive verification suite
- Phase 10: `RELEASE_SIGNOFF_v1.6.0.md`, `CHANGELOG.md`, `git tag v1.6.0`
