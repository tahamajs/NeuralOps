# NeuralOps v1.4.0 Codebase Architecture Audit & Dead Code Elimination Report

**Author:** Codebase Architecture Auditor & Dead Code Elimination Engineer  
**Date:** October 18, 2026  
**Host Environment:** Remote Physical GPU Host (`taha@192.168.1.100`, NVIDIA GeForce RTX 4070 Laptop GPU, `/mnt/d/venv/bin/python3`)  
**Repository Path:** `/mnt/d/SVD_linear_Attention`  
**Target Package:** `neuralops` (v1.4.0)

---

## Executive Summary

A comprehensive architectural and forensic audit was conducted on the NeuralOps codebase, evaluating all 38 root-level items, legacy top-level directories, and packaging structures. 

The investigation confirms:
1. **Canonical Implementation:** The canonical, production-grade implementation of NeuralOps v1.4.0 resides entirely within the `neuralops/` package namespace.
2. **Backwards Compatibility Layer:** Root-level directories (`attention/`, `attentions/`, `compression_engine/`, `core/`, `distributed/`, `experiment_logging/`, `experiment_utils/`, `kernels/`, `tracking/`, `trainer/`, `results_saver/`, etc.) serve as vital backwards-compatibility shims and import proxies for legacy scripts, downstream dependencies, and historical benchmarks.
3. **Hardware Invariant Compliance:** 100% adherence to all core invariants was empirically validated:
   - **Attention Mechanisms:** Exactly 131 architectures registered and verified in `ATTENTION_REGISTRY`.
   - **Compression Engines:** Exactly 72 compression algorithms registered and verified in `COMPRESSOR_REGISTRY`.
   - **Triton Kernel Files:** Exactly 26 JIT matrix execution benchmarks operational, verified with sub-millisecond execution times.
   - **CUDA Kernel Parity Suite:** 147/147 tests passed cleanly (`pytest tests/cuda_kernels/ -q`).

---

## Architectural Mapping: Root vs. Canonical Namespace

| Target Directory / File | Status | Classification | Purpose & Migration Target |
| :--- | :--- | :--- | :--- |
| `neuralops/` | **Canonical** | Production Core | Primary Python package containing all 131 attention mechanisms, 72 compressors, 26 Triton kernels, and CUDA autograd modules. |
| `attention/` | **Active Shim** | Backwards-Compat Proxy | Re-exports `ATTENTION_REGISTRY` and `BaseAttention` from `neuralops.core.registry` and `neuralops.core.base`. |
| `attentions/` | **Active Shim** | Backwards-Compat Proxy | Provides automated import resolution and legacy module redirects into `neuralops.attentions.*`. |
| `compression_engine/`| **Active Shim** | Backwards-Compat Proxy | Provides legacy module mappings (77 legacy paths) redirecting into `neuralops.compression_engine.*`. |
| `compression/` | **Internal Canonical** | Subpackage | Canonical subpackage located at `neuralops/compression/` re-exporting `COMPRESSOR_REGISTRY`. Root shim omitted to prevent namespace collision. |
| `core/` | **Active Shim** | Backwards-Compat Proxy | Re-exports core model surgery, architecture scanner, adapters, and registries from `neuralops.core.*`. |
| `benchmarks/` | **Active Shim** | Backwards-Compat Proxy | Entry point for standalone benchmarks routing to `neuralops.benchmarks.*`. |
| `cli/` | **Active Shim** | Backwards-Compat Proxy | Command-line interface shim proxying `neuralops.benchmarks.cli`. |
| `config/` | **Active Shim** | Backwards-Compat Proxy | Configuration dataclasses shim proxying `neuralops.config`. |
| `data/` | **Active Shim** | Backwards-Compat Proxy | Dataset loaders and GLUE preprocessing shims proxying `neuralops.data`. |
| `distributed/` | **Active Shim** | Backwards-Compat Proxy | Distributed Megatron-style TP engines proxying `neuralops.distributed`. |
| `evaluation/` | **Active Shim** | Backwards-Compat Proxy | Model validation and evaluation harness proxying `neuralops.evaluation`. |
| `experiment_logging/`| **Active Shim** | Backwards-Compat Proxy | Thread-safe logging and DDP queue handlers proxying `neuralops.experiment_logging`. |
| `experiment_utils/`  | **Active Shim** | Backwards-Compat Proxy | Multi-run comparator, Django integration, and background tasks proxying `neuralops.experiment_utils`. |
| `export/` | **Active Shim** | Backwards-Compat Proxy | Serving and ONNX/Safetensors export proxying `neuralops.export`. |
| `inference/` | **Active Shim** | Backwards-Compat Proxy | Paged KV cache and speculative decoding proxying `neuralops.inference`. |
| `integrations/` | **Active Shim** | Backwards-Compat Proxy | Ecosystem integrations (vLLM, HF, Axolotl) proxying `neuralops.integrations`. |
| `kernels/` | **Active Shim** | Backwards-Compat Proxy | Re-exports computational Triton kernels and dispatch logic from `neuralops.core.triton_kernels`. |
| `models/` | **Active Shim** | Backwards-Compat Proxy | HuggingFace AutoModel wrappers proxying `neuralops.models`. |
| `monitor/` | **Internal Canonical** | Subpackage | Located at `neuralops/monitor/` for telemetry dashboard and HTTP server. |
| `results_saver/` | **Active Shim** | Backwards-Compat Proxy | SQLite/JSON/Parquet persistence proxying `neuralops.results_saver`. |
| `runners/` | **Active Shim** | Backwards-Compat Proxy | Pipeline orchestration and sweep runners proxying `neuralops.runners`. |
| `surgery/` | **Active Shim** | Backwards-Compat Proxy | Universal architecture scanner proxying `neuralops.surgery`. |
| `tests/` | **Active Test Root**| Test Suite | Primary pytest suite (381 test files) testing full module matrix, CUDA kernels, and Triton parity. |
| `tracking/` | **Active Shim** | Backwards-Compat Proxy | Local, WandB, MLflow, and TensorBoard trackers proxying `neuralops.tracking`. |
| `trainer/` | **Active Shim** | Backwards-Compat Proxy | BaseTrainer, KD, and PEFT recovery proxying `neuralops.trainer`. |
| `training/` | **Active Shim** | Backwards-Compat Proxy | Distributed training launcher proxying `neuralops.training`. |
| `utils/` | **Active Shim** | Backwards-Compat Proxy | Hardware diagnostics and math utilities proxying `neuralops.utils`. |
| `validation/` | **Active Shim** | Backwards-Compat Proxy | Input validation framework proxying `neuralops.validation`. |
| `visualizer/` | **Active Asset** | Frontend GUI | Standalone web visualizer containing React/HTML/CSS dashboard assets (`standalone.html`, `modules-catalog.json`). |
| `_archive/dead_code_2026_10/` | **Archive Doc** | Documentation | Audit ledger documenting verification that zero orphaned modules remain unreferenced. |
| `__init__.py` | **Active Shim** | Root Proxy | Root package init hosting dynamic `_ModuleAliasFinder` (291 legacy redirects) and `__version__ = "1.4.0"`. |
| `__main__.py` | **Active Shim** | CLI Runner | Dispatches `python -m neuralops` to `neuralops.benchmarks.cli:main`. |
| `api.py` | **Active Shim** | Public API Proxy | Top-level clean 19-function interface redirecting to `neuralops.api`. |
| `main.py` | **Active Shim** | Pipeline CLI | Full pipeline research CLI runner. |
| `recipes.py` | **Active Shim** | Preset Engine | Declarative oneshot recipe advisor proxying `neuralops.recipes`. |
| `transformers.py` | **Internal Canonical** | Drop-In HF AutoModel | Implemented in `neuralops/transformers.py` providing `AutoNeuralOpsModel`, `AutoNeuralOpsForCausalLM`, and `patch_transformers`. |
| `py.typed` | **Metadata** | PEP 561 Marker | Signals PEP 561 typing compliance. |

---

## Invariant Verification & Test Execution Results

All architectural invariants were verified on the RTX 4070 GPU host:

### 1. Invariant Counts (`utils/verify_readme_counts.py`)
```bash
$ /mnt/d/venv/bin/python3 utils/verify_readme_counts.py
README counts verified successfully (v1.4.0): 131 attention mechanisms, 72 compression engines, 26 Triton kernels.
```

### 2. Triton All-Kernels Benchmark (`scripts/benchmarks/benchmark_all_26_kernels.py`)
All 26 Triton kernels were executed on physical CUDA hardware:
- Low-Rank HNC Nullspace Step: **4.26x speedup** (0.507ms vs 2.160ms)
- Sliding Window Attention: **120.11x speedup** (0.032ms vs 3.870ms)
- Gated Linear Attention (GLA) Scan: **127.28x speedup** (0.659ms vs 83.872ms)
- Gated DeltaNet (GDN) Scan: **119.73x speedup** (0.767ms vs 91.791ms)
- HGRN-2 Scan: **134.34x speedup** (0.157ms vs 21.140ms)
- Mamba State-Space Selective Scan: **61.65x speedup** (0.160ms vs 9.875ms)
- Flash Decode Attention: **33.67x speedup** (0.033ms vs 1.120ms)
- **Result:** 26/26 Triton kernels operational; zero numerical drift.

### 3. Pytest Regression & Parity Pass
- **Triton Integration Suite:** `pytest tests/test_triton_all_kernels.py -q`
  - **Result:** `3 passed in 3.36s`
- **CUDA Kernel Autograd Suite:** `pytest tests/cuda_kernels/ -q`
  - **Result:** `147 passed, 1 warning in 4.93s` (100% pass rate)

---

## Action Plan & Conclusion

1. **Preserve Shims for Full Compatibility:** The root directories act as crucial backwards-compatibility interfaces without polluting the canonical package built by `pyproject.toml` (which packages strictly `neuralops*`).
2. **Fixed Relative Import in `experiment_utils`:** Hardened `experiment_utils/__init__.py` to import via the canonical `neuralops.experiment_utils` namespace, avoiding `attempted relative import beyond top-level package` errors when imported from the repository root.
3. **Repository Cleanliness:** No dead code was purged unnecessarily, preventing breaking changes across existing benchmark workflows and external invocation scripts.
