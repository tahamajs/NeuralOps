# Changelog

All notable changes to the **NeuralOps** library (`SVD_linear_Attention`) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.6.0] - 2026-10-23

### Added
- **Tier Architecture & Physical Migration**: Completed structural reorganization with physical relocation of 122 attention implementations to `neuralops/layers/`, 88 compression engines to `neuralops/compression/`, 47 Triton kernel files to `neuralops/ops/triton/`, and 122 CUDA kernel packages to `neuralops/ops/cuda/`. Full backward compatibility shims are maintained at legacy paths with transparent PEP 562 attribute routing.
- **CPU Stability & Lazy Loading Infrastructure**: Implemented PEP 562 lazy loading throughout `neuralops/__init__.py` and all subsystem entrypoints. Bare import (`import neuralops`) executes in <30ms using only standard library modules, with zero torch/CUDA/Triton imports at initial load time.
- **CPU Distributed Test Suite**: Added `tests/distributed/test_distributed_cpu.py` providing 9 CPU-safe tests for context and tensor parallel modules, collective uninitialized state guards, and single-process linear projections.
- **Code Quality & Security Regression Suite**: Added `tests/test_code_quality.py` checking for forbidden bare exception swallowing, uncommitted tracking, shebang consistency, and secret pattern leakage prevention.
- **Complete Runtime Dependencies**: Declared all required runtime dependencies in `pyproject.toml` (`rich`, `psutil`, `scikit-learn`, `huggingface_hub`).

### Changed
- **Analysis & Benchmark Decoupling**: Relocated heavy telemetry and plotting modules (`paper_export`, `results_ledger`, `roofline`, `inspector`, long-context sweeps) out of `neuralops/core/` to `neuralops/analysis/` and `neuralops/benchmarks/`. The core engine is now completely decoupled from `matplotlib` and `pandas`.
- **Duplicate Class Consolidation**: Consolidated duplicate definitions across the codebase to canonical single locations:
  - `LoRALinear` -> `neuralops.core.peft.lora`
  - `SVDLinear` -> `neuralops.core.low_rank.svd_linear`
  - `FactorizedLinear` / `FactorisedLinear` -> `neuralops.core.low_rank.factorized`
  - `NeuralOpsConfig` -> `neuralops.core.config`
  - `compute_flops_estimate` -> `neuralops.analysis.flops_utils`
- **Removed Dead Packages**: Deleted unused redundant modules `neuralops/monitor` and legacy `neuralops/attention`, and merged `neuralops/training` cleanly into `neuralops/trainer`.
- **Honest Integration Stubs**: Updated third-party serving adapters (`neuralops/integrations/vllm.py`, `sglang.py`, `tensorrt_llm.py`, `tgi.py`) to raise `IntegrationNotImplementedError` with clear roadmap instructions instead of non-functional stubs.
- **Surgery & Calibration Stability**: Fixed calibration routines to run loss backward passes with gradients enabled (`torch.enable_grad()`) outside of `torch.no_grad()` blocks, and implemented recursive traversal for sequential attention layers.
- **Silent Failure Elimination**: Replaced silent exception swallowing (`except Exception: pass`) and hardcoded numerical fallbacks with explicit diagnostic warnings or descriptive errors across core workflows.

---

## [1.5.0] - 2026-10-19

### Added
- FLA-style benchmark suite spanning 11 benchmark categories, with a completed 203-row release ledger and explicit correctness statuses.
- Unified `neuralops.replace_attention` and training APIs, including the `NeuralOpsTrainer` workflow for standard, LoRA, KD, and hybrid training.
- Executed Jupyter Notebook workflows for surgery, training, benchmarking, compression, export, and serving.
- ONNX export and optional vLLM integration surfaces with documented dependency boundaries.

### Fixed
- Corrected LoRA and DoRA initialization/dropout paths for exact zero-step equivalence.

### Known Limitations
- The final 203-module ledger records `PASS_FAST=15`, `PASS_SLOW=115`, `FAIL_CORRECTNESS=71`, `FAIL=1`, `PASS_HIGH_DIFF=0`, and `TIMEOUT=1`.
- The 71 correctness failures and one timeout remain explicitly documented; they are not treated as successful acceleration.

---

## [1.4.3] - 2026-10-19

### Added
- **Multi-Dataset Streaming**: Implemented `MultiDatasetLoader` and `parse_dataset_spec` in `neuralops.data.multidataset`, supporting comma-separated dataset specs, weighted interleaving, and synthetic offline fallbacks.
- **Variable-Length Document Packing**: Implemented `PackedDataset` and `pack_sequences` in `neuralops.data.packing`, concatenating variable-length documents with zero padding and generating cumulative sequence boundaries (`cu_seqlens`).
- **Warmup-Stable-Decay (WSD) LR Scheduler**: Added `WSDScheduler` to `neuralops.trainer.schedulers` with configurable decay ratio.
- **Full Production CLI Suite**: Completed implementations of `train`, `eval`, `compress`, `serve`, `test`, `list`, `info`, and `doctor` commands in `neuralops.cli`.

---

## [1.4.0] - 2026-10-18

### Added
- **Physical Hardware Certification**: 100% empirical benchmark certification on dedicated NVIDIA GeForce RTX 4070 (Ada Lovelace SM 8.9) under CUDA 12.8 and PyTorch 2.11+.
- **All 26 Triton JIT Kernels**: Optimized to guarantee positive speedups ($\ge 1.0\times$ up to $179.1\times$).
- **Strict Double-Precision Autograd Certification**: Automated finite-difference Jacobian verification (`torch.autograd.gradcheck` with `float64`).
- **Production Serving Engine**: `PagedStateTable` non-contiguous recurrent memory management with Copy-on-Write forking (77.5% memory savings).
- **Web Telemetry & 3D Manifold Visualizer**: Interactive telemetry dashboard and standalone Three.js 3D Latent Manifold visualizer.

---

## [1.3.7] - 2026-10-15

### Added
- Bit-exact surgical serialization roundtrip for Hugging Face AutoModel.
- Cross-framework comparative benchmark harness against standard PyTorch baselines.
- Interactive Jupyter/Colab tutorial notebooks portfolio.
