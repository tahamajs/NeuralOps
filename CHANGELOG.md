# Changelog

All notable changes to the **NeuralOps** library (`SVD_linear_Attention`) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.3.0] - 2026-10-14

### Major Architecture & Engine Overhaul

`NeuralOps` v1.3.0 is a milestone release delivering complete architectural consolidation, unifying all subquadratic attention kernels, model compression algorithms, serving runtimes, distributed training loops, and public APIs into a production-grade framework.

### Added
- **19 Top-Level Canonical Functions (`neuralops`)**:
  - Model Surgery & Export: `load_model`, `replace_attention`, `compress`, `export_model`, `load_exported_model`.
  - Inspection & Discovery: `list_attentions`, `list_compressors`, `get_attention_class`, `get_compressor_class`, `list_families`.
  - Training, Distillation & Benchmarking: `train`, `distill`, `evaluate`, `benchmark`, `benchmark_all`.
  - System Diagnostics & Environment: `info`, `available_backends`, `has_triton`, `validate`.
- **Layer-Wise Hybrid Attention Surgery (`hybrid_spec`)**:
  - Ability to pass granular dictionary or list specifications into `neuralops.replace_attention` (e.g. Layers 0–3: dense attention, Layers 4–11: Cosformer, Layers 12–15: Low-Rank SVD).
- **Manifold & Orthogonal Matrix Optimizers**:
  - `Muon`: 5th-order Newton-Schulz polynomial approximation for orthogonalized 2D weight updates.
  - `RiemannianStiefelOptimizer` & `StiefelOptimizer`: Orthogonal parameter manifold optimization for unitary projection matrices.
- **Serving Adapters (`neuralops.export` / `neuralops.integrations`)**:
  - `VLLMAdapter`: Paged key-value cache wrapper routing dynamic blocks through `paged_attention_kernel`.
  - `TGIAdapter`: Deterministic Hugging Face Text Generation Inference runtime configuration.
  - `PagedKVCache` & `ContinuousBatcher` inference engines.
- **Dynamic OOM-Resilient Batch Probing (`probe_optimal_batch_size`)**:
  - Automated binary search identifying peak batch throughput without crashing execution environments.
- **Parametric Numerical Parity Test Suite (`tests/test_triton_parity.py`)**:
  - Parity tests verifying mathematical equivalence across all 26 Triton kernel modules against PyTorch CPU/MPS fallbacks.
- **Full `torch.compile` Custom Op Registration (`neuralops/core/triton_kernels/op_registry.py`)**:
  - Real FakeTensor and Meta-device registrations eliminating graph breaks during Inductor compilation.
- **PEP 561 Type Compliance**:
  - Bundled `neuralops/py.typed` package marker.

### Changed
- **Eliminated Python Module-vs-Package Shadowing Collisions**:
  - Removed `neuralops/trainer/optimizers.py` in favor of `neuralops/trainer/optimizers/`.
  - Removed `neuralops/data/benchmark_datasets.py` in favor of `neuralops/data/benchmark_datasets/`.
  - Removed `neuralops/utils/compat.py` in favor of `neuralops/utils/compat/`.
- **Pruned Dead Code & Workstation Artifacts (32+ Files)**:
  - Removed hardcoded local paths (`D:/SVD_Project`) in `neuralops/runners/`.
  - Cleaned obsolete `neuralops/benchmarks/_legacy/` and forwarding shims.
  - Removed archived test stubs in `neuralops/tests/_archive/`.
- **Refactored Attention Surgery for Grouped Query Attention (GQA) & Decoder Blocks**:
  - Handled $H_q \neq H_{kv}$ head replication.
  - Preserved tuple return signatures `(attn_output, attn_weights, past_key_values)`.

### Fixed
- Fixed non-contiguous tensor layout crashes in `linformer_attention` via `.reshape()` and `.contiguous()`.
- Added numerical stability guards (`eps=1e-6`, `torch.clamp`, `torch.nan_to_num`) in `nasc_attention` and `performer_attention`.
- Clamped residual branch scaling factors in `aegis_hybrid` and `split_head_hybrid_attention` to prevent vanishing/exploding gradients.
- Resolved HIBA mask exceptions when `attention_mask is None`.
- Fixed PyTorch 3.14+ deprecations across `GradScaler` and autograd graphs.

---

## [1.2.0] - 2026-10-09

### Added
- Unified CLI runtime: `run`, `plan`, `list`, `verify`, `resume`, `results`, `doctor`.
- 131 canonical attention mechanisms and 72 model compression engines verified.
- 26 Triton kernel modules with native PyTorch CPU/MPS fallbacks.
- Multi-dataset benchmark sweep covering SST-2, MNLI, AG News, and language modeling perplexity.

### Changed
- Reorganized codebase into clean modular subpackages.
- Preserved 657 legacy alias mappings via `neuralops/core/deprecated_aliases.json`.

---

## [1.1.0] - 2026-09-20

### Added
- Triton-accelerated fused recurrent scans: `gdn_scan`, `gla_scan`, `hgrn2_scan`, `fms_scan`.
- Knowledge distillation (KD) benchmark engine supporting temperature scaling and teacher-student logits matching.
- Initial model surgery injection for LLaMA and Mistral transformer backbones.

---

## [1.0.0] - 2026-06-15

### Added
- Initial public release of NeuralOps on GitHub and PyPI.
- Drop-in attention replacement engine (`neuralops.replace_attention`) for Hugging Face Transformers.
- SVD and low-rank compression pipeline (`neuralops.compress`).
- Basic benchmark runner across 8 supported transformer architectures.
