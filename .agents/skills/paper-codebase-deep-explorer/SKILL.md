---
name: paper-codebase-deep-explorer
description: "Systematically explore, index, grep, and cross-reference symbols, registries, kernels, and mathematical definitions across the NeuralOps repository for paper authoring"
version: "1.0.0"
tags: ["neuralops", "paper", "codebase", "exploration", "search", "indexing", "cross-reference"]
---

# Skill: Paper Codebase Deep Explorer & Cross-Referencing Architect

## When to Use This Skill
Use this skill when researching, writing, reviewing, or fact-checking academic manuscripts, conference submissions (MLSys, NeurIPS, ICLR), journal papers, or thesis chapters based on the **NeuralOps** codebase.

Apply this skill to:
- Trace mathematical claims in paper drafts directly to concrete source implementations in `neuralops/layers/`, `neuralops/compression/`, or `neuralops/ops/`.
- Navigate and index the **131 attention mechanisms** and **72 model compression engines** registered in `neuralops/core/registry.py`.
- Locate native CUDA C++ kernels, autotuned Triton JIT pipelines, and PyTorch reference fallbacks across `neuralops/ops/cuda/` and `neuralops/ops/triton/`.
- Cross-reference empirical numbers, latency benchmarks, VRAM usage, and perplexity measurements against canonical ledgers (`results/registry/master_ledger.sqlite`, `results/MASTER_RESULTS_MATRIX.csv`, and `academic/thesis/`).
- Conduct zero-overhead forensic code inspection using standard POSIX utilities (`grep`, `find`, `rg`, `sed`, `awk`, `git grep`, `sqlite3`) without executing unverified code paths.

---

## 1. Core Principles & Ground-Truth Contracts

When writing systems papers or empirical ML monographs, code claims and paper assertions must maintain bit-level integrity:

1. **Zero Hallucination of Symbols & Registries**:
   Never invent class names, registry keys, or kernel signatures. Every symbol referenced in an empirical paper section must resolve to a valid definition in `ATTENTION_REGISTRY`, `COMPRESSOR_REGISTRY`, or `neuralops/ops/`.
2. **Strict Fallback Attribution (3-Level Contract)**:
   When reporting microbenchmarks, always specify which tier generated the metric:
   - **Level 1**: Native compiled CUDA C++ / HIP (`neuralops/ops/cuda/`).
   - **Level 2**: Autotuned Triton JIT kernel (`neuralops/ops/triton/`).
   - **Level 3**: Vectorized PyTorch mathematical oracle (`neuralops/ops/pytorch/`).
3. **Canonical Ledgers as Single Source of Truth**:
   Every numerical claim (latency, throughput, parameter compression ratio, accuracy, perplexity) must be traceable to `results/registry/master_ledger.sqlite` or `results/MASTER_RESULTS_MATRIX.csv`.
4. **Environment-Agnostic Static Forensic Inspection**:
   Execute all codebase exploration via deterministic POSIX text processing and SQLite queries. Never rely on external hosts or unverified runtime environments to discover code structure.

---

## 2. Navigating the Unified Component Registries

NeuralOps organizes its core operator ecosystem through dynamic discovery registries defined in `neuralops/core/registry.py`.

```
                               ┌────────────────────────────────────────────────────────┐
                               │               neuralops/core/registry.py               │
                               └───────────┬────────────────────────────────┬───────────┘
                                           │                                │
                       ┌───────────────────┴───────────────┐  ┌─────────────┴─────────────────────┐
                       │        ATTENTION_REGISTRY         │  │        COMPRESSOR_REGISTRY        │
                       │    (Exactly 131 Canonical Keys)   │  │    (Exactly 72 Canonical Keys)    │
                       └───────────────────┬───────────────┘  └─────────────┬─────────────────────┘
                                           │                                │
                 ┌─────────────────────────┴─────────┐            ┌─────────┴─────────────────────┐
                 │ 7 Canonical Attention Families    │            │ 5 Canonical Compression       │
                 │ - Linear & Kernel Feature Maps    │            │   Families                    │
                 │ - State Space Models (SSM)        │            │ - Low-Rank & SVD Decomp.      │
                 │ - Low-Rank & Spectral SVD         │            │ - Structured & ADMM Pruning   │
                 │ - Block-Sparse & Sliding-Window   │            │ - Post-Training Quantization  │
                 │ - Mixture-of-Experts (MoE)        │            │ - KV-Cache State Eviction     │
                 │ - Gated Linear & Delta-Rules      │            │ - Tensor Decompositions       │
                 │ - Modern & Fused Fast Kernels     │            └───────────────────────────────┘
                 └───────────────────────────────────┘
```

### 2.1 Registry Architecture & Lazy Loading (PEP 562)

Eager loading of all 203 operator classes at import time would introduce substantial import latency. To eliminate this overhead, `neuralops/core/registry.py` implements a lazy-loading architecture:
- Operator classes are registered with decorators:
  ```python
  @ATTENTION_REGISTRY.register("linear_attention", supports_causal=True, supports_gqa=False)
  class LinearAttention(BaseAttention): ...
  
  @COMPRESSOR_REGISTRY.register("hnc_svd", hardware_accelerated=True)
  class HNCSVDCompressor(BaseCompressor): ...
  ```
- Concrete modules reside in `neuralops/layers/` (attentions) and `neuralops/compression/` (compressors), loaded dynamically upon first request.
- The shim `neuralops/attentions/` points to `neuralops/layers/` for backward compatibility.

### 2.2 Universal Parameter Synchronization Factory

When instantiating operators for paper benchmarks or ablations, the factory reconciles divergent parameter naming conventions:
- Dimension keys: `dim`, `hidden_size`, `d_model`, `embed_dim`
- Head count keys: `num_heads`, `num_attention_heads`, `n_head`
- Causal masking: `is_causal`, `causal`

The wrapper function `_make_universal_attention_class` in `neuralops/core/registry.py` automatically injects a `transformers.BertConfig` fallback if instantiated directly with positional or raw kwargs.

### 2.3 Attention Families & Source File Directory Map

| Attention Family | Key Characteristics & Mathematical Focus | Primary Directory | Representative Registry Keys |
| :--- | :--- | :--- | :--- |
| **Linear & Kernel** | Kernel feature maps $\phi(x)$, associative scan, $O(N)$ inference | `neuralops/layers/linear/` | `linear_attention`, `cosformer`, `performer`, `rfa`, `galerkin` |
| **SSM & Recurrent** | Continuous-time state space, selective scans, hardware-aware decay | `neuralops/layers/ssm/` | `mamba`, `mamba2`, `simple_ssm`, `selective_ssm`, `griffin_ssd` |
| **Low-Rank & SVD** | Dynamic SVD projection, spectral truncation, rank sweeps | `neuralops/layers/svd/` | `low_rank_svd`, `asvd`, `svd_compressed`, `rpca_sparse_svd` |
| **Sparse & Sliding** | Fixed/dynamic sparsity patterns, block-diagonal masks | `neuralops/layers/sparse/` | `sparse_attention`, `sliding_window`, `bigbird`, `longformer` |
| **MoE & Hybrid** | Sparse gating routers, multi-head expert selection, routing tokens | `neuralops/layers/hybrid/` | `moe_routing`, `samba_moe`, `bam_umoe`, `aegis_hybrid` |
| **Gated & Delta-Rule**| Input/output gates, recurrent delta update $S_t = S_{t-1} + v_t k_t^T$ | `neuralops/layers/gated/` | `gla`, `gdn_scan`, `hgrn2_scan`, `kda_attention`, `hla` |
| **Modern & Fused** | Flash decoding, grouped-query attention (GQA), RoPE offsets | `neuralops/layers/modern/`, `fast_kernels/` | `flash_gqa`, `paged_attention`, `flash_decode_stage2`, `fused_rope` |

### 2.4 Compressor Families & Source File Directory Map

| Compression Family | Compression Mechanism & Algorithmic Paradigm | Primary Directory | Representative Registry Keys |
| :--- | :--- | :--- | :--- |
| **Low-Rank & SVD** | Cholesky whitening, Riemannian Stiefel SVD, activation-aware rank | `neuralops/compression/svd/`, `low_rank/` | `hnc_svd`, `asvd_compressor`, `whitened_svd`, `cr_compressor` |
| **Pruning** | Magnitude pruning, ADMM structured channel pruning, entropy pruning | `neuralops/compression/pruning/` | `magnitude_pruner`, `admm_pruner`, `entropy_pruning_compressor` |
| **Quantization** | Uniform/non-uniform codebooks, FP8/FP4/INT4 packing, ternary scales | `neuralops/compression/quantization/`| `fp8_quantizer`, `fp4_gemm_compressor`, `ternary_weight` |
| **KV-Cache Eviction**| Token evictions, attention sink retention, cache notes | `neuralops/compression/kv_cache/` | `cachenotes_compressor`, `delta_kv`, `kvtc_compressor` |
| **Tensor Decomp.** | Tensor Train (TT), Tucker decomposition, random hashed projections | `neuralops/compression/tensor_decomp/`| `tt_compressor`, `roast_hashed_linear`, `flat_llm` |

### 2.5 Resolving Legacy Aliases

Legacy codebase iterations used historical naming variations. When locating symbols:
- Check `neuralops/core/legacy_aliases.py` and `neuralops/core/deprecated_aliases.json`.
- `ATTENTION_FILE_MAP`: Maps legacy shorthand keys (e.g. `gqa`, `galerkin`, `adasplash_inject`) to canonical filenames.
- `COMPRESSOR_FILE_MAP`: Maps legacy compressor names to canonical filenames.
- `DEPRECATED_ALIASES`: Maps deprecated module names (e.g. `chela_slay_attention` $\to$ `chela_slay`, `cosformer_attention` $\to$ `cosformer`, `hgrn2_outer` $\to$ `hgrn2_gated_outer`).

---

## 3. Locating Kernel Bindings & the 3-Level Fallback Chain

When authoring systems and kernel architecture sections, each operator must be mapped across all three execution levels.

```
       ┌────────────────────────────────────────────────────────┐
       │     Top-Level Operator Call (neuralops/layers/...)     │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │     Three-Level Dispatcher (neuralops/ops/cuda/_fallback.py)
       └───────┬───────────────────┼───────────────────┬────────┘
               │                   │                   │
               ▼ Level 1           ▼ Level 2           ▼ Level 3
        ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
        │   CUDA C++  │     │ Triton JIT  │     │   PyTorch   │
        │  Compiled   │     │  Autotuned  │     │ Reference   │
        │  Extension  │     │   Kernel    │     │   Oracle    │
        └─────────────┘     └─────────────┘     └─────────────┘
         ops/cuda/...        ops/triton/...      ops/pytorch/
```

### 3.1 Level 1: Native CUDA C++ Extensions (`neuralops/ops/cuda/`)

NeuralOps contains **112 native CUDA C++ kernel packages** housing **114 implementations**.
- **Package Anatomy**: Each subfolder in `neuralops/ops/cuda/<op_name>/` contains:
  - `__init__.py`: Package entrypoint exporting python bindings.
  - `_ops.py` or `<op_name>.py`: Interface wrapping C++ extension calls.
  - `kernel.cu`: Native CUDA source containing device kernels, thread-block tiling, warp shuffles (`__shfl_xor_sync`), and float4 vectorization.
  - `binding.cpp`: PyBind11 C++ bridge exposing C++ functions to Python.
  - `kernel.h` or `_common.cuh`: CUDA/C++ declarations and header templates.
- **JIT Compilation Driver**: `neuralops/ops/cuda/_bindings.py`
  - Function: `load_cuda_module(module_name, sources, extra_cflags, extra_cuda_cflags)`.
  - Content-addressed SHA-256 caching of compilation artifacts into `~/.cache/torch_extensions/`.
  - Detects GPU compute capability via `get_cuda_arch_flags()`.
- **Symbol Map**: `neuralops/ops/cuda/_map.py`
  - Contains `_KERNEL_SYMBOL_MAP` and `_SPECIAL_MAP` mapping external symbols to their hosting packages.

### 3.2 Level 2: Autotuned Triton JIT Kernels (`neuralops/ops/triton/`)

NeuralOps houses **26 canonical Triton kernel modules** containing **62 unified operators** grouped across 4 architectural tiers:
1. `TRITON_STANDALONE_KERNELS` (13 operators): Independent self-contained operators (`linear_attn`, `sparse_attn`, `low_rank`, `fused_kd`, etc.).
2. `TRITON_CORE_MODULES` (27 operators): Foundational sequence and state operations (`selective_scan`, `gla_scan`, `gdn_scan`, `hgrn2_scan`, `fms_scan`, etc.).
3. `TRITON_ROOT_MODULES` (2 operators): Root dispatch anchors (`op_registry`, `hardware_profile`).
4. `TRITON_EXTENDED_KERNELS` (20 operators): Advanced attention and normalization variants (`flash_decoding`, `fused_norm_linear`, `differential_attention`, etc.).

**Key Triton Exploration Files**:
- `neuralops/ops/triton/op_registry.py`: Registers operators via `@torch.library.custom_op("neuralops::<op>")` for PyTorch graph-capture and `torch.compile` compatibility.
- `neuralops/ops/triton/_common.py`: Common tiling offsets, PyTorch reference fallbacks, and runtime checks.
- `neuralops/ops/triton/autotuner.py` & `autotuned_configs.json`: Autotuning grid search spaces (block size, warps, pipeline stages).
- **Invariant**: Mainline Triton code strictly prohibits `tl.make_block_ptr` and `tl.advance` (reserved only for hardware-specific NPU dialects).

### 3.3 Level 3: Vectorized PyTorch Reference Engine (`neuralops/ops/pytorch/`)

The pure PyTorch reference tier serves as the mathematical ground truth for unit tests, autograd finite-difference verification, and CPU/MPS execution:
- Located in `neuralops/ops/pytorch/` and in `_common.py` fallbacks.
- Must execute with exact IEEE 754 precision without kernel dependencies.

### 3.4 Fallback Dispatcher & Runtime Telemetry (`neuralops/ops/cuda/_fallback.py`)

The central dispatch engine coordinates runtime backend selection:
- `dispatch(kernel_name, cuda_fn, triton_fn, pytorch_fn, *args, **kwargs)`
- **Zero Silent Fallback Law**: Any degradation from Level 1 $\to$ Level 2 $\to$ Level 3 MUST emit a structured `logger.warning` containing the failure reason.
- **Telemetry State**:
  - `_DISPATCH_STATS["cuda"]`, `_DISPATCH_STATS["triton"]`, `_DISPATCH_STATS["pytorch"]`
  - `_LAST_DISPATCH_BACKEND`, `_LAST_DISPATCH_ERROR`
- **Environment Overrides**:
  - `NEURALOPS_STRICT_CUDA=1`: Prohibits fallback; raises on Level 1 failure.
  - `FORCE_PYTORCH_FALLBACK=1`: Forces Level 3 across all operations.
  - `NEURALOPS_DEBUG_DISPATCH=1`: Emits real-time dispatch routing traces.

---

## 4. Cross-Referencing Thesis Chapters, Monographs & Results Ledgers

Paper authors must be able to move fluidly between thesis chapter sources, published monographs, and empirical ledgers.

```
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│     academic/thesis/chapters/   │       │   results/registry/             │
│  - chapter01_intro.tex          │       │   - master_ledger.sqlite        │
│  - chapter02_theory_part1/2.tex │◄─────►│   - master_ledger.jsonl         │
│  - chapter03_methodology_*.tex  │       │                                 │
│  - chapter04_results_*.tex      │       │   results/                      │
│  - chapter05_conclusion.tex     │       │   - MASTER_RESULTS_MATRIX.csv   │
└────────────────┬────────────────┘       └────────────────┬────────────────┘
                 │                                         │
                 ▼                                         ▼
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│   academic/paper/MODULES_CATALOG│       │   academic/thesis/              │
│   (203 19-Section Monographs)   │       │   - app_131_attentions_*.tex    │
│   build_paper/<module>/         │       │   - app_72_compressors_*.tex    │
└─────────────────────────────────┘       └─────────────────────────────────┘
```

### 4.1 Thesis Chapter Architecture (`academic/thesis/`)

Root document: `academic/thesis/810101504_main.tex` (presentation deck: `810101504_defense.tex`).

| Source File | Chapter Topic & Paper Scope | Key Cross-References & Concepts |
| :--- | :--- | :--- |
| `chapters/chapter01_intro.tex` | Research motivation, efficiency crisis, contributions | Scope of 131 attentions, 72 compressors, 3-level fallback |
| `chapters/chapter02_theory_part1.tex` | Attention foundations, kernel methods, associative scans | Taylor feature maps, Performer positive kernels, Recurrence duality |
| `chapters/chapter02_theory_part2.tex` | SVD low-rank geometry, Stiefel Riemannian optimization | Eckart-Young theorem, Cayley/QR retractions, Whitened covariance |
| `chapters/chapter03_methodology_part1.tex` | System architecture, dispatch engine, memory hierarchy | 3-Level fallback contract, float4 vectorization, bank conflicts |
| `chapters/chapter03_methodology_part2.tex` | CUDA C++ and Triton JIT microarchitectural kernels | Warp shuffle reductions, autotuner pipeline stages, shared mem |
| `chapters/chapter03_methodology_part3a.tex`| Multi-pillar knowledge distillation healing | Logit KL, hidden cosine alignment, attention entropy distillation |
| `chapters/chapter03_methodology_part3b.tex`| Test-time training (TTT) and post-surgery calibration | Fisher information updates, curvature-adapted learning rates |
| `chapters/chapter04_results_part1.tex` | 203 module empirical census, speedup leaderboards | Pass status (42 PASS_FAST, 161 PASS_SLOW), RTX 4070 metrics |
| `chapters/chapter04_results_part2.tex` | Microbenchmarks, roofline analysis, latency crossovers| Arithmetic intensity ($FLOPs/byte$), DRAM bandwidth saturation |
| `chapters/chapter04_results_part2b.tex`| Sequence length scaling horizons ($N=512$ to $32k$) | Quadratic attention memory cliffs vs linear attention stability |
| `chapters/chapter04_results_part3.tex` | Downstream accuracy benchmarks (SST-2, MNLI, AG News)| Accuracy retention vs parameter compression ratio ($CR$) |
| `chapters/chapter04_results_part4.tex` | Ablation studies, KD loss contributions, rank sweeps | Multi-pillar ablation matrix, condition number stability |
| `chapters/chapter05_conclusion.tex` | Summary of empirical findings, future research | Hardware-algorithm co-design, unified serving runtimes |
| `app_131_attentions_empirical.tex` | Complete 131-attention empirical ledger table | Full per-operator metrics table: CR, SST-2, PPL, VRAM, Throughput |
| `app_72_compressors_empirical.tex` | Complete 72-compressor empirical ledger table | Full per-compressor metrics table: CR, SST-2, PPL, VRAM, Speedup |
| `chapters/chapter_theorems_appendix.tex`| Formal mathematical proofs and bounds | Stiefel gradient pullback proofs, condition number invariance |
| `chapters/chapter_benchmarks_appendix.tex`| Hardware testbed specifications and seed configs | SM 8.9 Ada Lovelace architecture specs, CUDA toolkit versions |

### 4.2 Module Monograph Catalog (`academic/paper/MODULES_CATALOG.md`)

The repository maintains dedicated 19-section monographs in `academic/paper/build_paper/<module_name>/` for each evaluated module:
- `00_meta.tex`: Attribution, licensing, git commit hash.
- `03_background.tex` & `04_method.tex`: Algorithmic equations and tensor contracts.
- `05_architecture.tex`: Kernel dispatch hierarchy and backward graph.
- `07_results.tex` & `08_analysis.tex`: Empirical latency, VRAM, and roofline positioning.
- `12_reproducibility.tex`: Exact CLI reproduction commands.
- `B_implementation.tex`: Code listings for CUDA/Triton kernels and PyTorch reference.

### 4.3 Master Results Ledgers (`results/`)

All empirical results in papers must be traceable to the following files:

1. **`results/registry/master_ledger.sqlite`**:
   The authoritative relational database of verified hardware runs.
   - Table `results`:
     - `record_id`: Unique run identifier.
     - `module_name`: Registered module key.
     - `category`: `attention`, `compressor`, `ablation`, `baseline`.
     - `family`: Architectural family.
     - `status`: `VERIFIED`, `PASS_FAST`, `PASS_SLOW`.
     - `git_commit_sha`: Git commit hash of the executed code.
     - `hardware`: GPU/CPU execution device (e.g., `RTX 4070 Laptop (SM 8.9)`).
     - `accuracy`: SST-2 accuracy percentage.
     - `ppl`: WikiText-2 perplexity.
     - `latency_triton_ms`, `latency_pytorch_ms`, `speedup_x`.
     - `peak_vram_mb`, `throughput_tok_s`.
2. **`results/registry/master_ledger.jsonl`**:
   JSON Lines stream of physical execution events.
3. **`results/MASTER_RESULTS_MATRIX.csv`**:
   Canonical tabular CSV export containing the latest verified metrics.
4. **`results/SEQLEN_SCALING_MASTER.csv`**:
   Sequence scaling measurements across sequence lengths $N \in [512, 1024, 2048, 4096, 8192, 16384, 32768]$.
5. **`results/pareto_table.csv`**:
   Accuracy-latency Pareto frontier coordinates.

---

## 5. POSIX Forensic Inspection Recipes

Use these environment-agnostic command recipes to explore, index, and verify codebase elements.

### 5.1 Registry & Symbol Discovery

```bash
# 1. List all attention mechanisms registered via decorator
git grep -E '@ATTENTION_REGISTRY\.register\(' neuralops/

# 2. List all compressors registered via decorator
git grep -E '@COMPRESSOR_REGISTRY\.register\(' neuralops/

# 3. Search for a specific class definition across all layers
find neuralops/layers -name "*.py" -exec grep -Hn "class .*Attention" {} +

# 4. Find all capability flag declarations for attention operators
git grep -E "supports_causal|supports_gqa|supports_bidirectional" neuralops/layers/

# 5. Check whether a symbol has a legacy alias
grep -in -C 2 "symbol_name" neuralops/core/legacy_aliases.py neuralops/core/deprecated_aliases.json
```

### 5.2 Kernel Implementation Tracing

```bash
# 1. Locate all CUDA C++ kernel files (*.cu) for an operator
find neuralops/ops/cuda -name "*kernel*.cu" | grep -i "<operator_name>"

# 2. Inspect PyBind11 binding definitions for a CUDA package
grep -rn "PYBIND11_MODULE" neuralops/ops/cuda/<operator_name>/

# 3. Locate Triton JIT kernel functions decorated with @triton.jit
grep -rn "@triton.jit" neuralops/ops/triton/

# 4. Check whether an operator has a custom PyTorch library binding
grep -rn 'torch.library.custom_op("neuralops::' neuralops/ops/triton/

# 5. Inspect 3-level fallback dispatch calls for a kernel
grep -rn 'dispatch(' neuralops/ops/cuda/ -A 4 | grep -i "<operator_name>"
```

### 5.3 Empirical Results & Ledger Querying

```bash
# 1. Query the master SQLite ledger for a specific module's verified metrics
sqlite3 -header -column results/registry/master_ledger.sqlite \
  "SELECT module_name, family, accuracy, ppl, latency_triton_ms, speedup_x, peak_vram_mb, git_commit_sha \
   FROM results WHERE module_name = '<operator_name>' AND status LIKE 'VERIFIED%' LIMIT 1;"

# 2. Extract top 10 fastest attention mechanisms from the SQLite ledger
sqlite3 -header -column results/registry/master_ledger.sqlite \
  "SELECT module_name, family, latency_triton_ms, speedup_x, peak_vram_mb \
   FROM results WHERE category = 'attention' AND latency_triton_ms > 0 \
   ORDER BY latency_triton_ms ASC LIMIT 10;"

# 3. Filter CSV results for high-retention compression engines (>90% SST-2)
awk -F',' '$6 >= 90.0 {print $1, $3, $6, $10, $12}' results/MASTER_RESULTS_MATRIX.csv

# 4. Inspect sequence length scaling data for a model
grep "^<model_name>" results/SEQLEN_SCALING_MASTER.csv
```

### 5.4 LaTeX Source & Paper Cross-Referencing

```bash
# 1. Find the thesis chapter and line defining an equation or theorem
grep -rn "\\label{thm:<theorem_key>}" academic/thesis/chapters/

# 2. Search for all discussions of an operator in the thesis manuscript
grep -rn "<operator_key>" academic/thesis/chapters/

# 3. Verify that all LaTeX citations in a chapter exist in references.bib
# Extract all cite keys from a chapter and check references.bib
grep -oE '\\cite[a-z]*\{[^}]+\}' academic/thesis/chapters/chapter02_theory_part1.tex \
  | sed 's/\\cite[a-z]*{//;s/}//' | tr ',' '\n' | sort -u | while read -r key; do
    grep -q "{${key}," academic/thesis/references.bib || echo "Missing BibTeX entry: $key"
  done

# 4. Search for quantitative claims (e.g. speedup percentages) across the thesis
grep -En '[0-9]+\.[0-9]+(\\times|%|x)' academic/thesis/chapters/chapter04_results_*.tex
```

### 5.5 Repository Invariant Auditing via POSIX

```bash
# 1. Verify 131 registered attention operators in code
find neuralops/layers -name "*.py" ! -name "__init__.py" ! -name "base.py" | wc -l

# 2. Verify 72 registered compressor modules in code
find neuralops/compression -name "*.py" ! -name "__init__.py" ! -name "base.py" | wc -l

# 3. Verify 26 canonical Triton kernel modules
ls -1 neuralops/ops/triton/*.py | grep -v "__" | grep -v "_base" | grep -v "_types" | wc -l

# 4. Verify 112 native CUDA packages
find neuralops/ops/cuda -maxdepth 1 -mindepth 1 -type d ! -name "__*" | wc -l
```

---

## 6. Comprehensive Operator & Kernel Mapping Reference

Below is the verified correspondence matrix linking major operator families, registry keys, CUDA C++ packages, Triton implementations, and thesis documentation.

### 6.1 Flagship Attention Operators Mapping

| Registry Key | Mathematical Form | CUDA Package (`ops/cuda/`) | Triton Module (`ops/triton/`) | Thesis Section |
| :--- | :--- | :--- | :--- | :--- |
| `linear_attention` | $O = (Q (K^T V)) / (Q K^T \mathbf{1})$ | `linear_attention/` | `linear_attn.py` | Chap 2.1, 4.1 |
| `cosformer` | $\phi(x) = \text{ReLU}(x) \cos(\pi i / 2 M)$ | `cosformer/` | `cosformer_triton_kernel.py` | Chap 2.1, 4.2 |
| `gla` | $S_t = \alpha_t S_{t-1} + k_t^T v_t$ | `gla_scan/` | `gla_scan.py` | Chap 2.2, 4.2 |
| `gdn_scan` | Gated Delta Net recurrence | `gdn_scan/` | `gdn_scan.py` | Chap 2.2, 4.2 |
| `hgrn2_scan` | Hierarchical Gated RNN scan | `hgrn2_scan/` | `hgrn2_scan.py` | Chap 2.2, 4.1 |
| `selective_scan` | Mamba selective continuous scan | `selective_scan/` | `selective_scan.py` | Chap 2.2, 4.1 |
| `kda_attention` | Kronecker Delta log-space scan | `kronecker_gemm/` | `gla_scan.py` | Chap 2.2, 4.4 |
| `low_rank_svd` | Low-rank projected factorized att. | `low_rank_svd/` | `low_rank.py` | Chap 2.3, 4.3 |
| `sparse_attention`| Block-sparse static/adaptive mask | `sparse_attention/`| `block_sparse.py` | Chap 2.4, 4.2 |
| `moe_routing` | Top-$k$ soft routing GEMM | `moe_routing/` | `elementwise.py` | Chap 2.5, 4.1 |
| `flash_gqa` | Grouped query attention flash dec. | `flash_gqa/` | `gqa.py` | Chap 3.2, 4.2 |
| `differential_attention` | Diff-Transformer $A_1 - \lambda A_2$| `differential_attention/`| `differential_attention.py`| Chap 2.1, 4.1 |

### 6.2 Flagship Compression Engines Mapping

| Compressor Key | Algorithmic Mechanism | CUDA / C++ Implementation | Primary Source File | Thesis Section |
| :--- | :--- | :--- | :--- | :--- |
| `hnc_svd` | Whitened SVD with Stiefel opt. | `rpca_sparse_svd/` | `neuralops/compression/svd/hnc_svd.py` | Chap 2.3, 4.4 |
| `asvd_compressor` | Activation-aware SVD truncation | `low_rank_svd/` | `neuralops/compression/svd/asvd.py` | Chap 2.3, 4.3 |
| `whitened_svd` | Cholesky covariance pre-whitening | `low_rank_svd/` | `neuralops/compression/svd/whitened_svd.py` | Chap 2.3, 4.4 |
| `entropy_pruning`| Attention entropy-guided pruning | `entropy_pruning/` | `neuralops/compression/pruning/entropy.py` | Chap 3.3, 4.3 |
| `admm_pruner` | ADMM non-convex structured sparsity | `spectral_shrinkage/` | `neuralops/compression/pruning/admm.py` | Chap 3.3, 4.3 |
| `fp8_quantizer` | FP8 E4M3/E5M2 block scaling | `fp8_gemm/` | `neuralops/compression/quantization/fp8.py` | Chap 3.1, 4.2 |
| `cachenotes_compressor`| KV cache semantic eviction | `paged_kv_evict/` | `neuralops/compression/kv_cache/cachenotes.py` | Chap 3.2, 4.2b |
| `tt_compressor` | Tensor Train multi-way contraction | `tt_contraction/` | `neuralops/compression/tensor_decomp/tt.py` | Chap 2.3, 4.3 |

---

## 7. Verification & Paper Authoring Quality Gate

Before submitting any manuscript text, tables, or figures, execute the following quality gate:

```
[ ] 1. Every mathematical symbol matches the implementation tensor contracts:
       - Query, Key, Value shapes: (B, H, T, D) or (B, T, H, D).
       - Head dimension D is in SUPPORTED_HEAD_DIMS {32, 64, 128, 256}.
       - Recurrence state tensor dimensions match kernel shared memory allocations.

[ ] 2. Every quantitative claim matches the canonical ledgers:
       - SST-2 accuracy matches results/registry/master_ledger.sqlite within 0.01%.
       - Speedup factors (e.g. 1.74x) match relative latency measurements.
       - Memory consumption (Peak VRAM in MB) is verified against empirical logs.

[ ] 3. Execution backend is explicitly disambiguated:
       - Level 1 CUDA C++ speedups are distinguished from Level 2 Triton JIT speedups.
       - PyTorch Level 3 measurements are clearly identified as reference baselines.

[ ] 4. Canonical repository counts are preserved in text:
       - Exactly 131 attention mechanisms.
       - Exactly 72 model compression engines.
       - Exactly 26 canonical Triton kernel modules (housing 62 unified operators).
       - Exactly 112 native CUDA C++ kernel packages (housing 114 implementations).

[ ] 5. LaTeX typesetting conforms to publication standards:
       - All numeric values in tables use English math mode $ ... $ ($94.50\%$, $1.74\times$).
       - Zero overfull \hbox warnings (> 5pt).
       - All \cite and \ref labels resolve cleanly with 0 undefined references.
```
