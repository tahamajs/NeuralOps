# Wave 1 Audit Report: Attention Layers Sample (Agent B4)

**Agent ID:** Agent B4  
**Scope:** `neuralops/layers/**`, `tests/layers/**`, `audits/wave1/b4_layers.md`  
**Target Branch:** `fix/wave1-layers`  
**Date:** 2026-11-14  

---

## 1. Executive Summary

As part of the Wave 1 remediation, Agent B4 performed an audit on a representative sample of 20 attention mechanisms across dense, linear, sparse, modern, gated, and state-space model (SSM) families within `neuralops/layers/`.

Each operator was evaluated against standard benchmarking dimensions:
- Batch size: $B = 2$
- Heads: $H = 4$
- Sequence length: $N = 64$
- Head dimension: $D = 32$
- Total hidden dimension: $D_{\text{model}} = H \times D = 128$

All 20 sampled operators now pass forward execution, produce correct tensor shapes `(2, 64, 128)`, yield finite numerical values (zero NaNs or Infs), support PyTorch autograd backward passes, and pass automated smoke tests in `tests/layers/test_sampled_attentions.py`.

---

## 2. Sampled Attention Mechanisms Audit Matrix (20/20)

| # | Requested Name | Registered Name(s) | Class Name | Family | Input Shape $(B, N, D)$ | Output Shape | Finiteness (NaN/Inf free) | Autograd Grad Check | Status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `full_attention` | `full_attention`, `standard` | `FullAttention` | Dense | $(2, 64, 128)$ | $(2, 64, 128)$ | PASS (0 NaN, 0 Inf) | PASS | PASS |
| 2 | `cosformer` | `cosformer` | `CosformerAttention` | Linear | $(2, 64, 128)$ | $(2, 64, 128)$ | PASS (0 NaN, 0 Inf) | PASS | PASS |
| 3 | `gla` | `gla`, `gla_attention` | `GLAttention` | Linear | $(2, 64, 128)$ | $(2, 64, 128)$ | PASS (0 NaN, 0 Inf) | PASS | PASS |
| 4 | `mha` | `mha`, `full_attention` | `FullAttention` | Dense | $(2, 64, 128)$ | $(2, 64, 128)$ | PASS (0 NaN, 0 Inf) | PASS | PASS |
| 5 | `linear_attention` | `linear_attention`, `svd_linear_attention` | `SVDLinearAttention` | SVD / Linear | $(2, 64, 128)$ | $(2, 64, 128)$ | PASS (0 NaN, 0 Inf) | PASS | PASS |
| 6 | `sliding_window` | `sliding_window`, `sliding_window_attention` | `SlidingWindowAttention` | Modern | $(2, 64, 128)$ | $(2, 64, 128)$ | PASS (0 NaN, 0 Inf) | PASS | PASS (P0 Fixed) |
| 7 | `flash_attention` | `flash_attention`, `flash_attention_3` | `FlashAttention3` | Fast Kernels | $(2, 64, 128)$ | $(2, 64, 128)$ | PASS (0 NaN, 0 Inf) | PASS | PASS |
| 8 | `gqa` | `gqa`, `grouped_query_attention` | `GroupedQueryAttention` | Modern | $(2, 64, 128)$ | $(2, 64, 128)$ | PASS (0 NaN, 0 Inf) | PASS | PASS |
| 9 | `sparse_attention` | `sparse_attention`, `adaptive_sparse` | `AdaptiveSparseAttention` | Sparse | $(2, 64, 128)$ | $(2, 64, 128)$ | PASS (0 NaN, 0 Inf) | PASS | PASS |
| 10 | `nystrom` | `nystrom`, `nystrom_attention` | `NystromAttention` | Linear | $(2, 64, 128)$ | $(2, 64, 128)$ | PASS (0 NaN, 0 Inf) | PASS | PASS |
| 11 | `performer` | `performer`, `performer_attention` | `PerformerAttention` | Linear | $(2, 64, 128)$ | $(2, 64, 128)$ | PASS (0 NaN, 0 Inf) | PASS | PASS |
| 12 | `linformer` | `linformer`, `linformer_attention` | `LinformerAttention` | Linear | $(2, 64, 128)$ | $(2, 64, 128)$ | PASS (0 NaN, 0 Inf) | PASS | PASS |
| 13 | `rwkv` | `rwkv`, `rwkv6` | `RWKVAttention` | SSM / Recurrent | $(2, 64, 128)$ | $(2, 64, 128)$ | PASS (0 NaN, 0 Inf) | PASS | PASS (Layer Added) |
| 14 | `mamba` | `mamba`, `mamba_block` | `MambaBlock` | SSM | $(2, 64, 128)$ | $(2, 64, 128)$ | PASS (0 NaN, 0 Inf) | PASS | PASS |
| 15 | `mla` | `mla`, `mla_attention` | `MLAAttention` | Modern | $(2, 64, 128)$ | $(2, 64, 128)$ | PASS (0 NaN, 0 Inf) | PASS | PASS |
| 16 | `retnet` | `retnet`, `retnet_attention` | `RetNetAttention` | Linear / Retention | $(2, 64, 128)$ | $(2, 64, 128)$ | PASS (0 NaN, 0 Inf) | PASS | PASS (Layer Added) |
| 17 | `gated_deltanet` | `gated_deltanet` | `GatedDeltaNet` | Gated / Recurrent | $(2, 64, 128)$ | $(2, 64, 128)$ | PASS (0 NaN, 0 Inf) | PASS | PASS |
| 18 | `seesaw` | `seesaw`, `salt_attention` | `SALTAttention` | SVD / Adaptive | $(2, 64, 128)$ | $(2, 64, 128)$ | PASS (0 NaN, 0 Inf) | PASS | PASS |
| 19 | `softmax_attention` | `softmax_attention`, `full_attention` | `FullAttention` | Dense | $(2, 64, 128)$ | $(2, 64, 128)$ | PASS (0 NaN, 0 Inf) | PASS | PASS |
| 20 | `favor_plus` | `favor_plus`, `performer` | `PerformerAttention` | Linear | $(2, 64, 128)$ | $(2, 64, 128)$ | PASS (0 NaN, 0 Inf) | PASS | PASS |

---

## 3. P0 Bugs & Layer Remediation Details

### 3.1 P0 Crash: `SlidingWindowAttention` Parameter Type Mismatch
- **Location:** `neuralops/layers/modern/dynamic_conv_attention.py`
- **Issue:** When instantiated via registry or model replacement with a configuration object (`config`), `SlidingWindowAttention.__init__` accepted `window_size: int = 128` as its first positional parameter. When callers passed a config object (e.g. `BertConfig` or `MockConfig`), `self.window_size` became the config object itself. During `forward()`, `if N > self.window_size:` triggered a fatal exception:
  ```text
  TypeError: '>' not supported between instances of 'int' and 'BertConfig'
  ```
- **Remediation:**
  - Updated `SlidingWindowAttention` to inherit from `BaseAttention`.
  - Added polymorphic constructor accepting `config: Any = None`, extracting `hidden_size`, `num_attention_heads`, and `window_size` (defaulting to 128 if not an integer).
  - Explicitly registered aliases `"sliding_window"` and `"sliding_window_attention"`.

### 3.2 Missing BaseAttention Layers: `rwkv` and `retnet`
- **Issue:** Low-level Triton and CUDA kernels existed under `neuralops/ops/` (`rwkv6_scan` and `triton_multi_scale_retention`), but neither had standard `BaseAttention` neural network layer modules registered under `neuralops/layers/`. Callers attempting to benchmark or instantiate `rwkv` or `retnet` via `ATTENTION_REGISTRY` failed with `KeyError`.
- **Remediation:**
  - Implemented `neuralops/layers/ssm/rwkv_attention.py` (`RWKVAttention`), wrapping recurrent scan with CPU reference fallback, proper state tracking, and registration as `"rwkv"` and `"rwkv6"`.
  - Implemented `neuralops/layers/linear/retnet_attention.py` (`RetNetAttention`), providing multi-scale retention with decay masks, CPU reference fallback, and registration as `"retnet"` and `"retnet_attention"`.

### 3.3 Registry Canonical Alias Harmonization
- Registered canonical aliases for the sample specifications:
  - `FullAttention` registered as `"full_attention"`, `"mha"`, `"softmax_attention"`.
  - `SVDLinearAttention` registered as `"linear_attention"`.
  - `FlashAttention3` registered as `"flash_attention"`.
  - `GroupedQueryAttention` registered as `"gqa"`.
  - `AdaptiveSparseAttention` registered as `"sparse_attention"`.
  - `NystromAttention` registered as `"nystrom"`.
  - `PerformerAttention` registered as `"performer"`, `"favor_plus"`.
  - `LinformerAttention` registered as `"linformer"`.
  - `MambaBlock` registered as `"mamba"`.
  - `MLAAttention` registered as `"mla"`.
  - `GatedDeltaNet` registered as `"gated_deltanet"`.
  - `SALTAttention` registered as `"seesaw"`, `"salt_attention"`.

---

## 4. Extrapolation of Failure Risk for the Remaining 111 Layers

NeuralOps targets 131 canonical attention mechanisms in total. With 20 audited and verified in this sample, 111 remain. Based on the patterns identified during this audit, the risk distribution across the remaining layers is classified as follows:

### 4.1 High Risk: Circular Import Quarantine (16 layers, ~14.4%)
- **Mechanism:** As documented in Section 5 below, an out-of-scope circular import in `neuralops/core/attention_utils.py:32` prevents module discovery for 16 layers.
- **Affected layers:** `bigbird`, `fourier_mixer`, `hydra`, `longformer`, `deit`, `swin`, `fast_attention`, `memory_efficient_attention`, `local_attention`, `multi_query_attention`, `reformer`, `sinkhorn`, `synthesizer`, and 3 others.
- **Risk Assessment:** Critical. Any benchmark invocation requesting these layers will skip or raise `KeyError` until the core circular import is resolved.

### 4.2 Medium Risk: Constructor Signature Variance (~25 layers, ~22.5%)
- **Pattern:** Legacy layer classes often diverge from the canonical `BaseAttention(config, ...)` signature:
  - Some expect positional `hidden_size: int, num_heads: int`.
  - Some require explicit keyword arguments like `embed_dim`, `num_heads`, or `chunk_size`.
  - Some subclass `nn.Module` directly without `super().__init__(config)`.
- **Mitigation:** When Wave 2 expands layer coverage, all layers should adopt the polymorphic config/kwargs initialization pattern demonstrated in `SlidingWindowAttention`, `RWKVAttention`, and `RetNetAttention`.

### 4.3 Low Risk: GPU/Triton Hard Dependency without PyTorch Fallback (~15 layers, ~13.5%)
- **Pattern:** Certain fast kernel implementations assume CUDA availability and throw unhandled errors or missing kernel exceptions when evaluated on CPU or non-NVIDIA accelerators.
- **Mitigation:** Every layer module must implement or delegate to a PyTorch numerical reference fallback when `torch.cuda.is_available()` is `False` or Triton JIT is unsupported.

---

## 5. Out-of-Scope Bugs Found

Pursuant to the **Prime Directive** and **The Iron Law**, the following bug was discovered outside the owned scope (`neuralops/layers/**`, `tests/layers/**`) and was **NOT** modified:

### Out-of-Scope Bug 1: Circular Import in `neuralops/core/attention_utils.py`
- **File:** `neuralops/core/attention_utils.py` (Line 32)
- **Code:**
  ```python
  from neuralops.api import replace_attention
  ```
- **Call Chain:**
  1. `neuralops/layers/**` imports `from neuralops.core.attention_utils import apply_rotary_pos_emb, ...`
  2. `neuralops/core/attention_utils.py` executes line 32: `from neuralops.api import replace_attention`
  3. `neuralops/api.py` imports `from neuralops.registry import ATTENTION_REGISTRY`
  4. `neuralops/registry.py` executes `BenchmarkRegistry.populate_canonical_benchmarks()`, which loops over all files in `neuralops/layers/` and imports them.
  5. Subsequent layer files attempting to import from `neuralops.core.attention_utils` fail because `attention_utils` has not finished initializing its module namespace (cannot import `apply_rotary_pos_emb` or other symbols defined below line 32).
  6. The `registry.py` loader swallows these failures with `warnings.warn(f"Failed to load canonical operator {name}: {e}")`, leaving 16 operators unpopulated in `ATTENTION_REGISTRY`.
- **Recommended Fix (for Agent owning `neuralops/core/**`):**
  Move `from neuralops.api import replace_attention` inside the specific function in `attention_utils.py` where it is called (lazy import), or decouple replacement utilities from tensor/rope helper functions.
