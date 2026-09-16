# AG News Benchmark Sweep: Diagnostics & Failures Report

This report summarizes the outcome of analyzing all historical and current runs (over 330 unique modules across DistilBERT, RoBERTa, GPT-2, and T5).

## 1. Active AG News Sweep Status

During the current active multi-class sweep, the following modules are exhibiting critical failures or low accuracy.

### 🔴 Critical Failures & Crashes (0.00% Accuracy)
*These modules abort before finishing training due to OOMs, dimension mismatches, or numerical instability.*

| Module | Model | Status | Accuracy | Root Cause Analysis |
|---|---|---|---|---|
| `hgrn2_gated_deltanet` | `distilbert` | FAIL | 0.00% | Recurrence state explosion/numerical overflow in bidirectional context. |
| `hgrn2_layer` | `distilbert` | SKIP_BERT | 0.00% | Architecture requires strictly causal masking; incompatible with bidirectional encoders. |

### ⚠️ Sub-Threshold Accuracy (< 50%)
*These modules complete training but fail to recover basic representational capacity (the pass threshold is 31.63%, but anything under 50% on AG News indicates severe catastrophic forgetting).*

| Module | Model | Status | Accuracy | Suspected Issue |
|---|---|---|---|---|
| `mla` | `distilbert` | PASS_LOW | 31.60% | KV-projection compression ratio is too aggressive for dense classification. |
| `decay_linear_attention` | `distilbert` | PASS_NO_RECOVERY | 34.10% | Exponential decay matrix dampens key tokens in bidirectional context. |
| `gqa` | `distilbert` | PASS | 35.90% | Grouped query attention sharing causes semantic blurring in small models. |
| `galerkin_attention` | `distilbert` | PASS_NO_RECOVERY | 36.00% | Continuous kernel integral operator fails to capture localized cues. |
| `svd_linear` | `test_agnews_svd` | PASS | 36.30% | Unrecovered low-rank projection without sufficient LoRA adaptation steps. |
| `svd_llm_v2` | `distilbert` | PASS | 43.00% | Rank truncation discards high-variance sentiment/topic dimensions. |
| `svd_mamba` | `distilbert` | PASS_NO_RECOVERY | 49.60% | State-space selective scan under-trained on short sequence classification. |

---

## 2. Global Codebase Analysis (All Backbones)

Aggregating all historical runs reveals two persistent failure modes across the entire repository.

### Category A: The "Never Passing" Modules
There are **76 modules** that have **never** achieved >50% accuracy on any backbone or dataset. These fall into three structural groups:
1. **Incompatible SSMs on Bidirectional Backbones:** `jamba2`, `decay_gated_linear_attention`, `gfws_d`, and `spectral_residual_svd` crash on BERT/RoBERTa due to causal assumptions or float16 eigenvalue divergence.
2. **Dynamic Quantization Failures:** `dynamic_entropy`, `entropy_budget_allocation`, `erc_svd`, and `kmeans_quantize` crash consistently due to undefined behavior during backwards passes.
3. **Legacy `*_inject` Wrapper Stubs:** Modules like `adaptive_span_inject`, `adasplash_inject`, `cluster_inject`, etc., are failing because they are looking for Qwen-specific `.q_proj` hardcoded paths instead of using the new `UniversalAdapter`.

### Category B: The "Slow Death" Compressors
These modules execute without throwing errors but suffer massive catastrophic forgetting (average accuracy < 35%):
- **Early Aegis Prototypes (`aegis_v1` to `v4`)**: 9%–18% accuracy (gradients zero out in feature maps).
- **A3 & AASVD Compressors**: 10%–17% accuracy (fixed-rank SVD truncation destroys essential residual pathways).
- **Sparse & Windowed Compressors**: 9%–18% accuracy (dynamic span gates collapse to minimum window length).
- **Early MoE Variants (`bam_umoe`, `cur_moe`)**: 16%–35% accuracy (router collapse, all tokens go to one uninitialized expert).

## 3. Recommended Next Steps & Remediation

1. **Auto-Retries**: The orchestrator is already configured to automatically retry modules below 50% with `recovery_steps=50` (up from 25) and a higher learning rate.
2. **Deprecate `*_inject` Modules**: We should remove all standalone `*_inject` files and map those keys directly to the base classes that use `UniversalAdapter`.
3. **SSM/Recurrent Performance Fix**: I have successfully patched `GatedDeltaNet` and `GatedDeltaNetBlock` to re-enable gradient checkpointing (fixing the micro-batch cascading OOM) while explicitly freeing intermediate states (fixing the GPT-2 specific FP32 bloat).
