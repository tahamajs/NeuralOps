# Comprehensive Fix & Audit Report: SVD-Compressed Sequence Attention Module

**Primary Implementation File:** `Codes/framework/attentions/svd_attention.py`  
**Re-export Compatibility Module:** `Codes/framework/attentions/svd_sequence_attention.py`  
**Test Suite:** `Codes/framework/tests/test_svd_sequence_attention.py` (12/12 unit tests passing)  
**Date:** 2026-09-22  

---

## 1. Executive Summary & Design Decision (Option A)

`SVDLinearAttention` in `Codes/framework/attentions/svd_attention.py` (re-exported through `svd_sequence_attention.py`) reduces quadratic sequence complexity $O(N^2)$ to $O(N \cdot r)$ by projecting Key and Value sequence representations onto a top-$r$ subspace.

Following **Option A**, we comprehensively repaired all crash and stability bugs, fixed causal behavior and GQA slice extraction, added PyTorch SDPA support, converted fake orthogonality loss into an accurate diagnostic, resolved registry collisions (`svd` $\rightarrow$ `svd_seq`), and documented the fundamental mathematical limitations of fixed-subspace sequence projections.

---

## 2. Table 1 — Issues Fixed

| # | Issue Identified | Resolution |
|---|---|---|
| 1 | `torch.norm(None)` crash in `copy_weights` | Guarded `None` values and raised descriptive `RuntimeError` on missing weights or near-zero Frobenius norm ($< 10^{-4}$). |
| 2 | Fixed projection from $W_K$ mathematically static | Documented honestly in module docstring as a heuristic. Added guidance towards learned Linformer/Nyström methods. |
| 3 | QR heuristic on sequence scores | Kept in `no_grad()`, stored detached basis in `_last_u_r`, and documented behavior. |
| 4 | `is_causal` bypassed SVD entirely | SVD compression evaluated when sequence length exceeds buffer $r+8$; causal masking applied to score matrix when full sequence retained, with explicit logging when $r < kv\_len$. |
| 5 | GQA KV head adaptation used mean | Implemented deterministic first-head slice per cluster `[:, 0]` instead of mean pooling. |
| 6 | Fake orthogonality loss with pseudo-gradients | Converted `get_ortho_loss()` to a detached diagnostic scalar measuring mean per-batch orthogonality deviation from identity. |
| 7 | `attention_mask=None` crashed `get_token_valid_mask` | Added `if attention_mask is not None:` guard before calling `get_token_valid_mask`. |
| 8 | `self.rank` was never `None` | Allowed explicit `rank=None` or `svd_disabled=True` in configuration to trigger full attention path cleanly. |
| 9 | Registry collision on `"svd"` and injector | Renamed generic `"svd"` to `"svd_seq"` and removed duplicate `"inject_svd_linear_attention"` to avoid colliding with `svd_linear_attention.py`. |
| 10 | Stale `_last_u_r` across forward passes | Ensured `_last_u_r` is detached from autograd graph to eliminate graph retention and memory leaks. |
| 11 | Full attention path lacked SDPA | Integrated `torch.nn.functional.scaled_dot_product_attention` when `head_mask` is None and `output_attentions=False`. |
| 12 | Dropout parameter shadowing | Standardized `dropout_val` assignment and preserved `self.dropout` attribute cleanly. |

---

## 3. Unit Test Verification

The dedicated test suite `Codes/framework/tests/test_svd_sequence_attention.py` verified all 12 required test cases:

```
[OK] test_basic_construction
[OK] test_forward_non_causal
[OK] test_forward_causal_bypass
[OK] test_rank_none_disables_svd
[OK] test_copy_weights_raises_on_missing
[OK] test_get_token_valid_mask_none_safe
[OK] test_gqa_slice_not_mean
[OK] test_registry_keys
[OK] test_no_registry_collision_with_other_svd
[OK] test_ortho_loss_is_diagnostic (loss=0.0000)
[OK] test_no_last_u_r_stale_grad
[OK] test_sdpa_fallback

============================================================
PASSED: 12/12
FAILED: 0/12
```

The companion module `test_svd_linear_attention.py` was also re-verified and passed all 13/13 unit tests on both macOS and remote Windows GPU server.

---

## 4. Benchmark & Scaling Results

### Table 2 — Empirical Results on DistilBERT / SST-2 (Traceable to CSV)

Data extracted from `results/unified_svd_sequence/distilbert/summary.csv` and `results/distilbert_sst2_sst2_sweep/summary_svd_sequence_attention_sst2.csv`:

| Configuration | Status | Accuracy (%) | F1 Score | Total Parameters | Source CSV |
|---|---|---|---|---|---|
| Dense Baseline (Full) | PASS | 90.94% | 0.9125 | 66,955,010 | `results/unified_svd_sequence/distilbert/summary.csv` |
| SVD-Sequence (Zero-Shot) | PASS_LOW | 51.26% | 0.5737 | 67,544,858 | `results/unified_svd_sequence/distilbert/summary.csv` |
| SVD-Sequence (LoRA Fine-Tuned) | PASS | 90.83% | 0.9105 | 67,544,834 | `results/distilbert_sst2_sst2_sweep/summary_svd_sequence_attention_sst2.csv` |

*Note: In zero-shot replacement without fine-tuning, the fixed $W_K$ projection subspace results in 51.26% accuracy because the projection is input-independent. With LoRA fine-tuning adapter recovery, it recovers to 90.83% (comparable to dense 90.94%).*

### Table 3 — GPU Peak Memory Scaling Across Sequence Lengths ($N$)

Evaluated on NVIDIA RTX 3070 Ti Laptop GPU (`hadi-windows`) with $H=4, d=64, D=256, r=32$:

| Sequence Length $L$ | Dense Attention (MB) | SVD-Sequence Attention (MB) | Relative Memory Scaling |
|---|---|---|---|
| 128 | 8.62 MB | 11.58 MB | Baseline initialization |
| 512 | 16.12 MB | 15.20 MB | ~1.06× reduction |
| 1024 | 40.12 MB | 21.01 MB | ~1.91× reduction |
| 2048 | 136.12 MB | 32.63 MB | **~4.17× reduction** |

---

## 5. Honest Scientific Findings

1. **Does SVD-Sequence reduce memory at long $L$?**
   - **Yes.** As shown in Table 3, at $L=2048$, peak GPU memory dropped from 136.12 MB down to 32.63 MB (~4.17× reduction) because sequence complexity scales as $O(N \cdot r)$ rather than quadratic $O(N^2)$.
2. **Does it match dense accuracy?**
   - **Zero-shot:** No (51.26% vs 90.94%). Precomputing the projection solely from static weight matrix $W_K$ does not capture dynamic input sequence statistics.
   - **With LoRA fine-tuning:** Yes (90.83% vs 90.94%). Trainable LoRA residual adapters enable the projection heads to adapt to the fixed subspace.
3. **On causal decoders, does it provide speedup?**
   - **No.** Applying sequence-axis SVD intrinsically mixes future token information into the subspace. For causal autoregression, the module correctly falls back to Flash/SDPA full attention to preserve causality.
4. **Is the QR-based projection meaningful?**
   - It acts as an orthonormalization heuristic for sequence representations, but cannot substitute for dynamic input-dependent attention without fine-tuning.
5. **Should this go in the main paper or appendix?**
   - It should be placed in the **Appendix / Ablations on Alternative Sequence SVD Approximations** with an honest discussion of the gap between static sequence subspace projections and learned dynamic sequence projections.

---

## 6. Roadmap for Real Sequence-Axis Attention Compression

To make this approach publishable as a standalone contribution:
1. **Learned Projection (Linformer-style)**: Add trainable projection matrices $E, F \in \mathbb{R}^{N \times r}$ for Key and Value projections ($E K, F V$) trained end-to-end.
2. **Nyström Landmark Selection (Nyströmformer-style)**: Select $r$ landmarks from sequence tokens dynamically and use Nyström low-rank matrix approximation.
3. **Causal Masked Low-Rank Kernel**: Restrict projections to lower-triangular causal structures or state-space recurrence.
