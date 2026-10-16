# SPECTRA-v2 / MHA-SDPA Attention Module Audit & Fix Report

**Evaluator:** Antigravity AI  
**Date:** 2026-09-22  
**Target Modules:** `Codes/framework/attentions/spectra_v2.py`, `Codes/framework/attentions/spectra_v2_attention.py`

---

## 1. Executive Summary & Reality Check

The file `spectra_v2.py` previously claimed in its docstring and config to implement:
- Sparse per-head efficient context transformer
- Intra-block local attention
- Top-$k$ inter-block routing
- Relative position bias
- Per-head adaptive attention spans
- Periodic Random Fourier Feature (RFF) rectification

**Reality:** None of these components were implemented. In `forward()`, none of the 10 `SpectraConfig` attributes (`block_size`, `top_k_inter`, `use_rel_pos_bias`, `rel_pos_max_distance`, `adaptive_span`, `max_span`, `rectify_interval`, `rff_features`, `aks_enabled`) were referenced. The module was simply standard multi-head attention using PyTorch's `scaled_dot_product_attention` (SDPA).

To eliminate false claims while ensuring functional integrity and benchmark reproducibility:
1. **Renamed Canonical Key:** Registered as `mha_sdpa` with `spectra_v2` preserved as a deprecated alias.
2. **Clarified Docstring:** Updated with an honest disclaimer explaining that this is a clean MHA-SDPA wrapper.
3. **Resolved 11 Hidden Bugs:** Fixed missing logger import, undefined `is_cross` variable, ignored `attention_mask` in SDPA, ignored `head_mask` in SDPA, GQA weight copy averaging, loud Frobenius norm error handling, removed fake `get_ortho_loss`, removed dimension mismatch padding hacks, and purged unused config dataclasses.
4. **Verification:** All 9 unit tests in `Codes/framework/tests/test_spectra_v2.py` passed 100%.

---

## 2. Claim vs. Reality Table

| Feature | Claimed in Docstring | Actually Implemented |
|---|:---:|:---:|
| Sparse Attention | ✓ | ✗ |
| Intra-block Local Attention | ✓ | ✗ |
| Top-$k$ Inter-block Routing | ✓ | ✗ |
| Relative Position Bias | ✓ | ✗ |
| Per-head Adaptive Spans | ✓ | ✗ |
| Global RFF Rectification | ✓ | ✗ |
| Config Fields Declared vs. Used | 10 | 0 |

---

## 3. Bug Fixes

| # | Bug | Original State | Resolution |
|---|---|---|---|
| 1 | `logger` undefined | Used without import | Added `import logging; logger = logging.getLogger(__name__)` |
| 2 | `is_cross` undefined | Used in ternary condition without definition | Correctly defined: `is_cross = encoder_hidden_states is not None` |
| 3 | `attention_mask` ignored | SDPA called without mask argument | Formatted 2D/4D boolean mask and passed as `attn_mask` to SDPA |
| 4 | `head_mask` ignored | SDPA bypassed head masking entirely | Applied per-head mask post-SDPA across heads |
| 5 | Return type inconsistency | Varied between branches | Unified tuple return `(output, attn_weights_or_None)` |
| 6 | GQA weight copy | Repeated KV heads across Q groups | Mean-averaged head groups when source heads > target heads |
| 7 | Silent Frobenius norm failure | Warned and continued | Raises `RuntimeError` if non-zero source weight collapses to norm $< 10^{-4}$ |
| 8 | Fake `get_ortho_loss` | Returned constant `torch.tensor(0.0)` | Removed method completely |
| 9 | Shape mismatch padding | Masked dimensional errors via `F.pad` | Removed silent padding; raises `RuntimeError` on mismatch |
| 10 | Unused config dataclass | `SpectraConfig` declared 10 dead fields | Removed `SpectraConfig` dataclass |
| 11 | Misleading registry key | Named `spectra_v2` | Primary key `mha_sdpa`, alias `spectra_v2` |

---

## 4. Unit Test Verification

Running `python3 Codes/framework/tests/test_spectra_v2.py`:
```text
[OK] test_basic_construction
[OK] test_forward_shape
[OK] test_attention_mask_respected (diff=3.67e-01)
[OK] test_causal_flag (err=0.00e+00)
[OK] test_head_mask
[OK] test_gqa_copy_averaging
[OK] test_no_fake_loss
[OK] test_no_config_dataclass
[OK] test_deprecated_alias_works

============================================================
PASSED: 9/9
FAILED: 0/9
```

---

## 5. Roadmap for Full SPECTRA Architecture (Option C)

If the research team decides to build the genuine SPECTRA architecture in the future, the required development phases are estimated as follows:
- **Phase 1 (1 week):** Intra-block local attention with block reshaping ($B=64$).
- **Phase 2 (1 week):** Block summary extraction and top-$k$ inter-block sparse routing ($k=16$).
- **Phase 3 (1 week):** Learned relative position bias matrices capped at `rel_pos_max_distance`.
- **Phase 4 (1 week):** Soft-masking adaptive span kernels per attention head.
- **Phase 5 (1 week):** Periodic Random Fourier Feature (RFF) kernel rectification across layers.
- **Phase 6 (1-2 weeks):** Multi-task ablation suite on Long-Range Arena (LRA) and GLUE benchmarks.
- **Total Estimated Effort:** 6–8 engineer-weeks.
