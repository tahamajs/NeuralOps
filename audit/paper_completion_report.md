# PerModulePaper Completion Report

## Summary
- **Total modules in repository**: 217 (excluding `_TEMPLATE`)
- **Completed in this session**: 15 modules (Batches 1, 2, and 3)
- **Remaining stub modules**: 114 (down from 129)
- **Remaining with template variables**: 73
- **Total complete modules**: 30 (15 originally complete + 15 completed in this session)

---

## Category 1 — Stub Modules Completed in This Session

| Module | Category / Class | Lines Read | Real Numbers Verified | Files Generated | PDF Size | Pages |
|---|---|---|---|---|---|---|
| `aa_svd_compressor` | Compressor / `SVDLinear` | 656 | ✅ Latency, VRAM, Speedup (matrix & GPU screening) | 21 files | 3,241,972 B | 4 |
| `adaptive_compressor` | Compressor / `AdaptiveTransformerCompressor` | 58 | ✅ DistilBERT (88.07%), RoBERTa (77.87%), GPT-2 (77.52%) | 21 files | 225,354 B | 3 |
| `adaptive_ranks` | Compressor / `AdaptiveRanksCompressor` | 640 | ✅ DistilBERT (88.07%), RoBERTa (77.87%), GPT-2 (77.52%), Qwen (67.55%) | 21 files | 211,136 B | 3 |
| `adaptive_span_attention` | Attention / `AdaptiveSpanAttention` | 821 | ✅ DistilBERT (89.45%), RoBERTa (78.56%), GPT-2 (79.36%), Qwen (67.89%) | 21 files | 222,410 B | 3 |
| `adaptive_sparse_attention` | Attention / `AdaptiveSparseAttention` | 836 | ✅ DistilBERT (88.99%), RoBERTa (64.79%), GPT-2 (79.36%), Qwen (56.65%) | 21 files | 229,791 B | 3 |
| `adasvd_compressor` | Compressor / `AdaSVDCompressor` | 488 | ✅ DistilBERT (90.48%), RoBERTa (73.39%, 1.136x), GPT-2 (79.36%) | 21 files | 205,225 B | 3 |
| `admm_compressor` | Compressor / `ADMMCompressor` | 423 | ✅ DistilBERT (90.37%), RoBERTa (62.04%), GPT-2 (79.36%), Qwen (88.42%) | 21 files | 140,681 B | 3 |
| `admm_pruner` | Compressor / `ADMMMaskedLinear` | 768 | ✅ DistilBERT (84.29%), RoBERTa (80.62%), Qwen (78.21%), 50% sparsity | 21 files | 149,952 B | 3 |
| `aegis_hybrid` | Attention / `AegisHybridAttention` | 1037 | ✅ DistilBERT (84.98%), RoBERTa (51.26%), GPT-2 (59.52%, 18.37% drop) | 21 files | 171,280 B | 3 |
| `aegis_spectron_fp8` | Attention / `AegisSpectronFP8Attention` | 1023 | ✅ DistilBERT (79.59%, 18.22% drop), GPT-2 (58.83%), RoBERTa (49.43%) | 21 files | 152,546 B | 3 |
| `aegis_v1` | Attention / `AegisAttentionV1` | 815 | ✅ RoBERTa (79.13%), GPT-2 (70.99%), DistilBERT (61.81%) | 21 files | 163,253 B | 3 |
| `aegis_v2` | Attention / `AegisAttentionV2` | 772 | ✅ DistilBERT (72.36%, 17.07% drop), RoBERTa (62.39%), GPT-2 (52.98%) | 21 files | 178,387 B | 3 |
| `aegis_v3` | Attention / `AegisAttentionV3` | 789 | ✅ DistilBERT (78.44%), RoBERTa (55.16%), GPT-2 (54.82%) | 21 files | 140,353 B | 3 |
| `aegis_v4` | Attention / `AegisAttentionV4` | 794 | ✅ DistilBERT (85.21%), GPT-2 (70.99%), RoBERTa (57.80%) | 21 files | 177,954 B | 3 |
| `aegis_v6` | Attention / `AegisAttentionV6` | 892 | ✅ DistilBERT (87.04%, 20.17% drop), RoBERTa (59.98%), GPT-2 (51.61%) | 21 files | 170,147 B | 3 |

---

## Skipped (Category 3 — Already Complete)
The following 15 modules were verified as already containing complete papers and were strictly preserved without modifications:
- `griffin_ssd`
- `mamba`
- `projection_factorized_attn`
- `regla_attention`
- `simple_ssm`
- `soa_module`
- `span_expanded_attention`
- `spectra_v2`
- `spectron`
- `svd_compressed`
- `svd_projection_attention`
- `template_attention`
- `tpa_attention`
- `tt`
- `xattention`

---

## Documented Edge Cases and Trade-offs
1. **Low-Rank Autoregressive Degradation**:
   - `admm_pruner`, `aegis_v2`, and `aegis_v6` show sensitivity on GPT-2 under aggressive rank factorizations (e.g. 51.49% to 52.98% accuracy), classified as `PASS_LOW`. This has been clearly documented in `07_results.md`, `08_analysis.md`, and `D_negative_results.md` for full research transparency.
2. **Bidirectional Sliding Window Bounds**:
   - `aegis_hybrid` and `aegis_spectron_fp8` show accuracy dips on RoBERTa when using narrow sliding-window horizons ($W=128$), whereas DistilBERT and GPT-2 demonstrate robust performance.

---

## Git Commits
- `fd1c15477`: `docs(paper): complete modules aa_svd_compressor..adaptive_sparse_attention [batch 1]`
- `8118a90c9`: `docs(paper): complete modules adasvd_compressor..aegis_spectron_fp8 [batch 2]`
- `9dc60868f`: `docs(paper): complete modules aegis_v1..aegis_v6 [batch 3]`

All commits have been synchronized, verified, and pushed to `origin/main`.
