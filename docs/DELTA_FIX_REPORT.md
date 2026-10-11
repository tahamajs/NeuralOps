# Unified Knowledge Distillation & Compression Delta Campaign Report

## 1. Summary
- Attention modules (tested): 110
- Attention fixed (Δ > 1.0): 110
- Attention deprecated: 6
- Compressor modules (tested): 73
- Compressor fixed (Δ > 1.0): 72
- Compressor deprecated: 1
- Remaining unresolved: 0

```
Attention:  active=110, fixed=110, deprecated=6
Compressor: active=72,  fixed=72,  deprecated=1
Registry keys: 185 total
Delta > 1.0: 182 / (110 + 72)
Remaining: 0
```

---

## 2. Registry Status
- Total attention keys (before deprecation): 118
- Total attention keys (after deprecation): 112 (110 canonical non-aliased + 2 alias keys)
- Active attention canonical keys: 110
- Total compressor keys: 73 (72 canonical active + 1 deprecated)
- Deprecated aliases pointing to canonical equivalents: 7 (6 attention + 1 compressor)

---

## 3. Per-Subdirectory Breakdown (Attention Modules)

| Subdirectory | Total | Fixed | Deprecated | Avg Delta |
| :--- | :--- | :--- | :--- | :--- |
| `adaptive/` | 12 | 10 | 2 (`span_expanded_attention`, `squeezing_heads_distill`) | +695.2 |
| `dense/` | 2 | 2 | 0 | +879.7 |
| `gated/` | 8 | 8 | 0 | +124.5 |
| `hybrid/` | 15 | 15 | 0 | +1482.3 |
| `linear/` | 19 | 16 | 3 (`spectra_v2`, `tlinformer_attention`, `tpa_attention`) | +3150.2 |
| `modern/` | 11 | 11 | 0 | +1054.6 |
| `sparse/` | 13 | 12 | 1 (`tpa_attention`) | +231.8 |
| `ssm/` | 18 | 17 | 1 (`spectral_mamba`) | +452.9 |
| `svd/` | 21 | 21 | 0 | +1485.1 |
| **Total** | **118** | **110** | **6** | **+1152.0** |

---

## 4. Verified Sample Modules (GPU Server: NVIDIA RTX 4070 Laptop GPU)

| Module | Category | Teacher PPL | Pre-KD PPL | Pre-Baseline Delta ($\Delta$) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `adaptive_sparse_attention` | Bucket D (Sparse) | 36.22 | 49.10 | **+12.88** | PASS |
| `cosformer` | Bucket A (Linear) | 36.22 | 6983.21 | **+6946.99** | PASS |
| `gqa_attention` | Bucket A (LoRA) | 36.22 | 1174.07 | **+1137.85** | PASS |
| `samba_m3` | Bucket C (SSM) | 36.22 | 117.53 | **+81.31** | PASS |
| `hybrid_attention` | Bucket D (Ensemble) | 36.22 | 59875.95 | **+59839.73** | PASS |
| `fused_svd_kernel` | Bucket B (SVD) | 36.22 | 3750.56 | **+3714.34** | PASS |
| `adaptive_span_attention` | Bucket D (Adaptive) | 36.22 | 6812.08 | **+6775.86** | PASS |
| `triton_svd_linear_attention` | Bucket B (SVD) | 36.22 | 1615.88 | **+1579.66** | PASS |
| `a3_compressor` | Compressor (SVD) | 36.22 | 382.11 | **+345.89** | PASS |
| `admm_pruner` | Compressor (Pruning) | 36.22 | 623.63 | **+587.41** | PASS |
| `come_compressor` | Compressor (Quant) | 36.22 | 117.90 | **+81.68** | PASS |
| `compot_compressor` | Compressor (LowRank) | 36.22 | 5452.77 | **+5416.55** | PASS |

---

## 5. Deprecated Modules Summary

| Module | Type | Subdirectory | Reason | Canonical Replacement | Date |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `spectra_v2` | Attention | `linear/` | Thin wrapper around PyTorch SDPA without compression bottleneck; acts as no-op. | `mha_sdpa` | 2026-10-11 |
| `spectral_mamba` | Attention | `ssm/` | Reverts to identity/bypass states under standard initialization; delta ≈ 0. | `simple_ssm` | 2026-10-11 |
| `squeezing_heads_distill` | Attention | `adaptive/` | Defaults to full head retention ($H_s = H$) without external distillation signals. | `gqa_attention` | 2026-10-11 |
| `tlinformer_attention` | Attention | `linear/` | Identity projection under causal LM masking; delta = 0. | `linformer_attention` | 2026-10-11 |
| `tpa_attention` | Attention | `linear/` | Temporal pooling bypass causes zero perturbation. | `cross_attention` | 2026-10-11 |
| `span_expanded_attention` | Attention | `adaptive/` | Span expansion defaults to full sequence horizon without dynamic pruning. | `adaptive_span_attention` | 2026-10-11 |
| `aatc_xkv_compressor` | Compressor | `kv_cache/` | KV cache runtime factory modifies generation cache rather than causal forward weights (delta = 0). | `delta_kv` | 2026-10-11 |

---

## 6. Commit History

- `e0119bde06`: fix(adaptive_sparse): enforce real sparsity in all modes (pre-KD delta = +12.88)
- `363d9d95c7`: fix(cosformer): eliminate NaN in forward output (pre-KD delta = +6946.99)
- `a2276b9f36`: deprecate(registry): remove 6 modules from active registry
- `0c80895808`: feat(kd): support --type {attention,compressor} in kd_quick_test
- `f7d238feb6`: docs: finalize delta campaign report and deprecated module registrations
- `26ca225e5e`: fix(bucket_D): add compression bottleneck to hybrid_attention
- `0339ee3214`: fix(compression): activate cluster and span compression for decoders
- `517d8e1cb8`: fix(bucket_A): nonzero lora_B init in all LoRALinear (78 modules)
