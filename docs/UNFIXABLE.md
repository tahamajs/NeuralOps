# Deprecated & Unfixable Attention Modules

This document tracks modules identified as unfixable or unsuitable for delta/compression benchmarks after rigorous multi-pass evaluation and verification on the GPU server.

| Module | Subdirectory | Root Cause / Rationale | Canonical Alternative |
| :--- | :--- | :--- | :--- |
| `spectra_v2` | `spectral/` | Thin wrapper around PyTorch `scaled_dot_product_attention` by design; no compression, rank bottleneck, or sparsification. Delta ≈ 0. | `mha_sdpa` |
| `spectral_mamba` | `spectral/` | Reverts to identity/bypass states under standard initialization; delta ≈ 0. | `simple_ssm` |
| `squeezing_heads_distill` | `adaptive/` | Pre-initialized with full head retention ($H_s = H$) under default configuration without distillation signals; delta ≈ 0. | `gqa_attention` |
| `tlinformer_attention` | `linear/` | Identity projection under causal LM masking; delta ≈ 0. | `linformer_attention` |
| `tpa_attention` | `sparse/` | Marked deprecated in codebase; temporal pooling bypass causes delta = 0. | `cross_attention` |
| `mamba_svd_flash` | `svd/` | Deprecated flash SSM kernel variant superseded by simple_ssm. | `simple_ssm` |
| `aatc_xkv_compressor` | `kv_cache/` | Runtime dynamic KV-cache quantizer; does not modify static weight parameters, resulting in identical forward pass (delta = 0). | `cachenotes_compressor` |
| `hybridkv_compressor` | `kv_cache/` | Runtime dynamic key-value cache buffer compressor; static forward pass unchanged without generation cache (delta = 0). | `delta_kv` |

All listed modules have been mapped to canonical alternatives in `neuralops/core/deprecated_aliases.json` and registered in `eval_types.DEPRECATED_MODULES`.
