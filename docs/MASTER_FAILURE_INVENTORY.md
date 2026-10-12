# MASTER FAILURE INVENTORY

Generated from independent raw disk audit of `results/delta_sweep/*.log` and `.err` files on the GPU server.

## Summary by Root Cause Category

| Category | Count | Primary Fix Strategy |
|---|---|---|
| ROOT-CAUSE-1: Zero-Delta / Fallback / No-Op Forward | 16 | Ensure compression/projection modifies weights without silent fallback |
| ROOT-CAUSE-2: Tensor Shape / Dimension Mismatch (.view -> .reshape) | 2 | Replace `.view(...)` with `.reshape(...)` or `.contiguous().view(...)` |
| ROOT-CAUSE-3: Negative Delta (Post-KD Degradation / Explosion) | 8 | Warm-start gates / identity init / stabilize KD loss to prevent degradation |
| ROOT-CAUSE-4: Registry / Injection Target Mismatch | 2 | Align registry string and injector target block class name |
| ROOT-CAUSE-5: Index / Slice Bounds Violation | 0 | Clamp indices and validate tensor dimensions before indexing |
| ROOT-CAUSE-8: CUDA Out of Memory | 0 | Reduce test batch size (`--batch_size 4`) or gradient accumulation |
| ROOT-CAUSE-9: Numerical Instability / Severe Loss Divergence | 29 | Add eps clamping / nan_to_num guards / initialize residual gate safely |
| ROOT-CAUSE-10: Process Interrupted / Stalled during Sweep | 2 | Ensure compression/projection modifies weights without silent fallback |
| **Total Failing / Incomplete** | **59** | — |

## ROOT-CAUSE-1: Zero-Delta / Fallback / No-Op Forward (16 modules)

| Module | Type | Observed Delta | Specific Diagnostic Notes |
|---|---|---|---|
| `cosformer` | attention | 0.00 | Delta=0.00 |
| `cosformer_attention` | attention | 0.00 | Delta=0.00 |
| `mgie_budget` | compressor | 0.00 | Delta=0.00 |
| `residual_svd_compressor` | compressor | 0.00 | Delta=0.00 |
| `roast_hashed_linear` | compressor | 0.00 | Delta=0.00 |
| `rpca_compressor` | compressor | 0.00 | Delta=0.00 |
| `saes_svd_v2` | compressor | 0.00 | Delta=0.00 |
| `salr` | compressor | 0.00 | Delta=0.00 |
| `slicegpt` | compressor | 0.00 | Delta=0.00 |
| `slicegpt_pca` | compressor | 0.00 | Delta=0.00 |
| `subspace_projection` | compressor | 0.00 | Delta=0.00 |
| `svd_linear` | compressor | 0.00 | Delta=0.00 |
| `svd_llm_compressor` | compressor | 0.00 | Delta=0.00 |
| `swift_svd_act` | compressor | 0.00 | Delta=0.00 |
| `tomoe_compressor` | compressor | 0.00 | Delta=0.00 |
| `tt_compressor` | compressor | 0.00 | Delta=0.00 |

## ROOT-CAUSE-2: Tensor Shape / Dimension Mismatch (.view -> .reshape) (2 modules)

| Module | Type | Observed Delta | Specific Diagnostic Notes |
|---|---|---|---|
| `granite_hybrid` | attention | None | Non-contiguous view |
| `linformer_attention` | attention | None | Non-contiguous view |

## ROOT-CAUSE-3: Negative Delta (Post-KD Degradation / Explosion) (8 modules)

| Module | Type | Observed Delta | Specific Diagnostic Notes |
|---|---|---|---|
| `hla_attention` | attention | -1355.86 | Delta=-1355.86 |
| `nexus` | attention | -67082535795.75 | Delta=-67082535795.75 |
| `regla_attention` | attention | -1609.93 | Delta=-1609.93 |
| `tri_branch_hybrid` | attention | -9574.44 | Delta=-9574.44 |
| `aa_svd_compressor` | compressor | -3776.92 | Delta=-3776.92 |
| `lsc_compressor` | compressor | -26.56 | Delta=-26.56 |
| `soka_compressor` | compressor | -2.99 | Delta=-2.99 |
| `uniql_compressor` | compressor | -3.50 | Delta=-3.50 |

## ROOT-CAUSE-4: Registry / Injection Target Mismatch (2 modules)

| Module | Type | Observed Delta | Specific Diagnostic Notes |
|---|---|---|---|
| `full` | attention | None | Target lookup failure |
| `mamba_svd_flash` | attention | None | Target lookup failure |

## ROOT-CAUSE-9: Numerical Instability / Severe Loss Divergence (29 modules)

| Module | Type | Observed Delta | Specific Diagnostic Notes |
|---|---|---|---|
| `adaptive_sparse_attention` | unknown | None | Incomplete/Divergent |
| `adaptive_svd_attention` | unknown | None | Incomplete/Divergent |
| `adasplash_attention` | unknown | None | Incomplete/Divergent |
| `aegis_v6` | unknown | None | Incomplete/Divergent |
| `alibi` | unknown | None | Incomplete/Divergent |
| `asvd_attention` | unknown | None | Incomplete/Divergent |
| `dynamic_conv_attention` | attention | None | Incomplete/Divergent |
| `halo_regla` | attention | None | Incomplete/Divergent |
| `hamming_attention` | attention | None | Incomplete/Divergent |
| `hgrn2_gated_deltanet` | attention | None | Incomplete/Divergent |
| `hybrid_attention` | attention | None | Incomplete/Divergent |
| `hymba_transx` | attention | None | Incomplete/Divergent |
| `kda_attention` | attention | None | Incomplete/Divergent |
| `nasc_attention` | attention | None | Incomplete/Divergent |
| `performer_attention` | attention | None | Incomplete/Divergent |
| `rala_attention` | attention | None | Incomplete/Divergent |
| `rala_compressor` | attention | None | Incomplete/Divergent |
| `selective_ssm` | attention | None | Incomplete/Divergent |
| `simple_ssm` | attention | None | Incomplete/Divergent |
| `spectron_fp8` | attention | None | Incomplete/Divergent |
| `svd_mamba` | attention | None | Incomplete/Divergent |
| `zeros_attention` | attention | None | Incomplete/Divergent |
| `lillama_compressor` | compressor | None | Incomplete/Divergent |
| `span_expanded_attention` | unknown | None | Incomplete/Divergent |
| `spectra_v2` | unknown | None | Incomplete/Divergent |
| `spectral_mamba` | unknown | None | Incomplete/Divergent |
| `squeezing_heads_distill` | unknown | None | Incomplete/Divergent |
| `tlinformer_attention` | unknown | None | Incomplete/Divergent |
| `tpa_attention` | unknown | None | Incomplete/Divergent |

## ROOT-CAUSE-10: Process Interrupted / Stalled during Sweep (2 modules)

| Module | Type | Observed Delta | Specific Diagnostic Notes |
|---|---|---|---|
| `factorised_matrix_state` | attention | None | SIGINT/Timeout |
| `inject_leka_v1` | attention | None | SIGINT/Timeout |

