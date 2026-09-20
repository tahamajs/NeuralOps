# Audit Report: Literature and Citation Coverage Gaps

## Table A: Modules with Zero Bibliographic Citations in Paper Suite

Total Modules with Zero References: 147

| Module | Category | Expected Topic | Recommended Papers to Add |
| :--- | :---: | :---: | :--- |
| `a3` | PASS | T1 | Add canonical T1 references (e.g. SVD-LLM, ASVD, Mamba) |
| `aa_svd_compressor` | Cat A | T1 | Add canonical T1 references (e.g. SVD-LLM, ASVD, Mamba) |
| `adaptive_compressor` | PASS | T1 | Add canonical T1 references (e.g. SVD-LLM, ASVD, Mamba) |
| `adaptive_ranks` | PASS | T1 | Add canonical T1 references (e.g. SVD-LLM, ASVD, Mamba) |
| `adaptive_span_attention` | Cat A | T8 | Add canonical T8 references (e.g. SVD-LLM, ASVD, Mamba) |
| `adaptive_sparse_attention` | Cat C | T8 | Add canonical T8 references (e.g. SVD-LLM, ASVD, Mamba) |
| `adasvd_compressor` | Cat A | T1 | Add canonical T1 references (e.g. SVD-LLM, ASVD, Mamba) |
| `admm_compressor` | PASS | T5 | Add canonical T5 references (e.g. SVD-LLM, ASVD, Mamba) |
| `admm_pruner` | PASS | T5 | Add canonical T5 references (e.g. SVD-LLM, ASVD, Mamba) |
| `aegis_hybrid` | PASS | T10 | Add canonical T10 references (e.g. SVD-LLM, ASVD, Mamba) |
| `aegis_spectron_fp8` | PASS_LOW | T8 | Add canonical T8 references (e.g. SVD-LLM, ASVD, Mamba) |
| `aegis_v1` | Cat B | T1 | Add canonical T1 references (e.g. SVD-LLM, ASVD, Mamba) |
| `aegis_v2` | PASS | T1 | Add canonical T1 references (e.g. SVD-LLM, ASVD, Mamba) |
| `aegis_v3` | PASS | T1 | Add canonical T1 references (e.g. SVD-LLM, ASVD, Mamba) |
| `aegis_v4` | PASS | T1 | Add canonical T1 references (e.g. SVD-LLM, ASVD, Mamba) |
| `aegis_v6` | PASS | T1 | Add canonical T1 references (e.g. SVD-LLM, ASVD, Mamba) |
| `armor_pruner` | PASS | T5 | Add canonical T5 references (e.g. SVD-LLM, ASVD, Mamba) |
| `ast_pruner` | PASS | T5 | Add canonical T5 references (e.g. SVD-LLM, ASVD, Mamba) |
| `astro_quantizer` | PASS | T5 | Add canonical T5 references (e.g. SVD-LLM, ASVD, Mamba) |
| `block_transformer_hiba` | Cat A | T9 | Add canonical T9 references (e.g. SVD-LLM, ASVD, Mamba) |
| `bonsai_pruner` | Cat A | T5 | Add canonical T5 references (e.g. SVD-LLM, ASVD, Mamba) |
| `cachenotes_compressor` | PASS | T7 | Add canonical T7 references (e.g. SVD-LLM, ASVD, Mamba) |
| `come` | PASS | T1 | Add canonical T1 references (e.g. SVD-LLM, ASVD, Mamba) |
| `compot_compressor` | PASS | T1 | Add canonical T1 references (e.g. SVD-LLM, ASVD, Mamba) |
| `compressed_attention` | PASS_LOW | T8 | Add canonical T8 references (e.g. SVD-LLM, ASVD, Mamba) |
| `coopq_quantizer` | PASS | T8 | Add canonical T8 references (e.g. SVD-LLM, ASVD, Mamba) |
| `cr_compressor` | PASS | T1 | Add canonical T1 references (e.g. SVD-LLM, ASVD, Mamba) |
| `cur_compressor` | PASS | T1 | Add canonical T1 references (e.g. SVD-LLM, ASVD, Mamba) |
| `cur_linear` | PASS | T1 | Add canonical T1 references (e.g. SVD-LLM, ASVD, Mamba) |
| `cur_moe` | PASS | T4 | Add canonical T4 references (e.g. SVD-LLM, ASVD, Mamba) |
| ... and 117 additional modules | ... | ... | ... |

---

## Table B: Under-Covered Taxonomy Topics (< 5 Papers)

| Topic | Paper Count | Suggested Additional Papers |
| :---: | :---: | :--- |
| **T7** | 4 | Add specialized literature for T7 |
| **T9** | 2 | Add specialized literature for T9 |
| **T10** | 0 | Add specialized literature for T10 |
| **T12** | 4 | Add specialized literature for T12 |

---

## Table C: FAIL / PASS_LOW Modules Precedent Mapping

| Module | Failure Mode | Attempted Search Queries | Resolved Precedent Paper |
| :--- | :--- | :--- | :--- |
| `primal` | cuBLAS heuristic fail in FP16 | Schulz iteration FP16 cuBLASLt heuristic | **Xiong et al. (AAAI 2021)** |
| `sdrm_class` | Div-by-zero gradient NaN | Unconstrained SVD rank truncation singular value gradient divergence | **Wang et al. (ICLR 2024)** |
| `spectral_mamba` | Ambiguous tensor boolean check | Selective state space kwargs parsing | **Gu & Dao (2023)** |
| `spectraformer` | Kwargs unpacking mismatch | Fourier transform token mixing parameterization | **Lee-Thorp et al. (NAACL 2022)** |
| `inject_spectral_mamba` | Shared spectral wrapper kwargs | Injected spectral SSM layer | **Dao & Gu (ICML 2024)** |
| `hgrn2_outer` | cuBLAS non-contiguous stride fail | Batched outer product memory layout | **Qin et al. (2024)** |
| `mamba_svd_flash` | Multiplicative dimension explosion OOM | State dimension expansion d x r | **Gu et al. (ICLR 2022)** |
| `mamba / inject_mamba` | 34.1% MNLI (Causal scan collapse) | Causal SSM in bidirectional encoder | **Zhu et al. (ICML 2024)** |
| `kronecker_linear_class` | 34.0% MNLI (Gradient vanishing) | Kronecker factorization zero singular value plateaus | **Edalati et al. (EMNLP 2022)** |
| `phantom` | 32.1% MNLI (Exponential FP16 NaN) | Exponential subspace scaling FP16 overflow | **Micikevicius et al. (ICLR 2018)** |
| `mla` | 35.4% MNLI (Latent KV rank loss) | Post-hoc latent KV bottleneck rank starvation | **DeepSeek-AI (2024)** |
| `comp_attn_layer_energy` | 26.8% MNLI (Rank under-allocation) | Frobenius energy rank under-allocation on head | **Yuan et al. (2023)** |
| `aegis_spectron_fp8` | 27.7% MNLI (Dynamic scale clipping) | FP8 E4M3 outlier activation clipping | **Xiao et al. (ICML 2023)** |
| `cosformer` | 37.1% MNLI (Softmax linearization) | Softmax linearization cross-sentence alignment loss | **Qin et al. (ICLR 2022)** |
| `simple_ssm` | 50.9% SST-2 (Scalar decay capacity) | Single scalar diagonal recurrence expressivity | **Gu et al. (ICLR 2022)** |
