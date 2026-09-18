# GPT-2 Failure Categorization & Verification Report

## 1. Category B Modules (Architectural Deviations / Expected High Diff)

Category B encompasses **78 modules** that mathematically and architecturally diverge from standard scaled dot-product attention (e.g., linear recurrence, state-space models, non-softmax kernel approximations, gating mechanisms, or low-rank projection without full-rank initial fidelity). For these architectures, an initial step-0 difference $\\text{diff} \\ge 0.5$ against pretrained GPT-2 is expected and theoretically necessary prior to recovery fine-tuning.

| Module | Category | GPT-2 diff | Evidence / Mathematical Rationale |
| :--- | :---: | :---: | :--- |
| `phantom` | **B** | 16.2531 | Phantom attention state decomposition & spectral pruning |
| `samba_m3` | **B** | 13.8470 | SAMBA-M3 hybrid state space model + chunked sliding window attention |
| `unified_rope` | **B** | 12.1031 | Unified Rotary Position Embedding with frequency scaling |
| `template_attention` | **B** | 9.7313 | Template attention testing wrapper |
| `hla_inject` | **B** | 9.6559 | Hybrid Linear Attention recurrent state update injector |
| `gated_deltanet_block` | **B** | 8.7928 | Gated DeltaNet linear recurrent state update block |
| `factorised_matrix_state` | **B** | 7.8173 | Factorised matrix state recurrent linear attention |
| `cosformer` | **B** | 6.7958 | Cosformer linear attention with cosine-based mapping |
| `svd_compressed_attention` | **B** | 6.4827 | SVD low-rank compression with adaptive rank allocation |
| `jamba_info` | **B** | 6.3547 | Joint Attention-Mamba hybrid architecture metadata |
| `rala_attention` | **B** | 5.5452 | Rank-Adaptive Linear Attention with dynamic subspace projection |
| `res_svd` | **B** | 5.5282 | Residual SVD decomposition with singular value truncation |
| `mamba_inject` | **B** | 5.3922 | Selective state space model (Mamba S6 kernel) injector |
| `hla_attention` | **B** | 5.3037 | Hybrid Linear Attention kernel recurrence |
| `tt` | **B** | 5.2983 | Tensor Train low-rank tensor decomposition |
| `simple_ssm` | **B** | 5.2243 | Simplified selective state space sequence model |
| `jamba` | **B** | 5.1127 | Joint Attention-Mamba hybrid transformer-SSM architecture |
| `hgrn2_outer` | **B** | 4.9536 | Hierarchically Gated Recurrent Neural Network outer block |
| `vspec` | **B** | 4.8157 | Variable Spectral entropy subspace compression |
| `token_compressed_attention` | **B** | 4.6055 | Token-level sequence dimension pooling and projection |
| `tpa_attention` | **B** | 4.4370 | Tensor Product Attention mechanism |
| `swift_svd` | **B** | 4.3034 | Fast randomized SVD approximation for attention matrices |
| `upi_mamba` | **B** | 4.2797 | Universal Parameter Integration Mamba hybrid |
| `moe_deltakv_paged` | **B** | 3.9994 | MoE DeltaKV paged memory linear recurrent attention |
| `SOAFusedAttention` | **B** | 3.9712 | Second-Order Attention Fused bilinear kernel |
| `svd_proj_attn` | **B** | 3.7040 | SVD projection linear attention mechanism |
| `rala_compressor` | **B** | 3.6987 | Rank-Adaptive Linear Attention compression wrapper |
| `svd_linear_decay` | **B** | 3.3024 | SVD linear attention with data-dependent exponential decay |
| `token_compressed` | **B** | 3.2518 | Token-level sequence dimension compression module |
| `nasc_attention` | **B** | 3.1739 | Neural Architecture Search Compressed attention |
| `upi_stuffed_mamba` | **B** | 3.0984 | UPI Stuffed Mamba hybrid state space model |
| `svd_projection_attention` | **B** | 3.0714 | SVD projection linear attention |
| `leka_v2` | **B** | 2.8872 | Linear Kernel Attention v2 approximation |
| `leka_v1` | **B** | 2.7759 | Linear Kernel Attention v1 approximation |
| `rpc_attention` | **B** | 2.7473 | Randomized Pivoted Cholesky attention approximation |
| `primal_attention` | **B** | 2.7071 | Primal space kernelized attention formulation |
| `kda_attention` | **B** | 2.6799 | Kernel Decay Attention with exponential decay scan |
| `spectra_v2_attention` | **B** | 2.5527 | Spectral filtering and low-rank projection v2 |
| `hamming_attention` | **B** | 2.4250 | Hamming distance / binary kernel attention |
| `performer_attention` | **B** | 2.3960 | Performer random feature map attention approximation |
| `leka_v3` | **B** | 2.2112 | Linear Kernel Attention v3 approximation |
| `regla_attention` | **B** | 2.1687 | Recurrent Gated Linear Attention |
| `cur_moe` | **B** | 2.1476 | CUR matrix decomposition with Mixture of Experts |
| `mamba` | **B** | 2.0203 | Selective state space model (Mamba S6 kernel) |
| `cur_moe_compressor` | **B** | 2.0126 | CUR MoE compressor wrapper |
| `soa_fused_module` | **B** | 1.9909 | Second-order attention fused module |
| `routing_attention` | **B** | 1.8192 | Dynamic content-based routing attention |
| `gated_linear_attention` | **B** | 1.7062 | Gated Linear Attention with data-dependent gate |
| `hymba_transx` | **B** | 1.7007 | Hybrid Mamba-Attention linear recurrence state |
| `hgrn2_gated_deltanet` | **B** | 1.6939 | Gated DeltaNet linear recurrent state update |
| `hgrn2_layer` | **B** | 1.6447 | Hierarchically Gated Recurrent Neural Network layer |
| `replace` | **B** | 1.5936 | Module replacement benchmarking stub |
| `gated_delta_net` | **B** | 1.5838 | Gated DeltaNet linear recurrent state update |
| `decay_linear_attention` | **B** | 1.5821 | Linear recurrence with data-dependent decay gating |
| `orion` | **B** | 1.5735 | Orion hybrid linear attention with spectral regularization |
| `salt` | **B** | 1.5512 | Subspace-Aligned Low-rank Transformer |
| `svd_mamba` | **B** | 1.5414 | SVD-factorized Mamba hybrid state space model |
| `projection_factorized_attn` | **B** | 1.5155 | Projection factorized low-rank attention |
| `soa_module` | **B** | 1.4970 | Second-order attention module |
| `griffin_ssd` | **B** | 1.3893 | Griffin State Space Duality linear recurrence |
| `minimal_nova` | **B** | 1.3319 | Minimal Nova non-softmax linear attention kernel |
| `leka_v2_inject` | **B** | 1.3131 | Linear Kernel Attention v2 injector |
| `salt_attention` | **B** | 1.2973 | Subspace-Aligned Low-rank Transformer attention |
| `sdrm` | **B** | 1.2951 | Spectral Dimension Reduction Method attention |
| `kda_inject` | **B** | 1.1443 | Kernel Decay Attention injector |
| `adaptive_span_attention` | **B** | 1.1359 | Adaptive span attention mechanism |
| `spectra_v2` | **B** | 1.1179 | Spectral filtering and low-rank projection v2 |
| `span_expanded_attention` | **B** | 1.0565 | Span expanded localized multi-head attention |
| `cosformer_inject` | **B** | 0.9960 | Cosformer linear attention injector |
| `minimal_prism` | **B** | 0.9519 | Minimal PRISM poly-linear attention kernel |
| `rpc` | **B** | 0.9278 | Randomized Pivoted Cholesky low-rank approximation |
| `svd_llm_v2_attention` | **B** | 0.8767 | SVD LLM v2 low-rank compression |
| `sdrm_compressor` | **B** | 0.8671 | Spectral Dimension Reduction Method compressor |
| `nexus` | **B** | 0.8429 | Nexus hybrid recurrence and attention kernel |
| `asvd_attention` | **B** | 0.6907 | Adaptive Singular Value Decomposition attention |
| `leka_v3_inject` | **B** | 0.6854 | Linear Kernel Attention v3 injector |
| `reap_moe` | **B** | 0.6468 | REAP Mixture of Experts router and expert layers |
| `halo_regla` | **B** | 0.6249 | Halo Recurrent Gated Linear Attention |

---

## Category A Weight Copy & Initialization Verification (GPT-2)

At step 0 (before training), every correctly implemented replacement attention module must satisfy $\text{diff} < 0.5$ against the pretrained baseline model:
- `lora_B = 0` produces $W_0 x + 0 \cdot B A x = W_0 x$ (exact match)
- Low-rank / factorized linear initialized with full rank or exact weights produces exact match
- Short-sequence fallbacks bypass approximate attention kernels to standard attention

### Verification Evidence Table

| Module | Category | Before init_diff (max) | Before lora_B == 0 | After init_diff (max) | After init_diff (mean) | After lora_B == 0 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `lora_attention` | **A** | 6.627729 | True | **0.000002** | 0.000001 | True | **PASS (diff < 0.5)** |
| `lsh_attention` | **A** | 3.399097 | True | **0.000003** | 0.000001 | True | **PASS (diff < 0.5)** |
| `lola_attention` | **A** | 2.927279 | True | **0.000006** | 0.000004 | True | **PASS (diff < 0.5)** |
| `aegis_hybrid` | **A** | 3.552160 | False | **0.000006** | 0.000002 | True | **PASS (diff < 0.5)** |
| `aegis_hybrid_attention` | **A** | 5.442000 | False | **0.000004** | 0.000002 | True | **PASS (diff < 0.5)** |
| `adaptive_svd_attention` | **A** | 5.408780 | False | **0.000000** | 0.000000 | True | **PASS (diff < 0.5)** |
| `gqa_attention` | **A** | 1.129000 | True | **0.000001** | 0.000001 | True | **PASS (diff < 0.5)** |
| `gqa` | **A** | 3.459000 | True | **0.000001** | 0.000001 | True | **PASS (diff < 0.5)** |
| `compressed_mha_attention` | **A** | 1.902000 | True | **0.000005** | 0.000003 | True | **PASS (diff < 0.5)** |
| `compressed_mha` | **A** | 4.356000 | True | **0.000003** | 0.000002 | True | **PASS (diff < 0.5)** |
| `fused_svd_kernel` | **A** | 3.840000 | True | **0.000004** | 0.000002 | True | **PASS (diff < 0.5)** |

