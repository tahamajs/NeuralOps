# Attention & Compression Module Families

## SVD & Low-Rank Approximations

Spectral decompositions and singular value truncation with rank adaptation.

| Module Key | Paper Reference |
|:---|:---|
| `adaptive_svd` | N/A |
| `adaptive_svd_attention` | N/A |
| `adaptive_svd_attn` | N/A |
| `asvd` | ASVD: Activation-aware Singular Value Decomposition for Compressing LLMs |
| `asvd_attention` | N/A |
| `chela_slay` | CHELA: Channel Elimination and Layer Shedding for Accelerating Transformers |
| `chela_slay_attention` | N/A |
| `fused_svd_kernel` | Fused SVD Kernels for Hardware-Accelerated Neural Compression |
| `fused_svd_kernel_attention` | N/A |
| `inject_triton_svd_linear_attention` | N/A |
| `mamba_svd_flash` | N/A |
| `minimal_svd` | N/A |
| `naive_svd` | N/A |
| `residual_svd_attention` | N/A |
| `salt_attention` | N/A |
| `soa_attn` | N/A |
| `svd_compressed` | N/A |
| `svd_linear` | N/A |
| `svd_linear_attention` | N/A |
| `svd_linear_attn` | N/A |
| `svd_linear_decay` | N/A |
| `svd_llm_v2_attention` | N/A |
| `svd_mamba` | N/A |
| `svd_projection_attention` | N/A |
| `svd_seq` | N/A |
| `triton_svd_linear_attention` | N/A |
| `triton_svd_linear_attention_inject` | N/A |
| `aa_svd_compressor` | AASVD: Adaptive Activation SVD with Covariance Regularization |
| `adasvd_compressor` | N/A |
| `cr_compressor` | N/A |
| `cur_compressor` | CUR Matrix Decomposition for Transformer Compression |
| `cur_moe_compressor` | N/A |
| `direct_svd` | N/A |
| `dobi_svd` | N/A |
| `erc_svd` | N/A |
| `fwsvd_compressor` | N/A |
| `hybrid_svd_layer` | N/A |
| `mlorq_compressor` | N/A |
| `residual_svd_compressor` | N/A |
| `saes_svd_v2` | N/A |
| `svd_linear` | N/A |
| `svd_llm_compressor` | N/A |
| `swift_svd` | N/A |
| `swift_svd_act` | N/A |

## Linear & Subquadratic Attention

Kernelized feature maps and low-rank projections enabling O(N) context complexity.

| Module Key | Paper Reference |
|:---|:---|
| `cosformer` | Cosformer: Rethinking Softmax in Attention with Linear Complexity |
| `decay_linear_attention` | Linear Attention with Decay |
| `decay_linear_attn` | N/A |
| `galerkin_attention` | Galerkin Transformer: A Framework for Approximating Continuous Attention |
| `gated_linear_attention` | Gated Linear Attention Transformers with Hardware-Efficient Training |
| `inject_triton_cosformer` | N/A |
| `inject_triton_svd_linear_attention` | N/A |
| `linformer_attention` | Linformer: Self-Attention with Linear Complexity |
| `nystrom` | Nystromformer: A Nystrom-Based Algorithm for Approximating Self-Attention |
| `nystrom_attention` | Nystromformer: A Nystrom-Based Algorithm for Approximating Self-Attention |
| `performer` | Rethinking Attention with Performers |
| `performer_attention` | Rethinking Attention with Performers |
| `svd_linear_attention` | N/A |
| `tlinformer_attention` | N/A |
| `triton_cosformer` | N/A |
| `triton_cosformer_inject` | N/A |
| `triton_svd_linear_attention` | N/A |
| `triton_svd_linear_attention_inject` | N/A |

## Sparse & Dynamic Routing Attention

Dynamic key selection, block-sparse patterns, and multi-head routing heuristics.

| Module Key | Paper Reference |
|:---|:---|
| `adaptive_sparse` | N/A |
| `adaptive_sparse_attention` | N/A |
| `adaptive_sparse_attention_attention` | N/A |
| `adaptive_sparse_attn` | N/A |
| `cluster` | Cluster-Former: Clustering-based Sparse Transformer |
| `cluster_attention` | N/A |
| `halo_regla` | N/A |
| `mla_attention` | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model |
| `moba_attention` | N/A |
| `nsa` | Native Sparse Attention: Hardware-Aligned and Topologically Diverse Sparse Attention |
| `nsa_attention` | N/A |
| `routing_attention` | N/A |
| `bonsai_pruner` | Bonsai: Efficient Parameter Pruning for Transformers |
| `proxsparse` | N/A |
| `sparsegpt` | SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot |

## State-Space Models & Hybrid Recurrence

Continuous-time state-space parameterizations and linear recurrent neural models.

| Module Key | Paper Reference |
|:---|:---|
| `griffin_ssd` | Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models |
| `hgrn2_gated_deltanet` | HGRN2: Gated Linear RNNs with State Expansion |
| `hgrn2_gated_layer` | N/A |
| `hgrn2_gated_outer` | N/A |
| `hgrn2_outer` | HGRN2: Gated Linear RNNs with State Expansion |
| `hymba_transx` | Hymba: A Hybrid-Head Architecture for Small Language Models |
| `hymba_transx_attn` | N/A |
| `jamba` | Jamba: A Hybrid Transformer-Mamba Language Model |
| `jamba_info` | N/A |
| `mamba_block` | N/A |
| `mamba_svd_flash` | N/A |
| `samba_m3` | Samba: Simple Hybrid State Space Models for Efficient Language Modeling |
| `samba_moe` | Samba: Simple Hybrid State Space Models for Efficient Language Modeling |
| `selective_mamba` | N/A |
| `simple_ssm` | Simple Hardware-Efficient State Space Models |
| `spectral_mamba` | N/A |
| `svd_mamba` | N/A |
| `upi_mamba` | N/A |
| `upi_stuffed_mamba` | N/A |
| `mamba_shedder` | N/A |

## Grouped & Multi-Query Attention

Key-value head sharing to reduce inference memory footprint and memory-bandwidth saturation.

| Module Key | Paper Reference |
|:---|:---|
| `gqa` | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints |
| `gqa_attention` | N/A |
| `wgqa` | N/A |
| `wgqa_attention` | N/A |

