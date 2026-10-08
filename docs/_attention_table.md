| # | Key | Class | Module | Causal | GQA | RoPE | Description |
|---:|---|---|---|:---:|:---:|:---:|---|
| 1 | `adaptive_span_attention` | `AdaptiveSpanAttention` | `adaptive.adaptive_span_attention` | ✓ | ✓ | ✓ |  |
| 2 | `adaptive_sparse_attention` | `AdaptiveSparseAttention` | `sparse.adaptive_sparse_attention` | ✓ | ✓ | ✓ |  |
| 3 | `adaptive_sparse_attn` | `AdaptiveSparseAttention` | `sparse.adaptive_sparse_attention` | ✓ | ✓ | ✓ |  |
| 4 | `adaptive_svd` | `AdaptiveSVDAttention` | `svd.adaptive_svd_attention` | ✓ | ✓ | ✓ | Multi-head attention with adaptive SVD + residual factorized projections. |
| 5 | `adaptive_svd_attention` | `AdaptiveSVDAttention` | `svd.adaptive_svd_attention` | ✓ | ✓ | ✓ | Multi-head attention with adaptive SVD + residual factorized projections. |
| 6 | `adasplash` | `AdaSplashAttention` | `fast_kernels.adasplash_attention` | ✓ | ✓ | ✓ | AdaSplash Attention Layer with α-Entmax and Local Splash Radius. |
| 7 | `adasplash_attention` | `AdaSplashAttention` | `fast_kernels.adasplash_attention` | ✓ | ✓ | ✓ | AdaSplash Attention Layer with α-Entmax and Local Splash Radius. |
| 8 | `alibi_attention` | `ALiBiAttention` | `modern.alibi` | ✓ |  |  | Attention with linear biases instead of positional embeddings. |
| 9 | `asvd` | `ASVDAttention` | `svd.asvd_attention` |  |  |  | Hardened ASVD Attention with Factored Low-Rank Matrix Decomposition, GQA, RoPE,  |
| 10 | `asvd_attention` | `ASVDAttention` | `svd.asvd_attention` |  |  |  | Hardened ASVD Attention with Factored Low-Rank Matrix Decomposition, GQA, RoPE,  |
| 11 | `bam_umoe` | `BAMUMoEBlock` | `sparse.bam_umoe` |  |  |  |  |
| 12 | `block_transformer_hiba` | `BlockTransformerHIBA` | `adaptive.block_transformer_hiba` |  |  |  | Block Transformer with Hierarchical Intra-Block and Inter-Block Attention (HIBA) |
| 13 | `chela_slay` | `ChelaSlayAttention` | `linear.chela_slay` |  |  |  | CHELA-SLAY: Spherical Linearized Attention with Yat-Kernel. |
| 14 | `chela_slay_attention` | `ChelaSlayAttention` | `linear.chela_slay` |  |  |  | CHELA-SLAY: Spherical Linearized Attention with Yat-Kernel. |
| 15 | `cluster` | `ClusterAttention` | `sparse.cluster_attention` | ✓ | ✓ | ✓ |  |
| 16 | `cluster_attention` | `ClusterAttention` | `sparse.cluster_attention` | ✓ | ✓ | ✓ |  |
| 17 | `compressed_attention` | `CompressedAttentionLayer` | `svd.compressed_attention` | ✓ | ✓ | ✓ |  |
| 18 | `compressed_attention_layer` | `CompressedAttentionLayer` | `svd.compressed_attention` | ✓ | ✓ | ✓ |  |
| 19 | `compressed_attention_layer_energy` | `CompressedAttentionLayer2` | `svd.compressed_attention` | ✓ | ✓ | ✓ |  |
| 20 | `compressed_mha` | `CompressedDistilBertAttention` | `svd.compressed_mha_attention` | ✓ | ✓ | ✓ |  |
| 21 | `compressed_mha_attention` | `CompressedDistilBertAttention` | `svd.compressed_mha_attention` | ✓ | ✓ | ✓ |  |
| 22 | `cosformer` | `CosformerAttention` | `linear.cosformer` |  |  |  |  |
| 23 | `cross_attention` | `CrossAttention` | `modern.cross_attention` |  | ✓ |  | Multi-head cross attention between query sequence and encoder/context sequence. |
| 24 | `decay_linear_attention` | `DecayLinearAttention` | `gated.decay_linear_attention` |  |  |  | Decay Linear Attention Module with Data-Dependent Exponential Decay Gating. |
| 25 | `decay_linear_attn` | `DecayLinearAttention` | `gated.decay_linear_attention` |  |  |  | Decay Linear Attention Module with Data-Dependent Exponential Decay Gating. |
| 26 | `eat` | `EATBlock` | `adaptive.eat` |  |  |  |  |
| 27 | `eat_attention` | `EATBlock` | `adaptive.eat` |  |  |  |  |
| 28 | `expert_choice_moe` | `ExpertChoiceMoEAttention` | `sparse.expert_choice_moe` |  |  |  | Expert Choice MoE Multi-Head Attention Module. |
| 29 | `expert_choice_moe_attention` | `ExpertChoiceMoEAttention` | `sparse.expert_choice_moe` |  |  |  | Expert Choice MoE Multi-Head Attention Module. |
| 30 | `factorised_matrix_state` | `FactorisedMatrixState` | `svd.factorised_matrix_state` |  |  |  | Bidirectional & Causal Factorised Matrix State Recurrent Linear Attention. |
| 31 | `flash3` | `FlashAttention3` | `fast_kernels.flash_attention3` |  |  |  | FlashAttention-3 Multi-Head Exact Attention. |
| 32 | `flash3_fp8` | `FlashAttention3FP8` | `fast_kernels.flash_attention3_fp8` |  |  |  | FlashAttention-3 FP8 Hardware-Accelerated Multi-Head Attention. |
| 33 | `flash_attention3` | `FlashAttention3` | `fast_kernels.flash_attention3` |  |  |  | FlashAttention-3 Multi-Head Exact Attention. |
| 34 | `flash_attention3_fp8` | `FlashAttention3FP8` | `fast_kernels.flash_attention3_fp8` |  |  |  | FlashAttention-3 FP8 Hardware-Accelerated Multi-Head Attention. |
| 35 | `fms` | `FactorisedMatrixState` | `svd.factorised_matrix_state` |  |  |  | Bidirectional & Causal Factorised Matrix State Recurrent Linear Attention. |
| 36 | `full` | `FullAttention` | `dense.full_attention` | ✓ | ✓ | ✓ |  |
| 37 | `full_attention` | `FullAttention` | `dense.full_attention` | ✓ | ✓ | ✓ |  |
| 38 | `fused_svd_kernel` | `FusedSVDKernelAttention` | `svd.fused_svd_kernel` | ✓ | ✓ |  | Fused SVD-Kernel Linear Attention Module. |
| 39 | `fused_svd_kernel_attention` | `FusedSVDKernelAttention` | `svd.fused_svd_kernel` | ✓ | ✓ |  | Fused SVD-Kernel Linear Attention Module. |
| 40 | `fusion_attention` | `FUSIONATTN` | `hybrid.fusion_attn` | ✓ | ✓ |  | Hybrid sparse attention with block gating. |
| 41 | `fusion_attn` | `FUSIONATTN` | `hybrid.fusion_attn` | ✓ | ✓ |  | Hybrid sparse attention with block gating. |
| 42 | `galerkin_attention` | `GalerkinAttention` | `linear.galerkin_attention` | ✓ | ✓ |  |  |
| 43 | `gated_delta_net` | `GatedDeltaNet` | `gated.gated_delta_net` |  |  |  |  |
| 44 | `gated_deltanet_block` | `GatedDeltaNetBlock` | `gated.gated_deltanet_block` |  |  |  | Gated DeltaNet Transformer Block with Pretrained Weight Transfer. |
| 45 | `gated_hybrid_kernel` | `GatedHybridKernelAttention` | `hybrid.gated_hybrid_kernel_attention` | ✓ | ✓ |  |  |
| 46 | `gated_linear_attention` | `GatedLinearAttention` | `gated.gated_linear_attention` | ✓ | ✓ |  |  |
| 47 | `goldfinch` | `GoldFinchAHNBlock` | `ssm.goldfinch_ahn` |  |  |  | GoldFinch-AHN: hybrid sliding-window attention + gated recurrent memory. |
| 48 | `goldfinch_ahn` | `GoldFinchAHNBlock` | `ssm.goldfinch_ahn` |  |  |  | GoldFinch-AHN: hybrid sliding-window attention + gated recurrent memory. |
| 49 | `gqa_attention` | `GroupedQueryAttention` | `modern.gqa_attention` | ✓ | ✓ | ✓ |  |
| 50 | `granite_hybrid` | `GraniteHybridBlock` | `hybrid.granite_hybrid` |  |  |  | Granite Hybrid Attention-SSM Layer with MoE Feed-Forward and LoRA Adaptation. |
| 51 | `griffin_ssd` | `GriffinSSDBlock` | `ssm.griffin_ssd` |  |  |  | Griffin Hybrid Block: local window + SSD linear attention. |
| 52 | `grouped_query_attention` | `GroupedQueryAttention` | `modern.gqa_attention` | ✓ | ✓ | ✓ |  |
| 53 | `halo_regla` | `HALOReGLAAttention` | `hybrid.halo_regla` |  |  |  | Hybrid Dense Softmax + Linear Attention with adaptive fusion gate. |
| 54 | `hamming_attention` | `HammingAttention` | `sparse.hamming_attention` | ✓ | ✓ | ✓ |  |
| 55 | `hgrn2_gated_deltanet` | `HGRN2GatedDeltaNet` | `gated.hgrn2_gated_deltanet` |  |  |  | HGRN-2 + Gated DeltaNet hybrid with proper fusion. |
| 56 | `hgrn2_gated_layer` | `HGRN2Layer` | `gated.hgrn2_gated_layer` |  |  |  | HGRN-2 hybrid with local sliding-window attention. |
| 57 | `hgrn2_gated_outer` | `HGRN2GatedOuterBlock` | `gated.hgrn2_gated_outer` | ✓ | ✓ |  |  |
| 58 | `hgrn2_outer` | `HGRN2GatedOuterBlock` | `gated.hgrn2_gated_outer` | ✓ | ✓ |  |  |
| 59 | `hla_attention` | `HLAAttention` | `gated.hla_attention` | ✓ | ✓ |  | Hadamard Linear Attention (HLA) Bidirectional / Causal Module. |
| 60 | `hybrid_attention` | `HybridAttention` | `hybrid.hybrid_attention` | ✓ | ✓ | ✓ | Hybrid softmax + linear attention with per-head dynamic gated fusion. |
| 61 | `hybrid_local_kernel` | `HybridLocalKernelAttention` | `hybrid.hybrid_local_kernel_attention` | ✓ | ✓ |  |  |
| 62 | `hymba_transx` | `HymbaTransXBlock` | `ssm.hymba_transx` |  |  |  | HymbaTransX: Hybrid Differential Transformer & State Space Model Block. |
| 63 | `hymba_transx_attn` | `HymbaTransXBlock` | `ssm.hymba_transx` |  |  |  | HymbaTransX: Hybrid Differential Transformer & State Space Model Block. |
| 64 | `inject_triton_cosformer` | `TritonCosformerAttention` | `linear.triton_cosformer` |  |  |  | Triton-Accelerated Cosformer Linear Attention with Full Autograd Differentiabili |
| 65 | `inject_triton_svd_linear_attention` | `inject_triton_svd_linear_attention` | `svd.triton_svd_linear_attention` |  |  |  | Safe factory function to inject TritonSVDLinearAttention into a model. |
| 66 | `inject_xattention` | `inject_xattention` | `fast_kernels.xattention` |  |  |  |  |
| 67 | `jamba` | `Jamba2Block` | `adaptive.infini_attention` |  |  |  |  |
| 68 | `jamba_info` | `JambaInfoBlock` | `ssm.jamba_info` |  |  |  |  |
| 69 | `kda_attention` | `KDAAttention` | `gated.kda_attention` |  |  |  |  |
| 70 | `landmark_local` | `LandmarkLocalAttention` | `hybrid.landmark_local_attention` | ✓ | ✓ |  |  |
| 71 | `linformer_attention` | `LinformerAttention` | `linear.linformer_attention` | ✓ | ✓ |  |  |
| 72 | `lola_attention` | `LoLAAttention` | `adaptive.lola_attention` | ✓ | ✓ | ✓ |  |
| 73 | `lora_attention` | `LoRAAttention` | `svd.lora_attention` | ✓ | ✓ | ✓ | LoRA attention wrapper. |
| 74 | `lora_attn` | `LoRAAttention` | `svd.lora_attention` | ✓ | ✓ | ✓ | LoRA attention wrapper. |
| 75 | `lsh_attention` | `LSHAttention` | `sparse.lsh_attention` | ✓ | ✓ |  |  |
| 76 | `mamba_block` | `MambaBlock` | `ssm.mamba_block` | ✓ |  |  | Faithful S6 / Mamba block, production-hardened. |
| 77 | `mamba_svd_flash` | `MambaSVDFlash` | `ssm.mamba_svd_flash` |  |  |  | Hybrid SSM + MHA with per-chunk SVD and learnable per-head gating. |
| 78 | `minimal_nova` | `MinimalNOVAAttention` | `svd.minimal_nova` | ✓ | ✓ |  | MinimalNOVA: SVD-Factorized Low-Rank Projection Attention. |
| 79 | `minimal_svd` | `MinimalSVDAttention` | `svd.minimal_svd` | ✓ | ✓ |  | MinimalSVD: SVD-Factorized Low-Rank Projection Attention. |
| 80 | `mixattention_moska` | `MixAttentionMoSKA` | `hybrid.mixattention_moska` | ✓ | ✓ |  | MixAttention-MoSKA (FASA): frequency/variance-aware sparse attention. |
| 81 | `mla_attention` | `MLAAttention` | `modern.mla_attention` | ✓ | ✓ |  | Multi-Head Latent Attention (MLA) — DeepSeek-V2 style. |
| 82 | `moba_attention` | `MoBAAttention` | `sparse.moba_attention` | ✓ | ✓ |  | MoBA: Mixture of Block Attention for Long-Context Modeling. |
| 83 | `moe_deltakv_paged` | `MoEDeltaKVPagedAttention` | `sparse.moe_deltakv_paged` |  |  |  | MoE-DeltaKV-Paged Hybrid Attention. |
| 84 | `mqa_attention` | `MultiQueryAttention` | `modern.mqa` | ✓ | ✓ | ✓ | Multi-Query Attention — multiple query heads share single key and value heads. |
| 85 | `naive_svd` | `NaiveSVDAttention` | `svd.naive_svd` | ✓ | ✓ |  | Naive truncated SVD baseline attention module. |
| 86 | `nasc_attention` | `NASCAttention` | `linear.nasc_attention` | ✓ | ✓ |  | NASC: NEXUS-AEGIS Spectral Cascade Attention. |
| 87 | `nexus` | `NEXUSAttention` | `ssm.nexus` | ✓ | ✓ |  |  |
| 88 | `nha_class` | `NativeHybridAttention` | `modern.nha_attention` |  |  |  | Native Hybrid Attention: local sliding-window + global RFF, fused by a |
| 89 | `nsa` | `NativeSparseAttention` | `sparse.nsa_attention` |  |  |  |  |
| 90 | `nsa_attention` | `NativeSparseAttention` | `sparse.nsa_attention` |  |  |  |  |
| 91 | `nystrom` | `NystromAttention` | `linear.nystrom_attention` |  |  |  |  |
| 92 | `nystrom_attention` | `NystromAttention` | `linear.nystrom_attention` |  |  |  |  |
| 93 | `optimized_hybrid_attention` | `OptimizedHybridAttention` | `hybrid.optimized_hybrid_attention` |  |  |  |  |
| 94 | `orion` | `ORIONAttention` | `ssm.orion` |  |  |  |  |
| 95 | `padre_attention` | `PADReAttention` | `sparse.padre_attention` |  |  |  | Polynomial Attention with Dropout Regularization (PADRe). |
| 96 | `performer` | `PerformerAttention` | `linear.performer_attention` |  |  |  |  |
| 97 | `performer_attention` | `PerformerAttention` | `linear.performer_attention` |  |  |  |  |
| 98 | `phantom` | `PHANTOMAttention` | `ssm.phantom` |  |  |  |  |
| 99 | `positive_kernel_linear` | `PositiveKernelLinearAttention` | `hybrid.positive_kernel_linear_attention` | ✓ | ✓ |  |  |
| 100 | `primal_attention` | `PrimalAttention` | `linear.primal_attention` |  |  |  |  |
| 101 | `progressive_token_pruning` | `ProgressiveTokenPruning` | `adaptive.progressive_token_pruning` |  |  |  | Progressive Token Pruning attention. |
| 102 | `projection_factorized_attn` | `ProjectionFactorizedAttention` | `svd.projection_factorized_attn` | ✓ | ✓ |  | Projection Factorized SVD Linear Attention. |
| 103 | `qcqa_attention` | `QCQAAttention` | `modern.qcqa_attention` |  |  |  |  |
| 104 | `rala_attention` | `RALALinearAttention` | `linear.rala_attention` |  |  |  |  |
| 105 | `rala_compressor` | `RALALinearAttention` | `linear.rala_attention` |  |  |  |  |
| 106 | `reap_moe` | `REAPMoEAttention` | `sparse.reap_moe` |  |  |  |  |
| 107 | `refined_landmark` | `RefinedLandmarkAttention` | `hybrid.refined_landmark_attention` | ✓ | ✓ |  |  |
| 108 | `regla_attention` | `inject_regla` | `hybrid.regla_attention` |  |  |  |  |
| 109 | `residual_svd_attention` | `ResidualSVDAttention` | `svd.residual_svd_attention` |  |  |  |  |
| 110 | `routing_attention` | `inject_routing` | `sparse.routing_attention` |  |  |  |  |
| 111 | `rpc_attention` | `inject_rpc` | `linear.rpc_attention` |  |  |  |  |
| 112 | `salt_attention` | `SALTAttention` | `svd.salt_attention` | ✓ | ✓ |  | SALT Attention module replacing standard linear projections with SALTLinear. |
| 113 | `samba_m3` | `SAMBAM3Block` | `ssm.samba_m3` |  | ✓ |  | SAMBA-M3: Local Sliding-Window Attention + Spectral Mamba. |
| 114 | `samba_moe` | `SambaMoEBlock` | `ssm.samba_moe` | ✓ | ✓ |  | SambaMoE: SAMBA-M3 + REAP-MoE Hybrid Architecture with pretrained weight copying |
| 115 | `selective_mamba` | `SelectiveMamba` | `ssm.upi_stuffed_mamba` | ✓ |  |  | Selective Mamba SSM block. |
| 116 | `selective_ssm` | `SelectiveSSM` | `ssm.selective_ssm` | ✓ |  |  | Faithful Mamba-style selective SSM block with hardened execution paths. |
| 117 | `simple_ssm` | `SelectiveSSM` | `ssm.selective_ssm` | ✓ |  |  | Faithful Mamba-style selective SSM block with hardened execution paths. |
| 118 | `simplessm` | `SelectiveSSM` | `ssm.selective_ssm` | ✓ |  |  | Faithful Mamba-style selective SSM block with hardened execution paths. |
| 119 | `sink_attention` | `SinkAttention` | `adaptive.sink_attention` | ✓ | ✓ | ✓ | Attention with dedicated sink tokens that always get attention. |
| 120 | `sliding_window_attention` | `SlidingWindowAttention` | `modern.sliding_window` | ✓ | ✓ | ✓ | Sliding window attention — attends to last W tokens only per position. |
| 121 | `soa_attn` | `SOAAttention` | `fast_kernels.soa_attn` |  | ✓ |  | Shared Orthogonal Attention (SOA) with Per-Projection Latent Bottlenecks. |
| 122 | `span_expanded_attention` | `SpanExpandedAttention` | `adaptive.span_expanded_attention` |  |  |  |  |
| 123 | `spectra_v2` | `SpectraV2Attention` | `linear.spectra_v2` |  |  |  | MHA wrapper around PyTorch's scaled_dot_product_attention. |
| 124 | `spectraformer_attention` | `SpectraformerAttention` | `linear.spectraformer_attention` |  |  |  | Spectraformer Attention: Exact linear-time attention via random feature kerneliz |
| 125 | `spectral_mamba` | `SpectralMambaAttention` | `ssm.spectral_mamba` |  |  |  |  |
| 126 | `spectron` | `SpectronAttention` | `linear.spectron` | ✓ | ✓ |  |  |
| 127 | `spectron_fp8` | `SpectronFP8Attention` | `hybrid.spectron_fp8_attention` |  |  |  |  |
| 128 | `split_head_hybrid` | `SplitHeadHybridAttention` | `hybrid.split_head_hybrid_attention` | ✓ | ✓ |  |  |
| 129 | `squeezing_heads_distill` | `SqueezingHeadsDistiller` | `adaptive.squeezing_heads_distill` | ✓ | ✓ |  | Attention wrapper that keeps only the top-k most important heads |
| 130 | `svd_compressed` | `SVDCompressedAttention` | `svd.svd_compressed` |  | ✓ |  | Multi-head attention with truncated SVD projections, trainable LoRA adapters, |
| 131 | `svd_linear_attention` | `SVDLinearAttention` | `svd.svd_linear_attention` | ✓ | ✓ |  | SVD Factorized Linear Attention with optional exponential decay gating. |
| 132 | `svd_linear_attn` | `SVDLinearAttention` | `svd.svd_linear_attention` | ✓ | ✓ |  | SVD Factorized Linear Attention with optional exponential decay gating. |
| 133 | `svd_linear_decay` | `DecayGatedLinearAttention` | `svd.svd_linear_decay` | ✓ | ✓ |  | Decay Gated SVD Linear Attention. |
| 134 | `svd_llm_v2_attention` | `SVDLLMV2Attention` | `svd.svd_llm_v2_attention` |  | ✓ |  | SVD-factorized attention layer with explicit U, Sigma, V^H matrix buffers |
| 135 | `svd_mamba` | `SVDMamba` | `svd.svd_mamba` | ✓ |  |  | SVD-Mamba module with SVD-factorized projections, causal convolution, and select |
| 136 | `svd_projection_attention` | `SVDProjectionAttention` | `svd.svd_projection_attention` |  | ✓ |  | Multi-Head Attention using Factorized SVD Linear Projections. |
| 137 | `svd_seq` | `SVDLinearAttention` | `svd.svd_attention` |  | ✓ |  | Low-Rank SVD-Compressed Sequence Attention Mechanism. |
| 138 | `template_attention` | `inject_custom_attn` | `experimental.template_attention` |  |  |  |  |
| 139 | `tlinformer_attention` | `TLinFormerAttention` | `linear.tlinformer_attention` |  |  |  | TLinFormer: linear-time attention with learnable low-rank sequence projection. |
| 140 | `token_compressed_attention` | `TokenCompressedAttention` | `adaptive.token_compressed_attention` | ✓ | ✓ |  | Token-Compressed Attention v3. |
| 141 | `tpa_attention` | `TPAttention` | `linear.tpa_attention` | ✓ | ✓ |  | Tensor Product Attention v2 with asymmetric rank support. |
| 142 | `tri_branch_hybrid` | `TriBranchHybridAttention` | `hybrid.tri_branch_hybrid_attention` | ✓ | ✓ | ✓ |  |
| 143 | `triton_cosformer` | `TritonCosformerAttention` | `linear.triton_cosformer` |  |  |  | Triton-Accelerated Cosformer Linear Attention with Full Autograd Differentiabili |
| 144 | `triton_cosformer_inject` | `TritonCosformerAttention` | `linear.triton_cosformer` |  |  |  | Triton-Accelerated Cosformer Linear Attention with Full Autograd Differentiabili |
| 145 | `triton_svd_linear_attention` | `TritonSVDLinearAttention` | `svd.triton_svd_linear_attention` |  |  |  | Triton-Accelerated Linear Attention with Genuine SVD Low-Rank Projections, |
| 146 | `triton_svd_linear_attention_inject` | `inject_triton_svd_linear_attention` | `svd.triton_svd_linear_attention` |  |  |  | Safe factory function to inject TritonSVDLinearAttention into a model. |
| 147 | `unified_rope` | `UnifiedRoPE` | `modern.unified_rope` | ✓ | ✓ |  | Universal Rotary Position Embedding Attention module. |
| 148 | `upi_mamba` | `SelectiveMamba` | `ssm.upi_stuffed_mamba` | ✓ |  |  | Selective Mamba SSM block. |
| 149 | `upi_stuffed_mamba` | `SelectiveMamba` | `ssm.upi_stuffed_mamba` | ✓ |  |  | Selective Mamba SSM block. |
| 150 | `vig` | `ViGAttention` | `modern.vig_attention` | ✓ | ✓ | ✓ | Vision GNN-inspired 2D Grid Linear Attention with Learned Gated Fusion. |
| 151 | `vig_attention` | `ViGAttention` | `modern.vig_attention` | ✓ | ✓ | ✓ | Vision GNN-inspired 2D Grid Linear Attention with Learned Gated Fusion. |
| 152 | `vspec` | `VSpecAttention` | `ssm.vspec` | ✓ | ✓ | ✓ | VSpec++ Attention: Variational Spectral Token-Adaptive Attention Module. |
| 153 | `vspec_attention` | `VSpecAttention` | `ssm.vspec` | ✓ | ✓ | ✓ | VSpec++ Attention: Variational Spectral Token-Adaptive Attention Module. |
| 154 | `wgqa` | `WeightedGroupedQueryAttention` | `modern.wgqa_attention` | ✓ | ✓ | ✓ | Weighted Grouped-Query Attention (WGQA) with learnable per-KV-head importance we |
| 155 | `wgqa_attention` | `WeightedGroupedQueryAttention` | `modern.wgqa_attention` | ✓ | ✓ | ✓ | Weighted Grouped-Query Attention (WGQA) with learnable per-KV-head importance we |
| 156 | `x_attention` | `XAttention` | `fast_kernels.xattention` | ✓ | ✓ | ✓ |  |
| 157 | `xattention` | `XAttention` | `fast_kernels.xattention` | ✓ | ✓ | ✓ |  |
| 158 | `zeros` | `ZeroSAttention` | `dense.zeros_attention` | ✓ | ✓ | ✓ |  |
| 159 | `zeros_attention` | `ZeroSAttention` | `dense.zeros_attention` | ✓ | ✓ | ✓ |  |
