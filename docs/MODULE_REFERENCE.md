# Complete 232-Module Architectural Reference

Exhaustive registry documentation across all 159 attention mechanisms and 73 model compressor modules.

## 1. Attention Mechanisms (159 Modules)

| Registry Key | Class / Factory | Causal | GQA | RoPE | Reference Paper |
|:---|:---|:---:|:---:|:---:|:---|
| `adaptive_span_attention` | `AdaptiveSpanAttention` | ✓ | ✓ | ✓ | N/A |
| `adaptive_sparse` | `AdaptiveSparseAttention` | ✓ | ✓ | ✓ | N/A |
| `adaptive_sparse_attention` | `AdaptiveSparseAttention` | ✓ | ✓ | ✓ | N/A |
| `adaptive_sparse_attention_attention` | `AdaptiveSparseAttention` | ✓ | ✓ | ✓ | N/A |
| `adaptive_sparse_attn` | `AdaptiveSparseAttention` | ✓ | ✓ | ✓ | N/A |
| `adaptive_svd` | `AdaptiveSVDAttention` | ✓ | ✓ | ✓ | N/A |
| `adaptive_svd_attention` | `AdaptiveSVDAttention` | ✓ | ✓ | ✓ | N/A |
| `adaptive_svd_attn` | `AdaptiveSVDAttention` | ✓ | ✓ | ✓ | N/A |
| `adasplash` | `AdaSplashAttention` | ✓ | ✓ | ✓ | N/A |
| `adasplash_attention` | `AdaSplashAttention` | ✓ | ✓ | ✓ | N/A |
| `asvd` | `ASVDAttention` | - | - | - | ASVD: Activation-aware Singular Value Decomposition for Compressing LLMs |
| `asvd_attention` | `ASVDAttention` | - | - | - | N/A |
| `bam_umoe` | `BAMUMoEBlock` | - | - | - | BAM: Universal Mixture-of-Experts Compression |
| `block_transformer_hiba` | `BlockTransformerHIBA` | - | - | - | N/A |
| `chela_slay` | `ChelaSlayAttention` | - | - | - | CHELA: Channel Elimination and Layer Shedding for Accelerating Transformers |
| `chela_slay_attention` | `ChelaSlayAttention` | - | - | - | N/A |
| `cluster` | `ClusterAttention` | ✓ | ✓ | ✓ | Cluster-Former: Clustering-based Sparse Transformer |
| `cluster_attention` | `ClusterAttention` | ✓ | ✓ | ✓ | N/A |
| `compressed_attention` | `CompressedAttentionLayer` | ✓ | ✓ | ✓ | N/A |
| `compressed_attention_layer` | `CompressedAttentionLayer` | ✓ | ✓ | ✓ | N/A |
| `compressed_attention_layer_energy` | `CompressedAttentionLayer2` | ✓ | ✓ | ✓ | N/A |
| `compressed_mha` | `CompressedDistilBertAttention` | ✓ | ✓ | ✓ | N/A |
| `compressed_mha_attention` | `CompressedDistilBertAttention` | ✓ | ✓ | ✓ | N/A |
| `cosformer` | `CosformerAttention` | - | - | - | Cosformer: Rethinking Softmax in Attention with Linear Complexity |
| `decay_linear_attention` | `DecayLinearAttention` | - | - | - | Linear Attention with Decay |
| `decay_linear_attn` | `DecayLinearAttention` | - | - | - | N/A |
| `eat` | `EATBlock` | - | - | - | N/A |
| `eat_attention` | `EATBlock` | - | - | - | N/A |
| `expert_choice_moe` | `ExpertChoiceMoEAttention` | - | - | - | Mixture-of-Experts with Expert Choice Routing |
| `expert_choice_moe_attention` | `ExpertChoiceMoEAttention` | - | - | - | N/A |
| `factorised_matrix_state` | `FactorisedMatrixState` | - | - | - | N/A |
| `flash3` | `FlashAttention3` | - | - | - | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision |
| `flash3_fp8` | `FlashAttention3FP8` | - | - | - | N/A |
| `flash_attention3` | `FlashAttention3` | - | - | - | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision |
| `flash_attention3_fp8` | `FlashAttention3FP8` | - | - | - | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision |
| `fms` | `FactorisedMatrixState` | - | - | - | N/A |
| `full` | `FullAttention` | ✓ | ✓ | ✓ | N/A |
| `full_attention` | `FullAttention` | ✓ | ✓ | ✓ | N/A |
| `fused_svd_kernel` | `FusedSVDKernelAttention` | ✓ | ✓ | - | Fused SVD Kernels for Hardware-Accelerated Neural Compression |
| `fused_svd_kernel_attention` | `FusedSVDKernelAttention` | ✓ | ✓ | - | N/A |
| `fusion_attention` | `FUSIONATTN` | ✓ | ✓ | - | N/A |
| `fusion_attn` | `FUSIONATTN` | ✓ | ✓ | - | N/A |
| `galerkin_attention` | `GalerkinAttention` | ✓ | ✓ | - | Galerkin Transformer: A Framework for Approximating Continuous Attention |
| `gated_delta_net` | `GatedDeltaNet` | - | - | - | Gated Delta Networks: Improving Mamba2 with Delta Rule |
| `gated_deltanet_block` | `GatedDeltaNetBlock` | - | - | - | Gated Delta Networks: Improving Mamba2 with Delta Rule |
| `gated_hybrid_kernel` | `GatedHybridKernelAttention` | ✓ | ✓ | - | N/A |
| `gated_linear_attention` | `GatedLinearAttention` | ✓ | ✓ | - | Gated Linear Attention Transformers with Hardware-Efficient Training |
| `goldfinch` | `GoldFinchAHNBlock` | - | - | - | N/A |
| `goldfinch_ahn` | `GoldFinchAHNBlock` | - | - | - | N/A |
| `gqa` | `GroupedQueryAttention` | ✓ | ✓ | ✓ | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints |
| `gqa_attention` | `GroupedQueryAttention` | ✓ | ✓ | ✓ | N/A |
| `granite_hybrid` | `GraniteHybridBlock` | - | - | - | N/A |
| `griffin_ssd` | `GriffinSSDBlock` | - | - | - | Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models |
| `grouped_query_attention` | `GroupedQueryAttention` | ✓ | ✓ | ✓ | N/A |
| `halo_regla` | `HALOReGLAAttention` | - | - | - | N/A |
| `hamming_attention` | `HammingAttention` | ✓ | ✓ | ✓ | N/A |
| `hgrn2_gated_deltanet` | `HGRN2GatedDeltaNet` | - | - | - | HGRN2: Gated Linear RNNs with State Expansion |
| `hgrn2_gated_layer` | `HGRN2Layer` | - | - | - | N/A |
| `hgrn2_gated_outer` | `HGRN2GatedOuterBlock` | ✓ | ✓ | - | N/A |
| `hgrn2_outer` | `HGRN2GatedOuterBlock` | ✓ | ✓ | - | HGRN2: Gated Linear RNNs with State Expansion |
| `hla_attention` | `HLAAttention` | ✓ | ✓ | - | N/A |
| `hybrid_attention` | `HybridAttention` | ✓ | ✓ | ✓ | N/A |
| `hybrid_local_kernel` | `HybridLocalKernelAttention` | ✓ | ✓ | - | N/A |
| `hymba_transx` | `HymbaTransXBlock` | - | - | - | Hymba: A Hybrid-Head Architecture for Small Language Models |
| `hymba_transx_attn` | `HymbaTransXBlock` | - | - | - | N/A |
| `inject_triton_cosformer` | `TritonCosformerAttention` | - | - | - | N/A |
| `inject_triton_svd_linear_attention` | `inject_triton_svd_linear_attention` | - | - | - | N/A |
| `inject_xattention` | `inject_xattention` | - | - | - | N/A |
| `jamba` | `Jamba2Block` | - | - | - | Jamba: A Hybrid Transformer-Mamba Language Model |
| `jamba_info` | `JambaInfoBlock` | - | - | - | N/A |
| `kda_attention` | `KDAAttention` | - | - | - | N/A |
| `landmark_local` | `LandmarkLocalAttention` | ✓ | ✓ | - | N/A |
| `linformer_attention` | `LinformerAttention` | ✓ | ✓ | - | Linformer: Self-Attention with Linear Complexity |
| `lola_attention` | `LoLAAttention` | ✓ | ✓ | ✓ | N/A |
| `lora_attention` | `LoRAAttention` | ✓ | ✓ | ✓ | N/A |
| `lora_attn` | `LoRAAttention` | ✓ | ✓ | ✓ | N/A |
| `lsh_attention` | `LSHAttention` | ✓ | ✓ | - | Reformer: The Efficient Transformer |
| `mamba_block` | `MambaBlock` | ✓ | - | - | N/A |
| `mamba_svd_flash` | `MambaSVDFlash` | - | - | - | N/A |
| `minimal_nova` | `MinimalNOVAAttention` | ✓ | ✓ | - | N/A |
| `minimal_svd` | `MinimalSVDAttention` | ✓ | ✓ | - | N/A |
| `mixattention_moska` | `MixAttentionMoSKA` | ✓ | ✓ | - | N/A |
| `mla_attention` | `MLAAttention` | ✓ | ✓ | - | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model |
| `moba_attention` | `MoBAAttention` | ✓ | ✓ | - | N/A |
| `moe_deltakv_paged` | `MoEDeltaKVPagedAttention` | - | - | - | N/A |
| `naive_svd` | `NaiveSVDAttention` | ✓ | ✓ | - | N/A |
| `nasc_attention` | `NASCAttention` | ✓ | ✓ | - | N/A |
| `nexus` | `NEXUSAttention` | ✓ | ✓ | - | N/A |
| `nha_class` | `NativeHybridAttention` | - | - | - | N/A |
| `nsa` | `NativeSparseAttention` | - | - | - | Native Sparse Attention: Hardware-Aligned and Topologically Diverse Sparse Attention |
| `nsa_attention` | `NativeSparseAttention` | - | - | - | N/A |
| `nystrom` | `NystromAttention` | - | - | - | Nystromformer: A Nystrom-Based Algorithm for Approximating Self-Attention |
| `nystrom_attention` | `NystromAttention` | - | - | - | Nystromformer: A Nystrom-Based Algorithm for Approximating Self-Attention |
| `optimized_hybrid_attention` | `OptimizedHybridAttention` | - | - | - | N/A |
| `orion` | `ORIONAttention` | - | - | - | N/A |
| `padre_attention` | `PADReAttention` | - | - | - | N/A |
| `performer` | `PerformerAttention` | - | - | - | Rethinking Attention with Performers |
| `performer_attention` | `PerformerAttention` | - | - | - | Rethinking Attention with Performers |
| `phantom` | `PHANTOMAttention` | - | - | - | N/A |
| `positive_kernel_linear` | `PositiveKernelLinearAttention` | ✓ | ✓ | - | N/A |
| `primal_attention` | `PrimalAttention` | - | - | - | N/A |
| `progressive_token_pruning` | `ProgressiveTokenPruning` | - | - | - | N/A |
| `projection_factorized_attn` | `ProjectionFactorizedAttention` | ✓ | ✓ | - | N/A |
| `qcqa_attention` | `QCQAAttention` | - | - | - | N/A |
| `rala_attention` | `RALALinearAttention` | - | - | - | N/A |
| `rala_compressor` | `RALALinearAttention` | - | - | - | N/A |
| `reap_moe` | `REAPMoEAttention` | - | - | - | REAP: Efficient Mixture-of-Experts Pruning |
| `refined_landmark` | `RefinedLandmarkAttention` | ✓ | ✓ | - | N/A |
| `regla_attention` | `inject_regla` | - | - | - | N/A |
| `residual_svd_attention` | `ResidualSVDAttention` | - | - | - | N/A |
| `routing_attention` | `inject_routing` | - | - | - | N/A |
| `rpc_attention` | `inject_rpc` | - | - | - | N/A |
| `salt_attention` | `SALTAttention` | ✓ | ✓ | - | N/A |
| `samba_m3` | `SAMBAM3Block` | - | ✓ | - | Samba: Simple Hybrid State Space Models for Efficient Language Modeling |
| `samba_moe` | `SambaMoEBlock` | ✓ | ✓ | - | Samba: Simple Hybrid State Space Models for Efficient Language Modeling |
| `selective_mamba` | `SelectiveMamba` | ✓ | - | - | N/A |
| `selective_ssm` | `SelectiveSSM` | ✓ | - | - | N/A |
| `simple_ssm` | `SelectiveSSM` | ✓ | - | - | Simple Hardware-Efficient State Space Models |
| `simplessm` | `SelectiveSSM` | ✓ | - | - | N/A |
| `soa_attn` | `SOAAttention` | - | ✓ | - | N/A |
| `span_expanded_attention` | `SpanExpandedAttention` | - | - | - | N/A |
| `spectra_v2` | `SpectraV2Attention` | - | - | - | SpectraFormer: Spectral Domain Attention for Efficient Sequence Processing |
| `spectraformer_attention` | `SpectraformerAttention` | - | - | - | SpectraFormer: Spectral Domain Attention for Efficient Sequence Processing |
| `spectral_mamba` | `SpectralMambaAttention` | - | - | - | N/A |
| `spectron` | `SpectronAttention` | ✓ | ✓ | - | Spectron: Spectral Decomposition with Orthogonal Adapters |
| `spectron_fp8` | `SpectronFP8Attention` | - | - | - | N/A |
| `split_head_hybrid` | `SplitHeadHybridAttention` | ✓ | ✓ | - | N/A |
| `squeezing_heads_distill` | `SqueezingHeadsDistiller` | ✓ | ✓ | - | N/A |
| `svd_compressed` | `SVDCompressedAttention` | - | ✓ | - | N/A |
| `svd_linear` | `SVDLinearAttention` | ✓ | ✓ | - | N/A |
| `svd_linear_attention` | `SVDLinearAttention` | ✓ | ✓ | - | N/A |
| `svd_linear_attn` | `SVDLinearAttention` | ✓ | ✓ | - | N/A |
| `svd_linear_decay` | `DecayGatedLinearAttention` | ✓ | ✓ | - | N/A |
| `svd_llm_v2_attention` | `SVDLLMV2Attention` | - | ✓ | - | N/A |
| `svd_mamba` | `SVDMamba` | ✓ | - | - | N/A |
| `svd_projection_attention` | `SVDProjectionAttention` | - | ✓ | - | N/A |
| `svd_seq` | `SVDLinearAttention` | - | ✓ | - | N/A |
| `template_attention` | `inject_custom_attn` | - | - | - | N/A |
| `tlinformer_attention` | `TLinFormerAttention` | - | - | - | N/A |
| `token_compressed_attention` | `TokenCompressedAttention` | ✓ | ✓ | - | N/A |
| `tpa_attention` | `TPAttention` | ✓ | ✓ | - | N/A |
| `tri_branch_hybrid` | `TriBranchHybridAttention` | ✓ | ✓ | ✓ | N/A |
| `triton_cosformer` | `TritonCosformerAttention` | - | - | - | N/A |
| `triton_cosformer_inject` | `TritonCosformerAttention` | - | - | - | N/A |
| `triton_svd_linear_attention` | `TritonSVDLinearAttention` | - | - | - | N/A |
| `triton_svd_linear_attention_inject` | `inject_triton_svd_linear_attention` | - | - | - | N/A |
| `unified_rope` | `UnifiedRoPE` | ✓ | ✓ | - | N/A |
| `upi_mamba` | `SelectiveMamba` | ✓ | - | - | N/A |
| `upi_stuffed_mamba` | `SelectiveMamba` | ✓ | - | - | N/A |
| `vig` | `ViGAttention` | ✓ | ✓ | ✓ | N/A |
| `vig_attention` | `ViGAttention` | ✓ | ✓ | ✓ | N/A |
| `vspec` | `VSpecAttention` | ✓ | ✓ | ✓ | N/A |
| `vspec_attention` | `VSpecAttention` | ✓ | ✓ | ✓ | N/A |
| `wgqa` | `WeightedGroupedQueryAttention` | ✓ | ✓ | ✓ | N/A |
| `wgqa_attention` | `WeightedGroupedQueryAttention` | ✓ | ✓ | ✓ | N/A |
| `x_attention` | `XAttention` | ✓ | ✓ | ✓ | N/A |
| `xattention` | `XAttention` | ✓ | ✓ | ✓ | N/A |
| `zeros` | `ZeroSAttention` | ✓ | ✓ | ✓ | N/A |
| `zeros_attention` | `ZeroSAttention` | ✓ | ✓ | ✓ | N/A |

## 2. Compression Engine Modules (73 Modules)

| Registry Key | Class / Factory | Type | Reference Paper |
|:---|:---|:---|:---|:---|
| `a3_compressor` | `A3Compressor` | Compressor | N/A |
| `aa_svd_compressor` | `AASVDCompressor` | Compressor | AASVD: Adaptive Activation SVD with Covariance Regularization |
| `aatc_xkv_compressor` | `AATC_XKVCompressor` | Compressor | N/A |
| `adaptive_ranks` | `AdaptiveRanksCompressor` | Compressor | N/A |
| `adasvd_compressor` | `AdaSVDCompressor` | Compressor | N/A |
| `admm_compressor` | `ADMMCompressor` | Compressor | Progressive Weight Pruning of Deep Neural Networks via ADMM |
| `admm_pruner` | `ADMMPruner` | Compressor | Progressive Weight Pruning of Deep Neural Networks via ADMM |
| `armor_pruner` | `ARMORPruner` | Compressor | ARMOR: Robust Matrix Pruning for Pretrained Language Models |
| `ast_pruner` | `ASTPruner` | Compressor | AST: Attention-Structure-Preserving Pruning |
| `astro_quantizer` | `AstroQuantizer` | Compressor | N/A |
| `bonsai_pruner` | `bonsai_factory` | Compressor | Bonsai: Efficient Parameter Pruning for Transformers |
| `cachenotes_compressor` | `CacheNotesCompressor` | Compressor | CacheNotes: Compressing Transformer Key-Value Caches |
| `come_compressor` | `CoMeCompressor` | Compressor | N/A |
| `compot_compressor` | `compot_compressor_factory` | Compressor | N/A |
| `coopq_quantizer` | `coopq_factory` | Compressor | N/A |
| `cr_compressor` | `CRCompressor` | Compressor | N/A |
| `cur_compressor` | `CURCompressor` | Compressor | CUR Matrix Decomposition for Transformer Compression |
| `cur_moe_compressor` | `cur_moe_factory` | Compressor | N/A |
| `delta_kv` | `delta_kv_factory` | Compressor | Delta-KV: Incremental Key-Value State Updates for LLM Inference |
| `direct_svd` | `DirectSVDCompressor` | Compressor | N/A |
| `dobi_svd` | `dobi_factory` | Compressor | N/A |
| `drone_compressor` | `DRONECompressor` | Compressor | N/A |
| `dynamic_rank` | `dynamic_entropy_svd_factory` | Compressor | N/A |
| `enlighten_compressor` | `ENLightenCompressor` | Compressor | N/A |
| `entropy_budget_allocation` | `EntropyBudgetCompressor` | Compressor | N/A |
| `erc_svd` | `ERCCompressor` | Compressor | N/A |
| `faq_quantizer` | `FAQQuantizer` | Compressor | N/A |
| `flap` | `FLAPCompressor` | Compressor | FLAP: Fast Language Model Compression via Post-Training Structured Pruning |
| `flat_llm` | `FLATLLMCompressor` | Compressor | FlatLLM: Representation Flatness for LLM Compression |
| `frandi_compressor` | `frandi_factory` | Compressor | N/A |
| `fwsvd_compressor` | `FWSVDCompressor` | Compressor | N/A |
| `gfws` | `GFWSCompressor` | Compressor | N/A |
| `grecodec_compressor` | `grecodec_compressor_factory` | Compressor | N/A |
| `hashed_net_compressor` | `hashed_net_factory` | Compressor | Hashed Neural Networks |
| `huffman_coder` | `HuffmanCompressor` | Compressor | Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding |
| `hybrid_svd_layer` | `HybridSVDCompressor` | Compressor | N/A |
| `hybridkv_compressor` | `HybridKVCompressor` | Compressor | N/A |
| `kmeans_quantize` | `kmeans_factory` | Compressor | Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding |
| `kronecker_compressor` | `kronecker_factory` | Compressor | KroneckerBERT: Significant Compression of BERT Using Kronecker Decomposition |
| `kronecker_tt_lora` | `kronecker_tt_lora_factory` | Compressor | N/A |
| `kvtc_compressor` | `KVTCCompressor` | Compressor | N/A |
| `liftr` | `liftr_compressor_entry` | Compressor | N/A |
| `lillama_compressor` | `lillama_factory` | Compressor | LiLLaMA: Lightweight LLaMA Compression via Layer Pruning and SVD |
| `lsc_compressor` | `lsc_factory` | Compressor | N/A |
| `mamba_shedder` | `mamba_shedder_factory` | Compressor | N/A |
| `mgie_budget` | `MGIESVDBudgetCompressor` | Compressor | N/A |
| `mlorq_compressor` | `MLoRQCompressor` | Compressor | N/A |
| `modegpt_compressor` | `MoDeGPTCompressor` | Compressor | N/A |
| `nmf_compressor` | `NMFCompressor` | Compressor | N/A |
| `oats_compressor` | `OATSCompressor` | Compressor | N/A |
| `optimal_subspace_projection` | `OptimalSubspaceProjectionCompressor` | Compressor | N/A |
| `proxsparse` | `ProxSparsePruner` | Compressor | N/A |
| `residual_svd_compressor` | `ResidualSVDCompressor` | Compressor | N/A |
| `roast_hashed_linear` | `ROASTCompressor` | Compressor | ROAST: Randomized Compression of Neural Networks |
| `rotational_slicegpt_int4` | `RotationalSliceGPTINT4Compressor` | Compressor | N/A |
| `rpca_compressor` | `RPCACompressor` | Compressor | N/A |
| `saes_svd_v2` | `SAESSVDCompressorV2` | Compressor | N/A |
| `salr` | `SALRCompressor` | Compressor | SALR: Sensitivity-Aware Low-Rank Compression |
| `sdrm_compressor` | `sdrm_compressor_factory` | Compressor | N/A |
| `slicegpt` | `PCAActivationAwareSVDCompressor` | Compressor | SliceGPT: Compress Large Language Models by Deleting Rows and Columns |
| `slicegpt_pca` | `SliceGPTPCACompressor` | Compressor | N/A |
| `slim_compressor` | `SLiMCompressor` | Compressor | N/A |
| `soka_compressor` | `soka_factory` | Compressor | N/A |
| `sparsegpt` | `sparsegpt_factory` | Compressor | SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot |
| `subspace_projection` | `SubspaceProjectionCompressor` | Compressor | N/A |
| `svd_linear` | `SVDLinearCompressor` | Compressor | N/A |
| `svd_llm_compressor` | `SVDLLMCompressor` | Compressor | N/A |
| `swift_svd` | `SwiftSVDCompressor` | Compressor | N/A |
| `swift_svd_act` | `SwiftSVDActivationCompressor` | Compressor | N/A |
| `tomoe_compressor` | `tomoe_compressor_factory` | Compressor | N/A |
| `tt_compressor` | `TTCompressor` | Compressor | N/A |
| `uniql_compressor` | `UniQLCompressor` | Compressor | UniQL: Unified Quantization and Low-rank Decomposition |
| `zerokv_neo_compressor` | `ZeroKVNeoCompressor` | Compressor | N/A |
