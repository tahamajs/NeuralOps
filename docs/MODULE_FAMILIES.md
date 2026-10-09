# NeuralOps Module Families Breakdown (v1.2.0)

Total Registered Mechanisms: 232 (159 Attention + 73 Compression)

## Family Counts

- **`attentions.adaptive`**: 11 modules
- **`attentions.dense`**: 4 modules
- **`attentions.experimental`**: 1 modules
- **`attentions.fast_kernels`**: 10 modules
- **`attentions.gated`**: 11 modules
- **`attentions.hybrid`**: 16 modules
- **`attentions.linear`**: 22 modules
- **`attentions.modern`**: 14 modules
- **`attentions.sparse`**: 16 modules
- **`attentions.ssm`**: 22 modules
- **`attentions.svd`**: 32 modules
- **`compression.experimental`**: 5 modules
- **`compression.kv_cache`**: 6 modules
- **`compression.low_rank`**: 24 modules
- **`compression.pruning`**: 9 modules
- **`compression.quantization`**: 8 modules
- **`compression.svd`**: 18 modules
- **`compression.tensor_decomp`**: 3 modules

---

## Family: `attentions.adaptive` (11 modules)

| Name | Class |
|---|---|
| `adaptive_span_attention` | `AdaptiveSpanAttention` |
| `block_transformer_hiba` | `BlockTransformerHIBA` |
| `eat` | `EATBlock` |
| `eat_attention` | `EATBlock` |
| `jamba` | `Jamba2Block` |
| `lola_attention` | `LoLAAttention` |
| `progressive_token_pruning` | `ProgressiveTokenPruning` |
| `sink_attention` | `SinkAttention` |
| `span_expanded_attention` | `SpanExpandedAttention` |
| `squeezing_heads_distill` | `SqueezingHeadsDistiller` |
| `token_compressed_attention` | `TokenCompressedAttention` |

## Family: `attentions.dense` (4 modules)

| Name | Class |
|---|---|
| `full` | `FullAttention` |
| `full_attention` | `FullAttention` |
| `zeros` | `ZeroSAttention` |
| `zeros_attention` | `ZeroSAttention` |

## Family: `attentions.experimental` (1 modules)

| Name | Class |
|---|---|
| `template_attention` | `inject_custom_attn` |

## Family: `attentions.fast_kernels` (10 modules)

| Name | Class |
|---|---|
| `adasplash` | `AdaSplashAttention` |
| `adasplash_attention` | `AdaSplashAttention` |
| `flash3` | `FlashAttention3` |
| `flash3_fp8` | `FlashAttention3FP8` |
| `flash_attention3` | `FlashAttention3` |
| `flash_attention3_fp8` | `FlashAttention3FP8` |
| `inject_xattention` | `inject_xattention` |
| `soa_attn` | `SOAAttention` |
| `x_attention` | `XAttention` |
| `xattention` | `XAttention` |

## Family: `attentions.gated` (11 modules)

| Name | Class |
|---|---|
| `decay_linear_attention` | `DecayLinearAttention` |
| `decay_linear_attn` | `DecayLinearAttention` |
| `gated_delta_net` | `GatedDeltaNet` |
| `gated_deltanet_block` | `GatedDeltaNetBlock` |
| `gated_linear_attention` | `GatedLinearAttention` |
| `hgrn2_gated_deltanet` | `HGRN2GatedDeltaNet` |
| `hgrn2_gated_layer` | `HGRN2Layer` |
| `hgrn2_gated_outer` | `HGRN2GatedOuterBlock` |
| `hgrn2_outer` | `HGRN2GatedOuterBlock` |
| `hla_attention` | `HLAAttention` |
| `kda_attention` | `KDAAttention` |

## Family: `attentions.hybrid` (16 modules)

| Name | Class |
|---|---|
| `fusion_attention` | `FUSIONATTN` |
| `fusion_attn` | `FUSIONATTN` |
| `gated_hybrid_kernel` | `GatedHybridKernelAttention` |
| `granite_hybrid` | `GraniteHybridBlock` |
| `halo_regla` | `HALOReGLAAttention` |
| `hybrid_attention` | `HybridAttention` |
| `hybrid_local_kernel` | `HybridLocalKernelAttention` |
| `landmark_local` | `LandmarkLocalAttention` |
| `mixattention_moska` | `MixAttentionMoSKA` |
| `optimized_hybrid_attention` | `OptimizedHybridAttention` |
| `positive_kernel_linear` | `PositiveKernelLinearAttention` |
| `refined_landmark` | `RefinedLandmarkAttention` |
| `regla_attention` | `inject_regla` |
| `spectron_fp8` | `SpectronFP8Attention` |
| `split_head_hybrid` | `SplitHeadHybridAttention` |
| `tri_branch_hybrid` | `TriBranchHybridAttention` |

## Family: `attentions.linear` (22 modules)

| Name | Class |
|---|---|
| `chela_slay` | `ChelaSlayAttention` |
| `chela_slay_attention` | `ChelaSlayAttention` |
| `cosformer` | `CosformerAttention` |
| `galerkin_attention` | `GalerkinAttention` |
| `inject_triton_cosformer` | `inject_triton_cosformer` |
| `linformer_attention` | `LinformerAttention` |
| `nasc_attention` | `NASCAttention` |
| `nystrom` | `NystromAttention` |
| `nystrom_attention` | `NystromAttention` |
| `performer` | `PerformerAttention` |
| `performer_attention` | `PerformerAttention` |
| `primal_attention` | `PrimalAttention` |
| `rala_attention` | `RALALinearAttention` |
| `rala_compressor` | `RALALinearAttention` |
| `rpc_attention` | `inject_rpc` |
| `spectra_v2` | `SpectraV2Attention` |
| `spectraformer_attention` | `SpectraformerAttention` |
| `spectron` | `SpectronAttention` |
| `tlinformer_attention` | `TLinFormerAttention` |
| `tpa_attention` | `TPAttention` |
| `triton_cosformer` | `TritonCosformerAttention` |
| `triton_cosformer_inject` | `inject_triton_cosformer` |

## Family: `attentions.modern` (14 modules)

| Name | Class |
|---|---|
| `alibi_attention` | `ALiBiAttention` |
| `cross_attention` | `CrossAttention` |
| `gqa_attention` | `GroupedQueryAttention` |
| `grouped_query_attention` | `GroupedQueryAttention` |
| `mla_attention` | `MLAAttention` |
| `mqa_attention` | `MultiQueryAttention` |
| `nha_class` | `NativeHybridAttention` |
| `qcqa_attention` | `QCQAAttention` |
| `sliding_window_attention` | `SlidingWindowAttention` |
| `unified_rope` | `UnifiedRoPE` |
| `vig` | `ViGAttention` |
| `vig_attention` | `ViGAttention` |
| `wgqa` | `WeightedGroupedQueryAttention` |
| `wgqa_attention` | `WeightedGroupedQueryAttention` |

## Family: `attentions.sparse` (16 modules)

| Name | Class |
|---|---|
| `adaptive_sparse_attention` | `AdaptiveSparseAttention` |
| `adaptive_sparse_attn` | `AdaptiveSparseAttention` |
| `bam_umoe` | `BAMUMoEBlock` |
| `cluster` | `ClusterAttention` |
| `cluster_attention` | `ClusterAttention` |
| `expert_choice_moe` | `ExpertChoiceMoEAttention` |
| `expert_choice_moe_attention` | `ExpertChoiceMoEAttention` |
| `hamming_attention` | `HammingAttention` |
| `lsh_attention` | `LSHAttention` |
| `moba_attention` | `MoBAAttention` |
| `moe_deltakv_paged` | `MoEDeltaKVPagedAttention` |
| `nsa` | `NativeSparseAttention` |
| `nsa_attention` | `NativeSparseAttention` |
| `padre_attention` | `PADReAttention` |
| `reap_moe` | `REAPMoEAttention` |
| `routing_attention` | `inject_routing` |

## Family: `attentions.ssm` (22 modules)

| Name | Class |
|---|---|
| `goldfinch` | `GoldFinchAHNBlock` |
| `goldfinch_ahn` | `GoldFinchAHNBlock` |
| `griffin_ssd` | `GriffinSSDBlock` |
| `hymba_transx` | `HymbaTransXBlock` |
| `hymba_transx_attn` | `HymbaTransXBlock` |
| `jamba_info` | `JambaInfoBlock` |
| `mamba_block` | `MambaBlock` |
| `mamba_svd_flash` | `MambaSVDFlash` |
| `nexus` | `NEXUSAttention` |
| `orion` | `ORIONAttention` |
| `phantom` | `PHANTOMAttention` |
| `samba_m3` | `SAMBAM3Block` |
| `samba_moe` | `SambaMoEBlock` |
| `selective_mamba` | `SelectiveMamba` |
| `selective_ssm` | `SelectiveSSM` |
| `simple_ssm` | `SelectiveSSM` |
| `simplessm` | `SelectiveSSM` |
| `spectral_mamba` | `SpectralMambaAttention` |
| `upi_mamba` | `SelectiveMamba` |
| `upi_stuffed_mamba` | `SelectiveMamba` |
| `vspec` | `VSpecAttention` |
| `vspec_attention` | `VSpecAttention` |

## Family: `attentions.svd` (32 modules)

| Name | Class |
|---|---|
| `adaptive_svd` | `AdaptiveSVDAttention` |
| `adaptive_svd_attention` | `AdaptiveSVDAttention` |
| `asvd` | `ASVDAttention` |
| `asvd_attention` | `ASVDAttention` |
| `compressed_attention` | `CompressedAttentionLayer` |
| `compressed_attention_layer` | `CompressedAttentionLayer` |
| `compressed_attention_layer_energy` | `CompressedAttentionLayer2` |
| `compressed_mha` | `CompressedDistilBertAttention` |
| `compressed_mha_attention` | `CompressedDistilBertAttention` |
| `factorised_matrix_state` | `FactorisedMatrixState` |
| `fms` | `FactorisedMatrixState` |
| `fused_svd_kernel` | `FusedSVDKernelAttention` |
| `fused_svd_kernel_attention` | `FusedSVDKernelAttention` |
| `inject_triton_svd_linear_attention` | `inject_triton_svd_linear_attention` |
| `lora_attention` | `LoRAAttention` |
| `lora_attn` | `LoRAAttention` |
| `minimal_nova` | `MinimalNOVAAttention` |
| `minimal_svd` | `MinimalSVDAttention` |
| `naive_svd` | `NaiveSVDAttention` |
| `projection_factorized_attn` | `ProjectionFactorizedAttention` |
| `residual_svd_attention` | `ResidualSVDAttention` |
| `salt_attention` | `SALTAttention` |
| `svd_compressed` | `SVDCompressedAttention` |
| `svd_linear_attention` | `SVDLinearAttention` |
| `svd_linear_attn` | `SVDLinearAttention` |
| `svd_linear_decay` | `DecayGatedLinearAttention` |
| `svd_llm_v2_attention` | `SVDLLMV2Attention` |
| `svd_mamba` | `SVDMamba` |
| `svd_projection_attention` | `SVDProjectionAttention` |
| `svd_seq` | `SVDLinearAttention` |
| `triton_svd_linear_attention` | `TritonSVDLinearAttention` |
| `triton_svd_linear_attention_inject` | `inject_triton_svd_linear_attention` |

## Family: `compression.experimental` (5 modules)

| Name | Class |
|---|---|
| `flat_llm` | `FLATLLMCompressor` |
| `hashed_net_compressor` | `hashed_net_factory` |
| `liftr` | `liftr_compressor_entry` |
| `lillama_compressor` | `lillama_factory` |
| `roast_hashed_linear` | `ROASTCompressor` |

## Family: `compression.kv_cache` (6 modules)

| Name | Class |
|---|---|
| `aatc_xkv_compressor` | `AATC_XKVCompressor` |
| `cachenotes_compressor` | `CacheNotesCompressor` |
| `delta_kv` | `delta_kv_factory` |
| `hybridkv_compressor` | `HybridKVCompressor` |
| `kvtc_compressor` | `KVTCCompressor` |
| `zerokv_neo_compressor` | `ZeroKVNeoCompressor` |

## Family: `compression.low_rank` (24 modules)

| Name | Class |
|---|---|
| `a3_compressor` | `A3Compressor` |
| `come_compressor` | `CoMeCompressor` |
| `compot_compressor` | `compot_compressor_factory` |
| `cr_compressor` | `CRCompressor` |
| `cur_compressor` | `CURCompressor` |
| `cur_moe_compressor` | `cur_moe_factory` |
| `drone_compressor` | `DRONECompressor` |
| `enlighten_compressor` | `ENLightenCompressor` |
| `frandi_compressor` | `frandi_factory` |
| `grecodec_compressor` | `grecodec_compressor_factory` |
| `lsc_compressor` | `lsc_factory` |
| `mgie_budget` | `MGIESVDBudgetCompressor` |
| `mlorq_compressor` | `MLoRQCompressor` |
| `modegpt_compressor` | `MoDeGPTCompressor` |
| `nmf_compressor` | `NMFCompressor` |
| `oats_compressor` | `OATSCompressor` |
| `rpca_compressor` | `RPCACompressor` |
| `salr` | `SALRCompressor` |
| `sdrm_compressor` | `sdrm_compressor_factory` |
| `slim_compressor` | `SLiMCompressor` |
| `soka_compressor` | `soka_factory` |
| `svd_llm_compressor` | `SVDLLMCompressor` |
| `tomoe_compressor` | `tomoe_compressor_factory` |
| `uniql_compressor` | `UniQLCompressor` |

## Family: `compression.pruning` (9 modules)

| Name | Class |
|---|---|
| `admm_compressor` | `ADMMCompressor` |
| `admm_pruner` | `ADMMPruner` |
| `armor_pruner` | `ARMORPruner` |
| `ast_pruner` | `ASTPruner` |
| `bonsai_pruner` | `bonsai_factory` |
| `flap` | `FLAPCompressor` |
| `mamba_shedder` | `mamba_shedder_factory` |
| `proxsparse` | `ProxSparsePruner` |
| `sparsegpt` | `sparsegpt_factory` |

## Family: `compression.quantization` (8 modules)

| Name | Class |
|---|---|
| `astro_quantizer` | `AstroQuantizer` |
| `coopq_quantizer` | `coopq_factory` |
| `faq_quantizer` | `FAQQuantizer` |
| `huffman_coder` | `HuffmanCompressor` |
| `kmeans_quantize` | `kmeans_factory` |
| `rotational_slicegpt_int4` | `RotationalSliceGPTINT4Compressor` |
| `slicegpt` | `PCAActivationAwareSVDCompressor` |
| `slicegpt_pca` | `SliceGPTPCACompressor` |

## Family: `compression.svd` (18 modules)

| Name | Class |
|---|---|
| `aa_svd_compressor` | `AASVDCompressor` |
| `adaptive_ranks` | `AdaptiveRanksCompressor` |
| `adasvd_compressor` | `AdaSVDCompressor` |
| `direct_svd` | `DirectSVDCompressor` |
| `dobi_svd` | `dobi_factory` |
| `dynamic_rank` | `dynamic_entropy_svd_factory` |
| `entropy_budget_allocation` | `EntropyBudgetCompressor` |
| `erc_svd` | `ERCCompressor` |
| `fwsvd_compressor` | `FWSVDCompressor` |
| `gfws` | `GFWSCompressor` |
| `hybrid_svd_layer` | `HybridSVDCompressor` |
| `optimal_subspace_projection` | `OptimalSubspaceProjectionCompressor` |
| `residual_svd_compressor` | `ResidualSVDCompressor` |
| `saes_svd_v2` | `SAESSVDCompressorV2` |
| `subspace_projection` | `SubspaceProjectionCompressor` |
| `svd_linear` | `SVDLinearCompressor` |
| `swift_svd` | `SwiftSVDCompressor` |
| `swift_svd_act` | `SwiftSVDActivationCompressor` |

## Family: `compression.tensor_decomp` (3 modules)

| Name | Class |
|---|---|
| `kronecker_compressor` | `kronecker_factory` |
| `kronecker_tt_lora` | `kronecker_tt_lora_factory` |
| `tt_compressor` | `TTCompressor` |
