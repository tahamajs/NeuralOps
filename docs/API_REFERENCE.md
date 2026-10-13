# NeuralOps Complete API Reference (v1.2.0)

This reference documents all registered Attention mechanisms (130) and Compression modules (73).

## Table of Contents
- [Attention Mechanisms (130)](#attention-mechanisms)
- [Compression Modules (73)](#compression-modules)

---

## Attention Mechanisms

### `adaptive_span_attention`
- **Class**: `neuralops.attentions.adaptive.adaptive_span_attention.AdaptiveSpanAttention`
- **Summary**: Every custom attention module must inherit from this.

### `adaptive_sparse_attention`
- **Class**: `neuralops.attentions.sparse.adaptive_sparse_attention.AdaptiveSparseAttention`
- **Summary**: Every custom attention module must inherit from this.

### `adaptive_sparse_attn`
- **Class**: `neuralops.attentions.sparse.adaptive_sparse_attention.AdaptiveSparseAttention`
- **Summary**: Every custom attention module must inherit from this.

### `adaptive_svd`
- **Class**: `neuralops.attentions.svd.adaptive_svd_attention.AdaptiveSVDAttention`
- **Summary**: Multi-head attention with adaptive SVD + residual factorized projections.

### `adaptive_svd_attention`
- **Class**: `neuralops.attentions.svd.adaptive_svd_attention.AdaptiveSVDAttention`
- **Summary**: Multi-head attention with adaptive SVD + residual factorized projections.

### `adasplash`
- **Class**: `neuralops.attentions.fast_kernels.adasplash_attention.AdaSplashAttention`
- **Summary**: AdaSplash Attention Layer with α-Entmax and Local Splash Radius.

### `adasplash_attention`
- **Class**: `neuralops.attentions.fast_kernels.adasplash_attention.AdaSplashAttention`
- **Summary**: AdaSplash Attention Layer with α-Entmax and Local Splash Radius.

### `alibi_attention`
- **Class**: `neuralops.attentions.modern.alibi.ALiBiAttention`
- **Summary**: Attention with linear biases instead of positional embeddings.

### `asvd`
- **Class**: `neuralops.attentions.svd.asvd_attention.ASVDAttention`
- **Summary**: Hardened ASVD Attention with Factored Low-Rank Matrix Decomposition, GQA, RoPE, and LoRA.

### `asvd_attention`
- **Class**: `neuralops.attentions.svd.asvd_attention.ASVDAttention`
- **Summary**: Hardened ASVD Attention with Factored Low-Rank Matrix Decomposition, GQA, RoPE, and LoRA.

### `bam_umoe`
- **Class**: `neuralops.attentions.sparse.bam_umoe.BAMUMoEBlock`
- **Summary**: Every custom attention module must inherit from this.

### `block_transformer_hiba`
- **Class**: `neuralops.attentions.adaptive.block_transformer_hiba.BlockTransformerHIBA`
- **Summary**: Block Transformer with Hierarchical Intra-Block and Inter-Block Attention (HIBA).

### `chela_slay`
- **Class**: `neuralops.attentions.linear.chela_slay.ChelaSlayAttention`
- **Summary**: CHELA-SLAY: Spherical Linearized Attention with Yat-Kernel.

### `chela_slay_attention`
- **Class**: `neuralops.attentions.linear.chela_slay.ChelaSlayAttention`
- **Summary**: CHELA-SLAY: Spherical Linearized Attention with Yat-Kernel.

### `cluster`
- **Class**: `neuralops.attentions.sparse.cluster_attention.ClusterAttention`
- **Summary**: Every custom attention module must inherit from this.

### `cluster_attention`
- **Class**: `neuralops.attentions.sparse.cluster_attention.ClusterAttention`
- **Summary**: Every custom attention module must inherit from this.

### `compressed_attention`
- **Class**: `neuralops.attentions.svd.compressed_attention.CompressedAttentionLayer`
- **Summary**: Every custom attention module must inherit from this.

### `compressed_attention_layer`
- **Class**: `neuralops.attentions.svd.compressed_attention.CompressedAttentionLayer`
- **Summary**: Every custom attention module must inherit from this.

### `compressed_attention_layer_energy`
- **Class**: `neuralops.attentions.svd.compressed_attention.CompressedAttentionLayer2`
- **Summary**: Every custom attention module must inherit from this.

### `compressed_mha`
- **Class**: `neuralops.attentions.svd.compressed_mha_attention.CompressedDistilBertAttention`
- **Summary**: Every custom attention module must inherit from this.

### `compressed_mha_attention`
- **Class**: `neuralops.attentions.svd.compressed_mha_attention.CompressedDistilBertAttention`
- **Summary**: Every custom attention module must inherit from this.

### `cosformer`
- **Class**: `neuralops.attentions.linear.cosformer.CosformerAttention`
- **Summary**: Every custom attention module must inherit from this.

### `cross_attention`
- **Class**: `neuralops.attentions.modern.cross_attention.CrossAttention`
- **Summary**: Multi-head cross attention between query sequence and encoder/context sequence.

### `decay_linear_attention`
- **Class**: `neuralops.attentions.gated.decay_linear_attention.DecayLinearAttention`
- **Summary**: Decay Linear Attention Module with Data-Dependent Exponential Decay Gating.

### `decay_linear_attn`
- **Class**: `neuralops.attentions.gated.decay_linear_attention.DecayLinearAttention`
- **Summary**: Decay Linear Attention Module with Data-Dependent Exponential Decay Gating.

### `eat`
- **Class**: `neuralops.attentions.adaptive.eat.EATBlock`
- **Summary**: Every custom attention module must inherit from this.

### `eat_attention`
- **Class**: `neuralops.attentions.adaptive.eat.EATBlock`
- **Summary**: Every custom attention module must inherit from this.

### `expert_choice_moe`
- **Class**: `neuralops.attentions.sparse.expert_choice_moe.ExpertChoiceMoEAttention`
- **Summary**: Expert Choice MoE Multi-Head Attention Module.

### `expert_choice_moe_attention`
- **Class**: `neuralops.attentions.sparse.expert_choice_moe.ExpertChoiceMoEAttention`
- **Summary**: Expert Choice MoE Multi-Head Attention Module.

### `factorised_matrix_state`
- **Class**: `neuralops.attentions.svd.factorised_matrix_state.FactorisedMatrixState`
- **Summary**: Bidirectional & Causal Factorised Matrix State Recurrent Linear Attention.

### `flash3`
- **Class**: `neuralops.attentions.fast_kernels.flash_attention3.FlashAttention3`
- **Summary**: FlashAttention-3 Multi-Head Exact Attention.

### `flash3_fp8`
- **Class**: `neuralops.attentions.fast_kernels.flash_attention3_fp8.FlashAttention3FP8`
- **Summary**: FlashAttention-3 FP8 Hardware-Accelerated Multi-Head Attention.

### `flash_attention3`
- **Class**: `neuralops.attentions.fast_kernels.flash_attention3.FlashAttention3`
- **Summary**: FlashAttention-3 Multi-Head Exact Attention.

### `flash_attention3_fp8`
- **Class**: `neuralops.attentions.fast_kernels.flash_attention3_fp8.FlashAttention3FP8`
- **Summary**: FlashAttention-3 FP8 Hardware-Accelerated Multi-Head Attention.

### `fms`
- **Class**: `neuralops.attentions.svd.factorised_matrix_state.FactorisedMatrixState`
- **Summary**: Bidirectional & Causal Factorised Matrix State Recurrent Linear Attention.

### `full`
- **Class**: `neuralops.attentions.dense.full_attention.FullAttention`
- **Summary**: Every custom attention module must inherit from this.

### `full_attention`
- **Class**: `neuralops.attentions.dense.full_attention.FullAttention`
- **Summary**: Every custom attention module must inherit from this.

### `fused_svd_kernel`
- **Class**: `neuralops.attentions.svd.fused_svd_kernel.FusedSVDKernelAttention`
- **Summary**: Fused SVD-Kernel Linear Attention Module.

### `fused_svd_kernel_attention`
- **Class**: `neuralops.attentions.svd.fused_svd_kernel.FusedSVDKernelAttention`
- **Summary**: Fused SVD-Kernel Linear Attention Module.

### `fusion_attention`
- **Class**: `neuralops.attentions.hybrid.fusion_attn.FUSIONATTN`
- **Summary**: Hybrid sparse attention with block gating.

### `fusion_attn`
- **Class**: `neuralops.attentions.hybrid.fusion_attn.FUSIONATTN`
- **Summary**: Hybrid sparse attention with block gating.

### `galerkin_attention`
- **Class**: `neuralops.attentions.linear.galerkin_attention.GalerkinAttention`
- **Summary**: Every custom attention module must inherit from this.

### `gated_delta_net`
- **Class**: `neuralops.attentions.gated.gated_delta_net.GatedDeltaNet`
- **Summary**: Every custom attention module must inherit from this.

### `gated_deltanet_block`
- **Class**: `neuralops.attentions.gated.gated_deltanet_block.GatedDeltaNetBlock`
- **Summary**: Gated DeltaNet Transformer Block with Pretrained Weight Transfer.

### `gated_hybrid_kernel`
- **Class**: `neuralops.attentions.hybrid.gated_hybrid_kernel_attention.GatedHybridKernelAttention`
- **Summary**: Every custom attention module must inherit from this.

### `gated_linear_attention`
- **Class**: `neuralops.attentions.gated.gated_linear_attention.GatedLinearAttention`
- **Summary**: Every custom attention module must inherit from this.

### `goldfinch`
- **Class**: `neuralops.attentions.ssm.goldfinch_ahn.GoldFinchAHNBlock`
- **Summary**: GoldFinch-AHN: hybrid sliding-window attention + gated recurrent memory.

### `goldfinch_ahn`
- **Class**: `neuralops.attentions.ssm.goldfinch_ahn.GoldFinchAHNBlock`
- **Summary**: GoldFinch-AHN: hybrid sliding-window attention + gated recurrent memory.

### `gqa_attention`
- **Class**: `neuralops.attentions.modern.gqa_attention.GroupedQueryAttention`
- **Summary**: Every custom attention module must inherit from this.

### `granite_hybrid`
- **Class**: `neuralops.attentions.hybrid.granite_hybrid.GraniteHybridBlock`
- **Summary**: Granite Hybrid Attention-SSM Layer with MoE Feed-Forward and LoRA Adaptation.

### `griffin_ssd`
- **Class**: `neuralops.attentions.ssm.griffin_ssd.GriffinSSDBlock`
- **Summary**: Griffin Hybrid Block: local window + SSD linear attention.

### `grouped_query_attention`
- **Class**: `neuralops.attentions.modern.gqa_attention.GroupedQueryAttention`
- **Summary**: Every custom attention module must inherit from this.

### `halo_regla`
- **Class**: `neuralops.attentions.hybrid.halo_regla.HALOReGLAAttention`
- **Summary**: Hybrid Dense Softmax + Linear Attention with adaptive fusion gate.

### `hamming_attention`
- **Class**: `neuralops.attentions.sparse.hamming_attention.HammingAttention`
- **Summary**: Every custom attention module must inherit from this.

### `hgrn2_gated_deltanet`
- **Class**: `neuralops.attentions.gated.hgrn2_gated_deltanet.HGRN2GatedDeltaNet`
- **Summary**: HGRN-2 + Gated DeltaNet hybrid with proper fusion.

### `hgrn2_gated_layer`
- **Class**: `neuralops.attentions.gated.hgrn2_gated_layer.HGRN2Layer`
- **Summary**: HGRN-2 hybrid with local sliding-window attention.

### `hgrn2_gated_outer`
- **Class**: `neuralops.attentions.gated.hgrn2_gated_outer.HGRN2GatedOuterBlock`
- **Summary**: Every custom attention module must inherit from this.

### `hgrn2_outer`
- **Class**: `neuralops.attentions.gated.hgrn2_gated_outer.HGRN2GatedOuterBlock`
- **Summary**: Every custom attention module must inherit from this.

### `hla_attention`
- **Class**: `neuralops.attentions.gated.hla_attention.HLAAttention`
- **Summary**: Hadamard Linear Attention (HLA) Bidirectional / Causal Module.

### `hybrid_attention`
- **Class**: `neuralops.attentions.hybrid.hybrid_attention.HybridAttention`
- **Summary**: Hybrid softmax + linear attention with per-head dynamic gated fusion.

### `hybrid_local_kernel`
- **Class**: `neuralops.attentions.hybrid.hybrid_local_kernel_attention.HybridLocalKernelAttention`
- **Summary**: Every custom attention module must inherit from this.

### `hymba_transx`
- **Class**: `neuralops.attentions.ssm.hymba_transx.HymbaTransXBlock`
- **Summary**: HymbaTransX: Hybrid Differential Transformer & State Space Model Block.

### `hymba_transx_attn`
- **Class**: `neuralops.attentions.ssm.hymba_transx.HymbaTransXBlock`
- **Summary**: HymbaTransX: Hybrid Differential Transformer & State Space Model Block.

### `inject_triton_cosformer`
- **Class**: `neuralops.attentions.linear.triton_cosformer.TritonCosformerAttention`
- **Summary**: Triton-Accelerated Cosformer Linear Attention with Full Autograd Differentiability,

### `inject_triton_svd_linear_attention`
- **Class**: `neuralops.attentions.svd.triton_svd_linear_attention.inject_triton_svd_linear_attention`
- **Summary**: Safe factory function to inject TritonSVDLinearAttention into a model.

### `inject_xattention`
- **Class**: `neuralops.attentions.fast_kernels.xattention.inject_xattention`
- **Summary**: No docstring available.

### `jamba`
- **Class**: `neuralops.attentions.adaptive.infini_attention.Jamba2Block`
- **Summary**: Every custom attention module must inherit from this.

### `jamba_info`
- **Class**: `neuralops.attentions.ssm.jamba_info.JambaInfoBlock`
- **Summary**: Every custom attention module must inherit from this.

### `kda_attention`
- **Class**: `neuralops.attentions.gated.kda_attention.KDAAttention`
- **Summary**: Every custom attention module must inherit from this.

### `landmark_local`
- **Class**: `neuralops.attentions.hybrid.landmark_local_attention.LandmarkLocalAttention`
- **Summary**: Every custom attention module must inherit from this.

### `linformer_attention`
- **Class**: `neuralops.attentions.linear.linformer_attention.LinformerAttention`
- **Summary**: Every custom attention module must inherit from this.

### `lola_attention`
- **Class**: `neuralops.attentions.adaptive.lola_attention.LoLAAttention`
- **Summary**: Every custom attention module must inherit from this.

### `lora_attention`
- **Class**: `neuralops.attentions.svd.lora_attention.LoRAAttention`
- **Summary**: LoRA attention wrapper.

### `lora_attn`
- **Class**: `neuralops.attentions.svd.lora_attention.LoRAAttention`
- **Summary**: LoRA attention wrapper.

### `lsh_attention`
- **Class**: `neuralops.attentions.sparse.lsh_attention.LSHAttention`
- **Summary**: Every custom attention module must inherit from this.

### `mamba_block`
- **Class**: `neuralops.attentions.ssm.mamba_block.MambaBlock`
- **Summary**: Faithful S6 / Mamba block, production-hardened.

### `mamba_svd_flash`
- **Class**: `neuralops.attentions.ssm.mamba_svd_flash.MambaSVDFlash`
- **Summary**: Hybrid SSM + MHA with per-chunk SVD and learnable per-head gating.

### `minimal_nova`
- **Class**: `neuralops.attentions.svd.minimal_nova.MinimalNOVAAttention`
- **Summary**: MinimalNOVA: SVD-Factorized Low-Rank Projection Attention.

### `minimal_svd`
- **Class**: `neuralops.attentions.svd.minimal_svd.MinimalSVDAttention`
- **Summary**: MinimalSVD: SVD-Factorized Low-Rank Projection Attention.

### `mixattention_moska`
- **Class**: `neuralops.attentions.hybrid.mixattention_moska.MixAttentionMoSKA`
- **Summary**: MixAttention-MoSKA (FASA): frequency/variance-aware sparse attention.

### `mla_attention`
- **Class**: `neuralops.attentions.modern.mla_attention.MLAAttention`
- **Summary**: Multi-Head Latent Attention (MLA) — DeepSeek-V2 style.

### `moba_attention`
- **Class**: `neuralops.attentions.sparse.moba_attention.MoBAAttention`
- **Summary**: MoBA: Mixture of Block Attention for Long-Context Modeling.

### `moe_deltakv_paged`
- **Class**: `neuralops.attentions.sparse.moe_deltakv_paged.MoEDeltaKVPagedAttention`
- **Summary**: MoE-DeltaKV-Paged Hybrid Attention.

### `mqa_attention`
- **Class**: `neuralops.attentions.modern.mqa.MultiQueryAttention`
- **Summary**: Multi-Query Attention — multiple query heads share single key and value heads.

### `naive_svd`
- **Class**: `neuralops.attentions.svd.naive_svd.NaiveSVDAttention`
- **Summary**: Naive truncated SVD baseline attention module.

### `nasc_attention`
- **Class**: `neuralops.attentions.linear.nasc_attention.NASCAttention`
- **Summary**: NASC: NEXUS-AEGIS Spectral Cascade Attention.

### `nexus`
- **Class**: `neuralops.attentions.ssm.nexus.NEXUSAttention`
- **Summary**: Every custom attention module must inherit from this.

### `nha_class`
- **Class**: `neuralops.attentions.modern.nha_attention.NativeHybridAttention`
- **Summary**: Native Hybrid Attention: local sliding-window + global RFF, fused by a

### `nsa`
- **Class**: `neuralops.attentions.sparse.nsa_attention.NativeSparseAttention`
- **Summary**: Every custom attention module must inherit from this.

### `nsa_attention`
- **Class**: `neuralops.attentions.sparse.nsa_attention.NativeSparseAttention`
- **Summary**: Every custom attention module must inherit from this.

### `nystrom`
- **Class**: `neuralops.attentions.linear.nystrom_attention.NystromAttention`
- **Summary**: Every custom attention module must inherit from this.

### `nystrom_attention`
- **Class**: `neuralops.attentions.linear.nystrom_attention.NystromAttention`
- **Summary**: Every custom attention module must inherit from this.

### `optimized_hybrid_attention`
- **Class**: `neuralops.attentions.hybrid.optimized_hybrid_attention.OptimizedHybridAttention`
- **Summary**: Every custom attention module must inherit from this.

### `orion`
- **Class**: `neuralops.attentions.ssm.orion.ORIONAttention`
- **Summary**: Every custom attention module must inherit from this.

### `padre_attention`
- **Class**: `neuralops.attentions.sparse.padre_attention.PADReAttention`
- **Summary**: Polynomial Attention with Dropout Regularization (PADRe).

### `performer`
- **Class**: `neuralops.attentions.linear.performer_attention.PerformerAttention`
- **Summary**: Every custom attention module must inherit from this.

### `performer_attention`
- **Class**: `neuralops.attentions.linear.performer_attention.PerformerAttention`
- **Summary**: Every custom attention module must inherit from this.

### `phantom`
- **Class**: `neuralops.attentions.ssm.phantom.PHANTOMAttention`
- **Summary**: Every custom attention module must inherit from this.

### `positive_kernel_linear`
- **Class**: `neuralops.attentions.hybrid.positive_kernel_linear_attention.PositiveKernelLinearAttention`
- **Summary**: Every custom attention module must inherit from this.

### `primal_attention`
- **Class**: `neuralops.attentions.linear.primal_attention.PrimalAttention`
- **Summary**: Every custom attention module must inherit from this.

### `progressive_token_pruning`
- **Class**: `neuralops.attentions.adaptive.progressive_token_pruning.ProgressiveTokenPruning`
- **Summary**: Progressive Token Pruning attention.

### `projection_factorized_attn`
- **Class**: `neuralops.attentions.svd.projection_factorized_attn.ProjectionFactorizedAttention`
- **Summary**: Projection Factorized SVD Linear Attention.

### `qcqa_attention`
- **Class**: `neuralops.attentions.modern.qcqa_attention.QCQAAttention`
- **Summary**: Every custom attention module must inherit from this.

### `rala_attention`
- **Class**: `neuralops.attentions.linear.rala_attention.RALALinearAttention`
- **Summary**: Every custom attention module must inherit from this.

### `rala_compressor`
- **Class**: `neuralops.attentions.linear.rala_attention.RALALinearAttention`
- **Summary**: Every custom attention module must inherit from this.

### `reap_moe`
- **Class**: `neuralops.attentions.sparse.reap_moe.REAPMoEAttention`
- **Summary**: Every custom attention module must inherit from this.

### `refined_landmark`
- **Class**: `neuralops.attentions.hybrid.refined_landmark_attention.RefinedLandmarkAttention`
- **Summary**: Every custom attention module must inherit from this.

### `regla_attention`
- **Class**: `neuralops.attentions.hybrid.regla_attention.inject_regla`
- **Summary**: No docstring available.

### `residual_svd_attention`
- **Class**: `neuralops.attentions.svd.residual_svd_attention.ResidualSVDAttention`
- **Summary**: Every custom attention module must inherit from this.

### `routing_attention`
- **Class**: `neuralops.attentions.sparse.routing_attention.inject_routing`
- **Summary**: No docstring available.

### `rpc_attention`
- **Class**: `neuralops.attentions.linear.rpc_attention.inject_rpc`
- **Summary**: No docstring available.

### `salt_attention`
- **Class**: `neuralops.attentions.svd.salt_attention.SALTAttention`
- **Summary**: SALT Attention module replacing standard linear projections with SALTLinear.

### `samba_m3`
- **Class**: `neuralops.attentions.ssm.samba_m3.SAMBAM3Block`
- **Summary**: SAMBA-M3: Local Sliding-Window Attention + Spectral Mamba.

### `samba_moe`
- **Class**: `neuralops.attentions.ssm.samba_moe.SambaMoEBlock`
- **Summary**: SambaMoE: SAMBA-M3 + REAP-MoE Hybrid Architecture with pretrained weight copying.

### `selective_mamba`
- **Class**: `neuralops.attentions.ssm.upi_stuffed_mamba.SelectiveMamba`
- **Summary**: Selective Mamba SSM block.

### `selective_ssm`
- **Class**: `neuralops.attentions.ssm.selective_ssm.SelectiveSSM`
- **Summary**: Faithful Mamba-style selective SSM block with hardened execution paths.

### `simple_ssm`
- **Class**: `neuralops.attentions.ssm.selective_ssm.SelectiveSSM`
- **Summary**: Faithful Mamba-style selective SSM block with hardened execution paths.

### `simplessm`
- **Class**: `neuralops.attentions.ssm.selective_ssm.SelectiveSSM`
- **Summary**: Faithful Mamba-style selective SSM block with hardened execution paths.

### `sink_attention`
- **Class**: `neuralops.attentions.adaptive.sink_attention.SinkAttention`
- **Summary**: Attention with dedicated sink tokens that always get attention.

### `sliding_window_attention`
- **Class**: `neuralops.attentions.modern.sliding_window.SlidingWindowAttention`
- **Summary**: Sliding window attention — attends to last W tokens only per position.

### `soa_attn`
- **Class**: `neuralops.attentions.fast_kernels.soa_attn.SOAAttention`
- **Summary**: Shared Orthogonal Attention (SOA) with Per-Projection Latent Bottlenecks.

### `span_expanded_attention`
- **Class**: `neuralops.attentions.adaptive.span_expanded_attention.SpanExpandedAttention`
- **Summary**: Every custom attention module must inherit from this.

### `spectra_v2`
- **Class**: `neuralops.attentions.linear.spectra_v2.SpectraV2Attention`
- **Summary**: MHA wrapper around PyTorch's scaled_dot_product_attention.

### `spectraformer_attention`
- **Class**: `neuralops.attentions.linear.spectraformer_attention.SpectraformerAttention`
- **Summary**: Spectraformer Attention: Exact linear-time attention via random feature kernelization.

### `spectral_mamba`
- **Class**: `neuralops.attentions.ssm.spectral_mamba.SpectralMambaAttention`
- **Summary**: Every custom attention module must inherit from this.

### `spectron`
- **Class**: `neuralops.attentions.linear.spectron.SpectronAttention`
- **Summary**: Every custom attention module must inherit from this.

### `spectron_fp8`
- **Class**: `neuralops.attentions.hybrid.spectron_fp8_attention.SpectronFP8Attention`
- **Summary**: Every custom attention module must inherit from this.

### `split_head_hybrid`
- **Class**: `neuralops.attentions.hybrid.split_head_hybrid_attention.SplitHeadHybridAttention`
- **Summary**: Every custom attention module must inherit from this.

### `squeezing_heads_distill`
- **Class**: `neuralops.attentions.adaptive.squeezing_heads_distill.SqueezingHeadsDistiller`
- **Summary**: Attention wrapper that keeps only the top-k most important heads

### `svd_compressed`
- **Class**: `neuralops.attentions.svd.svd_compressed.SVDCompressedAttention`
- **Summary**: Multi-head attention with truncated SVD projections, trainable LoRA adapters,

### `svd_linear_attention`
- **Class**: `neuralops.attentions.svd.svd_linear_attention.SVDLinearAttention`
- **Summary**: SVD Factorized Linear Attention with optional exponential decay gating.

### `svd_linear_attn`
- **Class**: `neuralops.attentions.svd.svd_linear_attention.SVDLinearAttention`
- **Summary**: SVD Factorized Linear Attention with optional exponential decay gating.

### `svd_linear_decay`
- **Class**: `neuralops.attentions.svd.svd_linear_decay.DecayGatedLinearAttention`
- **Summary**: Decay Gated SVD Linear Attention.

### `svd_llm_v2_attention`
- **Class**: `neuralops.attentions.svd.svd_llm_v2_attention.SVDLLMV2Attention`
- **Summary**: SVD-factorized attention layer with explicit U, Sigma, V^H matrix buffers

### `svd_mamba`
- **Class**: `neuralops.attentions.svd.svd_mamba.SVDMamba`
- **Summary**: SVD-Mamba module with SVD-factorized projections, causal convolution, and selective SSM.

### `svd_projection_attention`
- **Class**: `neuralops.attentions.svd.svd_projection_attention.SVDProjectionAttention`
- **Summary**: Multi-Head Attention using Factorized SVD Linear Projections.

### `svd_seq`
- **Class**: `neuralops.attentions.svd.svd_attention.SVDLinearAttention`
- **Summary**: Low-Rank SVD-Compressed Sequence Attention Mechanism.

### `template_attention`
- **Class**: `neuralops.attentions.experimental.template_attention.inject_custom_attn`
- **Summary**: No docstring available.

### `tlinformer_attention`
- **Class**: `neuralops.attentions.linear.tlinformer_attention.TLinFormerAttention`
- **Summary**: TLinFormer: linear-time attention with learnable low-rank sequence projection.

### `token_compressed_attention`
- **Class**: `neuralops.attentions.adaptive.token_compressed_attention.TokenCompressedAttention`
- **Summary**: Token-Compressed Attention v3.

### `tpa_attention`
- **Class**: `neuralops.attentions.linear.tpa_attention.TPAttention`
- **Summary**: Tensor Product Attention v2 with asymmetric rank support.

### `tri_branch_hybrid`
- **Class**: `neuralops.attentions.hybrid.tri_branch_hybrid_attention.TriBranchHybridAttention`
- **Summary**: Every custom attention module must inherit from this.

### `triton_cosformer`
- **Class**: `neuralops.attentions.linear.triton_cosformer.TritonCosformerAttention`
- **Summary**: Triton-Accelerated Cosformer Linear Attention with Full Autograd Differentiability,

### `triton_cosformer_inject`
- **Class**: `neuralops.attentions.linear.triton_cosformer.TritonCosformerAttention`
- **Summary**: Triton-Accelerated Cosformer Linear Attention with Full Autograd Differentiability,

### `triton_svd_linear_attention`
- **Class**: `neuralops.attentions.svd.triton_svd_linear_attention.TritonSVDLinearAttention`
- **Summary**: Triton-Accelerated Linear Attention with Genuine SVD Low-Rank Projections,

### `triton_svd_linear_attention_inject`
- **Class**: `neuralops.attentions.svd.triton_svd_linear_attention.inject_triton_svd_linear_attention`
- **Summary**: Safe factory function to inject TritonSVDLinearAttention into a model.

### `unified_rope`
- **Class**: `neuralops.attentions.modern.unified_rope.UnifiedRoPE`
- **Summary**: Universal Rotary Position Embedding Attention module.

### `upi_mamba`
- **Class**: `neuralops.attentions.ssm.upi_stuffed_mamba.SelectiveMamba`
- **Summary**: Selective Mamba SSM block.

### `upi_stuffed_mamba`
- **Class**: `neuralops.attentions.ssm.upi_stuffed_mamba.SelectiveMamba`
- **Summary**: Selective Mamba SSM block.

### `vig`
- **Class**: `neuralops.attentions.modern.vig_attention.ViGAttention`
- **Summary**: Vision GNN-inspired 2D Grid Linear Attention with Learned Gated Fusion.

### `vig_attention`
- **Class**: `neuralops.attentions.modern.vig_attention.ViGAttention`
- **Summary**: Vision GNN-inspired 2D Grid Linear Attention with Learned Gated Fusion.

### `vspec`
- **Class**: `neuralops.attentions.ssm.vspec.VSpecAttention`
- **Summary**: VSpec++ Attention: Variational Spectral Token-Adaptive Attention Module.

### `vspec_attention`
- **Class**: `neuralops.attentions.ssm.vspec.VSpecAttention`
- **Summary**: VSpec++ Attention: Variational Spectral Token-Adaptive Attention Module.

### `wgqa`
- **Class**: `neuralops.attentions.modern.wgqa_attention.WeightedGroupedQueryAttention`
- **Summary**: Weighted Grouped-Query Attention (WGQA) with learnable per-KV-head importance weights,

### `wgqa_attention`
- **Class**: `neuralops.attentions.modern.wgqa_attention.WeightedGroupedQueryAttention`
- **Summary**: Weighted Grouped-Query Attention (WGQA) with learnable per-KV-head importance weights,

### `x_attention`
- **Class**: `neuralops.attentions.fast_kernels.xattention.XAttention`
- **Summary**: Every custom attention module must inherit from this.

### `xattention`
- **Class**: `neuralops.attentions.fast_kernels.xattention.XAttention`
- **Summary**: Every custom attention module must inherit from this.

### `zeros`
- **Class**: `neuralops.attentions.dense.zeros_attention.ZeroSAttention`
- **Summary**: Every custom attention module must inherit from this.

### `zeros_attention`
- **Class**: `neuralops.attentions.dense.zeros_attention.ZeroSAttention`
- **Summary**: Every custom attention module must inherit from this.

---

## Compression Modules

### `a3_compressor`
- **Class**: `neuralops.compression_engine.low_rank.a3_compressor.A3Compressor`
- **Summary**: A3: Activation-aware factorized SVD compressor.

### `aa_svd_compressor`
- **Class**: `neuralops.compression_engine.svd.aa_svd_compressor.AASVDCompressor`
- **Summary**: Base class for all compressors. Not a nn.Module to avoid parameter storage.

### `aatc_xkv_compressor`
- **Class**: `neuralops.compression_engine.kv_cache.aatc_xkv_compressor.AATC_XKVCompressor`
- **Summary**: Base class for all compressors. Not a nn.Module to avoid parameter storage.

### `adaptive_ranks`
- **Class**: `neuralops.compression_engine.svd.adaptive_ranks.AdaptiveRanksCompressor`
- **Summary**: Base class for all compressors. Not a nn.Module to avoid parameter storage.

### `adasvd_compressor`
- **Class**: `neuralops.compression_engine.svd.adasvd_compressor.AdaSVDCompressor`
- **Summary**: Base class for all compressors. Not a nn.Module to avoid parameter storage.

### `admm_compressor`
- **Class**: `neuralops.compression_engine.pruning.admm_compressor.ADMMCompressor`
- **Summary**: ADMM-based joint structured pruning + uniform quantization engine.

### `admm_pruner`
- **Class**: `neuralops.compression_engine.pruning.admm_pruner.ADMMPruner`
- **Summary**: Production-hardened ADMM Pruner with Layer-Wise Reconstruction Distillation,

### `armor_pruner`
- **Class**: `neuralops.compression_engine.pruning.armor_pruner.ARMORPruner`
- **Summary**: Production-hardened ARMOR Pruner.

### `ast_pruner`
- **Class**: `neuralops.compression_engine.pruning.ast_pruner.ASTPruner`
- **Summary**: AST Pruner: vectorized semi-structured 2:4 / N:M pruning with empirical

### `astro_quantizer`
- **Class**: `neuralops.compression_engine.quantization.astro_quantizer.AstroQuantizer`
- **Summary**: AstroQuantizer: Uniform min-max / symmetric quantization engine with Fisher-adaptive bit allocation.

### `bonsai_pruner`
- **Class**: `neuralops.compression_engine.pruning.bonsai_pruner.bonsai_factory`
- **Summary**: Factory for Bonsai pruner with progressive sparsity.

### `cachenotes_compressor`
- **Class**: `neuralops.compression_engine.kv_cache.cachenotes_compressor.CacheNotesCompressor`
- **Summary**: Base class for all compressors. Not a nn.Module to avoid parameter storage.

### `come_compressor`
- **Class**: `neuralops.compression_engine.low_rank.come_compressor.CoMeCompressor`
- **Summary**: CoMe Compressor: Layer shedding with learnable low-rank interpolation

### `compot_compressor`
- **Class**: `neuralops.compression_engine.low_rank.compot_compressor.compot_compressor_factory`
- **Summary**: Factory function for COMPOT compression.

### `coopq_quantizer`
- **Class**: `neuralops.compression_engine.quantization.coopq_quantizer.coopq_factory`
- **Summary**: CoopQ compressor factory function.

### `cr_compressor`
- **Class**: `neuralops.compression_engine.low_rank.cr_compressor.CRCompressor`
- **Summary**: CR (Column-Row / Interpolative Decomposition) Compressor.

### `cur_compressor`
- **Class**: `neuralops.compression_engine.low_rank.cur_compressor.CURCompressor`
- **Summary**: Base class for all compressors. Not a nn.Module to avoid parameter storage.

### `cur_moe_compressor`
- **Class**: `neuralops.compression_engine.low_rank.cur_moe_compressor.cur_moe_factory`
- **Summary**: Factory function for CUR-MoE compressor.

### `delta_kv`
- **Class**: `neuralops.compression_engine.kv_cache.delta_kv.delta_kv_factory`
- **Summary**: DeltaKV compressor factory.

### `direct_svd`
- **Class**: `neuralops.compression_engine.svd.direct_svd.DirectSVDCompressor`
- **Summary**: Base class for all compressors. Not a nn.Module to avoid parameter storage.

### `dobi_svd`
- **Class**: `neuralops.compression_engine.svd.dobi_svd.dobi_factory`
- **Summary**: Dobi factory function.

### `drone_compressor`
- **Class**: `neuralops.compression_engine.low_rank.drone_compressor.DRONECompressor`
- **Summary**: DRONE Compressor.

### `dynamic_rank`
- **Class**: `neuralops.compression_engine.svd.dynamic_rank.dynamic_entropy_svd_factory`
- **Summary**: Dynamic Entropy SVD factory function.

### `enlighten_compressor`
- **Class**: `neuralops.compression_engine.low_rank.enlighten_compressor.ENLightenCompressor`
- **Summary**: Base class for all compressors. Not a nn.Module to avoid parameter storage.

### `entropy_budget_allocation`
- **Class**: `neuralops.compression_engine.svd.entropy_budget_allocation.EntropyBudgetCompressor`
- **Summary**: Base class for all compressors. Not a nn.Module to avoid parameter storage.

### `erc_svd`
- **Class**: `neuralops.compression_engine.svd.erc_svd.ERCCompressor`
- **Summary**: Base class for all compressors. Not a nn.Module to avoid parameter storage.

### `faq_quantizer`
- **Class**: `neuralops.compression_engine.quantization.faq_quantizer.FAQQuantizer`
- **Summary**: FAQQuantizer: Feature-Activation-Aware Quantization.

### `flap`
- **Class**: `neuralops.compression_engine.pruning.flap.FLAPCompressor`
- **Summary**: FLAP Compressor: Fluctuation-based Adaptive Structured Pruning.

### `flat_llm`
- **Class**: `neuralops.compression_engine.experimental.flat_llm.FLATLLMCompressor`
- **Summary**: FLAT-LLM compressor: Activation-aware output-channel pruning.

### `frandi_compressor`
- **Class**: `neuralops.compression_engine.low_rank.frandi_compressor.frandi_factory`
- **Summary**: Factory function for FRanDI compressor.

### `fwsvd_compressor`
- **Class**: `neuralops.compression_engine.svd.fwsvd_compressor.FWSVDCompressor`
- **Summary**: Fisher-Weighted SVD Compressor.

### `gfws`
- **Class**: `neuralops.compression_engine.svd.gfws_d_compressor.GFWSCompressor`
- **Summary**: Base class for all compressors. Not a nn.Module to avoid parameter storage.

### `grecodec_compressor`
- **Class**: `neuralops.compression_engine.low_rank.grecodec_compressor.grecodec_compressor_factory`
- **Summary**: Factory function for GreCoDeC compressor.

### `hashed_net_compressor`
- **Class**: `neuralops.compression_engine.experimental.hashed_net_compressor.hashed_net_factory`
- **Summary**: No docstring available.

### `huffman_coder`
- **Class**: `neuralops.compression_engine.quantization.huffman_coder.HuffmanCompressor`
- **Summary**: HuffmanCompressor: Model-wide entropy compression profiler and simulator.

### `hybrid_svd_layer`
- **Class**: `neuralops.compression_engine.svd.hybrid_svd_layer.HybridSVDCompressor`
- **Summary**: Weight-based SVD compressor for Linear / Conv1D layers.

### `hybridkv_compressor`
- **Class**: `neuralops.compression_engine.kv_cache.hybridkv_compressor.HybridKVCompressor`
- **Summary**: HybridKV Compressor: Classifies projection heads into static/dynamic types

### `kmeans_quantize`
- **Class**: `neuralops.compression_engine.quantization.kmeans_quantize.kmeans_factory`
- **Summary**: No docstring available.

### `kronecker_compressor`
- **Class**: `neuralops.compression_engine.tensor_decomp.kronecker_compressor.kronecker_factory`
- **Summary**: Kronecker compressor factory.

### `kronecker_tt_lora`
- **Class**: `neuralops.compression_engine.tensor_decomp.kronecker_tt_lora.kronecker_tt_lora_factory`
- **Summary**: Factory for Kronecker-TT-LoRA compressor.

### `kvtc_compressor`
- **Class**: `neuralops.compression_engine.kv_cache.kvtc_compressor.KVTCCompressor`
- **Summary**: KVTC: KV Cache Transform Coding compressor (Static Weight PCA + Quantization).

### `liftr`
- **Class**: `neuralops.compression_engine.experimental.liftr.liftr_compressor_entry`
- **Summary**: Factory function for LIFTR compressor.

### `lillama_compressor`
- **Class**: `neuralops.compression_engine.experimental.lillama_compressor.lillama_factory`
- **Summary**: Factory function for LiLLaMA compressor.

### `lsc_compressor`
- **Class**: `neuralops.compression_engine.low_rank.lsc_compressor.lsc_factory`
- **Summary**: LSC Fourier spectral compressor factory.

### `mamba_shedder`
- **Class**: `neuralops.compression_engine.pruning.mamba_shedder.mamba_shedder_factory`
- **Summary**: No docstring available.

### `mgie_budget`
- **Class**: `neuralops.compression_engine.low_rank.mgie_svd_compressor.MGIESVDBudgetCompressor`
- **Summary**: MGIE-SVD Budget Compressor: Global Entropy-Proportional Rank Allocation.

### `mlorq_compressor`
- **Class**: `neuralops.compression_engine.low_rank.mlorq_compressor.MLoRQCompressor`
- **Summary**: MLoRQ Compressor: Multi-Layer Optimal Rank and Quantization under Memory Budget.

### `modegpt_compressor`
- **Class**: `neuralops.compression_engine.low_rank.modegpt_compressor.MoDeGPTCompressor`
- **Summary**: MoDeGPT: Modular Decomposition of Transformer Projection Blocks.

### `nmf_compressor`
- **Class**: `neuralops.compression_engine.low_rank.nmf_compressor.NMFCompressor`
- **Summary**: NMF Compressor: decomposes linear layer magnitudes into non-negative

### `oats_compressor`
- **Class**: `neuralops.compression_engine.low_rank.oats_compressor.OATSCompressor`
- **Summary**: OATS Compressor: Outlier-aware low-rank + structured sparse matrix decomposition.

### `optimal_subspace_projection`
- **Class**: `neuralops.compression_engine.svd.optimal_subspace_projection.OptimalSubspaceProjectionCompressor`
- **Summary**: Optimal Subspace Projection Compressor via activation-aware randomized SVD.

### `proxsparse`
- **Class**: `neuralops.compression_engine.pruning.proxsparse.ProxSparsePruner`
- **Summary**: ProxSparse: proximal gradient sparsity with structured / 2:4 / percentile

### `residual_svd_compressor`
- **Class**: `neuralops.compression_engine.svd.residual_svd_compressor.ResidualSVDCompressor`
- **Summary**: Spectral Residual LoRA (Res-SVD) Compressor.

### `roast_hashed_linear`
- **Class**: `neuralops.compression_engine.experimental.roast_hashed_linear.ROASTCompressor`
- **Summary**: Replace eligible linear and Conv1D layers with ROASTLinear (hardware-aware weight sharing).

### `rotational_slicegpt_int4`
- **Class**: `neuralops.compression_engine.quantization.rotational_slicegpt_int4.RotationalSliceGPTINT4Compressor`
- **Summary**: Rotational-SliceGPT-INT4 hybrid compression engine.

### `rpca_compressor`
- **Class**: `neuralops.compression_engine.low_rank.rpca_compressor.RPCACompressor`
- **Summary**: Base class for all compressors. Not a nn.Module to avoid parameter storage.

### `saes_svd_v2`
- **Class**: `neuralops.compression_engine.svd.saes_svd_v2.SAESSVDCompressorV2`
- **Summary**: Base class for all compressors. Not a nn.Module to avoid parameter storage.

### `salr`
- **Class**: `neuralops.compression_engine.low_rank.salr.SALRCompressor`
- **Summary**: Base class for all compressors. Not a nn.Module to avoid parameter storage.

### `sdrm_compressor`
- **Class**: `neuralops.compression_engine.low_rank.sdrm_compressor.sdrm_compressor_factory`
- **Summary**: SDRM compressor factory returning (compressed_model, stats) when requested.

### `slicegpt`
- **Class**: `neuralops.compression_engine.quantization.slicegpt.PCAActivationAwareSVDCompressor`
- **Summary**: Per-Layer Activation-Aware Low-Rank Compression (PCA-ASVD).

### `slicegpt_pca`
- **Class**: `neuralops.compression_engine.quantization.slicegpt_pca.SliceGPTPCACompressor`
- **Summary**: Production-hardened SliceGPT PCA-based compressor that performs activation-aware

### `slim_compressor`
- **Class**: `neuralops.compression_engine.low_rank.slim_compressor.SLiMCompressor`
- **Summary**: Base class for all compressors. Not a nn.Module to avoid parameter storage.

### `soka_compressor`
- **Class**: `neuralops.compression_engine.low_rank.soka_compressor.soka_factory`
- **Summary**: Factory function for SoKA compressor.

### `sparsegpt`
- **Class**: `neuralops.compression_engine.pruning.sparsegpt.sparsegpt_factory`
- **Summary**: Factory function for SparseGPT compressor.

### `subspace_projection`
- **Class**: `neuralops.compression_engine.svd.subspace_projection.SubspaceProjectionCompressor`
- **Summary**: Subspace Projection Compression Engine.

### `svd_linear`
- **Class**: `neuralops.compression_engine.svd.svd_linear.SVDLinearCompressor`
- **Summary**: Compresses model by replacing Linear/Conv1D layers with SVDLinearLayer.

### `svd_llm_compressor`
- **Class**: `neuralops.compression_engine.low_rank.svd_llm_compressor.SVDLLMCompressor`
- **Summary**: SVD-LLM (ICLR 2025) Compressor with Cholesky activation whitening,

### `swift_svd`
- **Class**: `neuralops.compression_engine.svd.swift_svd.SwiftSVDCompressor`
- **Summary**: Base class for all compressors. Not a nn.Module to avoid parameter storage.

### `swift_svd_act`
- **Class**: `neuralops.compression_engine.svd.swift_svd_act.SwiftSVDActivationCompressor`
- **Summary**: Swift-SVD Activation-Aware Compressor.

### `tomoe_compressor`
- **Class**: `neuralops.compression_engine.low_rank.tomoe_compressor.tomoe_compressor_factory`
- **Summary**: Factory function for ToMoE compressor.

### `tt_compressor`
- **Class**: `neuralops.compression_engine.tensor_decomp.tt_compressor.TTCompressor`
- **Summary**: Replaces Linear/Conv1D with TTLinearLayer when TT succeeds; otherwise

### `uniql_compressor`
- **Class**: `neuralops.compression_engine.low_rank.uniql_compressor.UniQLCompressor`
- **Summary**: UniQL: Joint Truncated SVD + Uniform Quantization compressor.

### `zerokv_neo_compressor`
- **Class**: `neuralops.compression_engine.kv_cache.zerokv_neo_compressor.ZeroKVNeoCompressor`
- **Summary**: ZeroKV-Neo Compressor: L2-norm weight pruning preserving attention sinks and heavy hitters.
