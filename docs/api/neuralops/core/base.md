Module neuralops.core.base
==========================
Base classes for attention modules and compressors.

Classes
-------

`BaseAttention(config: Any | None = None, rank: int | None = None, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Descendants

    * neuralops.attentions.adaptive_span_attention.AdaptiveSpanAttention
    * neuralops.attentions.adaptive_sparse_attention.AdaptiveSparseAttention
    * neuralops.attentions.adaptive_svd_attention.AdaptiveSVDAttention
    * neuralops.attentions.adasplash_attention.AdaSplashAttention
    * neuralops.attentions.aegis_hybrid.AegisHybridAttention
    * neuralops.attentions.aegis_spectron_fp8.AegisSpectronFP8Attention
    * neuralops.attentions.aegis_v2.AegisAttentionV2
    * neuralops.attentions.aegis_v5.AegisAttention
    * neuralops.attentions.aegis_v6.AegisAttentionV3
    * neuralops.attentions.asvd_attention.ASVDAttention
    * neuralops.attentions.bam_umoe.BAMUMoEBlock
    * neuralops.attentions.block_transformer_hiba.BlockTransformerHIBA
    * neuralops.attentions.chela_slay.ChelaSlayAttention
    * neuralops.attentions.cluster_attention.ClusterAttention
    * neuralops.attentions.compressed_attention.CompressedAttentionLayer
    * neuralops.attentions.compressed_mha_attention.CompressedDistilBertAttention
    * neuralops.attentions.cosformer.CosformerAttention
    * neuralops.attentions.decay_linear_attention.DecayLinearAttention
    * neuralops.attentions.eat.EATBlock
    * neuralops.attentions.expert_choice_moe.ExpertChoiceMoEAttention
    * neuralops.attentions.factorised_matrix_state.FactorisedMatrixState
    * neuralops.attentions.flash_attention3.FlashAttention3
    * neuralops.attentions.flash_attention3_fp8.FlashAttention3FP8
    * neuralops.attentions.full_attention.FullAttention
    * neuralops.attentions.fused_svd_kernel.FusedSVDKernelAttention
    * neuralops.attentions.fusion_attn.FUSIONATTN
    * neuralops.attentions.galerkin_attention.GalerkinAttention
    * neuralops.attentions.gated_delta_net.GatedDeltaNet
    * neuralops.attentions.gated_deltanet_block.GatedDeltaNetBlock
    * neuralops.attentions.gated_linear_attention.GatedLinearAttention
    * neuralops.attentions.goldfinch_ahn.GoldFinchAHNBlock
    * neuralops.attentions.gqa_attention.GroupedQueryAttention
    * neuralops.attentions.granite_hybrid.GraniteHybridBlock
    * neuralops.attentions.griffin_ssd.GriffinSSDBlock
    * neuralops.attentions.halo_regla.HALOReGLAAttention
    * neuralops.attentions.hamming_attention.HammingAttention
    * neuralops.attentions.hgrn2_gated_deltanet.HGRN2GatedDeltaNet
    * neuralops.attentions.hgrn2_layer.HGRN2Layer
    * neuralops.attentions.hgrn2_outer.HGRN2OuterBlock
    * neuralops.attentions.hla_attention.HLAAttention
    * neuralops.attentions.hybrid_attention.HybridAttention
    * neuralops.attentions.hymba_transx.HymbaTransXBlock
    * neuralops.attentions.infini_attention.InfiniAttention
    * neuralops.attentions.jamba.Jamba2Block
    * neuralops.attentions.jamba_info.JambaInfoBlock
    * neuralops.attentions.kda_attention.KDAAttention
    * neuralops.attentions.leka_v1.LEKAV1Attention
    * neuralops.attentions.leka_v2.LEKAv2Attention
    * neuralops.attentions.leka_v3.LEKAV3Attention
    * neuralops.attentions.linformer_attention.LinformerAttention
    * neuralops.attentions.lola_attention.LoLAAttention
    * neuralops.attentions.lora_attention.LoRAAttention
    * neuralops.attentions.lsh_attention.LSHAttention
    * neuralops.attentions.mamba_svd_flash.MambaSVDFlash
    * neuralops.attentions.minimal_nova.MinimalNOVAAttention
    * neuralops.attentions.minimal_svd.MinimalSVDAttention
    * neuralops.attentions.mixattention_moska.MixAttentionMoSKA
    * neuralops.attentions.mla_attention.MLAAttention
    * neuralops.attentions.moba_attention.MoBAAttention
    * neuralops.attentions.nasc_attention.NASCAttention
    * neuralops.attentions.nexus.NEXUSAttention
    * neuralops.attentions.nha_attention.NativeHybridAttention
    * neuralops.attentions.nsa_attention.NativeSparseAttention
    * neuralops.attentions.nystrom_attention.NystromAttention
    * neuralops.attentions.optimized_hybrid_attention.OptimizedHybridAttention
    * neuralops.attentions.orion.ORIONAttention
    * neuralops.attentions.padre_attention.PADReAttention
    * neuralops.attentions.performer_attention.PerformerAttention
    * neuralops.attentions.phantom.PHANTOMAttention
    * neuralops.attentions.primal_attention.PrimalAttention
    * neuralops.attentions.progressive_token_pruning.ProgressiveTokenPruning
    * neuralops.attentions.qcqa_attention.QCQAAttention
    * neuralops.attentions.rala_attention.RALALinearAttention
    * neuralops.attentions.reap_moe.REAPMoEAttention
    * neuralops.attentions.regla_attention.ReGLAAttention
    * neuralops.attentions.replace.ReplaceAttention
    * neuralops.attentions.res_svd.ResSVDAttention
    * neuralops.attentions.routing_attention.RoutingAttention
    * neuralops.attentions.rpc_attention.RPCAttention
    * neuralops.attentions.salt_attention.SALTAttention
    * neuralops.attentions.samba_moe.SambaMoEBlock
    * neuralops.attentions.selective_ssm.SelectiveSSM
    * neuralops.attentions.selective_ssm.SelectiveSSM
    * neuralops.attentions.soa_attn.SOAAttention
    * neuralops.attentions.span_expanded_attention.SpanExpandedAttention
    * neuralops.attentions.spectra_v2.SpectraV2Attention
    * neuralops.attentions.spectraformer_attention.SpectraformerAttention
    * neuralops.attentions.spectral_mamba.SpectralMambaAttention
    * neuralops.attentions.spectron.SpectronAttention
    * neuralops.attentions.squeezing_heads_distill.SqueezingHeadsDistiller
    * neuralops.attentions.svd_attention.SVDLinearAttention
    * neuralops.attentions.svd_compressed.SVDCompressedAttention
    * neuralops.attentions.svd_linear_attention.SVDLinearAttention
    * neuralops.attentions.svd_llm_v2_attention.SVDLLMV2Attention
    * neuralops.attentions.svd_mamba.SVDMamba
    * neuralops.attentions.svd_projection_attention.SVDProjectionAttention
    * neuralops.attentions.template_attention.YourCustomAttention
    * neuralops.attentions.tlinformer_attention.TLinFormerAttention
    * neuralops.attentions.token_compressed_attention.TokenCompressedAttention
    * neuralops.attentions.tpa_attention.TPAttention
    * neuralops.attentions.triton_cosformer.TritonCosformerAttention
    * neuralops.attentions.triton_svd_linear_attention.TritonSVDLinearAttention
    * neuralops.attentions.unified_rope.UnifiedRoPE
    * neuralops.attentions.upi_stuffed_mamba.SelectiveMamba
    * neuralops.attentions.vig_attention.ViGAttention
    * neuralops.attentions.vspec.VSpecAttention
    * neuralops.attentions.wgqa_attention.WeightedGroupedQueryAttention
    * neuralops.attentions.xattention.XAttention
    * neuralops.attentions.zeros_attention.ZeroSAttention
    * neuralops.compression_engine.nli_softmax.NLISoftmaxAttention
    * neuralops.core.replace_attention.ReplaceAttention

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Universal fallback: copy pretrained weights from old attention layer.

    `extra_repr(self) ‑> str`
    :   Extra repr.

    `forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, **kwargs) ‑> tuple[typing.Any, Any | None]`
    :   Must return (output, attention_weights_or_None).

`BaseCompressor(model: torch.nn.modules.module.Module, name: str = 'base', **kwargs)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Descendants

    * neuralops.compression_engine.a3_compressor.A3Compressor
    * neuralops.compression_engine.aa_svd_compressor.AASVDCompressor
    * neuralops.compression_engine.aatc_xkv_compressor.AATC_XKVCompressor
    * neuralops.compression_engine.adaptive_ranks.AdaptiveRanksCompressor
    * neuralops.compression_engine.adasvd_compressor.AdaSVDCompressor
    * neuralops.compression_engine.admm_compressor.ADMMCompressor
    * neuralops.compression_engine.admm_pruner.ADMMPruner
    * neuralops.compression_engine.armor_pruner.ARMORPruner
    * neuralops.compression_engine.ast_pruner.ASTPruner
    * neuralops.compression_engine.astro_quantizer.AstroQuantizer
    * neuralops.compression_engine.bonsai_pruner.BonsaiPruner
    * neuralops.compression_engine.cachenotes_compressor.CacheNotesCompressor
    * neuralops.compression_engine.come_compressor.CoMeCompressor
    * neuralops.compression_engine.compot_compressor.COMPOTCompressor
    * neuralops.compression_engine.coopq_quantizer.CoopQQuantizer
    * neuralops.compression_engine.cr_compressor.CRCompressor
    * neuralops.compression_engine.cur_compressor.CURCompressor
    * neuralops.compression_engine.cur_moe_compressor.CURMoECompressor
    * neuralops.compression_engine.delta_kv.DeltaKVCompressor
    * neuralops.compression_engine.direct_svd.DirectSVDCompressor
    * neuralops.compression_engine.dobi_svd.DobiSVDCompressor
    * neuralops.compression_engine.drone_compressor.DRONECompressor
    * neuralops.compression_engine.dynamic_rank.DynamicEntropySVDCompressor
    * neuralops.compression_engine.enlighten_compressor.ENLightenCompressor
    * neuralops.compression_engine.entropy_budget_allocation.EntropyBudgetCompressor
    * neuralops.compression_engine.erc_svd.ERCCompressor
    * neuralops.compression_engine.faq_quantizer.FAQQuantizer
    * neuralops.compression_engine.flap.FLAPCompressor
    * neuralops.compression_engine.flat_llm.FLATLLMCompressor
    * neuralops.compression_engine.frandi_compressor.FRanDICompressor
    * neuralops.compression_engine.fwsvd_compressor.FWSVDCompressor
    * neuralops.compression_engine.gfws_d_compressor.GFWSCompressor
    * neuralops.compression_engine.grecodec_compressor.GreCoDeCCompressor
    * neuralops.compression_engine.hashed_net_compressor.HashedNetCompressor
    * neuralops.compression_engine.huffman_coder.HuffmanCompressor
    * neuralops.compression_engine.hybrid_svd_layer.HybridSVDCompressor
    * neuralops.compression_engine.hybridkv_compressor.HybridKVCompressor
    * neuralops.compression_engine.kmeans_quantize.KMeansCompressor
    * neuralops.compression_engine.kronecker_compressor.KroneckerCompressor
    * neuralops.compression_engine.kronecker_tt_lora.KroneckerTTLoRACompressor
    * neuralops.compression_engine.kvtc_compressor.KVTCCompressor
    * neuralops.compression_engine.liftr.LIFTRCompressor
    * neuralops.compression_engine.lillama_compressor.LiLLaMADistiller
    * neuralops.compression_engine.lsc_compressor.LSCCompressor
    * neuralops.compression_engine.mamba_shedder.MambaShedderCompressor
    * neuralops.compression_engine.mgie_svd.MGIECompressor
    * neuralops.compression_engine.mgie_svd_compressor.MGIESVDBudgetCompressor
    * neuralops.compression_engine.mlorq_compressor.MLoRQCompressor
    * neuralops.compression_engine.modegpt_compressor.MoDeGPTCompressor
    * neuralops.compression_engine.nli_softmax.NLISoftmaxCompressor
    * neuralops.compression_engine.nmf_compressor.NMFCompressor
    * neuralops.compression_engine.oats_compressor.OATSCompressor
    * neuralops.compression_engine.optimal_subspace_projection.OptimalSubspaceProjectionCompressor
    * neuralops.compression_engine.proxsparse.ProxSparsePruner
    * neuralops.compression_engine.replaceme_compressor.ReplaceMeCompressor
    * neuralops.compression_engine.res_svd.ResSVDCompressor
    * neuralops.compression_engine.roast_hashed_linear.ROASTCompressor
    * neuralops.compression_engine.rotational_slicegpt_int4.RotationalSliceGPTINT4Compressor
    * neuralops.compression_engine.rpca_compressor.RPCACompressor
    * neuralops.compression_engine.saes_svd_v2.SAESSVDCompressorV2
    * neuralops.compression_engine.salr.SALRCompressor
    * neuralops.compression_engine.sdrm_compressor.SDRMCompressor
    * neuralops.compression_engine.slicegpt.SliceGPTCompressor
    * neuralops.compression_engine.slicegpt_pca.SliceGPTPCACompressor
    * neuralops.compression_engine.slim_compressor.SLiMCompressor
    * neuralops.compression_engine.soka_compressor.LowRankCompressor
    * neuralops.compression_engine.sparsegpt.SparseGPTCompressor
    * neuralops.compression_engine.subspace_projection.SubspaceProjectionCompressor
    * neuralops.compression_engine.svd_linear.SVDLinearCompressor
    * neuralops.compression_engine.svd_llm_compressor.SVDLLMCompressor
    * neuralops.compression_engine.swift_svd.SwiftSVDCompressor
    * neuralops.compression_engine.swift_svd_act.SwiftSVDActivationCompressor
    * neuralops.compression_engine.tomoe_compressor.ToMoECompressor
    * neuralops.compression_engine.tt_compressor.TTCompressor
    * neuralops.compression_engine.uniql_compressor.UniQLCompressor
    * neuralops.compression_engine.zerokv_neo_compressor.ZeroKVNeoCompressor

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Return (compressed_model, statistics).

    `get_compression_stats(self) ‑> dict[str, typing.Any]`
    :   Return the compression stats.

    `summary(self) ‑> str`
    :   Summary.