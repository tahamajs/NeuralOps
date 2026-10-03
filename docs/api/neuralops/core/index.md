Module neuralops.core
=====================
Efficient Transformer Toolkit – Core module.
Provides model‑agnostic architecture detection, attention injection,
compression pipeline, and base classes.

Sub-modules
-----------
* neuralops.core.adapters
* neuralops.core.architecture
* neuralops.core.attention_mask
* neuralops.core.attention_utils
* neuralops.core.base
* neuralops.core.common
* neuralops.core.compression_engine
* neuralops.core.config
* neuralops.core.gqa_utils
* neuralops.core.legacy_ensemble
* neuralops.core.mask_utils
* neuralops.core.memory
* neuralops.core.metrics
* neuralops.core.model_utils
* neuralops.core.model_wrapper
* neuralops.core.multi_mode_benchmark
* neuralops.core.pruning_utils
* neuralops.core.registry
* neuralops.core.replace_attention
* neuralops.core.snapshot_restore
* neuralops.core.svd_utils
* neuralops.core.t5_compat
* neuralops.core.triton_kernels
* neuralops.core.utils

Functions
---------

`_apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Applies Rotary Position Embedding (RoPE) to Query and Key tensors.

`_cfg_get(config: Any, key: str, default: Any = None) ‑> Any`
:   Safely retrieves a configuration attribute from a dict, object, or None.
    Handles nested fallbacks without throwing AttributeErrors.

`_detect_decoder(config: Any = None, is_decoder: bool | None = None, old_attn: nn.Module | None = None) ‑> bool`
:   Detects whether an attention layer should operate in causal decoder mode.

`activation_hooks(model: torch.nn.modules.module.Module, target_module_type: type = torch.nn.modules.linear.Linear, collect_input: bool = True, collect_output: bool = False) ‑> Any`
:   Context manager that collects activations for all modules of a given type.

`apply_attention_mask(scores: torch.Tensor, attention_mask: torch.Tensor | None, mask_value: float = -10000.0) ‑> torch.Tensor`
:   Apply attention mask to attention logits in a numerically robust and universal manner.
    
    Args:
        scores (torch.Tensor): Attention logits of shape `[Batch, Heads, SeqLen_Q, SeqLen_K]`.
        attention_mask (Optional[torch.Tensor]): Mask tensor in arbitrary standard format:
            - None: No-op, returns `scores` directly.
            - 2D `[B, N_k]`: Binary (1=keep, 0=mask) or boolean.
            - 3D `[B, 1, N_k]` or `[B, N_q, N_k]`.
            - 4D `[B, 1, 1, N_k]` or `[B, 1, N_q, N_k]`: Pre-computed additive float mask or binary.
        mask_value (float): Large negative value to assign to masked positions.
    
    Returns:
        torch.Tensor: Masked attention logits of same shape and dtype as `scores`.

`available_attention_types() ‑> list[str]`
:   Return all registered attention mechanism identifiers.

`cleanup_gpu_memory(force: bool = True) ‑> None`
:   Perform exhaustive GPU / MPS / CPU memory deallocation.
    Frees all cached blocks, calls cyclic garbage collector, and flushes allocators.

`compress_model(model: torch.nn.modules.module.Module, config: dict[str, typing.Any], tokenizer=None) ‑> torch.nn.modules.module.Module`
:   Convenience function to compress a model with a configuration dict.

`copy_attention_weights(src_module: torch.nn.modules.module.Module, dst_module: torch.nn.modules.module.Module, source_map: dict[str, list[str]] | None = None) ‑> None`
:   Copy pretrained attention weights from src to dst.
    Tries multiple possible attribute names for each projection.

`count_params(model: torch.nn.modules.module.Module, trainable_only: bool = False) ‑> int`
:   Count parameters.

`create_band_mask(seq_len: int, window_size: int, device: torch.device | None = None) ‑> torch.Tensor`
:   Create a sliding window band attention mask [SeqLen, SeqLen] where positions |i - j| <= window_size are True.

`create_causal_mask(seq_len: int, device: torch.device | None = None) ‑> torch.Tensor`
:   Create a lower-triangular causal attention mask [SeqLen, SeqLen] where True = attend, False = mask.

`detect_architecture(model: torch.nn.modules.module.Module) ‑> str`
:   Return canonical architecture name.

`format_number(num: int) ‑> str`
:   Format number.

`get_adapter(arch: str) ‑> Any`
:   Return universal adapter for any architecture.

`get_attention_modules(model: torch.nn.modules.module.Module) ‑> list[tuple[str, torch.nn.modules.module.Module]]`
:   Return (name, module) for all leaf attention sub-modules across any transformer architecture.
    
    Applies strict path-prefix deduplication so parent composite containers are never
    returned alongside their inner leaf attention blocks, preventing accidental deletion
    of LayerNorm, output projections, and residual streams.

`get_calibration_tensors(model: nn.Module, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: torch.device | None = None, sample_size: int = 8, seq_len: int = 32, hidden_dim: int | None = None) ‑> Any`
:   Generate or extract calibration inputs for a model.
    Handles tokenizers, DataLoaders, and synthetic tensor fallbacks.

`get_classifier_head(model: torch.nn.modules.module.Module) ‑> torch.nn.modules.module.Module | None`
:   Return the final classification layer.

`get_device(device_str: str = 'auto') ‑> torch.device`
:   Return the device.

`get_embedding_module(model: torch.nn.modules.module.Module) ‑> torch.nn.modules.module.Module | None`
:   Return the input word embedding module.

`get_ffn_modules(model: torch.nn.modules.module.Module) ‑> list[tuple[str, torch.nn.modules.module.Module]]`
:   Return (name, module) for all feed‑forward blocks.

`get_gpu_memory_stats() ‑> dict[str, float]`
:   Get detailed memory statistics in Megabytes (MB).

`get_parent_and_child_name(model: torch.nn.modules.module.Module, full_name: str) ‑> Any`
:   Return (parent_module, child_attribute_name), supporting ModuleList and ModuleDict.

`get_token_valid_mask(attention_mask: torch.Tensor | None, batch_size: int, seq_len: int, device: torch.device) ‑> torch.Tensor`
:   Extract a 1D boolean validity mask [Batch, SeqLen] where True = valid token, False = pad.
    
    Args:
        attention_mask (Optional[torch.Tensor]): Mask in any format.
        batch_size (int): Batch size B.
        seq_len (int): Sequence length N.
        device (torch.device): Device to place the mask on.
    
    Returns:
        torch.Tensor: Boolean tensor of shape `[Batch, SeqLen]`.

`get_transformer_layers(model: torch.nn.modules.module.Module) ‑> list[torch.nn.modules.module.Module]`
:   Return a flat list of transformer blocks.

`human_readable_time(seconds: float) ‑> str`
:   Human readable time.

`inject_attention(model: nn.Module, attention_type: str, rank: int | None = None, copy_weights: bool = True, **kwargs: Any) ‑> torch.nn.modules.module.Module`
:   Universal attention layer replacement engine for HuggingFace Transformers.

`list_all_components() ‑> dict`
:   Convenience function that returns all global registry keys.

`repeat_kv(hidden_states: torch.Tensor, n_rep: int) ‑> torch.Tensor`
:   Repeats Key/Value heads along the head dimension for Grouped-Query Attention (GQA).
    hidden_states: (batch, num_key_value_heads, seqlen, head_dim)
    Returns: (batch, num_attention_heads, seqlen, head_dim)

`replace_module(model: torch.nn.modules.module.Module, full_name: str, new_module: torch.nn.modules.module.Module) ‑> None`
:   Replace a sub-module identified by its dotted name, supporting ModuleList and ModuleDict.

`run_calibration_forward(model: torch.nn.modules.module.Module, tokenizer, texts: list[str], device: torch.device | None = None, max_length: int = 128) ‑> torch.nn.modules.module.Module`
:   Run a single forward pass on calibration texts; return model (in eval).

`selective_scan(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor | None = None, z: torch.Tensor | None = None, delta_bias: torch.Tensor | None = None, delta_softplus: bool = True, chunk_size: int = 32) ‑> torch.Tensor`
:   Dispatch to CUDA kernel if available, else stable chunked fallback.

`selective_scan_chunked(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor | None = None, z: torch.Tensor | None = None, delta_bias: torch.Tensor | None = None, delta_softplus: bool = True, chunk_size: int = 32) ‑> torch.Tensor`
:   Selective scan via numerically stable chunked recurrence.
    
    For chunk [s, e) of length C:
      dA[t] = Δ[t] · A <= 0
      dA_cum[t] = cumsum(dA)
      decay[j, i] = exp(dA_cum[j] - dA_cum[i]) for j >= i, else 0
      h_intra[j] = sum_i decay[j, i] · dB_u[i]
      h_carry = exp(dA_cum) · h_prev
      h_chunk = h_intra + h_carry
      y_chunk = sum_N(h_chunk · C)

`set_seed(seed: int, deterministic: bool = False) ‑> None`
:   Set the seed.

`softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) ‑> torch.Tensor`
:   Rectified non-negative sparse activation function with numerical stabilization.

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

`CompressionPipeline(model: torch.nn.modules.module.Module, config: dict[str, typing.Any], tokenizer=None)`
:   Apply a sequence of compression steps to a model.

    ### Methods

    `run(self) ‑> torch.nn.modules.module.Module`
    :   Execute the run operation and return results.

`FactorizedLinear(in_features: int, out_features: int, rank: int = 128, lora_rank: int = 32, bias: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Decomposed linear layer: W ≈ W2 @ W1 with optional LoRA residual branch.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `bias: torch.Tensor | None`
    :

    `weight: torch.Tensor`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `freeze_base_weights(self) ‑> None`
    :

    `reset_parameters(self) ‑> None`
    :

`GPUMemoryGuard()`
:   Context manager for zero-leak memory safety during execution.

`LoRALinear(base_or_in_features: nn.Module | int, out_features: int | None = None, bias: bool = True, lora_rank: int = 16, r: int | None = None, lora_alpha: float = 32.0, lora_dropout: float = 0.0, freeze_base: bool = False, merge_weights: bool = False, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Universal LoRA linear layer:
    Supports both wrapper mode: LoRALinear(base_layer, r=16, ...)
    and standalone mode: LoRALinear(in_features, out_features, bias=True, lora_rank=16, ...)
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `freeze_base_weights(self) ‑> None`
    :

    `merge(self) ‑> None`
    :

    `reset_parameters(self) ‑> None`
    :

    `unfreeze_base_weights(self) ‑> None`
    :

    `unmerge(self) ‑> None`
    :

`MultiModeBenchmarkRunner(model_name_or_path: str = 'distilbert-base-uncased', device: str | None = 'auto', dtype: str = 'float16', batch_size: int = 8, seq_len: int = 128, num_samples: int = 128, warmup_steps: int = 5, benchmark_steps: int = 15, output_dir: str = './benchmark_results', allow_synthetic: bool = False)`
:   Unified Multi-Mode Evaluation Runner.
    
    Executes systematic benchmarks across 12 distinct inference modes, measuring:
    - Perplexity (PPL)
    - Peak VRAM / Memory footprint
    - Latency (ms/token, ms/step)
    - Throughput (tokens/sec)
    - Compression Ratio & Memory Savings (%)

    ### Static methods

    `list_supported_modes() ‑> list[str]`
    :   Return all supported benchmark execution modes.

    ### Methods

    `benchmark_attention_module(self, attention_type: str, hidden_size: int = 64, num_heads: int = 4, seq_len: int = 64, batch_size: int = 2, num_iters: int = 10, **kwargs: Any) ‑> dict[str, typing.Any]`
    :   Directly benchmark a registered attention module forward pass.

    `export_csv(self, filepath: str | pathlib.Path | None = None) ‑> str`
    :   Export results to CSV file.

    `export_json(self, filepath: str | pathlib.Path | None = None) ‑> str`
    :   Export all benchmark outcomes to structured JSON.

    `print_summary_table(self)`
    :   Print an aligned ASCII summary table of the benchmark results.

    `run_all_modes(self, modes: list[str] | None = None, datasets: list[str] | None = None, rank: int = 32) ‑> list[neuralops.core.multi_mode_benchmark.ModeBenchmarkResult]`
    :   Execute evaluation sweep across all requested modes and datasets.

    `run_single_mode(self, mode: str, dataset_name: str = 'sst2', rank: int = 32) ‑> neuralops.core.multi_mode_benchmark.ModeBenchmarkResult`
    :   Run benchmark for a single execution mode.

    `to_dataframe(self) ‑> Any`
    :   Convert results to a pandas DataFrame.

`UnifiedTransformerConfig(attention_type: str = None, attention_rank: int | None = None, attention_kwargs: dict[str, typing.Any] | None = None, **kwargs)`
:   Lightweight container for the custom attention configuration.

    ### Static methods

    `from_dict(d: dict[str, typing.Any]) ‑> neuralops.core.model_wrapper.UnifiedTransformerConfig`
    :   From dict.

    ### Methods

    `to_dict(self) ‑> dict[str, typing.Any]`
    :   Serialise to a JSON‑compatible dict.

`UnifiedTransformerModel(model: torch.nn.modules.module.Module, custom_config: neuralops.core.model_wrapper.UnifiedTransformerConfig | None = None)`
:   A wrapper around a HuggingFace model that supports custom attention
    and can be saved/loaded via the HuggingFace Hub.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module
    * huggingface_hub.hub_mixin.PyTorchModelHubMixin
    * huggingface_hub.hub_mixin.ModelHubMixin

    ### Static methods

    `from_pretrained(model_id: str, task: str = 'sequence_classification', num_labels: int | None = None, **kwargs) ‑> neuralops.core.model_wrapper.UnifiedTransformerModel`
    :   Create a new UnifiedTransformerModel from a pretrained identifier.
        Automatically applies the custom attention if the model was saved with one.
        This is the main entry point for users.

    ### Methods

    `forward(self, *args, **kwargs) ‑> Any`
    :   Forward pass — compute module output from inputs.

    `freeze_layers(self, num_layers_to_freeze: int = 0) ‑> Any`
    :   Freeze layers.

    `get_parameter_count(self) ‑> dict[str, int]`
    :   Return the parameter count.

    `replace_attention(self, attention_type: str, rank: int | None = None, copy_weights: bool = True, **kwargs) ‑> Any`
    :   Inject a custom attention variant into all layers.
        This method also records the choice so it can be saved.

    `save_pretrained(self, save_directory: str) ‑> None`
    :   Convenience wrapper – delegates to the mixin.

    `unfreeze_all(self) ‑> Any`
    :   Unfreeze all.