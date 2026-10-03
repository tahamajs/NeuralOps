Module neuralops
================
NeuralOps Core ML Framework & Efficient Transformer Toolkit.

A unified, modular, and high-performance toolkit for custom attention mechanisms,
model compression (SVD, quantization, pruning, distillation), distributed training,
hyperparameter optimization, federated learning, data loading, benchmarking,
and interactive visualization.

Sub-modules
-----------
* neuralops.analysis
* neuralops.attentions
* neuralops.benchmarks
* neuralops.compression_engine
* neuralops.config
* neuralops.core
* neuralops.data
* neuralops.evaluation
* neuralops.experiment_logging
* neuralops.experiment_utils
* neuralops.main
* neuralops.pipeline
* neuralops.results_saver
* neuralops.runners
* neuralops.tests
* neuralops.tracking
* neuralops.trainer
* neuralops.utils

Functions
---------

`aggregate_results(results_list) ‑> Any`
:   Aggregate results.

`compress_model(model: torch.nn.modules.module.Module, config: dict[str, typing.Any], tokenizer=None) ‑> torch.nn.modules.module.Module`
:   Convenience function to compress a model with a configuration dict.

`create_dataloader(dataset, tokenizer, config: neuralops.data.base.DataConfig, **kwargs) ‑> torch.utils.data.dataloader.DataLoader`
:   Create a DataLoader from a HuggingFace Dataset, applying preprocessing if not already tokenized.
    Uses ProductionDataCollator to guarantee rock-solid batch dimensions and GPU Tensor Core alignment.

`detect_architecture(model: torch.nn.modules.module.Module) ‑> str`
:   Return canonical architecture name.

`fms_scan_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor) ‑> torch.Tensor`
:   Triton-accelerated FMS scan function. head_dim must be <= 64.

`generate_synthetic_data(output_path: str, num_samples: int = 10000, domain: str = 'movies', seed: int = 42) ‑> pandas.DataFrame`
:   Generate a single-domain synthetic classification dataset.

`get_calibration_dataloader(task_name: str = 'sst2', tokenizer: Any = None, batch_size: int = 8, num_samples: int = 128, seq_len: int = 128, split: str = 'train') ‑> Any`
:   Builds a calibration DataLoader from local disk-cached GLUE data.
    Used by data-aware SVD compression algorithms (ASVD, A3, FRanDI, DynamicRank,
    SwiftSVD, SliceGPT, etc.) to collect activation covariance and Fisher information.

`get_task_metrics(task_name: str) ‑> Any`
:   Return a compute_metrics function for a GLUE task.

`gla_scan_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lambdas: torch.Tensor) ‑> torch.Tensor`
:   Triton-accelerated GLA scan function. head_dim must be <= 64.

`inject_attention(model: nn.Module, attention_type: str, rank: int | None = None, copy_weights: bool = True, **kwargs: Any) ‑> torch.nn.modules.module.Module`
:   Universal attention layer replacement engine for HuggingFace Transformers.

`load_benchmark_dataset(dataset_name: str, tokenizer: Any = None, seq_len: int = 2048, batch_size: int = 8, split: str | None = None, max_samples: int = 1000, streaming: bool = True, cache_dir: str | None = None, shuffle: bool = False, num_workers: int = 0, use_synthetic: bool = False, seed: int = 42, return_dataloader: bool = True, **kwargs: Any) ‑> torch.utils.data.dataloader.DataLoader | list[dict[str, typing.Any]]`
:   Unified loader for 12+ NLP / LLM benchmark datasets.
    
    Supports:
      - wikitext2 / wikitext103
      - c4 / c4_en
      - ptb
      - pg19
      - gsm8k
      - humaneval
      - mmlu
      - pile / pile_val
      - arc (ARC-Easy & ARC-Challenge)
      - hellaswag
      - lambada
      - glue (sst2, mrpc, qqp, mnli, qnli, rte, wnli, cola, stsb)
    
    Args:
        dataset_name: Benchmark name or alias (e.g. 'wikitext2', 'c4', 'gsm8k', 'arc', 'glue/sst2').
        tokenizer: HuggingFace tokenizer instance (optional).
        seq_len: Maximum sequence length for tokenization.
        batch_size: DataLoader batch size.
        split: Dataset split ('train', 'validation', 'test'). Defaults to benchmark standard.
        max_samples: Maximum number of samples to load/stream.
        streaming: Whether to stream from HuggingFace Hub to avoid large downloads.
        cache_dir: Local directory to cache datasets.
        shuffle: Whether to shuffle samples.
        num_workers: Number of workers for DataLoader.
        use_synthetic: Force synthetic data generation for offline testing.
        seed: Random seed for reproducibility.
        return_dataloader: If True, returns PyTorch DataLoader; otherwise returns list of dicts.
        **kwargs: Additional arguments passed to `datasets.load_dataset`.
    
    Returns:
        torch.utils.data.DataLoader or List[Dict[str, Any]].

`load_dataset(config: neuralops.data.dataset_loader.ExperimentConfig) ‑> datasets.dataset_dict.DatasetDict`
:   Load a dataset based on ExperimentConfig.
    Returns DatasetDict with at least 'train' and optionally 'validation'/'test' splits.

`load_glue_local(task_name: str, base_path: str = None) ‑> datasets.dataset_dict.DatasetDict`
:   Load a GLUE task from local Parquet files, falling back to HuggingFace Datasets hub.
    Compatible with local, Kaggle, Colab, and remote GPU/HPC environments.

`preprocess_function(examples, tokenizer, config: neuralops.data.base.DataConfig) ‑> Any`
:   Generic preprocessing function handling all TaskTypes.
    Returns a dictionary with 'input_ids', 'attention_mask', and appropriate labels.

`run_universal_generation(output_dir: str | Path, total_samples_per_domain: int = 5000, domains: dict[str, Any] | None = None, seed: int = 42) ‑> pandas.DataFrame`
:   Generate synthetic dataset across multiple domains and combine them.

`selective_scan_fn(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor | None = None, delta_softplus: bool = True) ‑> torch.Tensor`
:   Triton selective scan functional interface.
    Shapes:
        u:     (B, L, D)
        delta: (B, L, D)
        A:     (D, N)
        B:     (B, L, N)
        C:     (B, L, N)
        D:     (D,) optional

`triton_dispatch(kernel_name: str, *args: Any, **kwargs: Any) ‑> Any`
:   Unified router that executes high-performance Triton kernels when supported,
    or transparently falls back to optimized PyTorch implementations.

Classes
-------

`ABTestResult(metric: str, mean_a: float, mean_b: float, difference: float, p_value: float, significant: bool, better_model: str, confidence_interval: tuple[float, float])`
:   ABTestResult(metric: str, mean_a: float, mean_b: float, difference: float, p_value: float, significant: bool, better_model: str, confidence_interval: tuple[float, float])

    ### Instance variables

    `better_model: str`
    :   The type of the None singleton.

    `confidence_interval: tuple[float, float]`
    :   The type of the None singleton.

    `difference: float`
    :   The type of the None singleton.

    `mean_a: float`
    :   The type of the None singleton.

    `mean_b: float`
    :   The type of the None singleton.

    `metric: str`
    :   The type of the None singleton.

    `p_value: float`
    :   The type of the None singleton.

    `significant: bool`
    :   The type of the None singleton.

`ABTester(alpha: float = 0.05)`
:   

    ### Methods

    `compare(self, results_a: dict[str, list[float]], results_b: dict[str, list[float]]) ‑> list[neuralops.evaluation.ab_testing.ABTestResult]`
    :   Compare.

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

`BaseTrainer()`
:   Abstract base class for all trainers.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * neuralops.trainer.custom_loop_trainer.CustomLoopTrainer
    * neuralops.trainer.hf_trainer.HuggingFaceTrainer
    * neuralops.trainer.rl_trainer.RLTrainer
    * neuralops.trainer.unsloth_trainer.UnslothTrainer

    ### Methods

    `evaluate(self) ‑> dict[str, typing.Any]`
    :   Run evaluate and return results.

    `train(self) ‑> dict[str, typing.Any]`
    :   Fit the model on the provided dataset.

`BenchmarkDatasetInfo(name: str, aliases: list[str], category: str, default_hf_path: str, default_subset: str | None = None, default_split: str = 'validation', text_formatter: Callable[[dict[str, Any]], dict[str, Any]] | None = None, synthetic_generator: Callable[[int, int], list[dict[str, Any]]] | None = None, description: str = '')`
:   Metadata describing a benchmark dataset and its default loading strategy.

    ### Instance variables

    `aliases: list[str]`
    :   The type of the None singleton.

    `category: str`
    :   The type of the None singleton.

    `default_hf_path: str`
    :   The type of the None singleton.

    `default_split: str`
    :   The type of the None singleton.

    `default_subset: str | None`
    :   The type of the None singleton.

    `description: str`
    :   The type of the None singleton.

    `name: str`
    :   The type of the None singleton.

    `synthetic_generator: Callable[[int, int], list[dict[str, typing.Any]]] | None`
    :   The type of the None singleton.

    `text_formatter: Callable[[dict[str, typing.Any]], dict[str, typing.Any]] | None`
    :   The type of the None singleton.

`BenchmarkDatasetRegistry()`
:   Registry maintaining metadata and formatters for NLP/LLM benchmarks.

    ### Static methods

    `get(name: str) ‑> neuralops.data.benchmark_datasets.BenchmarkDatasetInfo | None`
    :   Retrieve benchmark dataset info by name or alias.

    `is_registered(name: str) ‑> bool`
    :   Check if a benchmark dataset name is registered.

    `list_benchmarks() ‑> list[str]`
    :   List canonical names of all registered benchmarks.

    `list_datasets() ‑> list[str]`
    :   Alias for list_benchmarks.

    `register(info: BenchmarkDatasetInfo) ‑> None`
    :   Register a new benchmark dataset info.

`CompressionPipeline(model: torch.nn.modules.module.Module, config: dict[str, typing.Any], tokenizer=None)`
:   Apply a sequence of compression steps to a model.

    ### Methods

    `run(self) ‑> torch.nn.modules.module.Module`
    :   Execute the run operation and return results.

`CustomLoopTrainer(model: nn.Module, train_dataset: Any, eval_dataset: Any, config: dict[str, Any], results_manager: Any | None = None, device: torch.device | None = None, num_labels: int = 2, is_multichoice: bool = False, compute_metrics_fn: Callable | None = None, **kwargs: Any)`
:   Custom loop trainer optimized for maximum GPU compute throughput, mixed-precision training,
    and distributed data parallel (DDP) gradient accumulation without redundant all-reduce overhead.

    ### Ancestors (in MRO)

    * neuralops.trainer.base_trainer.BaseTrainer
    * abc.ABC

    ### Methods

    `train(self) ‑> dict[str, typing.Any]`
    :   Fit the model on the provided dataset with full DDP and accumulation safety.

    `train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler | None, gradient_accumulation_steps: int = 1, max_grad_norm: float = 1.0, amp_dtype: torch.dtype = torch.float32, use_amp: bool = False, pbar: Any | None = None) ‑> float`
    :   Execute a single training epoch with DDP gradient accumulation synchronization optimization.
        
        # FIX: Non-final micro-batches are wrapped in model.no_sync() or contextlib.nullcontext()
        # to eliminate redundant all-reduce synchronization across distributed workers during gradient accumulation.

`DataConfig(task_type: neuralops.data.base.TaskType = TaskType.SINGLE_CLASSIFICATION, task_name: str | None = None, max_length: int = 128, padding: Any = 'max_length', truncation: bool = True, return_tensors: str = 'pt', text_column: str | None = None, text_column_1: str | None = None, text_column_2: str | None = None, label_column: str | None = None, batch_size: int = 16, shuffle: bool = True, num_workers: int = 0, mlm_probability: float = 0.15, question_column: str | None = None, context_column: str | None = None, answer_column: str | None = None, tokens_column: str | None = None, tags_column: str | None = None, custom_preprocess: <class 'collections.abc.Callable'> | None = None, max_samples: int | None = None, split: str | None = None)`
:   DataConfig(task_type: neuralops.data.base.TaskType = <TaskType.SINGLE_CLASSIFICATION: 'single_classification'>, task_name: str | None = None, max_length: int = 128, padding: Any = 'max_length', truncation: bool = True, return_tensors: str = 'pt', text_column: str | None = None, text_column_1: str | None = None, text_column_2: str | None = None, label_column: str | None = None, batch_size: int = 16, shuffle: bool = True, num_workers: int = 0, mlm_probability: float = 0.15, question_column: str | None = None, context_column: str | None = None, answer_column: str | None = None, tokens_column: str | None = None, tags_column: str | None = None, custom_preprocess: collections.abc.Callable | None = None, max_samples: int | None = None, split: str | None = None)

    ### Instance variables

    `answer_column: str | None`
    :   The type of the None singleton.

    `batch_size: int`
    :   The type of the None singleton.

    `context_column: str | None`
    :   The type of the None singleton.

    `custom_preprocess: <class 'collections.abc.Callable'> | None`
    :   The type of the None singleton.

    `label_column: str | None`
    :   The type of the None singleton.

    `max_length: int`
    :   The type of the None singleton.

    `max_samples: int | None`
    :   The type of the None singleton.

    `mlm_probability: float`
    :   The type of the None singleton.

    `num_workers: int`
    :   The type of the None singleton.

    `padding: Any`
    :   The type of the None singleton.

    `question_column: str | None`
    :   The type of the None singleton.

    `return_tensors: str`
    :   The type of the None singleton.

    `shuffle: bool`
    :   The type of the None singleton.

    `split: str | None`
    :   The type of the None singleton.

    `tags_column: str | None`
    :   The type of the None singleton.

    `task_name: str | None`
    :   The type of the None singleton.

    `task_type: neuralops.data.base.TaskType`
    :   The type of the None singleton.

    `text_column: str | None`
    :   The type of the None singleton.

    `text_column_1: str | None`
    :   The type of the None singleton.

    `text_column_2: str | None`
    :   The type of the None singleton.

    `tokens_column: str | None`
    :   The type of the None singleton.

    `truncation: bool`
    :   The type of the None singleton.

`ExperimentConfig(dataset_name: str, dataset_path: str | None = None, task_type: str = 'single', text_columns: list[str] | None = None, label_column: str | None = None, num_labels: int = 2, metric: str = 'accuracy', max_seq_length: int = 128, split_ratio: float = 0.1, synthetic_samples: int = 5000, glue_base_path: str | None = None, **kwargs)`
:   Configuration for loading a dataset.
    
    Attributes:
        dataset_name: Name of the dataset (GLUE task, synthetic, file path, or HF identifier).
        dataset_path: Path to local dataset file (for custom or multi‑choice).
        task_type: "single", "pair", "multi_choice", "regression".
        text_columns: List of column names containing text (auto‑detected if None).
        label_column: Name of the label column (auto‑detected if None).
        num_labels: Number of classes (for classification).
        metric: Primary metric for evaluation (e.g., "accuracy").
        max_seq_length: Maximum sequence length for tokenization.
        split_ratio: Fraction of data to use for validation (for custom datasets).
        synthetic_samples: Number of synthetic samples to generate (if dataset_name="synthetic").
        glue_base_path: Base directory for local GLUE Parquet files.
        **kwargs: Additional arguments passed to underlying loaders.

`FairnessEvaluator(threshold: float = 0.05)`
:   Evaluate model fairness across protected groups.
    
    Usage:
        evaluator = FairnessEvaluator(threshold=0.05)
        report = evaluator.evaluate(predictions, labels, groups={"gender": gender_array})

    ### Methods

    `evaluate(self, predictions: numpy.ndarray, labels: numpy.ndarray, groups: dict[str, numpy.ndarray], metrics: list[str] | None = None, favorable_label: int = 1) ‑> neuralops.evaluation.fairness.FairnessReport`
    :   Args:
            predictions: predicted labels (or probabilities for the positive class)
            labels: ground truth
            groups: dict of attribute_name → array of group values per sample
            metrics: list of metric names (default: all)
            favorable_label: which label is considered "positive" (default 1)

`FairnessReport(metrics: dict[str, dict[str, float]], group_sizes: dict[str, int], disparities: dict[str, float], overall_score: float, worst_off_group: str, best_off_group: str, is_fair: bool, threshold: float, mitigation_suggestions: list[str] = <factory>)`
:   Complete fairness analysis for a model.

    ### Instance variables

    `best_off_group: str`
    :   The type of the None singleton.

    `disparities: dict[str, float]`
    :   The type of the None singleton.

    `group_sizes: dict[str, int]`
    :   The type of the None singleton.

    `is_fair: bool`
    :   The type of the None singleton.

    `metrics: dict[str, dict[str, float]]`
    :   The type of the None singleton.

    `mitigation_suggestions: list[str]`
    :   The type of the None singleton.

    `overall_score: float`
    :   The type of the None singleton.

    `threshold: float`
    :   The type of the None singleton.

    `worst_off_group: str`
    :   The type of the None singleton.

`HuggingFaceTrainer(model, train_dataset, eval_dataset, config: dict[str, typing.Any], results_manager, compute_metrics=None, **kwargs)`
:   Abstract base class for all trainers.

    ### Ancestors (in MRO)

    * neuralops.trainer.base_trainer.BaseTrainer
    * abc.ABC

    ### Instance variables

    `args: Any`
    :   Return training arguments.

    `state: Any`
    :   Return trainer state.

    ### Methods

    `add_callback(self, callback: Any) ‑> None`
    :   Add a callback to the trainer.

`LabelQualityDetector(threshold: float = 0.1, seed: int = 42)`
:   Detects label errors, outliers, and near‑duplicates using confident learning.
    
    Usage:
        detector = LabelQualityDetector()
        report = detector.analyze(labels=y_true, pred_probs=model_output_probs)
    
    No ground‑truth validation set is required – the method works entirely from
    model predictions and given labels.

    ### Methods

    `analyze(self, labels: numpy.ndarray, pred_probs: numpy.ndarray, class_names: list[str] | None = None) ‑> neuralops.evaluation.label_quality.LabelQualityReport`
    :   Run full label quality analysis.
        
        Args:
            labels: integer labels, shape (N,)
            pred_probs: model predicted probabilities, shape (N, K)
            class_names: optional list of class name strings

    `summary(self, report: neuralops.evaluation.label_quality.LabelQualityReport) ‑> dict[str, typing.Any]`
    :   Summary.

`LabelQualityReport(n_samples: int, n_label_issues: int, issue_rate: float, label_issue_indices: numpy.ndarray, label_quality_scores: numpy.ndarray, suggested_labels: numpy.ndarray | None = None, per_class_issues: dict[int, int] = <factory>, outlier_indices: numpy.ndarray = <factory>, near_duplicate_pairs: list[tuple[int, int]] = <factory>, overall_quality_score: float = 0.0)`
:   Complete label quality analysis for a dataset.

    ### Instance variables

    `issue_rate: float`
    :   The type of the None singleton.

    `label_issue_indices: numpy.ndarray`
    :   The type of the None singleton.

    `label_quality_scores: numpy.ndarray`
    :   The type of the None singleton.

    `n_label_issues: int`
    :   The type of the None singleton.

    `n_samples: int`
    :   The type of the None singleton.

    `near_duplicate_pairs: list[tuple[int, int]]`
    :   The type of the None singleton.

    `outlier_indices: numpy.ndarray`
    :   The type of the None singleton.

    `overall_quality_score: float`
    :   The type of the None singleton.

    `per_class_issues: dict[int, int]`
    :   The type of the None singleton.

    `suggested_labels: numpy.ndarray | None`
    :   The type of the None singleton.

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

`ResultWriter(output_dir: pathlib.Path | str)`
:   Centralised, organised result persistence for a single pipeline run.
    
    Initialise and create the standard directory tree under output_dir.

    ### Methods

    `save_config(self, cfg: Any) ‑> None`
    :   Serialise the ExperimentConfig (dataclass or dict) to config.json.

    `save_history(self, log_history: list[dict[str, typing.Any]]) ‑> None`
    :   Save the full HuggingFace trainer log_history list.

    `save_metrics(self, metrics: dict[str, typing.Any], tag: str = 'metrics') ‑> None`
    :   Save a metrics dict to metrics/<tag>.json with a timestamp.

    `save_plots(self, log_history: list[dict[str, typing.Any]]) ‑> None`
    :   Generate and save training/validation loss and metric curves.

    `save_system_info(self, elapsed_s: float = 0.0, n_params: int = 0, device: str = 'cpu') ‑> None`
    :   Record hardware, runtime, and parameter count metadata.

    `write_summary_md(self, exp_name: str, cfg: Any, metrics: dict[str, typing.Any], elapsed_s: float) ‑> None`
    :   Write a human-readable SUMMARY.md for quick inspection.

`ResultsSaver(run_id: str, base_dir: Path, config: dict[str, Any] | None = None)`
:   Usage:
        saver = ResultsSaver(run_id, base_dir, save_config_dict)
        saver.save_all(model, config, history, final_metrics, tokenizer, input_example)
        saver.push_all()

    ### Descendants

    * neuralops.utils.results_saver.ResultsSaver

    ### Methods

    `create_github_release(self, metrics: dict[str, Any]) ‑> str | None`
    :   Build and return github release.

    `dvc_track(self) ‑> None`
    :   Dvc track.

    `git_commit_and_push(self) ‑> None`
    :   Git commit and push.

    `log_metric_to_sqlite(self, step: int, metric_name: str, value: float, db_path: str | Path | None = None) ‑> None`
    :   Append a single step metric into SQLite database with multi-GPU concurrency safety.
        
        # FIX: Uses PRAGMA journal_mode=WAL; and PRAGMA synchronous=NORMAL; with file locking to ensure safe step metric logging.

    `notify(self, success: bool, metrics: dict[str, Any]) ‑> None`
    :   Notify.

    `push_all(self) ‑> None`
    :   Push all.

    `push_to_registries(self) ‑> None`
    :   Push to registries.

    `save_all(self, model, config, history, final_metrics, tokenizer=None, input_example=None) ‑> None`
    :   Save everything selected in the config.

    `save_config(self, config: Any) ‑> str`
    :   Save experiment configuration dictionary or object.

    `save_metrics(self, metrics: dict[str, Any]) ‑> str`
    :   Save a dictionary of evaluation/training metrics to JSON and SQLite.

    `save_model_weights(self, model: Any, filename: str = 'model.pt') ‑> str`
    :   Save PyTorch model state dict or checkpoint.

    `save_result(self, result: dict[str, Any] | Any, results_dir: str | Path | None = None, csv_filename: str = 'master_results.csv') ‑> str`
    :   Append a benchmark result to master CSV.

    `save_to_sqlite(self, db_path: str | Path | None = None, config: Any | None = None, metrics: dict[str, Any] | None = None, extra_metadata: dict[str, Any] | None = None) ‑> str`
    :   Save run metadata, configuration, and final metrics into SQLite database with multi-GPU write safety.
        
        # FIX: Uses PRAGMA journal_mode=WAL;, PRAGMA synchronous=NORMAL;, and file locking for multi-GPU database write safety.

    `should_save_heavy(self, final_metrics: dict[str, Any]) ‑> bool`
    :   Evaluate if metric meets auto_save_if_better_than threshold.

    `track_all(self, metrics: dict[str, Any], params: dict[str, Any]) ‑> None`
    :   Track all.

`RetrainDecision(should_retrain: bool, reason: str, trigger_type: neuralops.evaluation.retraining.TriggerType | None = None, urgency: str = 'low', details: dict[str, typing.Any] = <factory>)`
:   Output of the retraining decision engine.

    ### Instance variables

    `details: dict[str, typing.Any]`
    :   The type of the None singleton.

    `reason: str`
    :   The type of the None singleton.

    `should_retrain: bool`
    :   The type of the None singleton.

    `trigger_type: neuralops.evaluation.retraining.TriggerType | None`
    :   The type of the None singleton.

    `urgency: str`
    :   The type of the None singleton.

`RetrainPolicy(trigger_types: list[neuralops.evaluation.retraining.TriggerType] = <factory>, drift_threshold: float = 0.3, drift_methods: list[str] = <factory>, min_accuracy: float = 0.85, max_accuracy_degradation: float = 0.05, schedule_cron: str | None = None, max_days_since_last_train: int = 30, min_new_samples: int = 1000, max_new_samples: int = 100000, require_better_model: bool = True, min_improvement: float = 0.01, max_retrains_per_day: int = 3, validation_split: float = 0.2, notify_on_trigger: bool = True, notify_on_complete: bool = True)`
:   Configuration for when and how to retrain.

    ### Instance variables

    `drift_methods: list[str]`
    :   The type of the None singleton.

    `drift_threshold: float`
    :   The type of the None singleton.

    `max_accuracy_degradation: float`
    :   The type of the None singleton.

    `max_days_since_last_train: int`
    :   The type of the None singleton.

    `max_new_samples: int`
    :   The type of the None singleton.

    `max_retrains_per_day: int`
    :   The type of the None singleton.

    `min_accuracy: float`
    :   The type of the None singleton.

    `min_improvement: float`
    :   The type of the None singleton.

    `min_new_samples: int`
    :   The type of the None singleton.

    `notify_on_complete: bool`
    :   The type of the None singleton.

    `notify_on_trigger: bool`
    :   The type of the None singleton.

    `require_better_model: bool`
    :   The type of the None singleton.

    `schedule_cron: str | None`
    :   The type of the None singleton.

    `trigger_types: list[neuralops.evaluation.retraining.TriggerType]`
    :   The type of the None singleton.

    `validation_split: float`
    :   The type of the None singleton.

`RetrainingOrchestrator(policy: neuralops.evaluation.retraining.RetrainPolicy = None)`
:   Central decision engine for automated retraining.
    
    Usage:
        orch = RetrainingOrchestrator(RetrainPolicy())
        decision = orch.evaluate(
            current_metrics={"accuracy": 0.82},
            reference_metrics={"accuracy": 0.88},
            drift_report=drift_report,
            new_data_count=5000,
            days_since_last_train=35,
        )
        if decision.should_retrain:
            orch.trigger_retraining(run_id="model-v3", training_fn=my_train_func)

    ### Methods

    `evaluate(self, current_metrics: dict[str, float] | None = None, reference_metrics: dict[str, float] | None = None, drift_report: Any | None = None, new_data_count: int = 0, days_since_last_train: int = 0, manual_request: bool = False) ‑> neuralops.evaluation.retraining.RetrainDecision`
    :   Evaluate all configured triggers and return a retraining decision.

    `trigger_retraining(self, run_id: str, training_fn: <class 'collections.abc.Callable'>, decision: neuralops.evaluation.retraining.RetrainDecision, **training_kwargs) ‑> dict[str, typing.Any]`
    :   Execute retraining and validate the new model.

`TaskType(*values)`
:   Create a collection of name/value pairs.
    
    Example enumeration:
    
    >>> class Color(Enum):
    ...     RED = 1
    ...     BLUE = 2
    ...     GREEN = 3
    
    Access them by:
    
    - attribute access:
    
      >>> Color.RED
      <Color.RED: 1>
    
    - value lookup:
    
      >>> Color(1)
      <Color.RED: 1>
    
    - name lookup:
    
      >>> Color['RED']
      <Color.RED: 1>
    
    Enumerations can be iterated over, and know how many members they have:
    
    >>> len(Color)
    3
    
    >>> list(Color)
    [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]
    
    Methods can be added to enumerations, and members can have their own
    attributes -- see the documentation for details.

    ### Ancestors (in MRO)

    * enum.Enum

    ### Class variables

    `LANGUAGE_MODELING`
    :   The type of the None singleton.

    `PAIR_CLASSIFICATION`
    :   The type of the None singleton.

    `QA`
    :   The type of the None singleton.

    `REGRESSION`
    :   The type of the None singleton.

    `SEQ2SEQ`
    :   The type of the None singleton.

    `SINGLE_CLASSIFICATION`
    :   The type of the None singleton.

    `TOKEN_CLASSIFICATION`
    :   The type of the None singleton.

`TrainerRegistry()`
:   Registry for trainer classes.

    ### Static methods

    `create(name: str, **kwargs) ‑> neuralops.trainer.base_trainer.BaseTrainer`
    :   Create.

    `get(name: str) ‑> type[neuralops.trainer.base_trainer.BaseTrainer]`
    :   Get.

    `list() ‑> <classmethod(<function TrainerRegistry.list at 0x12b0a8b40>)>`
    :   List.

    `register(name: str) ‑> Any`
    :   Register.

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