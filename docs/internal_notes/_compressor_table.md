| # | Key | Class | Subpackage | Description |
|---:|---|---|---|---|
| 1 | `a3_compressor` | `A3Compressor` | `low_rank.a3_compressor` | A3: Activation-aware factorized SVD compressor. |
| 2 | `aa_svd_compressor` | `AASVDCompressor` | `svd.aa_svd_compressor` |  |
| 3 | `aatc_xkv_compressor` | `AATC_XKVCompressor` | `kv_cache.aatc_xkv_compressor` |  |
| 4 | `adaptive_ranks` | `AdaptiveRanksCompressor` | `svd.adaptive_ranks` |  |
| 5 | `adasvd_compressor` | `AdaSVDCompressor` | `svd.adasvd_compressor` |  |
| 6 | `admm_compressor` | `ADMMCompressor` | `pruning.admm_compressor` | ADMM-based joint structured pruning + uniform quantization engine. |
| 7 | `admm_pruner` | `ADMMPruner` | `pruning.admm_pruner` | Production-hardened ADMM Pruner with Layer-Wise Reconstruction Distillation, |
| 8 | `armor_pruner` | `ARMORPruner` | `pruning.armor_pruner` | Production-hardened ARMOR Pruner. |
| 9 | `ast_pruner` | `ASTPruner` | `pruning.ast_pruner` | AST Pruner: vectorized semi-structured 2:4 / N:M pruning with empirical |
| 10 | `astro_quantizer` | `AstroQuantizer` | `quantization.astro_quantizer` | AstroQuantizer: Uniform min-max / symmetric quantization engine with Fisher-adap |
| 11 | `bonsai_pruner` | `bonsai_factory` | `pruning.bonsai_pruner` | Factory for Bonsai pruner with progressive sparsity. |
| 12 | `cachenotes_compressor` | `CacheNotesCompressor` | `kv_cache.cachenotes_compressor` |  |
| 13 | `come_compressor` | `CoMeCompressor` | `low_rank.come_compressor` | CoMe Compressor: Layer shedding with learnable low-rank interpolation |
| 14 | `compot_compressor` | `compot_compressor_factory` | `low_rank.compot_compressor` | Factory function for COMPOT compression. |
| 15 | `coopq_quantizer` | `coopq_factory` | `quantization.coopq_quantizer` | CoopQ compressor factory function. |
| 16 | `cr_compressor` | `CRCompressor` | `low_rank.cr_compressor` | CR (Column-Row / Interpolative Decomposition) Compressor. |
| 17 | `cur_compressor` | `CURCompressor` | `low_rank.cur_compressor` |  |
| 18 | `cur_moe_compressor` | `cur_moe_factory` | `low_rank.cur_moe_compressor` | Factory function for CUR-MoE compressor. |
| 19 | `delta_kv` | `delta_kv_factory` | `kv_cache.delta_kv` | DeltaKV compressor factory. |
| 20 | `direct_svd` | `DirectSVDCompressor` | `svd.direct_svd` |  |
| 21 | `dobi_svd` | `dobi_factory` | `svd.dobi_svd` | Dobi factory function. |
| 22 | `drone_compressor` | `DRONECompressor` | `low_rank.drone_compressor` | DRONE Compressor. |
| 23 | `dynamic_rank` | `dynamic_entropy_svd_factory` | `svd.dynamic_rank` | Dynamic Entropy SVD factory function. |
| 24 | `enlighten_compressor` | `ENLightenCompressor` | `low_rank.enlighten_compressor` |  |
| 25 | `entropy_budget_allocation` | `EntropyBudgetCompressor` | `svd.entropy_budget_allocation` |  |
| 26 | `erc_svd` | `ERCCompressor` | `svd.erc_svd` |  |
| 27 | `faq_quantizer` | `FAQQuantizer` | `quantization.faq_quantizer` | FAQQuantizer: Feature-Activation-Aware Quantization. |
| 28 | `flap` | `FLAPCompressor` | `pruning.flap` | FLAP Compressor: Fluctuation-based Adaptive Structured Pruning. |
| 29 | `flat_llm` | `FLATLLMCompressor` | `experimental.flat_llm` | FLAT-LLM compressor: Activation-aware output-channel pruning. |
| 30 | `frandi_compressor` | `frandi_factory` | `low_rank.frandi_compressor` | Factory function for FRanDI compressor. |
| 31 | `fwsvd_compressor` | `FWSVDCompressor` | `svd.fwsvd_compressor` | Fisher-Weighted SVD Compressor. |
| 32 | `gfws` | `GFWSCompressor` | `svd.gfws_d_compressor` |  |
| 33 | `grecodec_compressor` | `grecodec_compressor_factory` | `low_rank.grecodec_compressor` | Factory function for GreCoDeC compressor. |
| 34 | `hashed_net_compressor` | `hashed_net_factory` | `experimental.hashed_net_compressor` |  |
| 35 | `huffman_coder` | `HuffmanCompressor` | `quantization.huffman_coder` | HuffmanCompressor: Model-wide entropy compression profiler and simulator. |
| 36 | `hybrid_svd_layer` | `HybridSVDCompressor` | `svd.hybrid_svd_layer` | Weight-based SVD compressor for Linear / Conv1D layers. |
| 37 | `hybridkv_compressor` | `HybridKVCompressor` | `kv_cache.hybridkv_compressor` | HybridKV Compressor: Classifies projection heads into static/dynamic types |
| 38 | `kmeans_quantize` | `kmeans_factory` | `quantization.kmeans_quantize` |  |
| 39 | `kronecker_compressor` | `kronecker_factory` | `tensor_decomp.kronecker_compressor` | Kronecker compressor factory. |
| 40 | `kronecker_tt_lora` | `kronecker_tt_lora_factory` | `tensor_decomp.kronecker_tt_lora` | Factory for Kronecker-TT-LoRA compressor. |
| 41 | `kvtc_compressor` | `KVTCCompressor` | `kv_cache.kvtc_compressor` | KVTC: KV Cache Transform Coding compressor (Static Weight PCA + Quantization). |
| 42 | `liftr` | `liftr_compressor_entry` | `experimental.liftr` | Factory function for LIFTR compressor. |
| 43 | `lillama_compressor` | `lillama_factory` | `experimental.lillama_compressor` | Factory function for LiLLaMA compressor. |
| 44 | `lsc_compressor` | `lsc_factory` | `low_rank.lsc_compressor` | LSC Fourier spectral compressor factory. |
| 45 | `mamba_shedder` | `mamba_shedder_factory` | `pruning.mamba_shedder` |  |
| 46 | `mgie_budget` | `MGIESVDBudgetCompressor` | `low_rank.mgie_svd_compressor` | MGIE-SVD Budget Compressor: Global Entropy-Proportional Rank Allocation. |
| 47 | `mlorq_compressor` | `MLoRQCompressor` | `low_rank.mlorq_compressor` | MLoRQ Compressor: Multi-Layer Optimal Rank and Quantization under Memory Budget. |
| 48 | `modegpt_compressor` | `MoDeGPTCompressor` | `low_rank.modegpt_compressor` | MoDeGPT: Modular Decomposition of Transformer Projection Blocks. |
| 49 | `nmf_compressor` | `NMFCompressor` | `low_rank.nmf_compressor` | NMF Compressor: decomposes linear layer magnitudes into non-negative |
| 50 | `oats_compressor` | `OATSCompressor` | `low_rank.oats_compressor` | OATS Compressor: Outlier-aware low-rank + structured sparse matrix decomposition |
| 51 | `optimal_subspace_projection` | `OptimalSubspaceProjectionCompressor` | `svd.optimal_subspace_projection` | Optimal Subspace Projection Compressor via activation-aware randomized SVD. |
| 52 | `proxsparse` | `ProxSparsePruner` | `pruning.proxsparse` | ProxSparse: proximal gradient sparsity with structured / 2:4 / percentile |
| 53 | `residual_svd_compressor` | `ResidualSVDCompressor` | `svd.residual_svd_compressor` |  |
| 54 | `roast_hashed_linear` | `ROASTCompressor` | `experimental.roast_hashed_linear` | Replace eligible linear and Conv1D layers with ROASTLinear (hardware-aware weigh |
| 55 | `rotational_slicegpt_int4` | `RotationalSliceGPTINT4Compressor` | `quantization.rotational_slicegpt_int4` | Rotational-SliceGPT-INT4 hybrid compression engine. |
| 56 | `rpca_compressor` | `RPCACompressor` | `low_rank.rpca_compressor` |  |
| 57 | `saes_svd_v2` | `SAESSVDCompressorV2` | `svd.saes_svd_v2` |  |
| 58 | `salr` | `SALRCompressor` | `low_rank.salr` |  |
| 59 | `sdrm_compressor` | `sdrm_compressor_factory` | `low_rank.sdrm_compressor` | SDRM compressor factory returning (compressed_model, stats) when requested. |
| 60 | `slicegpt` | `PCAActivationAwareSVDCompressor` | `quantization.slicegpt` | Per-Layer Activation-Aware Low-Rank Compression (PCA-ASVD). |
| 61 | `slicegpt_pca` | `SliceGPTPCACompressor` | `quantization.slicegpt_pca` | Production-hardened SliceGPT PCA-based compressor that performs activation-aware |
| 62 | `slim_compressor` | `SLiMCompressor` | `low_rank.slim_compressor` |  |
| 63 | `soka_compressor` | `soka_factory` | `low_rank.soka_compressor` | Factory function for SoKA compressor. |
| 64 | `sparsegpt` | `sparsegpt_factory` | `pruning.sparsegpt` | Factory function for SparseGPT compressor. |
| 65 | `subspace_projection` | `SubspaceProjectionCompressor` | `svd.subspace_projection` | Subspace Projection Compression Engine. |
| 66 | `svd_linear` | `SVDLinearCompressor` | `svd.svd_linear` | Compresses model by replacing Linear/Conv1D layers with SVDLinearLayer. |
| 67 | `svd_llm_compressor` | `SVDLLMCompressor` | `low_rank.svd_llm_compressor` | SVD-LLM (ICLR 2025) Compressor with Cholesky activation whitening, |
| 68 | `swift_svd` | `SwiftSVDCompressor` | `svd.swift_svd` |  |
| 69 | `swift_svd_act` | `SwiftSVDActivationCompressor` | `svd.swift_svd_act` | Swift-SVD Activation-Aware Compressor. |
| 70 | `tomoe_compressor` | `tomoe_compressor_factory` | `low_rank.tomoe_compressor` | Factory function for ToMoE compressor. |
| 71 | `tt_compressor` | `TTCompressor` | `tensor_decomp.tt_compressor` | Replaces Linear/Conv1D with TTLinearLayer when TT succeeds; otherwise |
| 72 | `uniql_compressor` | `UniQLCompressor` | `low_rank.uniql_compressor` | UniQL: Joint Truncated SVD + Uniform Quantization compressor. |
| 73 | `zerokv_neo_compressor` | `ZeroKVNeoCompressor` | `kv_cache.zerokv_neo_compressor` | ZeroKV-Neo Compressor: L2-norm weight pruning preserving attention sinks and hea |
