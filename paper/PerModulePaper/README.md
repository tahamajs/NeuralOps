# Per-Module Papers (217 Total)

This collection provides standalone, publication-quality short workshop papers for each of the **217** attention and compression mechanisms evaluated in our large-scale empirical study on DistilBERT-base across SST-2, MNLI, and AG News.

Every paper is compiled in the official **ICLR 2027 double-blind conference format** (`iclr2027_conference.sty`) with the standard review footer:
> *"Under review as a conference paper at ICLR 2027"*

## Master Index Table

| # | Module | Status (MNLI / SST-2) | PDF Paper | HF Base Model Checkpoint | Key Mechanism |
|---|--------|:---------------------:|:---------:|:------------------------:|---------------|
| 1 | `AdaptiveSVD` | `PASS` | [471 KB](AdaptiveSVD/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Adaptivesvd |
| 2 | `AdaptiveSpanAttention` | `REJECTED_EXPANSION` | [1992 KB](AdaptiveSpanAttention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Adaptivespanattention |
| 3 | `AdaptiveSparseAttention` | `PASS` | [1937 KB](AdaptiveSparseAttention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Adaptivesparseattention |
| 4 | `Galerkin` | `REJECTED_EXPANSION` | [2539 KB](Galerkin/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Galerkin |
| 5 | `HALO` | `EVALUATED` | [2558 KB](HALO/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Halo |
| 6 | `OH` | `EVALUATED` | [388 KB](OH/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Oh |
| 7 | `Primal` | `CRASH` | [2052 KB](Primal/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Primal |
| 8 | `SOAFusedAttention` | `PASS` | [359 KB](SOAFusedAttention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Soafusedattention |
| 9 | `SqueezingHeadsAttentionDistillationLayer` | `EVALUATED` | [2537 KB](SqueezingHeadsAttentionDistillationLayer/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Squeezingheadsattentiondistillationlayer |
| 10 | `a3` | `PASS` | [731 KB](a3/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | A3 |
| 11 | `aa_svd_compressor` | `EVALUATED` | [389 KB](aa_svd_compressor/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Aa Svd Compressor |
| 12 | `aasvd` | `PASS` | [412 KB](aasvd/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Aasvd |
| 13 | `adaptive_compressor` | `PASS` | [391 KB](adaptive_compressor/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Adaptive Compressor |
| 14 | `adaptive_ranks` | `EVALUATED` | [392 KB](adaptive_ranks/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Adaptive Ranks |
| 15 | `adaptive_span_attention` | `REJECTED_EXPANSION` | [395 KB](adaptive_span_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Adaptive Span Attention |
| 16 | `adaptive_sparse_attention` | `PASS` | [394 KB](adaptive_sparse_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Adaptive Sparse Attention |
| 17 | `adaptive_svd_attention` | `PASS` | [450 KB](adaptive_svd_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Adaptive Svd Attention |
| 18 | `adasplash` | `REJECTED_EXPANSION` | [439 KB](adasplash/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Adasplash |
| 19 | `adasvd_compressor` | `PASS` | [390 KB](adasvd_compressor/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Adasvd Compressor |
| 20 | `admm_compressor` | `PASS` | [389 KB](admm_compressor/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Admm Compressor |
| 21 | `admm_pruner` | `PASS` | [392 KB](admm_pruner/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Admm Pruner |
| 22 | `aegis_hybrid` | `PASS` | [391 KB](aegis_hybrid/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Aegis Hybrid |
| 23 | `aegis_spectron_fp8` | `PASS_LOW` | [392 KB](aegis_spectron_fp8/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Aegis Spectron Fp8 |
| 24 | `aegis_v1` | `REJECTED_EXPANSION` | [391 KB](aegis_v1/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Aegis V1 |
| 25 | `aegis_v2` | `PASS` | [391 KB](aegis_v2/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Aegis V2 |
| 26 | `aegis_v3` | `PASS` | [391 KB](aegis_v3/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Aegis V3 |
| 27 | `aegis_v4` | `PASS` | [391 KB](aegis_v4/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Aegis V4 |
| 28 | `aegis_v5` | `PASS` | [394 KB](aegis_v5/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Aegis V5 |
| 29 | `aegis_v6` | `PASS` | [391 KB](aegis_v6/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Aegis V6 |
| 30 | `armor_pruner` | `PASS` | [389 KB](armor_pruner/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Armor Pruner |
| 31 | `ast_pruner` | `PASS` | [389 KB](ast_pruner/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Ast Pruner |
| 32 | `astro_quantizer` | `PASS` | [389 KB](astro_quantizer/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Astro Quantizer |
| 33 | `asvd` | `REJECTED_EXPANSION` | [2784 KB](asvd/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Asvd |
| 34 | `bam_umoe` | `PASS` | [2534 KB](bam_umoe/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Bam Umoe |
| 35 | `block_transformer_hiba` | `REJECTED_EXPANSION` | [394 KB](block_transformer_hiba/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Block Transformer Hiba |
| 36 | `bonsai_pruner` | `PASS` | [389 KB](bonsai_pruner/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Bonsai Pruner |
| 37 | `cachenotes_compressor` | `PASS` | [390 KB](cachenotes_compressor/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Cachenotes Compressor |
| 38 | `chela_slay` | `REJECTED_EXPANSION` | [2610 KB](chela_slay/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Chela Slay |
| 39 | `cluster` | `REJECTED_EXPANSION` | [2506 KB](cluster/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Cluster |
| 40 | `come` | `PASS` | [388 KB](come/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Come |
| 41 | `compot_compressor` | `PASS` | [390 KB](compot_compressor/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Compot Compressor |
| 42 | `compressed_attention` | `PASS_LOW` | [391 KB](compressed_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Compressed Attention |
| 43 | `compressed_mha` | `PASS_LOW` | [2406 KB](compressed_mha/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Compressed Mha |
| 44 | `coopq_quantizer` | `PASS` | [389 KB](coopq_quantizer/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Coopq Quantizer |
| 45 | `cosformer` | `REJECTED_EXPANSION` | [2736 KB](cosformer/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Cosformer |
| 46 | `cr_compressor` | `PASS` | [389 KB](cr_compressor/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Cr Compressor |
| 47 | `cur_compressor` | `PASS` | [389 KB](cur_compressor/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Cur Compressor |
| 48 | `cur_linear` | `EVALUATED` | [391 KB](cur_linear/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Cur Linear |
| 49 | `cur_moe` | `PASS` | [389 KB](cur_moe/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Cur Moe |
| 50 | `cure` | `EVALUATED` | [391 KB](cure/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Cure |
| 51 | `decay_linear_attention` | `EVALUATED` | [395 KB](decay_linear_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Decay Linear Attention |
| 52 | `delta_kv` | `PASS` | [389 KB](delta_kv/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Delta Kv |
| 53 | `direct_svd` | `PASS` | [389 KB](direct_svd/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Direct Svd |
| 54 | `dobi` | `PASS` | [388 KB](dobi/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Dobi |
| 55 | `dobi_svd` | `EVALUATED` | [2542 KB](dobi_svd/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Dobi Svd |
| 56 | `dynamic_entropy` | `EVALUATED` | [392 KB](dynamic_entropy/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Dynamic Entropy |
| 57 | `dynamic_rank` | `PASS` | [389 KB](dynamic_rank/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Dynamic Rank |
| 58 | `eat` | `REJECTED_EXPANSION` | [2086 KB](eat/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Eat |
| 59 | `enlighten_compressor` | `PASS` | [390 KB](enlighten_compressor/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Enlighten Compressor |
| 60 | `entropy_budget_allocation` | `EVALUATED` | [391 KB](entropy_budget_allocation/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Entropy Budget Allocation |
| 61 | `erc_svd` | `EVALUATED` | [391 KB](erc_svd/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Erc Svd |
| 62 | `expert_choice_moe` | `PASS` | [2664 KB](expert_choice_moe/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Expert Choice Moe |
| 63 | `factorised_matrix_state` | `EVALUATED` | [391 KB](factorised_matrix_state/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Factorised Matrix State |
| 64 | `faq_quantizer` | `REJECTED_EXPANSION` | [391 KB](faq_quantizer/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Faq Quantizer |
| 65 | `flap` | `PASS` | [389 KB](flap/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Flap |
| 66 | `flash3` | `REJECTED_EXPANSION` | [373 KB](flash3/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Flash3 |
| 67 | `flash_attention3` | `EVALUATED` | [2470 KB](flash_attention3/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Flash Attention3 |
| 68 | `flash_attention3_fp8` | `EVALUATED` | [2503 KB](flash_attention3_fp8/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Flash Attention3 Fp8 |
| 69 | `flat_llm` | `PASS` | [389 KB](flat_llm/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Flat Llm |
| 70 | `frandi` | `PASS` | [388 KB](frandi/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Frandi |
| 71 | `full_attention` | `REJECTED_EXPANSION` | [390 KB](full_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Full Attention |
| 72 | `fused_svd_kernel` | `PASS` | [391 KB](fused_svd_kernel/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Fused Svd Kernel |
| 73 | `fusion` | `EVALUATED` | [1906 KB](fusion/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Fusion |
| 74 | `fusion_attn` | `REJECTED_EXPANSION` | [391 KB](fusion_attn/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Fusion Attn |
| 75 | `galerkin_attention` | `REJECTED_EXPANSION` | [391 KB](galerkin_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Galerkin Attention |
| 76 | `gated_delta_net` | `PASS` | [2547 KB](gated_delta_net/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Gated Delta Net |
| 77 | `gated_deltanet_block` | `PASS` | [2034 KB](gated_deltanet_block/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Gated Deltanet Block |
| 78 | `gated_linear_attention` | `EVALUATED` | [391 KB](gated_linear_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Gated Linear Attention |
| 79 | `gfws_d_compressor` | `EVALUATED` | [389 KB](gfws_d_compressor/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Gfws D Compressor |
| 80 | `goldfinch_ahn` | `EVALUATED` | [1823 KB](goldfinch_ahn/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Goldfinch Ahn |
| 81 | `gqa` | `REJECTED_EXPANSION` | [366 KB](gqa/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Gqa |
| 82 | `granite_hybrid` | `PASS` | [391 KB](granite_hybrid/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Granite Hybrid |
| 83 | `grecodec` | `PASS` | [388 KB](grecodec/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Grecodec |
| 84 | `griffin_ssd` | `REJECTED_EXPANSION` | [391 KB](griffin_ssd/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Griffin Ssd |
| 85 | `halo_regla` | `REJECTED_EXPANSION` | [391 KB](halo_regla/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Halo Regla |
| 86 | `hamming_attention` | `REJECTED_EXPANSION` | [390 KB](hamming_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Hamming Attention |
| 87 | `hashed_linear` | `EVALUATED` | [391 KB](hashed_linear/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Hashed Linear |
| 88 | `hashed_net_compressor` | `PASS` | [390 KB](hashed_net_compressor/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Hashed Net Compressor |
| 89 | `hgrn2_gated_deltanet` | `PASS` | [2362 KB](hgrn2_gated_deltanet/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Hgrn2 Gated Deltanet |
| 90 | `hgrn2_layer` | `PASS` | [2576 KB](hgrn2_layer/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Hgrn2 Layer |
| 91 | `hgrn2_outer` | `TIMEOUT` | [391 KB](hgrn2_outer/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Hgrn2 Outer |
| 92 | `hla_attention` | `REJECTED_EXPANSION` | [391 KB](hla_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Hla Attention |
| 93 | `huffman_coder` | `EVALUATED` | [392 KB](huffman_coder/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Huffman Coder |
| 94 | `hybrid_svd_layer` | `PASS` | [392 KB](hybrid_svd_layer/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Hybrid Svd Layer |
| 95 | `hybridkv` | `PASS` | [389 KB](hybridkv/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Hybridkv |
| 96 | `hymba_transx` | `REJECTED_EXPANSION` | [2454 KB](hymba_transx/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Hymba Transx |
| 97 | `infini` | `REJECTED_EXPANSION` | [2377 KB](infini/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Infini |
| 98 | `jamba` | `REJECTED_EXPANSION` | [390 KB](jamba/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Jamba |
| 99 | `jamba_info` | `REJECTED_EXPANSION` | [2374 KB](jamba_info/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Jamba Info |
| 100 | `kda_attention` | `REJECTED_EXPANSION` | [390 KB](kda_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Kda Attention |
| 101 | `kmeans_quantize` | `PASS` | [395 KB](kmeans_quantize/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Kmeans Quantize |
| 102 | `kronecker_compressor` | `PASS` | [390 KB](kronecker_compressor/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Kronecker Compressor |
| 103 | `kronecker_linear` | `EVALUATED` | [392 KB](kronecker_linear/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Kronecker Linear |
| 104 | `kronecker_tt_lora` | `REJECTED_EXPANSION` | [392 KB](kronecker_tt_lora/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Kronecker Tt Lora |
| 105 | `kvtc` | `PASS` | [389 KB](kvtc/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Kvtc |
| 106 | `leka_v1` | `REJECTED_EXPANSION` | [2688 KB](leka_v1/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Leka V1 |
| 107 | `leka_v2` | `REJECTED_EXPANSION` | [2843 KB](leka_v2/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Leka V2 |
| 108 | `leka_v3` | `REJECTED_EXPANSION` | [2790 KB](leka_v3/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Leka V3 |
| 109 | `liftr` | `PASS` | [389 KB](liftr/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Liftr |
| 110 | `lillama_compressor` | `PASS` | [390 KB](lillama_compressor/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Lillama Compressor |
| 111 | `linformer_attention` | `REJECTED_EXPANSION` | [372 KB](linformer_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Linformer Attention |
| 112 | `lola_attention` | `PASS` | [391 KB](lola_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Lola Attention |
| 113 | `lora_attention` | `EVALUATED` | [373 KB](lora_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Lora Attention |
| 114 | `lsc_compressor` | `PASS` | [389 KB](lsc_compressor/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Lsc Compressor |
| 115 | `lsh_attention` | `REJECTED_EXPANSION` | [391 KB](lsh_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Lsh Attention |
| 116 | `mamba` | `PASS_LOW` | [390 KB](mamba/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Mamba |
| 117 | `mamba_inject` | `EVALUATED` | [391 KB](mamba_inject/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Mamba Inject |
| 118 | `mamba_shedder` | `PASS` | [389 KB](mamba_shedder/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Mamba Shedder |
| 119 | `mamba_svd_flash` | `CRASH_NO_OUTPUT` | [391 KB](mamba_svd_flash/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Mamba Svd Flash |
| 120 | `mgie_svd` | `EVALUATED` | [390 KB](mgie_svd/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Mgie Svd |
| 121 | `minimal_aurora` | `PASS` | [2667 KB](minimal_aurora/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Minimal Aurora |
| 122 | `minimal_nova` | `PASS` | [2762 KB](minimal_nova/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Minimal Nova |
| 123 | `minimal_prism` | `PASS` | [2686 KB](minimal_prism/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Minimal Prism |
| 124 | `minimal_svd` | `PASS` | [2619 KB](minimal_svd/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Minimal Svd |
| 125 | `minimal_svd_v2` | `PASS` | [391 KB](minimal_svd_v2/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Minimal Svd V2 |
| 126 | `mixattention_moska` | `EVALUATED` | [365 KB](mixattention_moska/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Mixattention Moska |
| 127 | `mla_attention` | `PASS_LOW` | [2437 KB](mla_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Mla Attention |
| 128 | `mlorq_compressor` | `PASS` | [389 KB](mlorq_compressor/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Mlorq Compressor |
| 129 | `moba` | `REJECTED_EXPANSION` | [374 KB](moba/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Moba |
| 130 | `modegpt` | `PASS` | [388 KB](modegpt/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Modegpt |
| 131 | `moe_deltakv_paged` | `REJECTED_EXPANSION` | [385 KB](moe_deltakv_paged/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Moe Deltakv Paged |
| 132 | `nasc_attention` | `REJECTED_EXPANSION` | [376 KB](nasc_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Nasc Attention |
| 133 | `nexus` | `REJECTED_EXPANSION` | [367 KB](nexus/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Nexus |
| 134 | `nha` | `REJECTED_EXPANSION` | [2217 KB](nha/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Nha |
| 135 | `nli_softmax` | `REJECTED_EXPANSION` | [391 KB](nli_softmax/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Nli Softmax |
| 136 | `nmf_compressor` | `PASS` | [389 KB](nmf_compressor/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Nmf Compressor |
| 137 | `nsa` | `REJECTED_EXPANSION` | [1879 KB](nsa/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Nsa |
| 138 | `nystrom_attention` | `REJECTED_EXPANSION` | [1885 KB](nystrom_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Nystrom Attention |
| 139 | `oats` | `PASS` | [388 KB](oats/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Oats |
| 140 | `optimal_subspace_projection` | `PASS` | [390 KB](optimal_subspace_projection/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Optimal Subspace Projection |
| 141 | `optimized_hybrid_attention` | `EVALUATED` | [392 KB](optimized_hybrid_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Optimized Hybrid Attention |
| 142 | `orion` | `PASS` | [376 KB](orion/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Orion |
| 143 | `padre_attention` | `REJECTED_EXPANSION` | [351 KB](padre_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Padre Attention |
| 144 | `performer_attention` | `REJECTED_EXPANSION` | [375 KB](performer_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Performer Attention |
| 145 | `phantom` | `PASS_LOW` | [420 KB](phantom/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Phantom |
| 146 | `progressive_token_pruning` | `EVALUATED` | [391 KB](progressive_token_pruning/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Progressive Token Pruning |
| 147 | `projection_factorized_attn` | `PASS` | [392 KB](projection_factorized_attn/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Projection Factorized Attn |
| 148 | `proxsparse` | `PASS` | [389 KB](proxsparse/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Proxsparse |
| 149 | `proxsparse_pruner` | `PASS` | [392 KB](proxsparse_pruner/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Proxsparse Pruner |
| 150 | `pruning_utils` | `EVALUATED` | [390 KB](pruning_utils/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Pruning Utils |
| 151 | `ptp` | `REJECTED_EXPANSION` | [362 KB](ptp/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Ptp |
| 152 | `qcqa_attention` | `REJECTED_EXPANSION` | [390 KB](qcqa_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Qcqa Attention |
| 153 | `rala_attention` | `REJECTED_EXPANSION` | [391 KB](rala_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Rala Attention |
| 154 | `reap_moe` | `REJECTED_EXPANSION` | [360 KB](reap_moe/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Reap Moe |
| 155 | `regla_attention` | `REJECTED_EXPANSION` | [390 KB](regla_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Regla Attention |
| 156 | `replace` | `PASS` | [390 KB](replace/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Replace |
| 157 | `replaceme_compressor` | `PASS` | [390 KB](replaceme_compressor/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Replaceme Compressor |
| 158 | `res_svd` | `PASS` | [2551 KB](res_svd/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Res Svd |
| 159 | `roast_hashed_linear` | `EVALUATED` | [391 KB](roast_hashed_linear/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Roast Hashed Linear |
| 160 | `rotational_slicegpt_int4` | `EVALUATED` | [392 KB](rotational_slicegpt_int4/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Rotational Slicegpt Int4 |
| 161 | `routing_attention` | `REJECTED_EXPANSION` | [390 KB](routing_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Routing Attention |
| 162 | `rpc` | `REJECTED_EXPANSION` | [2476 KB](rpc/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Rpc |
| 163 | `rpca` | `PASS` | [388 KB](rpca/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Rpca |
| 164 | `saes_svd` | `EVALUATED` | [389 KB](saes_svd/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Saes Svd |
| 165 | `saes_svd_v2` | `EVALUATED` | [392 KB](saes_svd_v2/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Saes Svd V2 |
| 166 | `salr` | `PASS` | [388 KB](salr/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Salr |
| 167 | `salt` | `PASS` | [372 KB](salt/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Salt |
| 168 | `samba_m3` | `REJECTED_EXPANSION` | [389 KB](samba_m3/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Samba M3 |
| 169 | `samba_moe` | `REJECTED_EXPANSION` | [390 KB](samba_moe/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Samba Moe |
| 170 | `sdrm` | `FAIL` | [388 KB](sdrm/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Sdrm |
| 171 | `simple_ssm` | `REJECTED_EXPANSION` | [391 KB](simple_ssm/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Simple Ssm |
| 172 | `slicegpt` | `PASS` | [388 KB](slicegpt/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Slicegpt |
| 173 | `slicegpt_pca` | `PASS` | [389 KB](slicegpt_pca/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Slicegpt Pca |
| 174 | `slim_compressor` | `PASS` | [389 KB](slim_compressor/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Slim Compressor |
| 175 | `smart_weighted_ensemble` | `EVALUATED` | [391 KB](smart_weighted_ensemble/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Smart Weighted Ensemble |
| 176 | `soa_attn` | `PASS` | [391 KB](soa_attn/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Soa Attn |
| 177 | `soa_fused_module` | `EVALUATED` | [391 KB](soa_fused_module/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Soa Fused Module |
| 178 | `soa_module` | `EVALUATED` | [391 KB](soa_module/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Soa Module |
| 179 | `soka` | `PASS` | [388 KB](soka/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Soka |
| 180 | `span_expanded_attention` | `REJECTED_EXPANSION` | [391 KB](span_expanded_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Span Expanded Attention |
| 181 | `sparsegpt` | `PASS` | [388 KB](sparsegpt/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Sparsegpt |
| 182 | `spectra_v2` | `REJECTED_EXPANSION` | [391 KB](spectra_v2/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Spectra V2 |
| 183 | `spectraformer_attention` | `FAIL` | [391 KB](spectraformer_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Spectraformer Attention |
| 184 | `spectral_mamba` | `FAIL` | [391 KB](spectral_mamba/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Spectral Mamba |
| 185 | `spectron` | `PASS` | [390 KB](spectron/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Spectron |
| 186 | `squeezing_heads` | `REJECTED_EXPANSION` | [391 KB](squeezing_heads/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Squeezing Heads |
| 187 | `squeezing_heads_distill` | `EVALUATED` | [391 KB](squeezing_heads_distill/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Squeezing Heads Distill |
| 188 | `squeezing_heads_distiller` | `EVALUATED` | [392 KB](squeezing_heads_distiller/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Squeezing Heads Distiller |
| 189 | `subspace_projection` | `PASS` | [390 KB](subspace_projection/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Subspace Projection |
| 190 | `svd_attention` | `REJECTED_EXPANSION` | [390 KB](svd_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Svd Attention |
| 191 | `svd_compressed` | `PASS` | [391 KB](svd_compressed/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Svd Compressed |
| 192 | `svd_linear_attention` | `PASS` | [1355 KB](svd_linear_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Svd Linear Attention |
| 193 | `svd_linear_decay` | `PASS` | [391 KB](svd_linear_decay/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Svd Linear Decay |
| 194 | `svd_llm_v2_attention` | `PASS` | [2684 KB](svd_llm_v2_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Svd Llm V2 Attention |
| 195 | `svd_mamba` | `EVALUATED` | [2513 KB](svd_mamba/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Svd Mamba |
| 196 | `svd_proj_attn` | `EVALUATED` | [391 KB](svd_proj_attn/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Svd Proj Attn |
| 197 | `svd_projection_attention` | `EVALUATED` | [390 KB](svd_projection_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Svd Projection Attention |
| 198 | `svd_sequence_attention` | `EVALUATED` | [392 KB](svd_sequence_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Svd Sequence Attention |
| 199 | `swift_svd` | `PASS` | [389 KB](swift_svd/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Swift Svd |
| 200 | `swift_svd_act` | `PASS` | [389 KB](swift_svd_act/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Swift Svd Act |
| 201 | `template_attention` | `REJECTED_EXPANSION` | [391 KB](template_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Template Attention |
| 202 | `tlinformer_attention` | `REJECTED_EXPANSION` | [391 KB](tlinformer_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Tlinformer Attention |
| 203 | `token_compressed_attention` | `REJECTED_EXPANSION` | [377 KB](token_compressed_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Token Compressed Attention |
| 204 | `tomoe` | `PASS` | [391 KB](tomoe/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Tomoe |
| 205 | `tpa_attention` | `REJECTED_EXPANSION` | [390 KB](tpa_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Tpa Attention |
| 206 | `tt` | `PASS` | [388 KB](tt/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Tt |
| 207 | `unified_rope` | `REJECTED_EXPANSION` | [384 KB](unified_rope/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Unified Rope |
| 208 | `uniql_compressor` | `PASS` | [389 KB](uniql_compressor/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Uniql Compressor |
| 209 | `upi_mamba` | `REJECTED_EXPANSION` | [389 KB](upi_mamba/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Upi Mamba |
| 210 | `upi_stuffed_mamba` | `EVALUATED` | [391 KB](upi_stuffed_mamba/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Upi Stuffed Mamba |
| 211 | `vig_attention` | `REJECTED_EXPANSION` | [390 KB](vig_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Vig Attention |
| 212 | `vspec` | `PASS` | [361 KB](vspec/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Vspec |
| 213 | `weighted_ensemble` | `REJECTED_EXPANSION` | [391 KB](weighted_ensemble/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Weighted Ensemble |
| 214 | `wgqa` | `REJECTED_EXPANSION` | [390 KB](wgqa/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Wgqa |
| 215 | `xattention` | `REJECTED_EXPANSION` | [390 KB](xattention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Xattention |
| 216 | `zerokv_neo` | `EVALUATED` | [392 KB](zerokv_neo/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Zerokv Neo |
| 217 | `zeros_attention` | `REJECTED_EXPANSION` | [391 KB](zeros_attention/main.pdf) | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli) | Zeros Attention |
