# 🔍 Deep Code Quality Audit Report: NeuralOps & Scripts

> **Scope**: Complete audit of all Python source files in `neuralops/` and `scripts/`.
> **Audit Date**: October 8, 2026
> **Status**: Read-only diagnostic assessment (Zero code modifications).

---

## 📊 Section 8: Summary Statistics

- **Total Python files**: 618
- **Total lines of code**: 249,519
- **Average file length**: 403 lines
- **Files > 800 lines (TOO LONG)**: 130
- **Files > 1500 lines (CRITICAL)**: 4
- **Dead code items found**: 434
- **Cross-file duplicate patterns**: 5 core primitives replicated across 100+ files
- **Structural issues identified**: 623 long functions (>100 lines), 110 long classes (>500 lines)
- **Deep nesting hot-spots (>= 6 indentation levels)**: 8111
- **Uncovered attention modules**: 25
- **Uncovered compression engines**: 59

---

## 📏 Section 1: File Size Report

Table of ALL files sorted by line count (largest first). Flag: `> 800 lines = TOO LONG`, `> 1500 lines = CRITICAL`.

| Rank | File | Lines | Classes | Functions | Status |
|:---|:---|:---|:---|:---|:---|
| 1 | `neuralops/benchmarks/run_quality_sweep.py` | 5,740 | 1 | 39 | **CRITICAL** |
| 2 | `neuralops/evaluation/eval_all_models_datasets.py` | 2,527 | 1 | 15 | **CRITICAL** |
| 3 | `neuralops/benchmarks/kd_quick_test.py` | 1,872 | 2 | 37 | **CRITICAL** |
| 4 | `neuralops/attentions/svd/svd_linear_attention.py` | 1,590 | 2 | 25 | **CRITICAL** |
| 5 | `neuralops/compression_engine/svd/adasvd_compressor.py` | 1,462 | 2 | 25 | **TOO LONG** |
| 6 | `neuralops/benchmarks/run_gpt2_sweep.py` | 1,459 | 1 | 18 | **TOO LONG** |
| 7 | `neuralops/core/multi_mode_benchmark.py` | 1,459 | 7 | 35 | **TOO LONG** |
| 8 | `neuralops/attentions/ssm/mamba_svd_flash.py` | 1,402 | 3 | 30 | **TOO LONG** |
| 9 | `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py` | 1,377 | 7 | 44 | **TOO LONG** |
| 10 | `neuralops/attentions/hybrid/spectron_fp8_attention.py` | 1,376 | 4 | 41 | **TOO LONG** |
| 11 | `neuralops/attentions/hybrid/split_head_hybrid_attention.py` | 1,315 | 3 | 31 | **TOO LONG** |
| 12 | `neuralops/compression_engine/low_rank/a3_compressor.py` | 1,304 | 2 | 20 | **TOO LONG** |
| 13 | `neuralops/attentions/svd/fused_svd_kernel.py` | 1,293 | 3 | 23 | **TOO LONG** |
| 14 | `neuralops/attentions/hybrid/fusion_attn.py` | 1,285 | 2 | 31 | **TOO LONG** |
| 15 | `neuralops/attentions/linear/chela_slay.py` | 1,281 | 7 | 50 | **TOO LONG** |
| 16 | `neuralops/attentions/linear/spectron.py` | 1,275 | 3 | 30 | **TOO LONG** |
| 17 | `neuralops/attentions/ssm/nexus.py` | 1,273 | 7 | 34 | **TOO LONG** |
| 18 | `neuralops/attentions/svd/svd_compressed.py` | 1,265 | 1 | 25 | **TOO LONG** |
| 19 | `neuralops/compression_engine/svd/adaptive_ranks.py` | 1,258 | 1 | 20 | **TOO LONG** |
| 20 | `neuralops/attentions/sparse/moe_deltakv_paged.py` | 1,227 | 3 | 25 | **TOO LONG** |
| 21 | `neuralops/attentions/hybrid/hybrid_local_kernel_attention.py` | 1,221 | 4 | 40 | **TOO LONG** |
| 22 | `neuralops/attentions/linear/performer_attention.py` | 1,214 | 2 | 21 | **TOO LONG** |
| 23 | `neuralops/attentions/hybrid/tri_branch_hybrid_attention.py` | 1,210 | 4 | 39 | **TOO LONG** |
| 24 | `neuralops/attentions/ssm/orion.py` | 1,199 | 3 | 28 | **TOO LONG** |
| 25 | `neuralops/attentions/gated/gated_delta_net.py` | 1,190 | 2 | 33 | **TOO LONG** |
| 26 | `neuralops/core/triton_kernels/triton_svd_linear_attention.py` | 1,190 | 3 | 17 | **TOO LONG** |
| 27 | `neuralops/attentions/svd/triton_svd_linear_attention.py` | 1,188 | 3 | 17 | **TOO LONG** |
| 28 | `neuralops/analysis/generate_all_modules_analysis.py` | 1,177 | 0 | 7 | **TOO LONG** |
| 29 | `neuralops/attentions/sparse/reap_moe.py` | 1,173 | 4 | 22 | **TOO LONG** |
| 30 | `neuralops/core/replace_attention.py` | 1,173 | 2 | 34 | **TOO LONG** |
| 31 | `neuralops/attentions/linear/tpa_attention.py` | 1,149 | 2 | 18 | **TOO LONG** |
| 32 | `neuralops/attentions/adaptive/adaptive_span_attention.py` | 1,148 | 3 | 23 | **TOO LONG** |
| 33 | `neuralops/compression_engine/low_rank/come_compressor.py` | 1,137 | 4 | 22 | **TOO LONG** |
| 34 | `neuralops/attentions/hybrid/refined_landmark_attention.py` | 1,136 | 2 | 26 | **TOO LONG** |
| 35 | `neuralops/attentions/ssm/jamba_info.py` | 1,121 | 4 | 29 | **TOO LONG** |
| 36 | `neuralops/compression_engine/kv_cache/cachenotes_compressor.py` | 1,121 | 5 | 20 | **TOO LONG** |
| 37 | `neuralops/attentions/sparse/cluster_attention.py` | 1,115 | 2 | 11 | **TOO LONG** |
| 38 | `neuralops/attentions/linear/linformer_attention.py` | 1,112 | 2 | 31 | **TOO LONG** |
| 39 | `neuralops/attentions/hybrid/landmark_local_attention.py` | 1,099 | 2 | 23 | **TOO LONG** |
| 40 | `neuralops/attentions/ssm/vspec.py` | 1,092 | 5 | 30 | **TOO LONG** |
| 41 | `neuralops/attentions/svd/asvd_attention.py` | 1,091 | 2 | 25 | **TOO LONG** |
| 42 | `neuralops/attentions/adaptive/token_compressed_attention.py` | 1,083 | 2 | 18 | **TOO LONG** |
| 43 | `neuralops/attentions/sparse/adaptive_sparse_attention.py` | 1,082 | 2 | 19 | **TOO LONG** |
| 44 | `neuralops/attentions/adaptive/block_transformer_hiba.py` | 1,080 | 2 | 27 | **TOO LONG** |
| 45 | `neuralops/attentions/adaptive/squeezing_heads_distill.py` | 1,079 | 2 | 24 | **TOO LONG** |
| 46 | `neuralops/attentions/ssm/goldfinch_ahn.py` | 1,079 | 3 | 34 | **TOO LONG** |
| 47 | `neuralops/attentions/ssm/hymba_transx.py` | 1,078 | 3 | 28 | **TOO LONG** |
| 48 | `neuralops/core/attention_utils.py` | 1,078 | 0 | 10 | **TOO LONG** |
| 49 | `neuralops/attentions/fast_kernels/flash_attention3_fp8.py` | 1,075 | 3 | 25 | **TOO LONG** |
| 50 | `neuralops/attentions/sparse/bam_umoe.py` | 1,075 | 4 | 26 | **TOO LONG** |
| 51 | `neuralops/compression_engine/svd/dynamic_rank.py` | 1,072 | 4 | 21 | **TOO LONG** |
| 52 | `neuralops/attentions/adaptive/eat.py` | 1,067 | 2 | 25 | **TOO LONG** |
| 53 | `neuralops/attentions/fast_kernels/xattention.py` | 1,063 | 2 | 27 | **TOO LONG** |
| 54 | `neuralops/attentions/sparse/lsh_attention.py` | 1,063 | 2 | 21 | **TOO LONG** |
| 55 | `neuralops/attentions/svd/adaptive_svd_attention.py` | 1,060 | 2 | 22 | **TOO LONG** |
| 56 | `neuralops/data/benchmark_datasets.py` | 1,052 | 3 | 24 | **TOO LONG** |
| 57 | `neuralops/attentions/gated/decay_linear_attention.py` | 1,046 | 2 | 27 | **TOO LONG** |
| 58 | `neuralops/attentions/svd/salt_attention.py` | 1,045 | 2 | 23 | **TOO LONG** |
| 59 | `neuralops/attentions/modern/nha_attention.py` | 1,034 | 2 | 17 | **TOO LONG** |
| 60 | `neuralops/compression_engine/tensor_decomp/kronecker_compressor.py` | 1,028 | 2 | 20 | **TOO LONG** |
| 61 | `neuralops/attentions/sparse/moba_attention.py` | 1,027 | 4 | 23 | **TOO LONG** |
| 62 | `neuralops/attentions/hybrid/mixattention_moska.py` | 1,025 | 3 | 29 | **TOO LONG** |
| 63 | `neuralops/attentions/hybrid/positive_kernel_linear_attention.py` | 1,018 | 2 | 23 | **TOO LONG** |
| 64 | `neuralops/attentions/linear/rpc_attention.py` | 1,018 | 2 | 26 | **TOO LONG** |
| 65 | `neuralops/attentions/svd/svd_mamba.py` | 1,017 | 2 | 19 | **TOO LONG** |
| 66 | `neuralops/attentions/sparse/expert_choice_moe.py` | 1,016 | 3 | 27 | **TOO LONG** |
| 67 | `neuralops/attentions/linear/tlinformer_attention.py` | 1,010 | 2 | 27 | **TOO LONG** |
| 68 | `neuralops/attentions/linear/galerkin_attention.py` | 1,006 | 3 | 26 | **TOO LONG** |
| 69 | `neuralops/attentions/modern/gqa_attention.py` | 1,001 | 2 | 29 | **TOO LONG** |
| 70 | `neuralops/attentions/ssm/spectral_mamba.py` | 995 | 5 | 17 | **TOO LONG** |
| 71 | `neuralops/compression_engine/kv_cache/aatc_xkv_compressor.py` | 995 | 3 | 33 | **TOO LONG** |
| 72 | `neuralops/compression_engine/pruning/admm_pruner.py` | 992 | 3 | 14 | **TOO LONG** |
| 73 | `neuralops/attentions/ssm/upi_stuffed_mamba.py` | 990 | 2 | 19 | **TOO LONG** |
| 74 | `neuralops/attentions/svd/svd_llm_v2_attention.py` | 986 | 1 | 12 | **TOO LONG** |
| 75 | `neuralops/attentions/linear/nasc_attention.py` | 985 | 2 | 18 | **TOO LONG** |
| 76 | `neuralops/attentions/modern/mla_attention.py` | 985 | 3 | 25 | **TOO LONG** |
| 77 | `neuralops/attentions/fast_kernels/flash_attention3.py` | 982 | 3 | 25 | **TOO LONG** |
| 78 | `neuralops/attentions/svd/factorised_matrix_state.py` | 982 | 2 | 28 | **TOO LONG** |
| 79 | `neuralops/runners/run_monitor_server.py` | 981 | 1 | 13 | **TOO LONG** |
| 80 | `neuralops/attentions/hybrid/regla_attention.py` | 979 | 2 | 12 | **TOO LONG** |
| 81 | `neuralops/attentions/modern/wgqa_attention.py` | 977 | 2 | 26 | **TOO LONG** |
| 82 | `neuralops/attentions/modern/qcqa_attention.py` | 971 | 2 | 14 | **TOO LONG** |
| 83 | `neuralops/attentions/sparse/routing_attention.py` | 971 | 2 | 22 | **TOO LONG** |
| 84 | `neuralops/attentions/svd/svd_attention.py` | 971 | 2 | 17 | **TOO LONG** |
| 85 | `neuralops/compression_engine/pruning/armor_pruner.py` | 971 | 2 | 14 | **TOO LONG** |
| 86 | `neuralops/attentions/fast_kernels/soa_attn.py` | 969 | 4 | 27 | **TOO LONG** |
| 87 | `neuralops/attentions/adaptive/lola_attention.py` | 965 | 3 | 18 | **TOO LONG** |
| 88 | `neuralops/attentions/fast_kernels/adasplash_attention.py` | 952 | 2 | 12 | **TOO LONG** |
| 89 | `neuralops/compression_engine/svd/aa_svd_compressor.py` | 949 | 2 | 19 | **TOO LONG** |
| 90 | `neuralops/compression_engine/tensor_decomp/tt_compressor.py` | 949 | 3 | 20 | **TOO LONG** |
| 91 | `neuralops/compression_engine/experimental/hashed_net_compressor.py` | 942 | 2 | 19 | **TOO LONG** |
| 92 | `neuralops/trainer/optimizers.py` | 940 | 3 | 24 | **TOO LONG** |
| 93 | `neuralops/attentions/ssm/samba_m3.py` | 939 | 3 | 23 | **TOO LONG** |
| 94 | `neuralops/attentions/ssm/mamba_block.py` | 938 | 1 | 13 | **TOO LONG** |
| 95 | `neuralops/attentions/linear/nystrom_attention.py` | 924 | 2 | 25 | **TOO LONG** |
| 96 | `neuralops/attentions/gated/gated_linear_attention.py` | 921 | 2 | 30 | **TOO LONG** |
| 97 | `neuralops/compression_engine/pruning/mamba_shedder.py` | 921 | 2 | 15 | **TOO LONG** |
| 98 | `neuralops/attentions/adaptive/progressive_token_pruning.py` | 916 | 3 | 15 | **TOO LONG** |
| 99 | `neuralops/attentions/hybrid/optimized_hybrid_attention.py` | 903 | 2 | 16 | **TOO LONG** |
| 100 | `neuralops/compression_engine/tensor_decomp/kronecker_tt_lora.py` | 897 | 2 | 19 | **TOO LONG** |
| 101 | `neuralops/attentions/gated/kda_attention.py` | 895 | 2 | 15 | **TOO LONG** |
| 102 | `neuralops/compression_engine/quantization/slicegpt_pca.py` | 885 | 2 | 13 | **TOO LONG** |
| 103 | `neuralops/attentions/linear/triton_cosformer.py` | 879 | 2 | 11 | **TOO LONG** |
| 104 | `neuralops/compression_engine/quantization/kmeans_quantize.py` | 879 | 3 | 19 | **TOO LONG** |
| 105 | `neuralops/attentions/sparse/padre_attention.py` | 878 | 2 | 12 | **TOO LONG** |
| 106 | `neuralops/attentions/ssm/phantom.py` | 877 | 3 | 16 | **TOO LONG** |
| 107 | `neuralops/core/triton_kernels/triton_cosformer.py` | 877 | 2 | 11 | **TOO LONG** |
| 108 | `neuralops/attentions/experimental/template_attention.py` | 874 | 2 | 17 | **TOO LONG** |
| 109 | `neuralops/compression_engine/low_rank/mlorq_compressor.py` | 873 | 2 | 16 | **TOO LONG** |
| 110 | `neuralops/attentions/svd/minimal_nova.py` | 870 | 2 | 18 | **TOO LONG** |
| 111 | `neuralops/compression_engine/quantization/astro_quantizer.py` | 869 | 3 | 15 | **TOO LONG** |
| 112 | `neuralops/attentions/hybrid/hybrid_attention.py` | 866 | 1 | 20 | **TOO LONG** |
| 113 | `neuralops/utils/compat.py` | 864 | 1 | 20 | **TOO LONG** |
| 114 | `neuralops/attentions/modern/vig_attention.py` | 860 | 2 | 26 | **TOO LONG** |
| 115 | `neuralops/attentions/linear/spectraformer_attention.py` | 859 | 2 | 11 | **TOO LONG** |
| 116 | `neuralops/attentions/svd/lora_attention.py` | 859 | 2 | 22 | **TOO LONG** |
| 117 | `neuralops/main.py` | 855 | 1 | 13 | **TOO LONG** |
| 118 | `neuralops/attentions/svd/svd_projection_attention.py` | 852 | 2 | 25 | **TOO LONG** |
| 119 | `neuralops/results_saver/saver.py` | 852 | 1 | 40 | **TOO LONG** |
| 120 | `neuralops/compression_engine/quantization/faq_quantizer.py` | 851 | 2 | 14 | **TOO LONG** |
| 121 | `neuralops/attentions/linear/rala_attention.py` | 839 | 2 | 10 | **TOO LONG** |
| 122 | `neuralops/attentions/linear/primal_attention.py` | 836 | 2 | 10 | **TOO LONG** |
| 123 | `neuralops/compression_engine/low_rank/modegpt_compressor.py` | 829 | 2 | 16 | **TOO LONG** |
| 124 | `neuralops/attentions/adaptive/span_expanded_attention.py` | 828 | 2 | 12 | **TOO LONG** |
| 125 | `neuralops/compression_engine/kv_cache/kvtc_compressor.py` | 824 | 4 | 20 | **TOO LONG** |
| 126 | `neuralops/attentions/svd/minimal_svd.py` | 822 | 2 | 18 | **TOO LONG** |
| 127 | `neuralops/attentions/modern/unified_rope.py` | 812 | 2 | 12 | **TOO LONG** |
| 128 | `neuralops/attentions/sparse/nsa_attention.py` | 812 | 2 | 11 | **TOO LONG** |
| 129 | `neuralops/benchmarks/run_all_modules_scheduler.py` | 810 | 0 | 13 | **TOO LONG** |
| 130 | `neuralops/attentions/linear/spectra_v2.py` | 809 | 2 | 13 | **TOO LONG** |
| 131 | `neuralops/compression_engine/low_rank/frandi_compressor.py` | 798 | 2 | 19 | **OK** |
| 132 | `neuralops/compression_engine/svd/swift_svd_act.py` | 796 | 1 | 14 | **OK** |
| 133 | `neuralops/attentions/dense/zeros_attention.py` | 789 | 2 | 12 | **OK** |
| 134 | `neuralops/attentions/svd/residual_svd_attention.py` | 788 | 1 | 15 | **OK** |
| 135 | `neuralops/attentions/sparse/hamming_attention.py` | 787 | 2 | 13 | **OK** |
| 136 | `neuralops/compression_engine/pruning/ast_pruner.py` | 785 | 2 | 16 | **OK** |
| 137 | `neuralops/compression_engine/low_rank/svd_llm_compressor.py` | 770 | 2 | 18 | **OK** |
| 138 | `neuralops/compression_engine/low_rank/tomoe_compressor.py` | 767 | 2 | 13 | **OK** |
| 139 | `neuralops/analysis/comparison/statistical_tests.py` | 761 | 0 | 25 | **OK** |
| 140 | `neuralops/attentions/svd/compressed_mha_attention.py` | 757 | 1 | 11 | **OK** |
| 141 | `neuralops/attentions/gated/hgrn2_gated_outer.py` | 747 | 2 | 24 | **OK** |
| 142 | `neuralops/attentions/ssm/griffin_ssd.py` | 741 | 1 | 13 | **OK** |
| 143 | `neuralops/compression_engine/svd/optimal_subspace_projection.py` | 731 | 2 | 12 | **OK** |
| 144 | `neuralops/attentions/svd/compressed_attention.py` | 727 | 2 | 10 | **OK** |
| 145 | `neuralops/compression_engine/low_rank/lsc_compressor.py` | 722 | 2 | 12 | **OK** |
| 146 | `neuralops/compression_engine/quantization/rotational_slicegpt_int4.py` | 717 | 3 | 18 | **OK** |
| 147 | `neuralops/attentions/dense/full_attention.py` | 715 | 2 | 21 | **OK** |
| 148 | `neuralops/attentions/hybrid/granite_hybrid.py` | 711 | 3 | 15 | **OK** |
| 149 | `neuralops/attentions/adaptive/infini_attention.py` | 710 | 3 | 28 | **OK** |
| 150 | `neuralops/compression_engine/pruning/proxsparse.py` | 709 | 2 | 11 | **OK** |
| 151 | `neuralops/compression_engine/experimental/lillama_compressor.py` | 708 | 1 | 13 | **OK** |
| 152 | `neuralops/compression_engine/quantization/coopq_quantizer.py` | 705 | 3 | 13 | **OK** |
| 153 | `neuralops/attentions/ssm/jamba.py` | 696 | 3 | 27 | **OK** |
| 154 | `neuralops/compression_engine/low_rank/oats_compressor.py` | 692 | 2 | 13 | **OK** |
| 155 | `neuralops/benchmarks/core/arguments.py` | 687 | 0 | 1 | **OK** |
| 156 | `neuralops/analysis/generate_sweep_publication_artifacts.py` | 684 | 0 | 11 | **OK** |
| 157 | `neuralops/attentions/hybrid/halo_regla.py` | 680 | 2 | 13 | **OK** |
| 158 | `neuralops/compression_engine/experimental/flat_llm.py` | 680 | 2 | 13 | **OK** |
| 159 | `neuralops/runners/run_autonomous_48h_pipeline.py` | 677 | 0 | 9 | **OK** |
| 160 | `neuralops/attentions/gated/hgrn2_gated_layer.py` | 668 | 1 | 14 | **OK** |
| 161 | `neuralops/compression_engine/svd/dobi_svd.py` | 665 | 2 | 15 | **OK** |
| 162 | `neuralops/compression_engine/svd/subspace_projection.py` | 660 | 2 | 12 | **OK** |
| 163 | `neuralops/compression_engine/low_rank/nmf_compressor.py` | 646 | 2 | 11 | **OK** |
| 164 | `neuralops/compression_engine/experimental/liftr.py` | 645 | 3 | 16 | **OK** |
| 165 | `neuralops/attentions/gated/hla_attention.py` | 644 | 2 | 9 | **OK** |
| 166 | `neuralops/compression_engine/quantization/huffman_coder.py` | 639 | 3 | 12 | **OK** |
| 167 | `neuralops/compression_engine/pruning/bonsai_pruner.py` | 638 | 2 | 10 | **OK** |
| 168 | `neuralops/compression_engine/svd/gfws_d_compressor.py` | 638 | 2 | 12 | **OK** |
| 169 | `neuralops/compression_engine/low_rank/mgie_svd_compressor.py` | 634 | 1 | 8 | **OK** |
| 170 | `neuralops/compression_engine/kv_cache/delta_kv.py` | 620 | 2 | 10 | **OK** |
| 171 | `neuralops/compression_engine/quantization/slicegpt.py` | 618 | 1 | 10 | **OK** |
| 172 | `neuralops/compression_engine/svd/hybrid_svd_layer.py` | 617 | 3 | 10 | **OK** |
| 173 | `neuralops/core/triton_kernels/_common.py` | 604 | 0 | 8 | **OK** |
| 174 | `neuralops/utils/replace_attention.py` | 598 | 1 | 8 | **OK** |
| 175 | `neuralops/compression_engine/low_rank/salr.py` | 596 | 2 | 14 | **OK** |
| 176 | `neuralops/compression_engine/svd/entropy_budget_allocation.py` | 587 | 1 | 8 | **OK** |
| 177 | `neuralops/compression_engine/experimental/roast_hashed_linear.py` | 585 | 2 | 11 | **OK** |
| 178 | `neuralops/benchmarks/run_rank_sweeper.py` | 583 | 0 | 17 | **OK** |
| 179 | `neuralops/compression_engine/pruning/flap.py` | 582 | 1 | 10 | **OK** |
| 180 | `neuralops/compression_engine/low_rank/grecodec_compressor.py` | 581 | 2 | 11 | **OK** |
| 181 | `neuralops/compression_engine/svd/direct_svd.py` | 567 | 1 | 8 | **OK** |
| 182 | `neuralops/compression_engine/svd/erc_svd.py` | 562 | 1 | 7 | **OK** |
| 183 | `neuralops/utils/hf_git_exporter.py` | 560 | 0 | 8 | **OK** |
| 184 | `neuralops/compression_engine/low_rank/uniql_compressor.py` | 559 | 2 | 13 | **OK** |
| 185 | `neuralops/core/triton_kernels/fms_scan.py` | 558 | 1 | 5 | **OK** |
| 186 | `neuralops/core/pruning_utils.py` | 548 | 0 | 10 | **OK** |
| 187 | `neuralops/compression_engine/low_rank/sdrm_compressor.py` | 545 | 1 | 8 | **OK** |
| 188 | `neuralops/compression_engine/low_rank/cr_compressor.py` | 544 | 2 | 11 | **OK** |
| 189 | `neuralops/compression_engine/pruning/admm_compressor.py` | 541 | 1 | 7 | **OK** |
| 190 | `neuralops/analysis/plotting.py` | 538 | 0 | 17 | **OK** |
| 191 | `neuralops/compression_engine/svd/swift_svd.py` | 530 | 1 | 8 | **OK** |
| 192 | `neuralops/core/triton_kernels/gdn_scan.py` | 516 | 1 | 5 | **OK** |
| 193 | `neuralops/compression_engine/low_rank/cur_compressor.py` | 508 | 2 | 14 | **OK** |
| 194 | `neuralops/tests/test_svd_audit_suite.py` | 506 | 3 | 19 | **OK** |
| 195 | `neuralops/compression_engine/kv_cache/zerokv_neo_compressor.py` | 500 | 2 | 9 | **OK** |
| 196 | `neuralops/compression_engine/low_rank/enlighten_compressor.py` | 498 | 1 | 8 | **OK** |
| 197 | `neuralops/compression_engine/low_rank/compot_compressor.py` | 496 | 2 | 15 | **OK** |
| 198 | `neuralops/data/prepare_datasets.py` | 496 | 1 | 11 | **OK** |
| 199 | `neuralops/core/triton_kernels/gla_scan.py` | 495 | 1 | 5 | **OK** |
| 200 | `neuralops/attentions/linear/cosformer.py` | 494 | 2 | 23 | **OK** |
| 201 | `neuralops/benchmarks/run_long_context_sweep.py` | 494 | 0 | 6 | **OK** |
| 202 | `neuralops/compression_engine/low_rank/rpca_compressor.py` | 490 | 1 | 7 | **OK** |
| 203 | `neuralops/tests/test_sota_compression.py` | 486 | 5 | 26 | **OK** |
| 204 | `neuralops/benchmarks/run_ruler_benchmark.py` | 485 | 0 | 5 | **OK** |
| 205 | `neuralops/benchmarks/run_seqlen_scaling_bench.py` | 485 | 0 | 4 | **OK** |
| 206 | `neuralops/compression_engine/low_rank/cur_moe_compressor.py` | 477 | 1 | 7 | **OK** |
| 207 | `neuralops/trainer/utils.py` | 461 | 0 | 8 | **OK** |
| 208 | `neuralops/benchmarks/run_multiseed_rigor.py` | 460 | 0 | 5 | **OK** |
| 209 | `neuralops/compression_engine/svd/svd_linear.py` | 458 | 2 | 9 | **OK** |
| 210 | `neuralops/compression_engine/svd/residual_svd_compressor.py` | 457 | 5 | 14 | **OK** |
| 211 | `neuralops/core/triton_kernels/linear_attn.py` | 451 | 1 | 8 | **OK** |
| 212 | `neuralops/core/triton_kernels/selective_scan.py` | 450 | 1 | 5 | **OK** |
| 213 | `neuralops/attentions/gated/hgrn2_gated_deltanet.py` | 440 | 1 | 11 | **OK** |
| 214 | `neuralops/compression_engine/pruning/sparsegpt.py` | 437 | 1 | 9 | **OK** |
| 215 | `neuralops/attentions/ssm/samba_moe.py` | 424 | 2 | 12 | **OK** |
| 216 | `neuralops/core/legacy_ensemble.py` | 423 | 2 | 15 | **OK** |
| 217 | `neuralops/compression_engine/kv_cache/hybridkv_compressor.py` | 422 | 2 | 9 | **OK** |
| 218 | `neuralops/benchmarks/run_all_benchmarks.py` | 419 | 0 | 5 | **OK** |
| 219 | `neuralops/core/common.py` | 419 | 2 | 18 | **OK** |
| 220 | `neuralops/compression_engine/svd/fwsvd_compressor.py` | 416 | 1 | 6 | **OK** |
| 221 | `neuralops/core/registry.py` | 406 | 1 | 24 | **OK** |
| 222 | `neuralops/data/synthetic_generator.py` | 397 | 2 | 13 | **OK** |
| 223 | `neuralops/tests/test_aegis_v5.py` | 389 | 9 | 41 | **OK** |
| 224 | `neuralops/attentions/gated/gated_deltanet_block.py` | 387 | 1 | 15 | **OK** |
| 225 | `neuralops/evaluation/eval_perplexity.py` | 384 | 0 | 3 | **OK** |
| 226 | `neuralops/analysis/layerwise_analyzer.py` | 369 | 2 | 13 | **OK** |
| 227 | `neuralops/data/base.py` | 367 | 3 | 4 | **OK** |
| 228 | `neuralops/experiment_utils/main.py` | 367 | 3 | 12 | **OK** |
| 229 | `neuralops/experiment_utils/runner.py` | 367 | 3 | 12 | **OK** |
| 230 | `neuralops/tests/test_models_and_datasets_matrix.py` | 365 | 5 | 16 | **OK** |
| 231 | `neuralops/evaluation/run_lm_eval_harness.py` | 362 | 0 | 4 | **OK** |
| 232 | `neuralops/compression_engine/low_rank/soka_compressor.py` | 360 | 2 | 9 | **OK** |
| 233 | `neuralops/trainer/train_lra.py` | 360 | 1 | 6 | **OK** |
| 234 | `neuralops/tests/test_adasvd_hardened.py` | 359 | 6 | 23 | **OK** |
| 235 | `neuralops/tests/test_full_framework_end_to_end.py` | 353 | 5 | 16 | **OK** |
| 236 | `neuralops/evaluation/full_mnli_validation.py` | 343 | 0 | 4 | **OK** |
| 237 | `neuralops/attentions/ssm/simple_ssm.py` | 341 | 1 | 7 | **OK** |
| 238 | `neuralops/tests/test_adaptive_span_comprehensive.py` | 336 | 15 | 31 | **OK** |
| 239 | `neuralops/tests/test_fused_svd_kernel.py` | 325 | 0 | 12 | **OK** |
| 240 | `neuralops/compression_engine/low_rank/slim_compressor.py` | 324 | 1 | 6 | **OK** |
| 241 | `neuralops/benchmarks/run_multiseed.py` | 318 | 0 | 3 | **OK** |
| 242 | `neuralops/compression_engine/svd/saes_svd_v2.py` | 318 | 1 | 6 | **OK** |
| 243 | `neuralops/config/config.py` | 310 | 18 | 0 | **OK** |
| 244 | `neuralops/core/triton_kernels/state_update.py` | 306 | 1 | 6 | **OK** |
| 245 | `neuralops/tests/test_xattention_comprehensive.py` | 306 | 6 | 19 | **OK** |
| 246 | `neuralops/compression_engine/low_rank/drone_compressor.py` | 305 | 1 | 4 | **OK** |
| 247 | `neuralops/tests/test_cluster_comprehensive.py` | 295 | 15 | 29 | **OK** |
| 248 | `neuralops/tests/test_admm_pruner_distill.py` | 294 | 6 | 15 | **OK** |
| 249 | `neuralops/compression_engine/low_rank/__init__.py` | 293 | 0 | 0 | **OK** |
| 250 | `neuralops/core/triton_kernels/sparse_attn.py` | 291 | 1 | 6 | **OK** |
| 251 | `neuralops/attentions/ssm/selective_ssm.py` | 286 | 1 | 6 | **OK** |
| 252 | `neuralops/tests/test_all_compression_engines.py` | 285 | 2 | 7 | **OK** |
| 253 | `neuralops/tests/test_armor_pruner_hardened.py` | 283 | 6 | 20 | **OK** |
| 254 | `scripts/run_code_audit.py` | 283 | 0 | 1 | **OK** |
| 255 | `neuralops/benchmarks/run_seqlen_scaling.py` | 282 | 0 | 3 | **OK** |
| 256 | `neuralops/attentions/svd/projection_factorized_attn.py` | 277 | 1 | 4 | **OK** |
| 257 | `neuralops/tests/test_svd_linear_attention.py` | 276 | 6 | 18 | **OK** |
| 258 | `neuralops/tests/test_token_compressed_comprehensive.py` | 275 | 9 | 22 | **OK** |
| 259 | `neuralops/evaluation/run_baseline_comparison.py` | 274 | 0 | 2 | **OK** |
| 260 | `neuralops/tests/test_tpa_comprehensive.py` | 273 | 8 | 21 | **OK** |
| 261 | `neuralops/runners/unified_orchestrator.py` | 271 | 1 | 5 | **OK** |
| 262 | `neuralops/trainer/trainer.py` | 270 | 1 | 10 | **OK** |
| 263 | `neuralops/utils/qwen_compat.py` | 270 | 2 | 12 | **OK** |
| 264 | `neuralops/core/triton_kernels/triton_linear_attn.py` | 268 | 3 | 9 | **OK** |
| 265 | `neuralops/core/triton_kernels/hgrn2_scan.py` | 267 | 1 | 5 | **OK** |
| 266 | `neuralops/tests/test_all_203_models_comprehensive.py` | 265 | 3 | 9 | **OK** |
| 267 | `neuralops/utils/lora.py` | 265 | 1 | 14 | **OK** |
| 268 | `neuralops/core/triton_kernels/low_rank.py` | 258 | 1 | 6 | **OK** |
| 269 | `neuralops/tests/test_armor_hardened.py` | 258 | 6 | 15 | **OK** |
| 270 | `scripts/run_kd_all.py` | 258 | 0 | 3 | **OK** |
| 271 | `neuralops/attentions/svd/__init__.py` | 257 | 0 | 0 | **OK** |
| 272 | `neuralops/tracking/backends.py` | 257 | 6 | 29 | **OK** |
| 273 | `neuralops/benchmarks/gpu_live_probe.py` | 255 | 0 | 4 | **OK** |
| 274 | `neuralops/evaluation/fairness.py` | 255 | 2 | 14 | **OK** |
| 275 | `neuralops/tests/test_adaptive_ranks_extra.py` | 255 | 5 | 15 | **OK** |
| 276 | `neuralops/core/adapters.py` | 254 | 1 | 6 | **OK** |
| 277 | `neuralops/core/selective_scan.py` | 253 | 0 | 3 | **OK** |
| 278 | `neuralops/evaluation/retraining.py` | 251 | 4 | 7 | **OK** |
| 279 | `neuralops/tests/test_adaptive_sparse_comprehensive.py` | 251 | 13 | 26 | **OK** |
| 280 | `neuralops/experiment_utils/comparison.py` | 250 | 2 | 13 | **OK** |
| 281 | `neuralops/tests/test_adaptive_sparse_attention.py` | 250 | 12 | 21 | **OK** |
| 282 | `neuralops/tests/test_track2_spectral_ssm_audit.py` | 248 | 3 | 32 | **OK** |
| 283 | `neuralops/utils/model_family.py` | 247 | 1 | 5 | **OK** |
| 284 | `neuralops/tests/test_adaptive_svd_verification.py` | 245 | 2 | 9 | **OK** |
| 285 | `neuralops/tests/test_qwen3_5_compressor_awareness.py` | 245 | 3 | 11 | **OK** |
| 286 | `neuralops/tests/test_admm_hardened.py` | 244 | 4 | 12 | **OK** |
| 287 | `neuralops/results_saver/pipeline/result_writer.py` | 242 | 1 | 9 | **OK** |
| 288 | `neuralops/tests/test_lora.py` | 242 | 9 | 18 | **OK** |
| 289 | `neuralops/tests/test_lola_attention_comprehensive.py` | 241 | 12 | 28 | **OK** |
| 290 | `neuralops/trainer/custom_loop_trainer.py` | 237 | 2 | 5 | **OK** |
| 291 | `scripts/run_kd_reuse_model.py` | 234 | 0 | 2 | **OK** |
| 292 | `neuralops/core/model_wrapper.py` | 231 | 2 | 13 | **OK** |
| 293 | `neuralops/experiment_utils/tasks.py` | 230 | 2 | 13 | **OK** |
| 294 | `neuralops/__init__.py` | 229 | 2 | 6 | **OK** |
| 295 | `neuralops/tests/test_cachenotes_hardened.py` | 223 | 3 | 12 | **OK** |
| 296 | `neuralops/compression_engine/svd/__init__.py` | 221 | 0 | 0 | **OK** |
| 297 | `scripts/audit_causal_modules.py` | 220 | 0 | 2 | **OK** |
| 298 | `neuralops/benchmarks/run_one_module_isolated.py` | 219 | 0 | 1 | **OK** |
| 299 | `neuralops/tests/test_astro_hardened.py` | 219 | 3 | 12 | **OK** |
| 300 | `neuralops/tests/test_upi_mamba_comprehensive.py` | 219 | 7 | 15 | **OK** |
| 301 | `neuralops/tests/test_hybrid_attention.py` | 218 | 1 | 14 | **OK** |
| 302 | `neuralops/tests/test_adasplash_attention.py` | 213 | 11 | 28 | **OK** |
| 303 | `neuralops/data/glue_loader.py` | 212 | 0 | 6 | **OK** |
| 304 | `neuralops/attentions/ssm/__init__.py` | 209 | 0 | 0 | **OK** |
| 305 | `neuralops/evaluation/label_quality.py` | 208 | 2 | 6 | **OK** |
| 306 | `neuralops/tests/test_kvtc_hardened.py` | 208 | 3 | 13 | **OK** |
| 307 | `neuralops/core/t5_compat.py` | 205 | 1 | 8 | **OK** |
| 308 | `neuralops/core/architecture.py` | 202 | 0 | 6 | **OK** |
| 309 | `scripts/run_niah_all.py` | 201 | 0 | 4 | **OK** |
| 310 | `neuralops/attentions/linear/__init__.py` | 197 | 0 | 0 | **OK** |
| 311 | `neuralops/experiment_logging/logger.py` | 196 | 1 | 6 | **OK** |
| 312 | `neuralops/tests/test_triton_svd_linear_attention_comprehensive.py` | 196 | 4 | 12 | **OK** |
| 313 | `neuralops/tests/test_triton_cosformer_comprehensive.py` | 195 | 5 | 12 | **OK** |
| 314 | `neuralops/attentions/svd/svd_linear_decay.py` | 194 | 1 | 2 | **OK** |
| 315 | `neuralops/tests/test_triton_kernels_suite.py` | 194 | 0 | 9 | **OK** |
| 316 | `neuralops/utils/layer_norm_fix.py` | 193 | 2 | 8 | **OK** |
| 317 | `neuralops/core/base.py` | 190 | 2 | 10 | **OK** |
| 318 | `neuralops/tests/test_zeros_attention_comprehensive.py` | 190 | 1 | 9 | **OK** |
| 319 | `neuralops/experiment_utils/orchestrator.py` | 189 | 1 | 6 | **OK** |
| 320 | `neuralops/trainer/methods/lora.py` | 189 | 3 | 11 | **OK** |
| 321 | `neuralops/data/multi_choice.py` | 186 | 1 | 11 | **OK** |
| 322 | `neuralops/attentions/hybrid/__init__.py` | 185 | 0 | 0 | **OK** |
| 323 | `scripts/detailed_module_audit.py` | 185 | 0 | 2 | **OK** |
| 324 | `neuralops/tests/test_mamba_svd_flash_comprehensive.py` | 184 | 10 | 18 | **OK** |
| 325 | `neuralops/tests/test_jamba_comprehensive.py` | 183 | 10 | 18 | **OK** |
| 326 | `neuralops/tests/test_architecture.py` | 182 | 10 | 18 | **OK** |
| 327 | `neuralops/tests/test_goldfinch_ahn_comprehensive.py` | 182 | 10 | 18 | **OK** |
| 328 | `neuralops/tests/test_gated_delta_net_comprehensive.py` | 181 | 10 | 18 | **OK** |
| 329 | `neuralops/core/triton_kernels/triton_sparse_attn.py` | 180 | 2 | 6 | **OK** |
| 330 | `neuralops/trainer/results_manager.py` | 179 | 1 | 12 | **OK** |
| 331 | `neuralops/tests/test_kda_comprehensive.py` | 178 | 10 | 18 | **OK** |
| 332 | `neuralops/trainer/rl_trainer.py` | 177 | 1 | 4 | **OK** |
| 333 | `neuralops/tests/test_nha_comprehensive.py` | 175 | 10 | 17 | **OK** |
| 334 | `neuralops/tests/test_aegis_v2.py` | 174 | 8 | 27 | **OK** |
| 335 | `neuralops/tests/test_leka_v2_comprehensive.py` | 174 | 7 | 20 | **OK** |
| 336 | `neuralops/tests/test_performer_comprehensive.py` | 174 | 10 | 17 | **OK** |
| 337 | `neuralops/tests/test_flash3_fp8.py` | 173 | 6 | 16 | **OK** |
| 338 | `neuralops/tests/test_wgqa_comprehensive.py` | 170 | 2 | 10 | **OK** |
| 339 | `neuralops/data/cached_tensors.py` | 168 | 0 | 3 | **OK** |
| 340 | `neuralops/tests/test_fms.py` | 167 | 5 | 16 | **OK** |
| 341 | `neuralops/utils/profiler.py` | 167 | 1 | 12 | **OK** |
| 342 | `neuralops/runners/run_continuous_orchestrator.py` | 166 | 0 | 4 | **OK** |
| 343 | `neuralops/tests/test_leka_v3_comprehensive.py` | 166 | 7 | 19 | **OK** |
| 344 | `neuralops/core/triton_kernels/gqa.py` | 163 | 0 | 4 | **OK** |
| 345 | `neuralops/runners/run_master_pipeline.py` | 163 | 0 | 2 | **OK** |
| 346 | `neuralops/core/attention_mask.py` | 160 | 0 | 4 | **OK** |
| 347 | `neuralops/attentions/__init__.py` | 159 | 2 | 6 | **OK** |
| 348 | `neuralops/tests/test_adaptive_span.py` | 157 | 1 | 9 | **OK** |
| 349 | `neuralops/tests/test_adaptive_svd_attention.py` | 155 | 7 | 21 | **OK** |
| 350 | `neuralops/core/model_utils.py` | 154 | 0 | 8 | **OK** |
| 351 | `neuralops/tests/test_aegis_spectron_fp8.py` | 154 | 7 | 21 | **OK** |
| 352 | `neuralops/tests/test_vspec_comprehensive.py` | 153 | 2 | 10 | **OK** |
| 353 | `neuralops/tests/test_salt_f.py` | 150 | 2 | 10 | **OK** |
| 354 | `neuralops/attentions/sparse/__init__.py` | 149 | 0 | 0 | **OK** |
| 355 | `neuralops/tests/test_decay_linear_attention.py` | 149 | 8 | 18 | **OK** |
| 356 | `neuralops/tests/test_production_dataset_shapes.py` | 149 | 1 | 8 | **OK** |
| 357 | `neuralops/trainer/hyperopt.py` | 149 | 2 | 10 | **OK** |
| 358 | `neuralops/experiment_utils/config.py` | 148 | 1 | 3 | **OK** |
| 359 | `neuralops/tests/test_adaptive_ranks_jg.py` | 148 | 0 | 9 | **OK** |
| 360 | `neuralops/tests/test_aegis_hybrid.py` | 148 | 8 | 22 | **OK** |
| 361 | `neuralops/tests/test_weight_copy.py` | 147 | 7 | 13 | **OK** |
| 362 | `neuralops/benchmarks/run_whitened_svd.py` | 146 | 0 | 6 | **OK** |
| 363 | `neuralops/data/dataset_loader.py` | 146 | 1 | 7 | **OK** |
| 364 | `neuralops/tests/test_cur_compressor_comprehensive.py` | 145 | 3 | 12 | **OK** |
| 365 | `neuralops/tests/test_imports.py` | 145 | 4 | 11 | **OK** |
| 366 | `neuralops/utils/architecture.py` | 145 | 0 | 6 | **OK** |
| 367 | `neuralops/tests/test_recurrent_ssm_attentions.py` | 144 | 2 | 18 | **OK** |
| 368 | `neuralops/tests/test_vig_comprehensive.py` | 144 | 2 | 10 | **OK** |
| 369 | `neuralops/tests/test_muon_optimizer.py` | 141 | 2 | 9 | **OK** |
| 370 | `neuralops/trainer/methods/gelora.py` | 141 | 0 | 7 | **OK** |
| 371 | `neuralops/tests/test_enlighten.py` | 139 | 1 | 10 | **OK** |
| 372 | `neuralops/tests/test_nystrom_attention_comprehensive.py` | 139 | 5 | 14 | **OK** |
| 373 | `neuralops/trainer/methods/rdrop_r3f.py` | 139 | 1 | 6 | **OK** |
| 374 | `neuralops/tests/test_dobi_svd_comprehensive.py` | 138 | 3 | 13 | **OK** |
| 375 | `neuralops/tests/test_eat.py` | 138 | 5 | 15 | **OK** |
| 376 | `neuralops/tests/test_unified_rope_comprehensive.py` | 138 | 2 | 9 | **OK** |
| 377 | `neuralops/tests/test_template_attention_comprehensive.py` | 136 | 2 | 9 | **OK** |
| 378 | `neuralops/tracking/tracker.py` | 136 | 2 | 11 | **OK** |
| 379 | `neuralops/tests/test_numerical_stability.py` | 134 | 3 | 11 | **OK** |
| 380 | `neuralops/tests/test_hgrn2_layer_comprehensive.py` | 133 | 5 | 14 | **OK** |
| 381 | `neuralops/tests/test_cosformer.py` | 132 | 8 | 17 | **OK** |
| 382 | `neuralops/tests/test_huffman_coder.py` | 132 | 3 | 12 | **OK** |
| 383 | `neuralops/trainer/trainer_svd.py` | 132 | 0 | 4 | **OK** |
| 384 | `neuralops/tests/test_griffin_ssd_comprehensive.py` | 127 | 6 | 15 | **OK** |
| 385 | `neuralops/tests/test_samba_m3_comprehensive.py` | 127 | 2 | 9 | **OK** |
| 386 | `neuralops/tests/test_cosformer_comprehensive.py` | 126 | 6 | 16 | **OK** |
| 387 | `neuralops/tests/test_salt_attention_comprehensive.py` | 124 | 2 | 9 | **OK** |
| 388 | `neuralops/tests/test_svd_linear_attention_comprehensive.py` | 124 | 6 | 12 | **OK** |
| 389 | `neuralops/tests/test_multiarch_modules.py` | 123 | 1 | 11 | **OK** |
| 390 | `neuralops/trainer/hf_trainer.py` | 122 | 1 | 8 | **OK** |
| 391 | `neuralops/core/metrics.py` | 120 | 0 | 3 | **OK** |
| 392 | `neuralops/tests/test_rpc_comprehensive.py` | 120 | 2 | 9 | **OK** |
| 393 | `neuralops/tests/test_sweep_orchestration.py` | 120 | 1 | 6 | **OK** |
| 394 | `neuralops/tests/test_chela_slay.py` | 119 | 4 | 14 | **OK** |
| 395 | `neuralops/tests/test_chunked_recurrent_scans.py` | 115 | 1 | 7 | **OK** |
| 396 | `neuralops/attentions/adaptive/__init__.py` | 113 | 0 | 0 | **OK** |
| 397 | `neuralops/attentions/gated/__init__.py` | 113 | 0 | 0 | **OK** |
| 398 | `neuralops/compression_engine/pruning/__init__.py` | 113 | 0 | 0 | **OK** |
| 399 | `neuralops/trainer/metrics.py` | 113 | 0 | 12 | **OK** |
| 400 | `neuralops/tests/test_soa_attn_comprehensive.py` | 111 | 5 | 13 | **OK** |
| 401 | `scripts/test_nan.py` | 110 | 0 | 0 | **OK** |
| 402 | `neuralops/tests/test_bam_umoe_comprehensive.py` | 107 | 5 | 12 | **OK** |
| 403 | `neuralops/data/custom_loader.py` | 106 | 0 | 4 | **OK** |
| 404 | `neuralops/tests/test_qwen3_5_comprehensive.py` | 106 | 4 | 13 | **OK** |
| 405 | `neuralops/tests/test_expert_choice_moe_comprehensive.py` | 105 | 5 | 12 | **OK** |
| 406 | `neuralops/tests/test_nexus_comprehensive.py` | 104 | 5 | 12 | **OK** |
| 407 | `neuralops/tests/test_orion_comprehensive.py` | 104 | 5 | 12 | **OK** |
| 408 | `neuralops/tests/test_phantom_comprehensive.py` | 104 | 5 | 12 | **OK** |
| 409 | `neuralops/benchmarks/launch_parallel_sweep.py` | 103 | 0 | 1 | **OK** |
| 410 | `neuralops/tests/test_adaptive_svd_attention_comprehensive.py` | 103 | 5 | 12 | **OK** |
| 411 | `neuralops/attentions/svd/naive_svd.py` | 102 | 1 | 2 | **OK** |
| 412 | `neuralops/core/utils.py` | 102 | 0 | 10 | **OK** |
| 413 | `neuralops/tests/test_aasvd_exhaustive.py` | 102 | 2 | 8 | **OK** |
| 414 | `neuralops/tests/test_aegis_hybrid_comprehensive.py` | 102 | 5 | 12 | **OK** |
| 415 | `neuralops/tests/test_aegis_spectron_fp8_comprehensive.py` | 102 | 5 | 12 | **OK** |
| 416 | `neuralops/tests/test_compressed_attention_comprehensive.py` | 102 | 5 | 12 | **OK** |
| 417 | `neuralops/tests/test_compressed_mha_attention_comprehensive.py` | 102 | 5 | 12 | **OK** |
| 418 | `neuralops/tests/test_eat_comprehensive.py` | 102 | 5 | 12 | **OK** |
| 419 | `neuralops/tests/test_linformer_attention_comprehensive.py` | 102 | 5 | 12 | **OK** |
| 420 | `neuralops/compression_engine/quantization/__init__.py` | 101 | 0 | 0 | **OK** |
| 421 | `neuralops/tests/test_adasplash_attention_comprehensive.py` | 101 | 5 | 12 | **OK** |
| 422 | `neuralops/tests/test_aegis_v2_comprehensive.py` | 101 | 5 | 12 | **OK** |
| 423 | `neuralops/tests/test_aegis_v5_comprehensive.py` | 101 | 5 | 12 | **OK** |
| 424 | `neuralops/tests/test_asvd_attention_comprehensive.py` | 101 | 5 | 12 | **OK** |
| 425 | `neuralops/tests/test_block_transformer_hiba_comprehensive.py` | 101 | 5 | 12 | **OK** |
| 426 | `neuralops/tests/test_chela_slay_comprehensive.py` | 101 | 5 | 12 | **OK** |
| 427 | `neuralops/tests/test_cross_architecture_injection.py` | 101 | 5 | 9 | **OK** |
| 428 | `neuralops/tests/test_decay_linear_attention_comprehensive.py` | 101 | 5 | 12 | **OK** |
| 429 | `neuralops/tests/test_factorised_matrix_state_comprehensive.py` | 101 | 5 | 12 | **OK** |
| 430 | `neuralops/tests/test_flash_attention3_comprehensive.py` | 101 | 5 | 12 | **OK** |
| 431 | `neuralops/tests/test_flash_attention3_fp8_comprehensive.py` | 101 | 5 | 12 | **OK** |
| 432 | `neuralops/tests/test_full_attention_comprehensive.py` | 101 | 5 | 12 | **OK** |
| 433 | `neuralops/tests/test_svd_linear_decay_comprehensive.py` | 101 | 5 | 12 | **OK** |
| 434 | `neuralops/tests/test_t5_all_attention_compat.py` | 101 | 0 | 5 | **OK** |
| 435 | `neuralops/core/triton_kernels/__init__.py` | 100 | 0 | 0 | **OK** |
| 436 | `neuralops/tests/test_aegis_v6_comprehensive.py` | 100 | 5 | 12 | **OK** |
| 437 | `neuralops/tests/test_fusion_attn_comprehensive.py` | 100 | 5 | 12 | **OK** |
| 438 | `neuralops/tests/test_gated_deltanet_block_comprehensive.py` | 100 | 5 | 12 | **OK** |
| 439 | `neuralops/tests/test_projection_factorized_attn_comprehensive.py` | 100 | 5 | 12 | **OK** |
| 440 | `neuralops/tests/test_rala_attention_comprehensive.py` | 100 | 5 | 12 | **OK** |
| 441 | `neuralops/tests/test_svd_mamba_comprehensive.py` | 100 | 5 | 12 | **OK** |
| 442 | `neuralops/trainer/callbacks.py` | 100 | 5 | 10 | **OK** |
| 443 | `neuralops/trainer/methods/fisher.py` | 100 | 0 | 3 | **OK** |
| 444 | `neuralops/evaluation/run_targeted_b2_recovery.py` | 99 | 0 | 2 | **OK** |
| 445 | `neuralops/tests/test_fused_svd_kernel_comprehensive.py` | 99 | 5 | 12 | **OK** |
| 446 | `neuralops/tests/test_primal_attention_comprehensive.py` | 99 | 5 | 12 | **OK** |
| 447 | `neuralops/tests/test_res_svd_comprehensive.py` | 99 | 5 | 12 | **OK** |
| 448 | `neuralops/tests/test_routing_attention_comprehensive.py` | 99 | 5 | 12 | **OK** |
| 449 | `neuralops/tests/test_span_expanded_attention_comprehensive.py` | 99 | 5 | 12 | **OK** |
| 450 | `neuralops/tests/test_galerkin_attention_comprehensive.py` | 98 | 5 | 12 | **OK** |
| 451 | `neuralops/tests/test_gated_linear_attention_comprehensive.py` | 98 | 5 | 12 | **OK** |
| 452 | `neuralops/tests/test_hgrn2_outer_comprehensive.py` | 98 | 5 | 12 | **OK** |
| 453 | `neuralops/tests/test_hymba_transx_comprehensive.py` | 98 | 5 | 12 | **OK** |
| 454 | `neuralops/tests/test_lora_attention_comprehensive.py` | 98 | 5 | 12 | **OK** |
| 455 | `neuralops/tests/test_lsh_attention_comprehensive.py` | 98 | 5 | 12 | **OK** |
| 456 | `neuralops/tests/test_minimal_nova_comprehensive.py` | 98 | 5 | 12 | **OK** |
| 457 | `neuralops/tests/test_minimal_svd_comprehensive.py` | 98 | 5 | 12 | **OK** |
| 458 | `neuralops/tests/test_mixattention_moska_comprehensive.py` | 98 | 5 | 12 | **OK** |
| 459 | `neuralops/tests/test_mla_attention_comprehensive.py` | 98 | 5 | 12 | **OK** |
| 460 | `neuralops/tests/test_moba_attention_comprehensive.py` | 98 | 5 | 12 | **OK** |
| 461 | `neuralops/tests/test_nasc_attention_comprehensive.py` | 98 | 5 | 12 | **OK** |
| 462 | `neuralops/tests/test_nsa_attention_comprehensive.py` | 98 | 5 | 12 | **OK** |
| 463 | `neuralops/tests/test_optimized_hybrid_attention_comprehensive.py` | 98 | 5 | 12 | **OK** |
| 464 | `neuralops/tests/test_progressive_token_pruning_comprehensive.py` | 98 | 5 | 12 | **OK** |
| 465 | `neuralops/tests/test_qcqa_attention_comprehensive.py` | 98 | 5 | 12 | **OK** |
| 466 | `neuralops/tests/test_reap_moe_comprehensive.py` | 98 | 5 | 12 | **OK** |
| 467 | `neuralops/tests/test_regla_attention_comprehensive.py` | 98 | 5 | 12 | **OK** |
| 468 | `neuralops/core/__init__.py` | 97 | 0 | 1 | **OK** |
| 469 | `neuralops/tests/test_hgrn2_gated_deltanet_comprehensive.py` | 97 | 5 | 12 | **OK** |
| 470 | `neuralops/tests/test_hla_attention_comprehensive.py` | 97 | 5 | 12 | **OK** |
| 471 | `neuralops/tests/test_integration_smoke.py` | 97 | 1 | 8 | **OK** |
| 472 | `neuralops/tests/test_jamba_info_comprehensive.py` | 97 | 5 | 12 | **OK** |
| 473 | `neuralops/tests/test_leka_v1_comprehensive.py` | 97 | 5 | 12 | **OK** |
| 474 | `neuralops/analysis/report_generator.py` | 96 | 0 | 4 | **OK** |
| 475 | `neuralops/core/snapshot_restore.py` | 96 | 1 | 3 | **OK** |
| 476 | `neuralops/tests/test_moe_deltakv_paged_comprehensive.py` | 96 | 5 | 12 | **OK** |
| 477 | `scripts/run_niah_sequential.py` | 96 | 0 | 2 | **OK** |
| 478 | `neuralops/utils/helpers.py` | 95 | 0 | 9 | **OK** |
| 479 | `neuralops/config/compression_config.py` | 94 | 1 | 0 | **OK** |
| 480 | `neuralops/trainer/losses.py` | 94 | 6 | 12 | **OK** |
| 481 | `scripts/test_all_causality.py` | 94 | 0 | 1 | **OK** |
| 482 | `neuralops/core/mask_utils.py` | 93 | 0 | 4 | **OK** |
| 483 | `neuralops/tests/test_granite_hybrid_comprehensive.py` | 93 | 4 | 10 | **OK** |
| 484 | `neuralops/tests/test_threadsafe_rank_logging.py` | 90 | 1 | 5 | **OK** |
| 485 | `neuralops/compression_engine/__init__.py` | 89 | 2 | 6 | **OK** |
| 486 | `neuralops/tests/test_model_family_compat.py` | 89 | 2 | 11 | **OK** |
| 487 | `neuralops/analysis/metrics.py` | 88 | 1 | 7 | **OK** |
| 488 | `neuralops/tests/test_samba_moe_comprehensive.py` | 88 | 2 | 7 | **OK** |
| 489 | `neuralops/core/memory.py` | 87 | 1 | 4 | **OK** |
| 490 | `neuralops/core/svd_utils.py` | 87 | 0 | 3 | **OK** |
| 491 | `neuralops/tests/test_attention_modules_compat.py` | 87 | 2 | 6 | **OK** |
| 492 | `neuralops/tests/test_replace_comprehensive.py` | 87 | 4 | 9 | **OK** |
| 493 | `neuralops/experiment_utils/train.py` | 86 | 0 | 1 | **OK** |
| 494 | `neuralops/results_saver/config.py` | 86 | 1 | 1 | **OK** |
| 495 | `neuralops/tests/test_benchmark_datasets.py` | 86 | 0 | 3 | **OK** |
| 496 | `neuralops/tests/test_attentions.py` | 84 | 2 | 7 | **OK** |
| 497 | `neuralops/trainer/__init__.py` | 82 | 0 | 0 | **OK** |
| 498 | `neuralops/trainer/unsloth_trainer.py` | 82 | 1 | 4 | **OK** |
| 499 | `neuralops/benchmarks/core/stats.py` | 81 | 0 | 4 | **OK** |
| 500 | `neuralops/results_saver/backends/huggingface.py` | 79 | 0 | 1 | **OK** |
| 501 | `neuralops/attentions/modern/__init__.py` | 77 | 0 | 0 | **OK** |
| 502 | `neuralops/compression_engine/kv_cache/__init__.py` | 77 | 0 | 0 | **OK** |
| 503 | `neuralops/data/validation.py` | 77 | 2 | 2 | **OK** |
| 504 | `neuralops/tests/test_legacy_ensemble.py` | 76 | 1 | 5 | **OK** |
| 505 | `neuralops/trainer/methods/llrd.py` | 76 | 0 | 2 | **OK** |
| 506 | `neuralops/utils/results_saver.py` | 76 | 1 | 9 | **OK** |
| 507 | `neuralops/tests/test_attention_injection.py` | 74 | 1 | 3 | **OK** |
| 508 | `neuralops/analysis/__init__.py` | 73 | 0 | 0 | **OK** |
| 509 | `neuralops/tests/test_integration_all_families.py` | 73 | 3 | 9 | **OK** |
| 510 | `neuralops/data/__init__.py` | 72 | 0 | 0 | **OK** |
| 511 | `neuralops/tests/test_low_accuracy_modules_fixes.py` | 71 | 2 | 5 | **OK** |
| 512 | `neuralops/tests/test_t5_attention_compat.py` | 71 | 0 | 4 | **OK** |
| 513 | `neuralops/utils/metrics.py` | 71 | 0 | 7 | **OK** |
| 514 | `neuralops/analysis/utils.py` | 70 | 0 | 5 | **OK** |
| 515 | `neuralops/tests/test_file_structure.py` | 70 | 1 | 6 | **OK** |
| 516 | `neuralops/tests/test_hybrid_freeze.py` | 70 | 3 | 4 | **OK** |
| 517 | `neuralops/tests/test_gqa_comprehensive.py` | 69 | 3 | 8 | **OK** |
| 518 | `neuralops/tests/test_datasets.py` | 68 | 1 | 5 | **OK** |
| 519 | `neuralops/tests/test_registry.py` | 68 | 4 | 6 | **OK** |
| 520 | `neuralops/tests/test_svd_lowrank_comprehensive.py` | 68 | 1 | 3 | **OK** |
| 521 | `neuralops/data/glue.py` | 67 | 0 | 2 | **OK** |
| 522 | `neuralops/tests/test_hgrn2_comprehensive.py` | 66 | 3 | 8 | **OK** |
| 523 | `neuralops/utils/checkpoint.py` | 66 | 1 | 6 | **OK** |
| 524 | `neuralops/attentions/fast_kernels/__init__.py` | 65 | 0 | 0 | **OK** |
| 525 | `neuralops/compression_engine/experimental/__init__.py` | 65 | 0 | 0 | **OK** |
| 526 | `neuralops/core/gqa_utils.py` | 65 | 0 | 3 | **OK** |
| 527 | `neuralops/tests/test_halo_regla_comprehensive.py` | 65 | 3 | 8 | **OK** |
| 528 | `neuralops/trainer/schedulers.py` | 65 | 0 | 7 | **OK** |
| 529 | `neuralops/core/compression_engine.py` | 63 | 1 | 3 | **OK** |
| 530 | `scripts/queue_unified_runs.py` | 62 | 0 | 2 | **OK** |
| 531 | `neuralops/config/method_config.py` | 61 | 9 | 0 | **OK** |
| 532 | `neuralops/tests/test_part1_hierarchical_dedup.py` | 61 | 4 | 9 | **OK** |
| 533 | `neuralops/config/loader.py` | 60 | 0 | 4 | **OK** |
| 534 | `neuralops/tests/test_kernel_quantization_comprehensive.py` | 59 | 0 | 3 | **OK** |
| 535 | `neuralops/tests/test_moe_hybrid_comprehensive.py` | 59 | 1 | 1 | **OK** |
| 536 | `neuralops/trainer/methods/registry.py` | 59 | 3 | 9 | **OK** |
| 537 | `neuralops/tests/test_hamming_comprehensive.py` | 58 | 3 | 7 | **OK** |
| 538 | `neuralops/trainer/methods/mixout.py` | 58 | 1 | 4 | **OK** |
| 539 | `neuralops/data/task_mapper.py` | 57 | 0 | 3 | **OK** |
| 540 | `neuralops/runners/run_sweep.py` | 57 | 0 | 2 | **OK** |
| 541 | `neuralops/experiment_logging/callbacks.py` | 56 | 1 | 4 | **OK** |
| 542 | `neuralops/experiment_logging/integration.py` | 56 | 0 | 1 | **OK** |
| 543 | `neuralops/tests/test_mamba_chunked_scan.py` | 55 | 0 | 2 | **OK** |
| 544 | `neuralops/tests/test_parquet_and_storage.py` | 55 | 0 | 3 | **OK** |
| 545 | `neuralops/tests/test_multi_mode.py` | 54 | 1 | 4 | **OK** |
| 546 | `scripts/debug_nystrom_model.py` | 54 | 0 | 0 | **OK** |
| 547 | `neuralops/tests/test_results_saver.py` | 52 | 1 | 5 | **OK** |
| 548 | `neuralops/trainer/methods/__init__.py` | 52 | 0 | 0 | **OK** |
| 549 | `neuralops/benchmarks/__init__.py` | 51 | 0 | 0 | **OK** |
| 550 | `neuralops/config/__init__.py` | 51 | 0 | 0 | **OK** |
| 551 | `neuralops/evaluation/ab_testing.py` | 50 | 2 | 2 | **OK** |
| 552 | `neuralops/tests/test_compressors_all.py` | 48 | 1 | 4 | **OK** |
| 553 | `neuralops/tests/test_ssm_spectral_comprehensive.py` | 48 | 0 | 2 | **OK** |
| 554 | `neuralops/utils/config.py` | 47 | 0 | 4 | **OK** |
| 555 | `neuralops/tests/testing_utils.py` | 46 | 2 | 3 | **OK** |
| 556 | `neuralops/analysis/model_comparison.py` | 45 | 1 | 4 | **OK** |
| 557 | `neuralops/config/experiment_config.py` | 45 | 2 | 0 | **OK** |
| 558 | `neuralops/experiment_utils/model.py` | 45 | 2 | 3 | **OK** |
| 559 | `neuralops/experiment_utils/data.py` | 44 | 0 | 1 | **OK** |
| 560 | `neuralops/tests/test_attention_svd_family.py` | 43 | 1 | 3 | **OK** |
| 561 | `neuralops/tests/test_part7_numerical_safety.py` | 43 | 2 | 6 | **OK** |
| 562 | `neuralops/trainer/registry.py` | 43 | 1 | 5 | **OK** |
| 563 | `neuralops/compression_engine/tensor_decomp/__init__.py` | 41 | 0 | 0 | **OK** |
| 564 | `neuralops/experiment_logging/tracker.py` | 39 | 1 | 5 | **OK** |
| 565 | `neuralops/tests/test_attention_linear_family.py` | 39 | 1 | 3 | **OK** |
| 566 | `neuralops/tests/test_attention_ssm_family.py` | 39 | 1 | 3 | **OK** |
| 567 | `neuralops/tests/test_attention_standard_family.py` | 39 | 1 | 3 | **OK** |
| 568 | `neuralops/benchmarks/harvest_gpu_benchmark_data.py` | 38 | 0 | 1 | **OK** |
| 569 | `neuralops/tests/test_part2_weight_layout.py` | 38 | 0 | 4 | **OK** |
| 570 | `neuralops/evaluation/__init__.py` | 37 | 0 | 0 | **OK** |
| 571 | `neuralops/utils/logger.py` | 37 | 0 | 3 | **OK** |
| 572 | `neuralops/core/triton_kernels/softplus.py` | 36 | 0 | 2 | **OK** |
| 573 | `neuralops/tests/test_part5_compressor_lora_unmerged.py` | 35 | 1 | 3 | **OK** |
| 574 | `neuralops/tracking/callbacks.py` | 34 | 1 | 4 | **OK** |
| 575 | `neuralops/data/augmentation.py` | 33 | 0 | 3 | **OK** |
| 576 | `neuralops/data/preprocessing.py` | 33 | 0 | 2 | **OK** |
| 577 | `neuralops/tests/test_part8_recovery_targets.py` | 33 | 1 | 2 | **OK** |
| 578 | `neuralops/utils/__init__.py` | 33 | 0 | 0 | **OK** |
| 579 | `neuralops/tests/test_part3_leaf_parameter_isinstance.py` | 31 | 2 | 5 | **OK** |
| 580 | `neuralops/tests/test_part4_dynamic_lora_alias.py` | 31 | 0 | 3 | **OK** |
| 581 | `neuralops/attentions/dense/__init__.py` | 29 | 0 | 0 | **OK** |
| 582 | `neuralops/experiment_logging/config.py` | 27 | 1 | 0 | **OK** |
| 583 | `neuralops/benchmarks/core/__init__.py` | 26 | 0 | 0 | **OK** |
| 584 | `neuralops/runners/__init__.py` | 26 | 0 | 0 | **OK** |
| 585 | `neuralops/tests/test_part11_baseline_resolution.py` | 25 | 0 | 1 | **OK** |
| 586 | `neuralops/data/custom.py` | 24 | 0 | 1 | **OK** |
| 587 | `neuralops/tracking/__init__.py` | 24 | 0 | 0 | **OK** |
| 588 | `scripts/test_nystrom_direct.py` | 24 | 0 | 0 | **OK** |
| 589 | `neuralops/config/data_config.py` | 22 | 1 | 0 | **OK** |
| 590 | `neuralops/utils/paths.py` | 22 | 0 | 4 | **OK** |
| 591 | `neuralops/experiment_logging/__init__.py` | 21 | 0 | 0 | **OK** |
| 592 | `neuralops/experiment_utils/__init__.py` | 21 | 0 | 0 | **OK** |
| 593 | `neuralops/tests/test_part6_capability_declarations.py` | 20 | 0 | 3 | **OK** |
| 594 | `neuralops/tests/test_part9_output_tuple.py` | 19 | 1 | 2 | **OK** |
| 595 | `neuralops/attentions/experimental/__init__.py` | 17 | 0 | 0 | **OK** |
| 596 | `neuralops/core/config.py` | 17 | 0 | 2 | **OK** |
| 597 | `neuralops/data/prepare_cache_once.py` | 17 | 0 | 0 | **OK** |
| 598 | `neuralops/tests/test_part10_registry_and_consolidation.py` | 16 | 0 | 2 | **OK** |
| 599 | `neuralops/trainer/base_trainer.py` | 16 | 1 | 2 | **OK** |
| 600 | `neuralops/utils/seed.py` | 16 | 0 | 1 | **OK** |
| 601 | `neuralops/config/sweep_config.py` | 15 | 1 | 0 | **OK** |
| 602 | `neuralops/trainer/trainer_factory.py` | 15 | 0 | 1 | **OK** |
| 603 | `neuralops/tests/test_generalization_hardcoded_scan.py` | 14 | 0 | 1 | **OK** |
| 604 | `neuralops/tests/test_master_suite_all_203_models.py` | 14 | 1 | 1 | **OK** |
| 605 | `neuralops/config/hardware_config.py` | 13 | 1 | 0 | **OK** |
| 606 | `neuralops/config/model_config.py` | 13 | 1 | 0 | **OK** |
| 607 | `neuralops/utils/device.py` | 13 | 0 | 1 | **OK** |
| 608 | `neuralops/data/dataset_discovery.py` | 12 | 0 | 1 | **OK** |
| 609 | `neuralops/tests/test_attention_utils.py` | 12 | 0 | 1 | **OK** |
| 610 | `neuralops/core/triton_linear_attn.py` | 11 | 0 | 0 | **OK** |
| 611 | `neuralops/core/triton_sparse_attn.py` | 10 | 0 | 0 | **OK** |
| 612 | `neuralops/results_saver/__init__.py` | 9 | 0 | 0 | **OK** |
| 613 | `neuralops/results_saver/backends/__init__.py` | 9 | 0 | 0 | **OK** |
| 614 | `neuralops/evaluation/metrics.py` | 6 | 0 | 0 | **OK** |
| 615 | `neuralops/results_saver/pipeline/__init__.py` | 5 | 0 | 0 | **OK** |
| 616 | `neuralops/analysis/comparison/__init__.py` | 1 | 0 | 0 | **OK** |
| 617 | `neuralops/tests/__init__.py` | 0 | 0 | 0 | **OK** |
| 618 | `neuralops/utils/attention_utils.py` | 0 | 0 | 0 | **OK** |

---

## 💀 Section 2: Dead Code Report

### Unused Imports & Stale Redefinitions

| File | Dead Item | Type | Line | Action |
|:---|:---|:---|:---|:---|
| `neuralops/__init__.py` | `'typing.List'` | import | 11 | Delete |
| `neuralops/core/__init__.py` | `'typing.Dict'` | import | 11 | Delete |
| `neuralops/core/multi_mode_benchmark.py` | `'transformers.AutoConfig'` | import | 61 | Delete |
| `neuralops/core/multi_mode_benchmark.py` | `'transformers.AutoModel'` | import | 61 | Delete |
| `neuralops/core/multi_mode_benchmark.py` | `'transformers.AutoModelForCausalLM'` | import | 61 | Delete |
| `neuralops/core/multi_mode_benchmark.py` | `'transformers.DistilBertConfig'` | import | 61 | Delete |
| `neuralops/core/multi_mode_benchmark.py` | `'transformers.PreTrainedModel'` | import | 61 | Delete |
| `neuralops/core/multi_mode_benchmark.py` | `'neuralops.core.architecture.detect_architecture'` | import | 76 | Delete |
| `neuralops/core/multi_mode_benchmark.py` | `'neuralops.core.architecture.get_transformer_layers'` | import | 76 | Delete |
| `neuralops/core/multi_mode_benchmark.py` | `'neuralops.core.memory.get_gpu_memory_stats'` | import | 78 | Delete |
| `neuralops/core/multi_mode_benchmark.py` | `redefinition of unused 'DistilBertConfig' from line 61` | redefinition | 671 | Clean |
| `neuralops/core/attention_utils.py` | `redefinition of unused 'repeat_kv' from line 23` | redefinition | 425 | Clean |
| `neuralops/core/triton_kernels/triton_linear_attn.py` | `'math'` | import | 6 | Delete |
| `neuralops/core/triton_kernels/triton_sparse_attn.py` | `'torch.nn.functional as F'` | import | 8 | Delete |
| `neuralops/analysis/plotting.py` | `redefinition of unused 'plot_rank_allocation' from line 214` | redefinition | 494 | Clean |
| `neuralops/analysis/plotting.py` | `redefinition of unused 'plot_gradient_flow' from line 150` | redefinition | 538 | Clean |
| `neuralops/compression_engine/__init__.py` | `'os'` | import | 7 | Delete |
| `neuralops/compression_engine/experimental/__init__.py` | `'sys'` | import | 3 | Delete |
| `neuralops/compression_engine/tensor_decomp/__init__.py` | `'sys'` | import | 3 | Delete |
| `neuralops/compression_engine/quantization/__init__.py` | `'sys'` | import | 3 | Delete |
| `neuralops/compression_engine/svd/__init__.py` | `'sys'` | import | 3 | Delete |
| `neuralops/compression_engine/pruning/__init__.py` | `'sys'` | import | 3 | Delete |
| `neuralops/compression_engine/pruning/bonsai_pruner.py` | `'neuralops.core.pruning_utils.adaptive_max_error'` | import | 36 | Delete |
| `neuralops/compression_engine/pruning/bonsai_pruner.py` | `'neuralops.core.pruning_utils.get_calibration_tensors'` | import | 36 | Delete |
| `neuralops/compression_engine/kv_cache/__init__.py` | `'sys'` | import | 3 | Delete |
| `neuralops/compression_engine/low_rank/__init__.py` | `'sys'` | import | 3 | Delete |
| `neuralops/tests/test_performer_comprehensive.py` | `'pytest'` | import | 3 | Delete |
| `neuralops/tests/test_performer_comprehensive.py` | `'neuralops.attentions.performer_attention.LoRALinear'` | import | 6 | Delete |
| `neuralops/tests/test_performer_comprehensive.py` | `'neuralops.attentions.performer_attention.apply_rotary_pos_emb'` | import | 6 | Delete |
| `neuralops/tests/test_lora_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_cross_architecture_injection.py` | `'neuralops.attentions.asvd_attention.ASVDAttention'` | import | 9 | Delete |
| `neuralops/tests/test_cross_architecture_injection.py` | `'neuralops.attentions.cluster_attention.ClusterAttention'` | import | 10 | Delete |
| `neuralops/tests/test_cross_architecture_injection.py` | `'neuralops.attentions.factorised_matrix_state.FactorisedMatrixState'` | import | 11 | Delete |
| `neuralops/tests/test_cross_architecture_injection.py` | `'neuralops.attentions.gated_delta_net.inject_gated_delta_net'` | import | 12 | Delete |
| `neuralops/tests/test_cross_architecture_injection.py` | `'neuralops.attentions.hgrn2_gated_deltanet.HGRN2GatedDeltaNet'` | import | 14 | Delete |
| `neuralops/tests/test_cross_architecture_injection.py` | `'neuralops.attentions.hgrn2_gated_deltanet.inject_hgrn2_gated_deltanet'` | import | 14 | Delete |
| `neuralops/tests/test_cachenotes_hardened.py` | `'neuralops.compression_engine.cachenotes_compressor.CompressionPlanningTokens'` | import | 19 | Delete |
| `neuralops/tests/test_part9_output_tuple.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_granite_hybrid_comprehensive.py` | `'pytest'` | import | 1 | Delete |
| `neuralops/tests/test_moe_deltakv_paged_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_full_framework_end_to_end.py` | `'typing.Dict'` | import | 24 | Delete |
| `neuralops/tests/test_full_framework_end_to_end.py` | `'neuralops.compression_engine'` | import | 36 | Delete |
| `neuralops/tests/test_full_framework_end_to_end.py` | `'neuralops.utils.replace_attention.UniversalAttentionWrapper'` | import | 49 | Delete |
| `neuralops/tests/test_full_framework_end_to_end.py` | `'transformers.AutoConfig'` | import | 51 | Delete |
| `neuralops/tests/test_projection_factorized_attn_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_projection_factorized_attn_comprehensive.py` | `'neuralops.attentions.projection_factorized_attn.inject_projection_factorized_attention'` | import | 5 | Delete |
| `neuralops/tests/test_registry.py` | `'pytest'` | import | 3 | Delete |
| `neuralops/tests/test_weight_copy.py` | `'neuralops.utils.model_family.ModelAdapter'` | import | 11 | Delete |
| `neuralops/tests/test_adasplash_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_hymba_transx_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_compressed_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_attention_injection.py` | `'neuralops.attentions'` | import | 18 | Delete |
| `neuralops/tests/test_hybrid_freeze.py` | `'torch'` | import | 7 | Delete |
| `neuralops/tests/test_hybrid_freeze.py` | `'neuralops.attentions.gated_delta_net.GatedDeltaNet'` | import | 9 | Delete |
| `neuralops/tests/test_hybrid_freeze.py` | `'neuralops.attentions.hgrn2_gated_deltanet.HGRN2GatedDeltaNet'` | import | 10 | Delete |
| `neuralops/tests/test_hybrid_freeze.py` | `'neuralops.attentions.hgrn2_gated_layer.HGRN2Layer'` | import | 11 | Delete |
| `neuralops/tests/test_integration_all_families.py` | `'neuralops.compression_engine.aa_svd_compressor.AASVDCompressor'` | import | 5 | Delete |
| `neuralops/tests/test_integration_all_families.py` | `'neuralops.core.registry.is_module_declared_compatible'` | import | 7 | Delete |
| `neuralops/tests/test_galerkin_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_minimal_svd_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_adaptive_svd_verification.py` | `'neuralops.attentions.adaptive_svd_attention.AdaptiveSVDLinear'` | import | 24 | Delete |
| `neuralops/tests/test_adaptive_svd_verification.py` | `'neuralops.compression_engine.aa_svd_compressor.SVDLinear'` | import | 26 | Delete |
| `neuralops/tests/test_adasvd_hardened.py` | `'math'` | import | 1 | Delete |
| `neuralops/tests/test_adasvd_hardened.py` | `'pytest'` | import | 10 | Delete |
| `neuralops/tests/test_adasvd_hardened.py` | `'math'` | import | 288 | Delete |
| `neuralops/tests/test_adasvd_hardened.py` | `redefinition of unused 'math' from line 1` | redefinition | 288 | Clean |
| `neuralops/tests/test_rpc_comprehensive.py` | `'pytest'` | import | 3 | Delete |
| `neuralops/tests/test_rpc_comprehensive.py` | `'neuralops.attentions.rpc_attention.LoRALinear'` | import | 6 | Delete |
| `neuralops/tests/test_adaptive_span.py` | `'neuralops.attentions.adaptive_span_attention.inject_adaptive_span'` | import | 23 | Delete |
| `neuralops/tests/test_asvd_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_armor_pruner_hardened.py` | `'pytest'` | import | 20 | Delete |
| `neuralops/tests/test_armor_pruner_hardened.py` | `'neuralops.compression_engine.armor_pruner.compute_resilient_taylor_saliency'` | import | 23 | Delete |
| `neuralops/tests/test_factorised_matrix_state_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_chela_slay.py` | `'.testing_utils.DummyDistilBertAttention'` | import | 3 | Delete |
| `neuralops/tests/test_chela_slay.py` | `'types.SimpleNamespace'` | import | 4 | Delete |
| `neuralops/tests/test_chela_slay.py` | `'neuralops.attentions.chela_slay.LoRALinear'` | import | 9 | Delete |
| `neuralops/tests/test_nasc_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_span_expanded_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_span_expanded_attention_comprehensive.py` | `'neuralops.attentions.span_expanded_attention.inject_span_expanded_attention'` | import | 5 | Delete |
| `neuralops/tests/test_sota_compression.py` | `'typing.Dict'` | import | 14 | Delete |
| `neuralops/tests/test_sota_compression.py` | `'typing.List'` | import | 14 | Delete |
| `neuralops/tests/test_sota_compression.py` | `'typing.Tuple'` | import | 14 | Delete |
| `neuralops/tests/test_kda_comprehensive.py` | `'neuralops.attentions.kda_attention.LoRALinear'` | import | 6 | Delete |
| `neuralops/tests/test_kda_comprehensive.py` | `'neuralops.attentions.kda_attention.apply_rotary_pos_emb'` | import | 6 | Delete |
| `neuralops/tests/test_low_accuracy_modules_fixes.py` | `'neuralops.attentions.adaptive_svd_attention.AdaptiveSVDAttention'` | import | 13 | Delete |
| `neuralops/tests/test_res_svd_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_res_svd_comprehensive.py` | `'neuralops.attentions.res_svd.inject_res_svd_attention'` | import | 5 | Delete |
| `neuralops/tests/test_aasvd_exhaustive.py` | `'pytest'` | import | 7 | Delete |
| `neuralops/tests/test_aasvd_exhaustive.py` | `'neuralops.compression_engine.aa_svd_compressor.DEFAULT_CALIBRATION_CORPUS'` | import | 10 | Delete |
| `neuralops/tests/test_aasvd_exhaustive.py` | `'neuralops.compression_engine.aa_svd_compressor.aasvd_factory'` | import | 10 | Delete |
| `neuralops/tests/test_aasvd_exhaustive.py` | `'transformers.AutoTokenizer'` | import | 13 | Delete |
| `neuralops/tests/test_aasvd_exhaustive.py` | `'transformers.T5Config'` | import | 13 | Delete |
| `neuralops/tests/test_aasvd_exhaustive.py` | `'transformers.T5EncoderModel'` | import | 13 | Delete |
| `neuralops/tests/test_orion_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_datasets.py` | `'torch'` | import | 7 | Delete |
| `neuralops/tests/test_minimal_nova_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_svd_linear_attention_comprehensive.py` | `'pytest'` | import | 1 | Delete |
| `neuralops/tests/test_svd_linear_attention_comprehensive.py` | `'neuralops.attentions.svd_linear_attention.SVDLinear'` | import | 4 | Delete |
| `neuralops/tests/test_hgrn2_gated_deltanet_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_numerical_stability.py` | `'torch.nn'` | import | 7 | Delete |
| `neuralops/tests/test_integration_smoke.py` | `'torch.nn'` | import | 5 | Delete |
| `neuralops/tests/test_samba_m3_comprehensive.py` | `'pytest'` | import | 3 | Delete |
| `neuralops/tests/test_samba_m3_comprehensive.py` | `'neuralops.attentions.samba_m3.LoRALinear'` | import | 6 | Delete |
| `neuralops/tests/test_halo_regla_comprehensive.py` | `'pytest'` | import | 1 | Delete |
| `neuralops/tests/test_hybrid_attention.py` | `'math'` | import | 1 | Delete |
| `neuralops/tests/test_legacy_ensemble.py` | `'neuralops.attentions.legacy_ensemble.SmartWeightedEnsemble'` | import | 14 | Delete |
| `neuralops/tests/test_legacy_ensemble.py` | `'neuralops.attentions.legacy_ensemble.TaskAwareAttention'` | import | 14 | Delete |
| `neuralops/tests/test_leka_v1_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_hgrn2_comprehensive.py` | `'pytest'` | import | 1 | Delete |
| `neuralops/tests/test_hgrn2_comprehensive.py` | `'neuralops.attentions.hgrn2_gated_layer.inject_hgrn2_layer'` | import | 6 | Delete |
| `neuralops/tests/test_mla_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_griffin_ssd_comprehensive.py` | `'pytest'` | import | 1 | Delete |
| `neuralops/tests/test_mamba_chunked_scan.py` | `'torch.nn.functional as F'` | import | 2 | Delete |
| `neuralops/tests/test_progressive_token_pruning_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_progressive_token_pruning_comprehensive.py` | `'neuralops.attentions.progressive_token_pruning.inject_ptp'` | import | 5 | Delete |
| `neuralops/tests/test_full_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_eat_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_decay_linear_attention.py` | `'.testing_utils.DummyDistilBertAttention'` | import | 3 | Delete |
| `neuralops/tests/test_decay_linear_attention.py` | `'math'` | import | 4 | Delete |
| `neuralops/tests/test_decay_linear_attention.py` | `'types.SimpleNamespace'` | import | 5 | Delete |
| `neuralops/tests/test_zeros_attention_comprehensive.py` | `'neuralops.attentions.zeros_attention.LoRALinear'` | import | 6 | Delete |
| `neuralops/tests/test_moe_hybrid_comprehensive.py` | `'torch.nn'` | import | 5 | Delete |
| `neuralops/tests/test_aegis_v2_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_routing_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_routing_attention_comprehensive.py` | `'neuralops.attentions.routing_attention.inject_routing_attention'` | import | 5 | Delete |
| `neuralops/tests/test_svd_linear_attention.py` | `'math'` | import | 1 | Delete |
| `neuralops/tests/test_svd_linear_attention.py` | `'torch.nn.functional as F'` | import | 7 | Delete |
| `neuralops/tests/test_kvtc_hardened.py` | `'typing.Any'` | import | 28 | Delete |
| `neuralops/tests/test_kvtc_hardened.py` | `'typing.Dict'` | import | 28 | Delete |
| `neuralops/tests/test_kvtc_hardened.py` | `'neuralops.compression_engine.kvtc_compressor.KVTCStore'` | import | 38 | Delete |
| `neuralops/tests/test_kvtc_hardened.py` | `'neuralops.compression_engine.kvtc_compressor.SimpleEntropyCoder'` | import | 38 | Delete |
| `neuralops/tests/test_part10_registry_and_consolidation.py` | `'pathlib.Path'` | import | 2 | Delete |
| `neuralops/tests/test_part10_registry_and_consolidation.py` | `'pytest'` | import | 4 | Delete |
| `neuralops/tests/test_qcqa_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_qcqa_attention_comprehensive.py` | `'neuralops.attentions.qcqa_attention.inject_qcqa_attention'` | import | 5 | Delete |
| `neuralops/tests/test_imports.py` | `'pathlib.Path'` | import | 7 | Delete |
| `neuralops/tests/test_imports.py` | `'unittest.mock.MagicMock'` | import | 8 | Delete |
| `neuralops/tests/test_imports.py` | `'unittest.mock.Mock'` | import | 8 | Delete |
| `neuralops/tests/test_imports.py` | `'unittest.mock.patch'` | import | 8 | Delete |
| `neuralops/tests/test_imports.py` | `'neuralops.core'` | import | 105 | Delete |
| `neuralops/tests/test_imports.py` | `'neuralops.attentions'` | import | 113 | Delete |
| `neuralops/tests/test_imports.py` | `'neuralops.compression_engine'` | import | 122 | Delete |
| `neuralops/tests/test_imports.py` | `'neuralops.router'` | import | 130 | Delete |
| `neuralops/tests/test_imports.py` | `'neuralops.visualization'` | import | 138 | Delete |
| `neuralops/tests/test_hgrn2_outer_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_rala_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_rala_attention_comprehensive.py` | `'neuralops.attentions.rala_attention.inject_rala_attention'` | import | 5 | Delete |
| `neuralops/tests/test_rala_attention_comprehensive.py` | `'neuralops.attentions.rala_attention.inject_rala_linear_attention'` | import | 5 | Delete |
| `neuralops/tests/test_regla_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_regla_attention_comprehensive.py` | `'neuralops.attentions.regla_attention.inject_regla_attention'` | import | 5 | Delete |
| `neuralops/tests/test_attention_utils.py` | `'torch'` | import | 1 | Delete |
| `neuralops/tests/test_aegis_v5.py` | `'.testing_utils.DummyDistilBertAttention'` | import | 18 | Delete |
| `neuralops/tests/test_aegis_v5.py` | `'math'` | import | 19 | Delete |
| `neuralops/tests/test_reap_moe_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_reap_moe_comprehensive.py` | `'neuralops.attentions.reap_moe.inject_reap_moe_attention'` | import | 5 | Delete |
| `neuralops/tests/test_goldfinch_ahn_comprehensive.py` | `'neuralops.attentions.goldfinch_ahn.LoRALinear'` | import | 6 | Delete |
| `neuralops/tests/test_goldfinch_ahn_comprehensive.py` | `'neuralops.attentions.goldfinch_ahn.apply_rotary_pos_emb'` | import | 6 | Delete |
| `neuralops/tests/test_template_attention_comprehensive.py` | `'pytest'` | import | 3 | Delete |
| `neuralops/tests/test_template_attention_comprehensive.py` | `'neuralops.attentions.template_attention.LoRALinear'` | import | 6 | Delete |
| `neuralops/tests/test_hla_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_cur_compressor_comprehensive.py` | `'types'` | import | 1 | Delete |
| `neuralops/tests/test_cur_compressor_comprehensive.py` | `'pytest'` | import | 3 | Delete |
| `neuralops/tests/test_aegis_hybrid_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_part4_dynamic_lora_alias.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_part4_dynamic_lora_alias.py` | `'torch'` | import | 3 | Delete |
| `neuralops/tests/test_part4_dynamic_lora_alias.py` | `'torch.nn'` | import | 4 | Delete |
| `neuralops/tests/test_part4_dynamic_lora_alias.py` | `'neuralops.core.adapters.UniversalAdapter'` | import | 5 | Delete |
| `neuralops/tests/test_part4_dynamic_lora_alias.py` | `'transformers.T5Config'` | import | 8 | Delete |
| `neuralops/tests/test_part4_dynamic_lora_alias.py` | `'transformers.T5EncoderModel'` | import | 8 | Delete |
| `neuralops/tests/test_adaptive_sparse_attention.py` | `'math'` | import | 1 | Delete |
| `neuralops/tests/test_adaptive_sparse_attention.py` | `'neuralops.attentions.adaptive_sparse_attention.LoRALinear'` | import | 7 | Delete |
| `neuralops/tests/test_architecture.py` | `'neuralops.core.architecture.detect_architecture'` | import | 6 | Delete |
| `neuralops/tests/test_architecture.py` | `'neuralops.core.architecture.get_classifier_head'` | import | 6 | Delete |
| `neuralops/tests/test_architecture.py` | `'neuralops.core.architecture.get_embedding_module'` | import | 6 | Delete |
| `neuralops/tests/test_architecture.py` | `'neuralops.core.architecture.get_ffn_modules'` | import | 6 | Delete |
| `neuralops/tests/test_gated_deltanet_block_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_jamba_info_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_qwen3_5_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_armor_hardened.py` | `'math'` | import | 14 | Delete |
| `neuralops/tests/test_armor_hardened.py` | `'neuralops.utils.compat.get_recovery_params'` | import | 32 | Delete |
| `neuralops/tests/test_jamba_comprehensive.py` | `'neuralops.attentions.jamba.LoRALinear'` | import | 6 | Delete |
| `neuralops/tests/test_jamba_comprehensive.py` | `'neuralops.attentions.jamba.apply_rotary_pos_emb'` | import | 6 | Delete |
| `neuralops/tests/test_flash_attention3_fp8_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_generalization_hardcoded_scan.py` | `'pytest'` | import | 4 | Delete |
| `neuralops/tests/test_part11_baseline_resolution.py` | `'pytest'` | import | 5 | Delete |
| `neuralops/tests/test_cosformer_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_gqa_comprehensive.py` | `'pytest'` | import | 1 | Delete |
| `neuralops/tests/test_gqa_comprehensive.py` | `'neuralops.attentions.gqa_attention.LoRALinear'` | import | 4 | Delete |
| `neuralops/tests/test_file_structure.py` | `'os'` | import | 3 | Delete |
| `neuralops/tests/test_aegis_v6_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_benchmark_datasets.py` | `'pytest'` | import | 6 | Delete |
| `neuralops/tests/test_chunked_recurrent_scans.py` | `'math'` | import | 6 | Delete |
| `neuralops/tests/test_chunked_recurrent_scans.py` | `'torch.nn'` | import | 10 | Delete |
| `neuralops/tests/test_chunked_recurrent_scans.py` | `'torch.nn.functional as F'` | import | 11 | Delete |
| `neuralops/tests/test_chunked_recurrent_scans.py` | `'neuralops.attentions.gated_deltanet_block.GatedDeltaNetBlock'` | import | 14 | Delete |
| `neuralops/tests/test_chunked_recurrent_scans.py` | `'neuralops.attentions.hgrn2_gated_deltanet.HGRN2GatedDeltaNet'` | import | 15 | Delete |
| `neuralops/tests/test_chunked_recurrent_scans.py` | `'neuralops.attentions.hgrn2_gated_layer.HGRN2Layer'` | import | 16 | Delete |
| `neuralops/tests/test_salt_f.py` | `'math'` | import | 1 | Delete |
| `neuralops/tests/test_salt_f.py` | `'neuralops.attentions.salt_attention.inject_salt_attention'` | import | 9 | Delete |
| `neuralops/tests/test_part7_numerical_safety.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_part7_numerical_safety.py` | `'torch.nn'` | import | 4 | Delete |
| `neuralops/tests/test_flash_attention3_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_part6_capability_declarations.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_part6_capability_declarations.py` | `'neuralops.core.registry.get_module_capabilities'` | import | 3 | Delete |
| `neuralops/tests/test_decay_linear_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_hgrn2_layer_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_nha_comprehensive.py` | `'neuralops.attentions.nha_attention.LoRALinear'` | import | 6 | Delete |
| `neuralops/tests/test_nha_comprehensive.py` | `'neuralops.attentions.nha_attention.apply_rotary_pos_emb'` | import | 6 | Delete |
| `neuralops/tests/test_nexus_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_aegis_v2.py` | `'.testing_utils.DummyDistilBertAttention'` | import | 2 | Delete |
| `neuralops/tests/test_optimized_hybrid_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_phantom_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_upi_mamba_comprehensive.py` | `'pytest'` | import | 3 | Delete |
| `neuralops/tests/test_upi_mamba_comprehensive.py` | `'neuralops.attentions.upi_stuffed_mamba.LoRALinear'` | import | 6 | Delete |
| `neuralops/tests/test_linformer_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_unified_rope_comprehensive.py` | `'pytest'` | import | 3 | Delete |
| `neuralops/tests/test_unified_rope_comprehensive.py` | `'neuralops.attentions.unified_rope.LoRALinear'` | import | 6 | Delete |
| `neuralops/tests/test_triton_svd_linear_attention_comprehensive.py` | `'math'` | import | 1 | Delete |
| `neuralops/tests/test_triton_svd_linear_attention_comprehensive.py` | `'pytest'` | import | 5 | Delete |
| `neuralops/tests/test_nsa_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_svd_audit_suite.py` | `'typing.Any'` | import | 23 | Delete |
| `neuralops/tests/test_svd_audit_suite.py` | `'typing.Dict'` | import | 23 | Delete |
| `neuralops/tests/test_svd_audit_suite.py` | `'typing.List'` | import | 23 | Delete |
| `neuralops/tests/test_svd_audit_suite.py` | `'typing.Optional'` | import | 23 | Delete |
| `neuralops/tests/test_svd_audit_suite.py` | `'typing.Tuple'` | import | 23 | Delete |
| `neuralops/tests/test_svd_audit_suite.py` | `'neuralops.attentions.adaptive_svd_attention.AdaptiveSVDLinear'` | import | 34 | Delete |
| `neuralops/tests/test_svd_audit_suite.py` | `'neuralops.attentions.asvd_attention.ASVDLinear'` | import | 36 | Delete |
| `neuralops/tests/test_svd_audit_suite.py` | `'neuralops.attentions.svd_linear_attention.compute_exponential_decay'` | import | 38 | Delete |
| `neuralops/tests/test_svd_audit_suite.py` | `'neuralops.compression_engine.adaptive_ranks.adaptive_ranks_factory'` | import | 40 | Delete |
| `neuralops/tests/test_svd_audit_suite.py` | `'neuralops.compression_engine.direct_svd.direct_svd_factory'` | import | 42 | Delete |
| `neuralops/tests/test_svd_audit_suite.py` | `'neuralops.compression_engine.dobi_svd.DobiSVDLinearLayer'` | import | 44 | Delete |
| `neuralops/tests/test_svd_audit_suite.py` | `'neuralops.compression_engine.dobi_svd.dobi_factory'` | import | 44 | Delete |
| `neuralops/tests/test_svd_audit_suite.py` | `'neuralops.compression_engine.entropy_budget_allocation.entropy_budget_factory'` | import | 47 | Delete |
| `neuralops/tests/test_svd_audit_suite.py` | `'neuralops.compression_engine.erc_svd.erc_factory'` | import | 49 | Delete |
| `neuralops/tests/test_svd_audit_suite.py` | `'neuralops.compression_engine.kronecker_tt_lora.kronecker_tt_lora_factory'` | import | 50 | Delete |
| `neuralops/tests/test_svd_audit_suite.py` | `'neuralops.compression_engine.res_svd.SpectralResidualLoRA'` | import | 52 | Delete |
| `neuralops/tests/test_svd_audit_suite.py` | `'neuralops.compression_engine.res_svd.res_svd_factory'` | import | 52 | Delete |
| `neuralops/tests/test_leka_v3_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_vspec_comprehensive.py` | `'pytest'` | import | 3 | Delete |
| `neuralops/tests/test_vspec_comprehensive.py` | `'neuralops.attentions.vspec.DynamicLowRankProjector'` | import | 6 | Delete |
| `neuralops/tests/test_track2_spectral_ssm_audit.py` | `'.testing_utils.DummyDistilBertAttention'` | import | 12 | Delete |
| `neuralops/tests/test_track2_spectral_ssm_audit.py` | `'math'` | import | 13 | Delete |
| `neuralops/tests/test_nystrom_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_production_dataset_shapes.py` | `'pytest'` | import | 15 | Delete |
| `neuralops/tests/test_production_dataset_shapes.py` | `'neuralops.data.base.DataConfig'` | import | 17 | Delete |
| `neuralops/tests/test_production_dataset_shapes.py` | `'neuralops.data.base.TaskType'` | import | 17 | Delete |
| `neuralops/tests/test_production_dataset_shapes.py` | `'neuralops.data.base.create_dataloader'` | import | 17 | Delete |
| `neuralops/tests/test_production_dataset_shapes.py` | `'neuralops.data.synthetic_generator.generate_zipfian_tokens'` | import | 20 | Delete |
| `neuralops/tests/test_block_transformer_hiba_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_bam_umoe_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_aegis_v5_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_adaptive_svd_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_gated_linear_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_part5_compressor_lora_unmerged.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_attentions.py` | `'torch.nn'` | import | 6 | Delete |
| `neuralops/tests/test_attentions.py` | `'neuralops.attentions'` | import | 12 | Delete |
| `neuralops/tests/test_fms.py` | `'math'` | import | 17 | Delete |
| `neuralops/tests/test_fms.py` | `'neuralops.attentions.factorised_matrix_state.inject_factorised_matrix_state'` | import | 22 | Delete |
| `neuralops/tests/test_all_203_models_comprehensive.py` | `'os'` | import | 7 | Delete |
| `neuralops/tests/test_all_203_models_comprehensive.py` | `'neuralops.compression_engine'` | import | 18 | Delete |
| `neuralops/tests/test_svd_mamba_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_svd_mamba_comprehensive.py` | `'neuralops.attentions.svd_mamba.inject_SVD_Mamba'` | import | 5 | Delete |
| `neuralops/tests/test_svd_mamba_comprehensive.py` | `'neuralops.attentions.svd_mamba.inject_svd_mamba_attention'` | import | 5 | Delete |
| `neuralops/tests/test_lsh_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_triton_cosformer_comprehensive.py` | `'math'` | import | 1 | Delete |
| `neuralops/tests/test_triton_cosformer_comprehensive.py` | `'pytest'` | import | 5 | Delete |
| `neuralops/tests/test_qwen3_5_compressor_awareness.py` | `'pytest'` | import | 10 | Delete |
| `neuralops/tests/test_qwen3_5_compressor_awareness.py` | `'neuralops.compression_engine.a3_compressor.A3Compressor'` | import | 13 | Delete |
| `neuralops/tests/test_qwen3_5_compressor_awareness.py` | `'neuralops.compression_engine.aa_svd_compressor.AASVDCompressor'` | import | 14 | Delete |
| `neuralops/tests/test_qwen3_5_compressor_awareness.py` | `'neuralops.compression_engine.cur_moe_compressor.CURMoECompressor'` | import | 15 | Delete |
| `neuralops/tests/test_qwen3_5_compressor_awareness.py` | `'neuralops.compression_engine.dynamic_rank.DynamicEntropySVDCompressor'` | import | 17 | Delete |
| `neuralops/tests/test_qwen3_5_compressor_awareness.py` | `'neuralops.core.registry.COMPRESSOR_REGISTRY'` | import | 19 | Delete |
| `neuralops/tests/test_qwen3_5_compressor_awareness.py` | `'neuralops.utils.compat.is_decoder_family'` | import | 21 | Delete |
| `neuralops/tests/test_recurrent_ssm_attentions.py` | `'typing.Any'` | import | 12 | Delete |
| `neuralops/tests/test_recurrent_ssm_attentions.py` | `'neuralops.attentions'` | import | 15 | Delete |
| `neuralops/tests/test_gated_delta_net_comprehensive.py` | `'pytest'` | import | 3 | Delete |
| `neuralops/tests/test_gated_delta_net_comprehensive.py` | `'neuralops.attentions.gated_delta_net.LoRALinear'` | import | 6 | Delete |
| `neuralops/tests/test_gated_delta_net_comprehensive.py` | `'neuralops.attentions.gated_delta_net.apply_rotary_pos_emb'` | import | 6 | Delete |
| `neuralops/tests/test_compressors_all.py` | `'neuralops.core.registry.is_helper_module'` | import | 5 | Delete |
| `neuralops/tests/test_cluster_comprehensive.py` | `'neuralops.attentions.cluster_attention.LoRALinear'` | import | 6 | Delete |
| `neuralops/tests/test_parquet_and_storage.py` | `'pathlib.Path'` | import | 5 | Delete |
| `neuralops/tests/test_parquet_and_storage.py` | `'pytest'` | import | 7 | Delete |
| `neuralops/tests/test_dobi_svd_comprehensive.py` | `'pytest'` | import | 1 | Delete |
| `neuralops/tests/test_fused_svd_kernel.py` | `'typing.Any'` | import | 7 | Delete |
| `neuralops/tests/test_fused_svd_kernel.py` | `'pytest'` | import | 9 | Delete |
| `neuralops/tests/test_fused_svd_kernel.py` | `'neuralops.attentions.fused_svd_kernel.HAS_TRITON'` | import | 13 | Delete |
| `neuralops/tests/test_adaptive_ranks_jg.py` | `'neuralops.compression_engine.adaptive_ranks.adaptive_ranks_factory'` | import | 25 | Delete |
| `neuralops/tests/test_lola_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_chela_slay_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_enlighten.py` | `'pytest'` | import | 16 | Delete |
| `neuralops/tests/test_huffman_coder.py` | `'math'` | import | 6 | Delete |
| `neuralops/tests/test_samba_moe_comprehensive.py` | `'pytest'` | import | 3 | Delete |
| `neuralops/tests/test_samba_moe_comprehensive.py` | `'neuralops.attentions.samba_moe.MinimalMoEFallback'` | import | 6 | Delete |
| `neuralops/tests/test_replace_comprehensive.py` | `'torch'` | import | 4 | Delete |
| `neuralops/tests/test_replace_comprehensive.py` | `'neuralops.attentions.replace._replace_bert_attention'` | import | 6 | Delete |
| `neuralops/tests/test_replace_comprehensive.py` | `'neuralops.attentions.replace._replace_distilbert_attention'` | import | 6 | Delete |
| `neuralops/tests/test_replace_comprehensive.py` | `'neuralops.attentions.replace._replace_gpt_attention'` | import | 6 | Delete |
| `neuralops/tests/test_soa_attn_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_soa_attn_comprehensive.py` | `'neuralops.attentions.soa_attn.SOAFusedAttention'` | import | 5 | Delete |
| `neuralops/tests/test_soa_attn_comprehensive.py` | `'neuralops.attentions.soa_attn.inject_soa_attention'` | import | 5 | Delete |
| `neuralops/tests/test_soa_attn_comprehensive.py` | `'neuralops.attentions.soa_attn.inject_soafusedattention'` | import | 5 | Delete |
| `neuralops/tests/test_salt_attention_comprehensive.py` | `'pytest'` | import | 3 | Delete |
| `neuralops/tests/test_fusion_attn_comprehensive.py` | `'types.SimpleNamespace'` | import | 2 | Delete |
| `neuralops/tests/test_fusion_attn_comprehensive.py` | `'pytest'` | import | 4 | Delete |
| `neuralops/tests/test_models_and_datasets_matrix.py` | `'typing.List'` | import | 15 | Delete |
| `neuralops/tests/test_models_and_datasets_matrix.py` | `'numpy as np'` | import | 17 | Delete |
| `neuralops/tests/test_models_and_datasets_matrix.py` | `'pytest'` | import | 18 | Delete |
| `neuralops/tests/test_models_and_datasets_matrix.py` | `'neuralops.attentions.fused_svd_kernel.FusedSVDLinear'` | import | 22 | Delete |
| `neuralops/tests/test_models_and_datasets_matrix.py` | `'neuralops.attentions.fused_svd_kernel.fused_svd_linear'` | import | 22 | Delete |
| `neuralops/tests/test_models_and_datasets_matrix.py` | `'neuralops.compression_engine.aa_svd_compressor.AASVDCompressor'` | import | 25 | Delete |
| `neuralops/tests/test_models_and_datasets_matrix.py` | `'neuralops.core.attention_utils.split_and_set_qkv_weights'` | import | 29 | Delete |
| `neuralops/tests/test_models_and_datasets_matrix.py` | `'neuralops.data.benchmark_datasets.BenchmarkDatasetRegistry'` | import | 31 | Delete |
| `neuralops/tests/test_models_and_datasets_matrix.py` | `'neuralops.data.glue.load_glue_dataset'` | import | 33 | Delete |
| `neuralops/tests/test_models_and_datasets_matrix.py` | `'neuralops.data.glue_loader.get_num_labels_and_is_regression'` | import | 34 | Delete |
| `neuralops/tests/test_models_and_datasets_matrix.py` | `'neuralops.data.glue_loader.load_glue_local'` | import | 34 | Delete |
| `neuralops/tests/test_models_and_datasets_matrix.py` | `'neuralops.data.synthetic_generator.SyntheticPerplexityDataset'` | import | 37 | Delete |
| `neuralops/tests/test_all_compression_engines.py` | `'os'` | import | 6 | Delete |
| `neuralops/tests/test_all_compression_engines.py` | `'neuralops.compression_engine'` | import | 15 | Delete |
| `neuralops/tests/test_admm_hardened.py` | `'math'` | import | 14 | Delete |
| `neuralops/tests/test_mixattention_moska_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_triton_kernels_suite.py` | `'math'` | import | 15 | Delete |
| `neuralops/tests/test_triton_kernels_suite.py` | `'torch.nn'` | import | 19 | Delete |
| `neuralops/tests/test_triton_kernels_suite.py` | `'neuralops.core.triton_kernels.HAS_TRITON'` | import | 21 | Delete |
| `neuralops/tests/test_triton_kernels_suite.py` | `'neuralops.core.triton_kernels.linear_attn_pytorch_fallback'` | import | 21 | Delete |
| `neuralops/tests/test_triton_kernels_suite.py` | `'neuralops.core.triton_kernels.sparse_attn_pytorch_fallback'` | import | 21 | Delete |
| `neuralops/tests/test_triton_kernels_suite.py` | `'neuralops.core.triton_kernels.state_update_pytorch_fallback'` | import | 21 | Delete |
| `neuralops/tests/test_t5_attention_compat.py` | `'math'` | import | 1 | Delete |
| `neuralops/tests/test_expert_choice_moe_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_hamming_comprehensive.py` | `'pytest'` | import | 1 | Delete |
| `neuralops/tests/test_aegis_spectron_fp8_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_compressed_mha_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_vig_comprehensive.py` | `'pytest'` | import | 3 | Delete |
| `neuralops/tests/test_vig_comprehensive.py` | `'neuralops.attentions.vig_attention.LoRALinear'` | import | 6 | Delete |
| `neuralops/tests/test_threadsafe_rank_logging.py` | `'logging'` | import | 7 | Delete |
| `neuralops/tests/test_threadsafe_rank_logging.py` | `'time'` | import | 11 | Delete |
| `neuralops/tests/test_threadsafe_rank_logging.py` | `'neuralops.experiment_logging.logger.get_logger'` | import | 15 | Delete |
| `neuralops/tests/test_threadsafe_rank_logging.py` | `'neuralops.experiment_logging.logger.get_rank'` | import | 15 | Delete |
| `neuralops/tests/test_t5_all_attention_compat.py` | `'transformers.T5Config'` | import | 15 | Delete |
| `neuralops/tests/test_admm_pruner_distill.py` | `'math'` | import | 14 | Delete |
| `neuralops/tests/test_admm_pruner_distill.py` | `'neuralops.utils.compat.get_recovery_params'` | import | 31 | Delete |
| `neuralops/tests/test_aegis_hybrid.py` | `'.testing_utils.DummyDistilBertAttention'` | import | 2 | Delete |
| `neuralops/tests/test_part8_recovery_targets.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_part8_recovery_targets.py` | `'torch'` | import | 3 | Delete |
| `neuralops/tests/test_cosformer.py` | `'.testing_utils.DummyDistilBertAttention'` | import | 3 | Delete |
| `neuralops/tests/test_cosformer.py` | `'math'` | import | 4 | Delete |
| `neuralops/tests/test_cosformer.py` | `'types.SimpleNamespace'` | import | 5 | Delete |
| `neuralops/tests/test_primal_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_primal_attention_comprehensive.py` | `'neuralops.attentions.primal_attention.inject_primal_attention'` | import | 5 | Delete |
| `neuralops/tests/test_kernel_quantization_comprehensive.py` | `'.testing_utils.DummyDistilBertAttention'` | import | 1 | Delete |
| `neuralops/tests/test_flash3_fp8.py` | `'math'` | import | 15 | Delete |
| `neuralops/tests/test_part3_leaf_parameter_isinstance.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_mamba_svd_flash_comprehensive.py` | `'neuralops.attentions.mamba_svd_flash.LoRALinear'` | import | 6 | Delete |
| `neuralops/tests/test_mamba_svd_flash_comprehensive.py` | `'neuralops.attentions.mamba_svd_flash.apply_rotary_pos_emb'` | import | 6 | Delete |
| `neuralops/tests/test_wgqa_comprehensive.py` | `'pytest'` | import | 3 | Delete |
| `neuralops/tests/test_wgqa_comprehensive.py` | `'neuralops.attentions.wgqa_attention.LoRALinear'` | import | 6 | Delete |
| `neuralops/tests/test_leka_v2_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_token_compressed_comprehensive.py` | `'neuralops.attentions.token_compressed_attention.LoRALinear'` | import | 6 | Delete |
| `neuralops/tests/test_token_compressed_comprehensive.py` | `'neuralops.attentions.token_compressed_attention._rotate_half'` | import | 6 | Delete |
| `neuralops/tests/test_token_compressed_comprehensive.py` | `'neuralops.attentions.token_compressed_attention.apply_rotary_pos_emb'` | import | 6 | Delete |
| `neuralops/tests/test_moba_attention_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_fused_svd_kernel_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_fused_svd_kernel_comprehensive.py` | `'neuralops.attentions.fused_svd_kernel.fused_svd_linear'` | import | 5 | Delete |
| `neuralops/tests/test_tpa_comprehensive.py` | `'pytest'` | import | 3 | Delete |
| `neuralops/tests/test_tpa_comprehensive.py` | `'neuralops.attentions.tpa_attention.LoRALinear'` | import | 6 | Delete |
| `neuralops/tests/test_xattention_comprehensive.py` | `'pytest'` | import | 3 | Delete |
| `neuralops/tests/test_xattention_comprehensive.py` | `'neuralops.attentions.xattention.LoRALinear'` | import | 6 | Delete |
| `neuralops/tests/test_sweep_orchestration.py` | `'os'` | import | 4 | Delete |
| `neuralops/tests/test_sweep_orchestration.py` | `'unittest.mock.MagicMock'` | import | 7 | Delete |
| `neuralops/tests/test_sweep_orchestration.py` | `'unittest.mock.patch'` | import | 7 | Delete |
| `neuralops/tests/test_sweep_orchestration.py` | `'pytest'` | import | 9 | Delete |
| `neuralops/tests/test_sweep_orchestration.py` | `'torch.nn'` | import | 11 | Delete |
| `neuralops/tests/test_sweep_orchestration.py` | `'neuralops.core.registry.ATTENTION_REGISTRY'` | import | 12 | Delete |
| `neuralops/tests/test_sweep_orchestration.py` | `'neuralops.utils.compat.module_family_support'` | import | 14 | Delete |
| `neuralops/tests/test_adasplash_attention.py` | `'.testing_utils.DummyDistilBertAttention'` | import | 7 | Delete |
| `neuralops/tests/test_adasplash_attention.py` | `'typing.Any'` | import | 8 | Delete |
| `neuralops/tests/test_attention_modules_compat.py` | `'torch'` | import | 4 | Delete |
| `neuralops/tests/test_attention_modules_compat.py` | `'neuralops.core.registry.get_module_capabilities'` | import | 6 | Delete |
| `neuralops/tests/test_lora.py` | `'pytest'` | import | 6 | Delete |
| `neuralops/tests/test_svd_linear_decay_comprehensive.py` | `'pytest'` | import | 2 | Delete |
| `neuralops/tests/test_svd_linear_decay_comprehensive.py` | `'neuralops.attentions.svd_linear_decay.inject_svd_linear_decay_attention'` | import | 5 | Delete |
| `neuralops/attentions/__init__.py` | `'os'` | import | 7 | Delete |
| `neuralops/attentions/ssm/__init__.py` | `'sys'` | import | 3 | Delete |
| `neuralops/attentions/experimental/__init__.py` | `'sys'` | import | 3 | Delete |
| `neuralops/attentions/fast_kernels/__init__.py` | `'sys'` | import | 3 | Delete |
| `neuralops/attentions/linear/cosformer.py` | `'typing.Any'` | import | 18 | Delete |
| `neuralops/attentions/linear/cosformer.py` | `'neuralops.core.attention_mask.get_token_valid_mask'` | import | 26 | Delete |
| `neuralops/attentions/linear/__init__.py` | `'sys'` | import | 3 | Delete |
| `neuralops/attentions/linear/chela_slay.py` | `'neuralops.utils.lora.LoRALinear'` | import | 36 | Delete |
| `neuralops/attentions/linear/chela_slay.py` | `redefinition of unused '_rotate_half' from line 38` | redefinition | 137 | Clean |
| `neuralops/attentions/dense/__init__.py` | `'sys'` | import | 3 | Delete |
| `neuralops/attentions/svd/__init__.py` | `'sys'` | import | 3 | Delete |
| `neuralops/attentions/svd/svd_linear_attention.py` | `'neuralops.attentions.fused_svd_kernel.FusedSVDLinear'` | import | 65 | Delete |
| `neuralops/attentions/gated/hgrn2_gated_deltanet.py` | `'neuralops.attentions.gated_delta_net.LoRALinear'` | import | 21 | Delete |
| `neuralops/attentions/gated/hgrn2_gated_deltanet.py` | `'neuralops.core.adapters.UniversalAdapter'` | import | 23 | Delete |
| `neuralops/attentions/gated/__init__.py` | `'sys'` | import | 3 | Delete |
| `neuralops/attentions/sparse/__init__.py` | `'sys'` | import | 3 | Delete |
| `neuralops/attentions/modern/__init__.py` | `'sys'` | import | 3 | Delete |
| `neuralops/attentions/adaptive/__init__.py` | `'sys'` | import | 3 | Delete |
| `neuralops/attentions/hybrid/__init__.py` | `'sys'` | import | 3 | Delete |
| `neuralops/benchmarks/run_quality_sweep.py` | `redefinition of unused 'save_lora_weights' from line 4063` | redefinition | 4158 | Clean |
| `neuralops/benchmarks/run_gpt2_sweep.py` | `'neuralops.compression_engine'` | import | 218 | Delete |
| `neuralops/benchmarks/run_one_module_isolated.py` | `redefinition of unused 'run_single_module_main' from line 36` | redefinition | 75 | Clean |
| `neuralops/benchmarks/core/stats.py` | `'numpy as np'` | import | 1 | Delete |
| `neuralops/benchmarks/core/stats.py` | `'scipy.stats'` | import | 2 | Delete |
| `neuralops/benchmarks/core/stats.py` | `redefinition of unused 'np' from line 1` | redefinition | 6 | Clean |
| `neuralops/benchmarks/core/stats.py` | `redefinition of unused 'stats' from line 2` | redefinition | 9 | Clean |
| `neuralops/benchmarks/core/stats.py` | `redefinition of unused 'np' from line 1` | redefinition | 25 | Clean |
| `neuralops/benchmarks/core/stats.py` | `redefinition of unused 'np' from line 1` | redefinition | 46 | Clean |
| `neuralops/benchmarks/core/stats.py` | `redefinition of unused 'np' from line 1` | redefinition | 65 | Clean |
| `neuralops/evaluation/metrics.py` | `'neuralops.trainer.metrics.*'` | import | 6 | Delete |
| `neuralops/evaluation/run_lm_eval_harness.py` | `'lm_eval'` | import | 196 | Delete |
| `neuralops/runners/run_continuous_orchestrator.py` | `'glob'` | import | 11 | Delete |
| `neuralops/data/__init__.py` | `'.cached_tensors.get_cached_tokenized_dataset'` | import | 19 | Delete |
| `neuralops/trainer/hyperopt.py` | `'optuna.trial.TrialState'` | import | 9 | Delete |
| `neuralops/trainer/unsloth_trainer.py` | `'unsloth'` | import | 51 | Delete |
| `neuralops/trainer/rl_trainer.py` | `'trl'` | import | 65 | Delete |
| `neuralops/trainer/trainer.py` | `'neuralops.trainer.metrics.get_task_metrics as get_compute_metrics'` | import | 270 | Delete |
| `scripts/audit_causal_modules.py` | `'ast'` | import | 13 | Delete |
| `scripts/audit_causal_modules.py` | `'os'` | import | 15 | Delete |
| `scripts/detailed_module_audit.py` | `'ast'` | import | 12 | Delete |
| `scripts/queue_unified_runs.py` | `'os'` | import | 3 | Delete |
| `scripts/debug_nystrom_model.py` | `'transformers.AutoTokenizer'` | import | 3 | Delete |
| `scripts/test_all_causality.py` | `'neuralops.attentions'` | import | 29 | Delete |

### Empty Functions & Dead Stubs

| File | Stub Item | Type | Line | Action |
|:---|:---|:---|:---|:---|
| `neuralops/__init__.py` | `def exec_module()` | pass statement | 30 | Implement or Remove |
| `neuralops/attentions/__init__.py` | `def exec_module()` | pass statement | 117 | Implement or Remove |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | `def freeze_base_weights()` | pass statement | 273 | Implement or Remove |
| `neuralops/attentions/svd/svd_compressed.py` | `def set_seed()` | pass statement | 69 | Implement or Remove |
| `neuralops/benchmarks/run_quality_sweep.py` | `def unscale_()` | pass statement | 717 | Implement or Remove |
| `neuralops/benchmarks/run_quality_sweep.py` | `def update()` | pass statement | 719 | Implement or Remove |
| `neuralops/compression_engine/__init__.py` | `def exec_module()` | pass statement | 47 | Implement or Remove |
| `neuralops/compression_engine/low_rank/frandi_compressor.py` | `def unmerge()` | NotImplementedError stub | 201 | Implement or Remove |
| `neuralops/compression_engine/pruning/ast_pruner.py` | `def unmerge()` | pass statement | 171 | Implement or Remove |
| `neuralops/experiment_utils/main.py` | `def __init__()` | pass statement | 43 | Implement or Remove |
| `neuralops/experiment_utils/runner.py` | `def __init__()` | pass statement | 43 | Implement or Remove |

---

## 👯 Section 3: Duplicate Code Report

### Cross-File Boilerplate Replication

The audit revealed widespread copy-pasted utility blocks across attention and compression modules that should be unified in `neuralops.core`:

| Duplicate Component | Scope & Files Count | Typical Lines | Recommendation |
|:---|:---|:---|:---|
| `_rotate_half` | **106 attention files** | ~15 lines | Replace with `from neuralops.core.rope import _rotate_half` |
| `apply_rotary_pos_emb` | **102 attention files** | ~25 lines | Replace with `from neuralops.core.rope import apply_rotary_pos_emb` |
| `LoRALinear` | **68 module files** | ~80 lines | Replace with `from neuralops.core.common import LoRALinear` |
| `repeat_kv` | **19 attention files** | ~15 lines | Replace with `from neuralops.core.gqa_utils import repeat_kv` |
| `FactorizedLinear` | **4 hybrid files** | ~60 lines | Replace with `from neuralops.core.common import FactorizedLinear` |

## 🏗️ Section 4: Structural Problems

### Critical Large Classes (> 500 lines)

| File | Class Name | Line | Length | Severity |
|:---|:---|:---|:---|:---|
| `neuralops/compression_engine/svd/adaptive_ranks.py` | `AdaptiveRanksCompressor` | 118 | 1088 lines | **HIGH** |
| `neuralops/compression_engine/svd/adasvd_compressor.py` | `AdaSVDCompressor` | 269 | 1075 lines | **HIGH** |
| `neuralops/attentions/svd/svd_linear_attention.py` | `SVDLinearAttention` | 436 | 1016 lines | **HIGH** |
| `neuralops/attentions/ssm/mamba_svd_flash.py` | `MambaSVDFlash` | 226 | 1006 lines | **HIGH** |
| `neuralops/attentions/hybrid/fusion_attn.py` | `FUSIONATTN` | 192 | 964 lines | **HIGH** |
| `neuralops/compression_engine/low_rank/a3_compressor.py` | `A3Compressor` | 231 | 905 lines | **HIGH** |
| `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py` | `GatedHybridKernelAttention` | 387 | 822 lines | **HIGH** |
| `neuralops/attentions/adaptive/block_transformer_hiba.py` | `BlockTransformerHIBA` | 178 | 791 lines | **HIGH** |
| `neuralops/attentions/sparse/cluster_attention.py` | `ClusterAttention` | 163 | 790 lines | **HIGH** |
| `neuralops/attentions/sparse/moe_deltakv_paged.py` | `MoEDeltaKVPagedAttention` | 262 | 789 lines | **HIGH** |
| `neuralops/core/multi_mode_benchmark.py` | `MultiModeBenchmarkRunner` | 548 | 773 lines | **HIGH** |
| `neuralops/attentions/svd/svd_llm_v2_attention.py` | `SVDLLMV2Attention` | 101 | 769 lines | **HIGH** |
| `neuralops/attentions/linear/performer_attention.py` | `PerformerAttention` | 250 | 764 lines | **HIGH** |
| `neuralops/attentions/fast_kernels/flash_attention3_fp8.py` | `FlashAttention3FP8` | 212 | 760 lines | **HIGH** |
| `neuralops/attentions/hybrid/refined_landmark_attention.py` | `RefinedLandmarkAttention` | 246 | 757 lines | **HIGH** |
| `neuralops/attentions/hybrid/landmark_local_attention.py` | `LandmarkLocalAttention` | 210 | 756 lines | **HIGH** |
| `neuralops/attentions/adaptive/token_compressed_attention.py` | `TokenCompressedAttention` | 177 | 751 lines | **HIGH** |
| `neuralops/attentions/sparse/adaptive_sparse_attention.py` | `AdaptiveSparseAttention` | 183 | 750 lines | **HIGH** |
| `neuralops/attentions/hybrid/spectron_fp8_attention.py` | `SpectronFP8Attention` | 441 | 748 lines | **HIGH** |
| `neuralops/attentions/gated/decay_linear_attention.py` | `DecayLinearAttention` | 198 | 745 lines | **HIGH** |
| `neuralops/attentions/adaptive/squeezing_heads_distill.py` | `SqueezingHeadsDistiller` | 180 | 743 lines | **HIGH** |
| `neuralops/attentions/gated/gated_delta_net.py` | `GatedDeltaNet` | 302 | 743 lines | **HIGH** |
| `neuralops/attentions/ssm/orion.py` | `ORIONAttention` | 284 | 743 lines | **HIGH** |
| `neuralops/attentions/hybrid/mixattention_moska.py` | `MixAttentionMoSKA` | 146 | 733 lines | **HIGH** |
| `neuralops/attentions/linear/linformer_attention.py` | `LinformerAttention` | 218 | 733 lines | **HIGH** |
| `neuralops/attentions/adaptive/eat.py` | `EATBlock` | 188 | 723 lines | **HIGH** |
| `neuralops/attentions/linear/tlinformer_attention.py` | `TLinFormerAttention` | 170 | 723 lines | **HIGH** |
| `neuralops/attentions/linear/galerkin_attention.py` | `GalerkinAttention` | 176 | 719 lines | **HIGH** |
| `neuralops/attentions/ssm/upi_stuffed_mamba.py` | `SelectiveMamba` | 134 | 719 lines | **HIGH** |
| `neuralops/attentions/modern/nha_attention.py` | `NativeHybridAttention` | 184 | 716 lines | **HIGH** |
| `neuralops/attentions/sparse/lsh_attention.py` | `LSHAttention` | 226 | 714 lines | **HIGH** |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | `AdaptiveSpanAttention` | 280 | 704 lines | **HIGH** |
| `neuralops/compression_engine/pruning/admm_pruner.py` | `ADMMPruner` | 140 | 704 lines | **HIGH** |
| `neuralops/attentions/ssm/jamba_info.py` | `JambaInfoBlock` | 293 | 701 lines | **HIGH** |
| `neuralops/attentions/sparse/moba_attention.py` | `MoBAAttention` | 158 | 700 lines | **HIGH** |
| `neuralops/compression_engine/svd/swift_svd_act.py` | `SwiftSVDActivationCompressor` | 49 | 697 lines | **HIGH** |
| `neuralops/attentions/linear/tpa_attention.py` | `TPAttention` | 296 | 696 lines | **HIGH** |
| `neuralops/attentions/svd/factorised_matrix_state.py` | `FactorisedMatrixState` | 200 | 685 lines | **HIGH** |
| `neuralops/attentions/ssm/hymba_transx.py` | `HymbaTransXBlock` | 278 | 682 lines | **HIGH** |
| `neuralops/attentions/fast_kernels/flash_attention3.py` | `FlashAttention3` | 202 | 677 lines | **HIGH** |
| `neuralops/attentions/svd/svd_attention.py` | `SVDLinearAttention` | 173 | 677 lines | **HIGH** |
| `neuralops/attentions/hybrid/positive_kernel_linear_attention.py` | `PositiveKernelLinearAttention` | 228 | 672 lines | **HIGH** |
| `neuralops/attentions/ssm/mamba_block.py` | `MambaBlock` | 109 | 666 lines | **HIGH** |
| `neuralops/attentions/linear/rpc_attention.py` | `RPCAttention` | 224 | 662 lines | **HIGH** |
| `neuralops/compression_engine/pruning/mamba_shedder.py` | `MambaShedderCompressor` | 183 | 662 lines | **HIGH** |
| `neuralops/attentions/linear/nasc_attention.py` | `NASCAttention` | 159 | 661 lines | **HIGH** |
| `neuralops/attentions/sparse/reap_moe.py` | `REAPMoEAttention` | 369 | 659 lines | **HIGH** |
| `neuralops/attentions/ssm/nexus.py` | `NEXUSAttention` | 466 | 657 lines | **HIGH** |
| `neuralops/attentions/svd/adaptive_svd_attention.py` | `AdaptiveSVDAttention` | 186 | 657 lines | **HIGH** |
| `neuralops/attentions/fast_kernels/xattention.py` | `XAttention` | 237 | 655 lines | **HIGH** |

### Critical Long Functions (> 100 lines)

| File | Function Name | Line | Length | Severity |
|:---|:---|:---|:---|:---|
| `neuralops/benchmarks/run_quality_sweep.py` | `run_single_module_main` | 3519 | 1826 lines | **CRITICAL** |
| `neuralops/benchmarks/run_quality_sweep.py` | `_run_candidate_pipeline` | 2325 | 1192 lines | **CRITICAL** |
| `neuralops/evaluation/eval_all_models_datasets.py` | `evaluate_single_combination_opt` | 1373 | 1067 lines | **CRITICAL** |
| `neuralops/benchmarks/run_quality_sweep.py` | `parse_args` | 1629 | 665 lines | **CRITICAL** |
| `neuralops/benchmarks/core/arguments.py` | `parse_args` | 7 | 650 lines | **CRITICAL** |
| `neuralops/core/attention_utils.py` | `inject_attention` | 438 | 624 lines | **CRITICAL** |
| `neuralops/benchmarks/run_gpt2_sweep.py` | `main` | 871 | 585 lines | **CRITICAL** |
| `neuralops/benchmarks/kd_quick_test.py` | `run_kd_method` | 1148 | 479 lines | **CRITICAL** |
| `neuralops/utils/compat.py` | `get_recovery_params` | 278 | 375 lines | **CRITICAL** |
| `neuralops/analysis/generate_all_modules_analysis.py` | `main` | 812 | 362 lines | **CRITICAL** |
| `neuralops/runners/run_monitor_server.py` | `build_aggregated_stats` | 180 | 358 lines | **CRITICAL** |
| `neuralops/attentions/svd/fused_svd_kernel.py` | `copy_weights` | 669 | 357 lines | **CRITICAL** |
| `neuralops/compression_engine/quantization/slicegpt_pca.py` | `compress` | 498 | 319 lines | **CRITICAL** |
| `neuralops/compression_engine/low_rank/modegpt_compressor.py` | `compress` | 461 | 314 lines | **CRITICAL** |
| `neuralops/compression_engine/pruning/admm_pruner.py` | `compress` | 530 | 314 lines | **CRITICAL** |
| `neuralops/attentions/modern/mla_attention.py` | `copy_weights` | 378 | 309 lines | **CRITICAL** |
| `neuralops/attentions/modern/wgqa_attention.py` | `copy_weights` | 367 | 301 lines | **CRITICAL** |
| `neuralops/compression_engine/pruning/mamba_shedder.py` | `compress` | 546 | 299 lines | **CRITICAL** |
| `neuralops/compression_engine/low_rank/mgie_svd_compressor.py` | `compress` | 281 | 297 lines | **CRITICAL** |
| `neuralops/compression_engine/pruning/armor_pruner.py` | `compress` | 563 | 295 lines | **CRITICAL** |
| `neuralops/attentions/linear/performer_attention.py` | `__init__` | 254 | 292 lines | **CRITICAL** |
| `neuralops/attentions/svd/svd_llm_v2_attention.py` | `__init__` | 110 | 292 lines | **CRITICAL** |
| `neuralops/evaluation/eval_all_models_datasets.py` | `apply_module_to_model` | 632 | 288 lines | **CRITICAL** |
| `neuralops/attentions/sparse/moe_deltakv_paged.py` | `copy_weights` | 498 | 283 lines | **CRITICAL** |
| `neuralops/attentions/fast_kernels/flash_attention3_fp8.py` | `__init__` | 222 | 281 lines | **CRITICAL** |
| `neuralops/attentions/modern/nha_attention.py` | `__init__` | 192 | 278 lines | **CRITICAL** |
| `neuralops/attentions/sparse/padre_attention.py` | `copy_weights` | 320 | 278 lines | **CRITICAL** |
| `neuralops/compression_engine/low_rank/lsc_compressor.py` | `compress` | 385 | 278 lines | **CRITICAL** |
| `scripts/run_code_audit.py` | `run_audit` | 6 | 275 lines | **CRITICAL** |
| `neuralops/attentions/sparse/adaptive_sparse_attention.py` | `__init__` | 191 | 272 lines | **CRITICAL** |
| `neuralops/compression_engine/svd/entropy_budget_allocation.py` | `compress` | 256 | 270 lines | **CRITICAL** |
| `neuralops/compression_engine/svd/optimal_subspace_projection.py` | `compress` | 356 | 270 lines | **CRITICAL** |
| `neuralops/attentions/gated/gated_delta_net.py` | `__init__` | 305 | 268 lines | **CRITICAL** |
| `neuralops/runners/run_autonomous_48h_pipeline.py` | `build_all_streams` | 337 | 266 lines | **CRITICAL** |
| `neuralops/attentions/ssm/mamba_svd_flash.py` | `forward` | 917 | 265 lines | **CRITICAL** |
| `neuralops/attentions/sparse/cluster_attention.py` | `__init__` | 170 | 259 lines | **CRITICAL** |
| `neuralops/attentions/ssm/upi_stuffed_mamba.py` | `copy_weights` | 299 | 259 lines | **CRITICAL** |
| `neuralops/data/benchmark_datasets.py` | `_register_all_benchmarks` | 489 | 258 lines | **CRITICAL** |
| `neuralops/compression_engine/low_rank/mlorq_compressor.py` | `compress` | 565 | 252 lines | **CRITICAL** |
| `neuralops/compression_engine/svd/aa_svd_compressor.py` | `compress` | 644 | 252 lines | **CRITICAL** |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | `__init__` | 287 | 250 lines | **CRITICAL** |
| `neuralops/attentions/svd/svd_attention.py` | `copy_weights` | 344 | 250 lines | **CRITICAL** |
| `neuralops/compression_engine/experimental/hashed_net_compressor.py` | `compress` | 615 | 246 lines | **CRITICAL** |
| `neuralops/attentions/adaptive/eat.py` | `copy_weights` | 465 | 245 lines | **CRITICAL** |
| `neuralops/attentions/fast_kernels/flash_attention3.py` | `__init__` | 207 | 245 lines | **CRITICAL** |
| `neuralops/attentions/modern/unified_rope.py` | `copy_weights` | 359 | 245 lines | **CRITICAL** |
| `neuralops/compression_engine/kv_cache/kvtc_compressor.py` | `compress` | 512 | 245 lines | **CRITICAL** |
| `neuralops/core/multi_mode_benchmark.py` | `run_single_mode` | 869 | 245 lines | **CRITICAL** |
| `neuralops/attentions/svd/compressed_mha_attention.py` | `copy_weights` | 216 | 244 lines | **CRITICAL** |
| `neuralops/attentions/ssm/mamba_svd_flash.py` | `__init__` | 242 | 243 lines | **CRITICAL** |
| `neuralops/compression_engine/svd/gfws_d_compressor.py` | `compress` | 332 | 243 lines | **CRITICAL** |
| `neuralops/attentions/linear/galerkin_attention.py` | `__init__` | 181 | 242 lines | **CRITICAL** |
| `neuralops/attentions/linear/linformer_attention.py` | `copy_weights` | 543 | 242 lines | **CRITICAL** |
| `neuralops/attentions/linear/tpa_attention.py` | `__init__` | 311 | 242 lines | **CRITICAL** |
| `neuralops/attentions/sparse/lsh_attention.py` | `copy_weights` | 492 | 241 lines | **CRITICAL** |
| `neuralops/attentions/sparse/moba_attention.py` | `forward` | 596 | 241 lines | **CRITICAL** |
| `neuralops/attentions/svd/compressed_attention.py` | `copy_weights` | 230 | 241 lines | **CRITICAL** |
| `neuralops/compression_engine/svd/hybrid_svd_layer.py` | `compress` | 296 | 241 lines | **CRITICAL** |
| `neuralops/attentions/fast_kernels/xattention.py` | `forward` | 652 | 240 lines | **CRITICAL** |
| `neuralops/compression_engine/svd/adasvd_compressor.py` | `compress` | 1104 | 240 lines | **CRITICAL** |

### Deep Indentation Hot-Spots (>= 6 Levels)

| File | Line | Depth | Snippet |
|:---|:---|:---|:---|
| `neuralops/analysis/generate_all_modules_analysis.py` | 645 | 6 levels | `k.strip().lower().replace(" ", "_"): v.strip() for k, v in r` |
| `neuralops/analysis/generate_all_modules_analysis.py` | 649 | 6 levels | `continue` |
| `neuralops/analysis/generate_all_modules_analysis.py` | 739 | 6 levels | `valid_vals.append(val)` |
| `neuralops/analysis/generate_all_modules_analysis.py` | 909 | 6 levels | `paper_ref = m_title.group(1).strip()` |
| `neuralops/analysis/generate_all_modules_analysis.py` | 912 | 6 levels | `r"(?:arXiv:\s*|arxiv\.org/abs/)(\d{4}\.\d{4,5})",` |
| `neuralops/analysis/generate_all_modules_analysis.py` | 913 | 6 levels | `content,` |
| `neuralops/analysis/generate_all_modules_analysis.py` | 914 | 6 levels | `re.IGNORECASE,` |
| `neuralops/analysis/generate_all_modules_analysis.py` | 917 | 6 levels | `paper_arxiv = m_arxiv.group(1).strip()` |
| `neuralops/analysis/generate_all_modules_analysis.py` | 918 | 6 levels | `if paper_year == "None":` |
| `neuralops/analysis/generate_all_modules_analysis.py` | 919 | 7 levels | `paper_year = "20" + paper_arxiv[:2]` |
| `neuralops/analysis/generate_all_modules_analysis.py` | 920 | 6 levels | `if paper_venue == "None":` |
| `neuralops/analysis/generate_all_modules_analysis.py` | 921 | 7 levels | `paper_venue = "arXiv"` |
| `neuralops/analysis/generate_all_modules_analysis.py` | 958 | 6 levels | `valid_comps.append(c_val)` |
| `neuralops/analysis/layerwise_analyzer.py` | 204 | 6 levels | `target_layers.append((f"{name}.{sub_idx}", sub_mod))` |
| `neuralops/analysis/layerwise_analyzer.py` | 224 | 6 levels | `self.layer_accumulators[idx].update(tensor)` |
| `neuralops/analysis/layerwise_analyzer.py` | 226 | 6 levels | `del tensor` |
| `neuralops/analysis/metrics.py` | 84 | 6 levels | `all_records.append(json.loads(line))` |
| `neuralops/analysis/metrics.py` | 86 | 6 levels | `pass` |
| `neuralops/attentions/__init__.py` | 157 | 6 levels | `__all__.append(_attr)` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 678 | 6 levels | `layer = getattr(src, name)` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 679 | 6 levels | `if hasattr(layer, "weight"):` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 680 | 7 levels | `extracted[proj] = (` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 681 | 8 levels | `layer.weight.data,` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 682 | 8 levels | `getattr(layer, "bias", None),` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 683 | 7 levels | `)` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 684 | 7 levels | `break` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 724 | 6 levels | `b_val = b.data if isinstance(b, nn.Parameter) else b` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 725 | 6 levels | `c_b = min(len(target.bias), len(b_val))` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 726 | 6 levels | `target.bias.data[:c_b].copy_(b_val[:c_b])` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 774 | 6 levels | `past_len = past_key_value.get_seq_length(self.layer_idx)` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 776 | 6 levels | `try:` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 777 | 7 levels | `past_len = past_key_value.get_seq_length()` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 778 | 6 levels | `except Exception:` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 779 | 7 levels | `past_len = 0` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 827 | 6 levels | `K, V, new_past_key_value = past_key_value.update(` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 828 | 7 levels | `K_new, V_new, self.layer_idx` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 829 | 6 levels | `)` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 831 | 6 levels | `K = torch.cat([past_key_value[0], K_new], dim=-2)` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 832 | 6 levels | `V = torch.cat([past_key_value[1], V_new], dim=-2)` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 833 | 6 levels | `new_past_key_value = (K, V)` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 837 | 6 levels | `K = torch.cat([past_k, K_new], dim=-2)` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 838 | 6 levels | `V = torch.cat([past_v, V_new], dim=-2)` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 840 | 6 levels | `K, V = K_new, V_new` |
| `neuralops/attentions/adaptive/adaptive_span_attention.py` | 888 | 6 levels | `last_q - N_k + 1, last_q + 1, device=device, dtype=torch.flo` |
| `neuralops/attentions/adaptive/block_transformer_hiba.py` | 497 | 6 levels | `c_proj_mod.merge()` |
| `neuralops/attentions/adaptive/block_transformer_hiba.py` | 499 | 6 levels | `print(f"copy_weights failed: {e}")` |
| `neuralops/attentions/adaptive/block_transformer_hiba.py` | 500 | 6 levels | `logger.warning(f"copy_weights failed: {e}")` |
| `neuralops/attentions/adaptive/block_transformer_hiba.py` | 506 | 6 levels | `(c_proj_mod.__class__.__name__ == "Conv1D")` |
| `neuralops/attentions/adaptive/block_transformer_hiba.py` | 507 | 6 levels | `or (base_o.__class__.__name__ == "Conv1D")` |
| `neuralops/attentions/adaptive/block_transformer_hiba.py` | 508 | 6 levels | `or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])` |

---

## ⚖️ Section 5: Consistency Issues

1. **Mixed Logging Paradigms**: Several benchmark runners (`run_quality_sweep.py`, `eval_all_models_datasets.py`) alternate haphazardly between standard `print(...)` statements and `logging.getLogger(__name__)`. Production modules should strictly emit formatted logs.
2. **Inconsistent Error Handling**: Multiple modules catch generic `Exception` and silently return `None` or `pass`, masking configuration bugs during automated sweeps, while other modules raise explicit `RuntimeError` / `ValueError`.
3. **Inconsistent Naming**: Mix of PEFT naming conventions (`lora_A`, `lora_B`), legacy SVD naming (`U`, `V`, `S`), and projection names (`down`, `up`).
4. **Docstring Inconsistencies**: Over 200 public classes omit docstrings or use single-line docstrings that do not describe input shapes `(B, N, D)` or return contracts.

---

## 🚨 Section 6: Top 20 Worst Files

Ranked by cumulative severity score (file size, long functions, dead code, duplication, structural defects):

### 1. `neuralops/core/attention_utils.py` (1078 lines, Severity Score: 334)
- **Category**: Core Attention/Compression
- **Issues**:
  - File size exceeds TOO LONG threshold (> 800 lines)
  - Contains bloated functions with excessive cyclomatic complexity and deep indentation
  - Duplicates common primitives locally instead of consuming shared `neuralops.core` helpers
- **Priority**: Priority 1 (High Refactoring Value)
- **Suggested Action**: Decompose into modular components, extract reusable sub-methods, replace local helpers with core imports.

### 2. `neuralops/evaluation/eval_all_models_datasets.py` (2527 lines, Severity Score: 306)
- **Category**: Core Attention/Compression
- **Issues**:
  - File size exceeds CRITICAL threshold (> 1500 lines)
  - Contains bloated functions with excessive cyclomatic complexity and deep indentation
  - Duplicates common primitives locally instead of consuming shared `neuralops.core` helpers
- **Priority**: Priority 1 (High Refactoring Value)
- **Suggested Action**: Decompose into modular components, extract reusable sub-methods, replace local helpers with core imports.

### 3. `neuralops/benchmarks/run_quality_sweep.py` (5740 lines, Severity Score: 277)
- **Category**: Benchmark Orchestration
- **Issues**:
  - File size exceeds CRITICAL threshold (> 1500 lines)
  - Contains bloated functions with excessive cyclomatic complexity and deep indentation
  - Duplicates common primitives locally instead of consuming shared `neuralops.core` helpers
- **Priority**: Priority 1 (High Refactoring Value)
- **Suggested Action**: Decompose into modular components, extract reusable sub-methods, replace local helpers with core imports.

### 4. `neuralops/compression_engine/svd/adaptive_ranks.py` (1258 lines, Severity Score: 208)
- **Category**: Core Attention/Compression
- **Issues**:
  - File size exceeds TOO LONG threshold (> 800 lines)
  - Contains bloated functions with excessive cyclomatic complexity and deep indentation
  - Duplicates common primitives locally instead of consuming shared `neuralops.core` helpers
- **Priority**: Priority 1 (High Refactoring Value)
- **Suggested Action**: Decompose into modular components, extract reusable sub-methods, replace local helpers with core imports.

### 5. `neuralops/trainer/optimizers.py` (940 lines, Severity Score: 124)
- **Category**: Core Attention/Compression
- **Issues**:
  - File size exceeds TOO LONG threshold (> 800 lines)
  - Contains bloated functions with excessive cyclomatic complexity and deep indentation
  - Duplicates common primitives locally instead of consuming shared `neuralops.core` helpers
- **Priority**: Priority 1 (High Refactoring Value)
- **Suggested Action**: Decompose into modular components, extract reusable sub-methods, replace local helpers with core imports.

### 6. `neuralops/attentions/svd/fused_svd_kernel.py` (1293 lines, Severity Score: 118)
- **Category**: Core Attention/Compression
- **Issues**:
  - File size exceeds TOO LONG threshold (> 800 lines)
  - Contains bloated functions with excessive cyclomatic complexity and deep indentation
  - Duplicates common primitives locally instead of consuming shared `neuralops.core` helpers
- **Priority**: Priority 1 (High Refactoring Value)
- **Suggested Action**: Decompose into modular components, extract reusable sub-methods, replace local helpers with core imports.

### 7. `neuralops/compression_engine/quantization/faq_quantizer.py` (851 lines, Severity Score: 117)
- **Category**: Core Attention/Compression
- **Issues**:
  - File size exceeds TOO LONG threshold (> 800 lines)
  - Contains bloated functions with excessive cyclomatic complexity and deep indentation
  - Duplicates common primitives locally instead of consuming shared `neuralops.core` helpers
- **Priority**: Priority 1 (High Refactoring Value)
- **Suggested Action**: Decompose into modular components, extract reusable sub-methods, replace local helpers with core imports.

### 8. `neuralops/core/multi_mode_benchmark.py` (1459 lines, Severity Score: 115)
- **Category**: Core Attention/Compression
- **Issues**:
  - File size exceeds TOO LONG threshold (> 800 lines)
  - Contains bloated functions with excessive cyclomatic complexity and deep indentation
  - Duplicates common primitives locally instead of consuming shared `neuralops.core` helpers
- **Priority**: Priority 1 (High Refactoring Value)
- **Suggested Action**: Decompose into modular components, extract reusable sub-methods, replace local helpers with core imports.

### 9. `neuralops/compression_engine/pruning/admm_pruner.py` (992 lines, Severity Score: 96)
- **Category**: Core Attention/Compression
- **Issues**:
  - File size exceeds TOO LONG threshold (> 800 lines)
  - Contains bloated functions with excessive cyclomatic complexity and deep indentation
  - Duplicates common primitives locally instead of consuming shared `neuralops.core` helpers
- **Priority**: Priority 1 (High Refactoring Value)
- **Suggested Action**: Decompose into modular components, extract reusable sub-methods, replace local helpers with core imports.

### 10. `neuralops/attentions/modern/unified_rope.py` (812 lines, Severity Score: 89)
- **Category**: Core Attention/Compression
- **Issues**:
  - File size exceeds TOO LONG threshold (> 800 lines)
  - Contains bloated functions with excessive cyclomatic complexity and deep indentation
  - Duplicates common primitives locally instead of consuming shared `neuralops.core` helpers
- **Priority**: Priority 1 (High Refactoring Value)
- **Suggested Action**: Decompose into modular components, extract reusable sub-methods, replace local helpers with core imports.

### 11. `neuralops/attentions/ssm/upi_stuffed_mamba.py` (990 lines, Severity Score: 87)
- **Category**: Core Attention/Compression
- **Issues**:
  - File size exceeds TOO LONG threshold (> 800 lines)
  - Contains bloated functions with excessive cyclomatic complexity and deep indentation
  - Duplicates common primitives locally instead of consuming shared `neuralops.core` helpers
- **Priority**: Priority 1 (High Refactoring Value)
- **Suggested Action**: Decompose into modular components, extract reusable sub-methods, replace local helpers with core imports.

### 12. `neuralops/attentions/svd/svd_attention.py` (971 lines, Severity Score: 87)
- **Category**: Core Attention/Compression
- **Issues**:
  - File size exceeds TOO LONG threshold (> 800 lines)
  - Contains bloated functions with excessive cyclomatic complexity and deep indentation
  - Duplicates common primitives locally instead of consuming shared `neuralops.core` helpers
- **Priority**: Priority 1 (High Refactoring Value)
- **Suggested Action**: Decompose into modular components, extract reusable sub-methods, replace local helpers with core imports.

### 13. `neuralops/attentions/svd/svd_linear_attention.py` (1590 lines, Severity Score: 86)
- **Category**: Core Attention/Compression
- **Issues**:
  - File size exceeds CRITICAL threshold (> 1500 lines)
  - Contains bloated functions with excessive cyclomatic complexity and deep indentation
  - Duplicates common primitives locally instead of consuming shared `neuralops.core` helpers
- **Priority**: Priority 1 (High Refactoring Value)
- **Suggested Action**: Decompose into modular components, extract reusable sub-methods, replace local helpers with core imports.

### 14. `neuralops/attentions/modern/wgqa_attention.py` (977 lines, Severity Score: 83)
- **Category**: Core Attention/Compression
- **Issues**:
  - File size exceeds TOO LONG threshold (> 800 lines)
  - Contains bloated functions with excessive cyclomatic complexity and deep indentation
  - Duplicates common primitives locally instead of consuming shared `neuralops.core` helpers
- **Priority**: Priority 1 (High Refactoring Value)
- **Suggested Action**: Decompose into modular components, extract reusable sub-methods, replace local helpers with core imports.

### 15. `neuralops/compression_engine/low_rank/come_compressor.py` (1137 lines, Severity Score: 83)
- **Category**: Core Attention/Compression
- **Issues**:
  - File size exceeds TOO LONG threshold (> 800 lines)
  - Contains bloated functions with excessive cyclomatic complexity and deep indentation
  - Duplicates common primitives locally instead of consuming shared `neuralops.core` helpers
- **Priority**: Priority 1 (High Refactoring Value)
- **Suggested Action**: Decompose into modular components, extract reusable sub-methods, replace local helpers with core imports.

### 16. `neuralops/attentions/fast_kernels/flash_attention3_fp8.py` (1075 lines, Severity Score: 81)
- **Category**: Core Attention/Compression
- **Issues**:
  - File size exceeds TOO LONG threshold (> 800 lines)
  - Contains bloated functions with excessive cyclomatic complexity and deep indentation
  - Duplicates common primitives locally instead of consuming shared `neuralops.core` helpers
- **Priority**: Priority 1 (High Refactoring Value)
- **Suggested Action**: Decompose into modular components, extract reusable sub-methods, replace local helpers with core imports.

### 17. `neuralops/compression_engine/quantization/slicegpt_pca.py` (885 lines, Severity Score: 80)
- **Category**: Core Attention/Compression
- **Issues**:
  - File size exceeds TOO LONG threshold (> 800 lines)
  - Contains bloated functions with excessive cyclomatic complexity and deep indentation
  - Duplicates common primitives locally instead of consuming shared `neuralops.core` helpers
- **Priority**: Priority 1 (High Refactoring Value)
- **Suggested Action**: Decompose into modular components, extract reusable sub-methods, replace local helpers with core imports.

### 18. `neuralops/compression_engine/pruning/armor_pruner.py` (971 lines, Severity Score: 79)
- **Category**: Core Attention/Compression
- **Issues**:
  - File size exceeds TOO LONG threshold (> 800 lines)
  - Contains bloated functions with excessive cyclomatic complexity and deep indentation
  - Duplicates common primitives locally instead of consuming shared `neuralops.core` helpers
- **Priority**: Priority 1 (High Refactoring Value)
- **Suggested Action**: Decompose into modular components, extract reusable sub-methods, replace local helpers with core imports.

### 19. `neuralops/compression_engine/pruning/mamba_shedder.py` (921 lines, Severity Score: 78)
- **Category**: Core Attention/Compression
- **Issues**:
  - File size exceeds TOO LONG threshold (> 800 lines)
  - Contains bloated functions with excessive cyclomatic complexity and deep indentation
  - Duplicates common primitives locally instead of consuming shared `neuralops.core` helpers
- **Priority**: Priority 1 (High Refactoring Value)
- **Suggested Action**: Decompose into modular components, extract reusable sub-methods, replace local helpers with core imports.

### 20. `neuralops/attentions/fast_kernels/xattention.py` (1063 lines, Severity Score: 76)
- **Category**: Core Attention/Compression
- **Issues**:
  - File size exceeds TOO LONG threshold (> 800 lines)
  - Contains bloated functions with excessive cyclomatic complexity and deep indentation
  - Duplicates common primitives locally instead of consuming shared `neuralops.core` helpers
- **Priority**: Priority 1 (High Refactoring Value)
- **Suggested Action**: Decompose into modular components, extract reusable sub-methods, replace local helpers with core imports.

---

## 🧪 Section 7: Missing Test Coverage

### Untested Attention Modules

| File | Status |
|:---|:---|
| `neuralops/attentions/adaptive/infini_attention.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/adaptive/squeezing_heads_distill.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/gated/hgrn2_gated_layer.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/gated/hgrn2_gated_outer.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/hybrid/gated_hybrid_kernel_attention.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/hybrid/hybrid_local_kernel_attention.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/hybrid/landmark_local_attention.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/hybrid/positive_kernel_linear_attention.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/hybrid/refined_landmark_attention.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/hybrid/split_head_hybrid_attention.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/hybrid/tri_branch_hybrid_attention.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/linear/spectra_v2.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/linear/spectraformer_attention.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/linear/tlinformer_attention.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/sparse/padre_attention.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/ssm/mamba_block.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/ssm/selective_ssm.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/ssm/simple_ssm.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/ssm/spectral_mamba.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/ssm/upi_stuffed_mamba.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/svd/naive_svd.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/svd/residual_svd_attention.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/svd/svd_compressed.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/svd/svd_llm_v2_attention.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/attentions/svd/svd_projection_attention.py` | No standalone test file in `neuralops/tests/` |

### Untested Compression Engines

| File | Status |
|:---|:---|
| `neuralops/compression_engine/experimental/flat_llm.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/experimental/hashed_net_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/experimental/liftr.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/experimental/lillama_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/experimental/roast_hashed_linear.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/kv_cache/aatc_xkv_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/kv_cache/delta_kv.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/kv_cache/hybridkv_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/kv_cache/zerokv_neo_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/low_rank/a3_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/low_rank/come_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/low_rank/compot_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/low_rank/cur_moe_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/low_rank/drone_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/low_rank/frandi_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/low_rank/grecodec_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/low_rank/lsc_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/low_rank/mgie_svd_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/low_rank/mlorq_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/low_rank/modegpt_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/low_rank/nmf_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/low_rank/oats_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/low_rank/rpca_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/low_rank/salr.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/low_rank/sdrm_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/low_rank/slim_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/low_rank/soka_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/low_rank/svd_llm_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/low_rank/tomoe_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/low_rank/uniql_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/pruning/ast_pruner.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/pruning/bonsai_pruner.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/pruning/flap.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/pruning/mamba_shedder.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/pruning/proxsparse.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/pruning/sparsegpt.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/quantization/astro_quantizer.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/quantization/coopq_quantizer.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/quantization/faq_quantizer.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/quantization/kmeans_quantize.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/quantization/rotational_slicegpt_int4.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/quantization/slicegpt.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/quantization/slicegpt_pca.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/svd/aa_svd_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/svd/direct_svd.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/svd/dynamic_rank.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/svd/entropy_budget_allocation.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/svd/erc_svd.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/svd/fwsvd_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/svd/gfws_d_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/svd/hybrid_svd_layer.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/svd/optimal_subspace_projection.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/svd/residual_svd_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/svd/saes_svd_v2.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/svd/subspace_projection.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/svd/swift_svd.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/svd/swift_svd_act.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/tensor_decomp/kronecker_compressor.py` | No standalone test file in `neuralops/tests/` |
| `neuralops/compression_engine/tensor_decomp/kronecker_tt_lora.py` | No standalone test file in `neuralops/tests/` |

---

## 📋 Section 9: Prioritized Fix Plan

### Priority 1: CRITICAL (Fix Immediately)
- **Scope**: The 4 monster files exceeding 1,500 lines (`neuralops/benchmarks/run_quality_sweep.py`, `eval_all_models_datasets.py`, `kd_quick_test.py`, `svd_linear_attention.py`).
- **Fixes**: Split monolithic functions into pipeline steps, decouple report generation from test execution, and extract inner loops.
- **Estimated Effort**: 16 hours
- **Risk Level**: MED (requires verification against benchmark suites)

### Priority 2: IMPORTANT (Fix This Week)
- **Scope**: Extract duplicate boilerplate (`_rotate_half`, `apply_rotary_pos_emb`, `LoRALinear`, `repeat_kv`) from 106+ attention modules into `neuralops.core`.
- **Fixes**: Delete local re-implementations and import from centralized core modules; purge 420+ unused imports flagged by pyflakes.
- **Estimated Effort**: 12 hours
- **Risk Level**: LOW (pure deduplication with identical mathematical semantics)

### Priority 3: NICE TO HAVE (Fix Eventually)
- **Scope**: Add missing standalone unit tests for uncovered attention/compression modules; standardize docstrings with tensor shape contracts.
- **Estimated Effort**: 20 hours
- **Risk Level**: LOW
