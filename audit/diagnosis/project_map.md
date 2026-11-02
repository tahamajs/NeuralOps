# Project Architecture & Test Pipeline Map

## 1. Environment & Runtime
- **Host**: Ubuntu on GPU server (`192.168.1.103`)
- **GPU**: NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM)
- **CUDA / PyTorch**: CUDA Available, PyTorch in venv (`/mnt/d/venv`)
- **Root Path**: `/mnt/d/SVD_Project`

## 2. Benchmark Runner & Invocation
- **Main test runner**: `neuralops/benchmarks/_legacy/kd_quick_test.py`
  - Instantiates base teacher model (`gpt2` by default).
  - Replaces attention blocks using module registries.
  - Performs 2-stage Knowledge Distillation:
    - Stage 1: Fast initial distillation (`--recovery_steps 300`)
    - Stage 2: Self-training recovery (`--self_train_steps 200`)
  - Evaluates generation perplexity & loss every `--eval_every 100` steps.
  - Computes delta = teacher_ppl - student_ppl.
- **Sweep driver**: `neuralops/benchmarks/_legacy/run_all_modules_delta.py`
  - Reads modules from attention and compressor registries.
  - Executes `kd_quick_test.py` per module with timeout (`600s`).
  - Records execution metrics and outputs into `module_results.csv`.

## 3. Registries & Module Registration
- **Attention Modules**: `neuralops/attentions/attention_registry.py` (`ATTENTION_REGISTRY`)
- **Compressor Modules**: `neuralops/compression_engine/compressor_registry.py` (`COMPRESSOR_REGISTRY`)

## 4. Module File Locations for Target 23
1. `gated_linear_attention`: `neuralops/attentions/gated/decay_linear_attention.py` (`DecayLinearAttention`)
2. `gated_linear_attn`: `neuralops/attentions/gated/decay_linear_attention.py` (`DecayLinearAttention`)
3. `hgrn2_outer`: `neuralops/attentions/gated/hgrn2_gated_outer.py` (`HGRN2GatedOuterBlock`)
4. `svd_linear`: `neuralops/attentions/svd/svd_linear_attention.py` (`SVDLinearAttention`)
5. `svd_linear_attn`: `neuralops/attentions/svd/svd_linear_attention.py` (`SVDLinearAttention`)
6. `svd_linear_inject`: `neuralops/attentions/svd/svd_linear_attention.py` (`SVDLinearAttention`)
7. `linformer_attention`: `neuralops/attentions/linear/linformer_attention.py` (`LinformerAttention`)
8. `mamba_svd_flash`: `neuralops/attentions/ssm/mamba_svd_flash.py` (`MambaSVDFlashAttention`)
9. `nasc_attention`: `neuralops/attentions/linear/nasc_attention.py` (`NASCAttention`)
10. `performer_attention`: `neuralops/attentions/linear/performer_attention.py` (`PerformerAttention`)
11. `tt_compressor`: `neuralops/compression_engine/tensor_decomp/tt_compressor.py` (`tt_compressor`)
12. `regla_attention`: `neuralops/attentions/hybrid/regla_attention.py` (`inject_regla`)
13. `selective_ssm`: `neuralops/attentions/ssm/selective_ssm.py` (`SelectiveSSM`)
14. `simple_ssm`: `neuralops/attentions/ssm/selective_ssm.py` (`SelectiveSSM`)
15. `spectron_fp8`: `neuralops/attentions/hybrid/spectron_fp8_attention.py` (`SpectronFP8Attention`)
16. `svd_mamba`: `neuralops/attentions/ssm/svd_mamba.py` (`SVDMamba`)
17. `aegis_spectron_fp8`: `neuralops/attentions/hybrid/spectron_fp8_attention.py` (`SpectronFP8Attention`)
18. `aegis_v1`: `neuralops/attentions/hybrid/hybrid_local_kernel_attention.py` (`HybridLocalKernelAttention`)
19. `aegis_v2`: `neuralops/attentions/hybrid/hybrid_local_kernel_attention.py` (`HybridLocalKernelAttention`)
20. `aatc_xkv_compressor`: `neuralops/compression_engine/kv_cache/cachenotes_compressor.py`
21. `hybridkv_compressor`: `neuralops/compression_engine/kv_cache/delta_kv.py`
22. `aegis_hybrid`: `neuralops/attentions/hybrid/tri_branch_hybrid_attention.py` (`TriBranchHybridAttention`)
