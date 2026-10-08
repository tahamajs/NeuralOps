# Unused Imports & Variables Audit (Round 2)

## Summary

- **Total issues**: 95
- **F401 (unused imports)**: 81
- **F841 (unused variables)**: 12

## Per-file breakdown

| File | Issues | F401 | F841 |
| :--- | :---: | :---: | :---: |
| `neuralops/core/triton_kernels/paged_attention.py` | 5 | 0 | 5 |
| `neuralops/benchmarks/eval_suite.py` | 4 | 4 | 0 |
| `neuralops/inference/paged_cache.py` | 4 | 4 | 0 |
| `neuralops/core/replace_attention.py` | 3 | 3 | 0 |
| `neuralops/inference/engine.py` | 3 | 2 | 1 |
| `scripts/gen_api_docs.py` | 3 | 3 | 0 |
| `neuralops/attentions/gated/hgrn2_gated_deltanet.py` | 2 | 2 | 0 |
| `neuralops/attentions/linear/cosformer.py` | 2 | 2 | 0 |
| `neuralops/attentions/modern/alibi.py` | 2 | 1 | 1 |
| `neuralops/attentions/modern/sliding_window.py` | 2 | 2 | 0 |
| `neuralops/benchmarks/registry.py` | 2 | 2 | 0 |
| `neuralops/compression_engine/pruning/bonsai_pruner.py` | 2 | 2 | 0 |
| `neuralops/core/triton_kernels/flash_decoding.py` | 2 | 0 | 2 |
| `neuralops/integrations/__init__.py` | 2 | 2 | 0 |
| `neuralops/training/trainer.py` | 2 | 2 | 0 |
| `scripts/normalize_results.py` | 2 | 2 | 0 |
| `neuralops/__init__.py` | 1 | 1 | 0 |
| `neuralops/analysis/__init__.py` | 1 | 1 | 0 |
| `neuralops/analysis/module_analysis/parsers.py` | 1 | 1 | 0 |
| `neuralops/attentions/__init__.py` | 1 | 1 | 0 |
| `neuralops/attentions/adaptive/__init__.py` | 1 | 1 | 0 |
| `neuralops/attentions/adaptive/lola_attention.py` | 1 | 1 | 0 |
| `neuralops/attentions/adaptive/sink_attention.py` | 1 | 1 | 0 |
| `neuralops/attentions/dense/__init__.py` | 1 | 1 | 0 |
| `neuralops/attentions/experimental/__init__.py` | 1 | 1 | 0 |
| `neuralops/attentions/fast_kernels/__init__.py` | 1 | 1 | 0 |
| `neuralops/attentions/gated/__init__.py` | 1 | 1 | 0 |
| `neuralops/attentions/hybrid/__init__.py` | 1 | 1 | 0 |
| `neuralops/attentions/linear/__init__.py` | 1 | 1 | 0 |
| `neuralops/attentions/linear/chela_slay.py` | 1 | 1 | 0 |
| `neuralops/attentions/modern/__init__.py` | 1 | 1 | 0 |
| `neuralops/attentions/sparse/__init__.py` | 1 | 1 | 0 |
| `neuralops/attentions/sparse/expert_choice_moe.py` | 1 | 0 | 1 |
| `neuralops/attentions/ssm/__init__.py` | 1 | 1 | 0 |
| `neuralops/attentions/ssm/goldfinch_ahn.py` | 1 | 0 | 1 |
| `neuralops/attentions/svd/__init__.py` | 1 | 1 | 0 |
| `neuralops/attentions/svd/svd_linear_attention.py` | 1 | 1 | 0 |
| `neuralops/benchmarks/cli.py` | 1 | 1 | 0 |
| `neuralops/benchmarks/discovery.py` | 1 | 1 | 0 |
| `neuralops/compression_engine/__init__.py` | 1 | 1 | 0 |
| `neuralops/compression_engine/experimental/__init__.py` | 1 | 1 | 0 |
| `neuralops/compression_engine/kv_cache/__init__.py` | 1 | 1 | 0 |
| `neuralops/compression_engine/low_rank/__init__.py` | 1 | 1 | 0 |
| `neuralops/compression_engine/low_rank/dora_qlora.py` | 1 | 1 | 0 |
| `neuralops/compression_engine/pruning/__init__.py` | 1 | 1 | 0 |
| `neuralops/compression_engine/quantization/__init__.py` | 1 | 1 | 0 |
| `neuralops/compression_engine/quantization/awq.py` | 1 | 0 | 1 |
| `neuralops/compression_engine/quantization/gptq.py` | 1 | 1 | 0 |
| `neuralops/compression_engine/svd/__init__.py` | 1 | 1 | 0 |
| `neuralops/compression_engine/tensor_decomp/__init__.py` | 1 | 1 | 0 |
| `neuralops/core/__init__.py` | 1 | 1 | 0 |
| `neuralops/core/_attention_extractors.py` | 1 | 1 | 0 |
| `neuralops/core/_attention_injectors.py` | 1 | 1 | 0 |
| `neuralops/core/_attention_surgery.py` | 1 | 1 | 0 |
| `neuralops/core/cuda_graph.py` | 1 | 1 | 0 |
| `neuralops/core/triton_kernels/fms_scan.py` | 1 | 1 | 0 |
| `neuralops/core/triton_kernels/gdn_scan.py` | 1 | 1 | 0 |
| `neuralops/core/triton_kernels/gla_scan.py` | 1 | 1 | 0 |
| `neuralops/core/triton_kernels/hgrn2_scan.py` | 1 | 1 | 0 |
| `neuralops/core/triton_kernels/selective_scan.py` | 1 | 1 | 0 |
| `neuralops/evaluation/eval_all_models_datasets.py` | 1 | 1 | 0 |
| `neuralops/integrations/distributed_compile.py` | 1 | 1 | 0 |
| `neuralops/integrations/hf_automodel.py` | 1 | 1 | 0 |
| `neuralops/integrations/serving.py` | 1 | 1 | 0 |
| `neuralops/trainer/__init__.py` | 1 | 1 | 0 |
| `neuralops/utils/__init__.py` | 1 | 1 | 0 |
| `scripts/normalize_models.py` | 1 | 1 | 0 |

## Full list of issues

```text
neuralops/__init__.py:11:25: F401 `typing.List` imported but unused
neuralops/analysis/__init__.py:36:5: F401 `.utils.get_best_configurations` imported but unused; consider removing, adding to `__all__`, or using a redundant alias
neuralops/analysis/module_analysis/parsers.py:12:5: F401 [*] `neuralops.analysis.module_analysis.constants.ALIASES` imported but unused
neuralops/attentions/__init__.py:7:8: F401 `os` imported but unused
neuralops/attentions/adaptive/__init__.py:3:8: F401 `sys` imported but unused
neuralops/attentions/adaptive/lola_attention.py:23:33: F401 [*] `neuralops.core.rope._rotate_half` imported but unused
neuralops/attentions/adaptive/sink_attention.py:4:20: F401 [*] `torch.nn` imported but unused
neuralops/attentions/dense/__init__.py:3:8: F401 `sys` imported but unused
neuralops/attentions/experimental/__init__.py:3:8: F401 `sys` imported but unused
neuralops/attentions/fast_kernels/__init__.py:3:8: F401 `sys` imported but unused
neuralops/attentions/gated/__init__.py:3:8: F401 `sys` imported but unused
neuralops/attentions/gated/hgrn2_gated_deltanet.py:21:65: F401 [*] `neuralops.attentions.gated_delta_net.LoRALinear` imported but unused
neuralops/attentions/gated/hgrn2_gated_deltanet.py:23:37: F401 [*] `neuralops.core.adapters.UniversalAdapter` imported but unused
neuralops/attentions/hybrid/__init__.py:3:8: F401 `sys` imported but unused
neuralops/attentions/linear/__init__.py:3:8: F401 `sys` imported but unused
neuralops/attentions/linear/chela_slay.py:36:34: F401 [*] `neuralops.utils.lora.LoRALinear` imported but unused
neuralops/attentions/linear/cosformer.py:12:20: F401 [*] `typing.Any` imported but unused
neuralops/attentions/linear/cosformer.py:20:43: F401 [*] `neuralops.core.attention_mask.get_token_valid_mask` imported but unused
neuralops/attentions/modern/__init__.py:3:8: F401 `sys` imported but unused
neuralops/attentions/modern/alibi.py:5:20: F401 [*] `torch.nn` imported but unused
neuralops/attentions/modern/alibi.py:52:9: F841 Local variable `bias` is assigned to but never used
neuralops/attentions/modern/sliding_window.py:3:8: F401 [*] `math` imported but unused
neuralops/attentions/modern/sliding_window.py:6:31: F401 [*] `torch.nn.functional` imported but unused
neuralops/attentions/sparse/__init__.py:3:8: F401 `sys` imported but unused
neuralops/attentions/sparse/expert_choice_moe.py:219:9: F841 Local variable `T` is assigned to but never used
neuralops/attentions/ssm/__init__.py:3:8: F401 `sys` imported but unused
neuralops/attentions/ssm/goldfinch_ahn.py:242:9: F841 Local variable `cum_log_omg` is assigned to but never used
neuralops/attentions/svd/__init__.py:3:8: F401 `sys` imported but unused
neuralops/attentions/svd/svd_linear_attention.py:66:55: F401 `neuralops.attentions.fused_svd_kernel.FusedSVDLinear` imported but unused; consider using `importlib.util.find_spec` to test for availability
neuralops/benchmarks/cli.py:137:12: F401 [*] `neuralops.compression_engine` imported but unused
neuralops/benchmarks/discovery.py:110:12: F401 [*] `neuralops.compression_engine` imported but unused
neuralops/benchmarks/eval_suite.py:3:30: F401 [*] `typing.Any` imported but unused
neuralops/benchmarks/eval_suite.py:3:35: F401 [*] `typing.List` imported but unused
neuralops/benchmarks/eval_suite.py:3:41: F401 [*] `typing.Dict` imported but unused
neuralops/benchmarks/eval_suite.py:4:8: F401 [*] `random` imported but unused
neuralops/benchmarks/registry.py:34:16: F401 [*] `neuralops.attentions` imported but unused
neuralops/benchmarks/registry.py:40:16: F401 [*] `neuralops.compression_engine` imported but unused
neuralops/compression_engine/__init__.py:7:8: F401 `os` imported but unused
neuralops/compression_engine/experimental/__init__.py:3:8: F401 `sys` imported but unused
neuralops/compression_engine/kv_cache/__init__.py:3:8: F401 `sys` imported but unused
neuralops/compression_engine/low_rank/__init__.py:3:8: F401 `sys` imported but unused
neuralops/compression_engine/low_rank/dora_qlora.py:3:8: F401 [*] `math` imported but unused
neuralops/compression_engine/pruning/__init__.py:3:8: F401 `sys` imported but unused
neuralops/compression_engine/pruning/bonsai_pruner.py:37:5: F401 [*] `neuralops.core.pruning_utils.adaptive_max_error` imported but unused
neuralops/compression_engine/pruning/bonsai_pruner.py:40:5: F401 [*] `neuralops.core.pruning_utils.get_calibration_tensors` imported but unused
neuralops/compression_engine/quantization/__init__.py:3:8: F401 `sys` imported but unused
neuralops/compression_engine/quantization/awq.py:27:9: F841 Local variable `q_levels` is assigned to but never used
neuralops/compression_engine/quantization/gptq.py:3:8: F401 [*] `math` imported but unused
neuralops/compression_engine/svd/__init__.py:3:8: F401 `sys` imported but unused
neuralops/compression_engine/tensor_decomp/__init__.py:3:8: F401 `sys` imported but unused
neuralops/core/__init__.py:11:25: F401 `typing.Dict` imported but unused
neuralops/core/_attention_extractors.py:11:8: F401 [*] `torch` imported but unused
neuralops/core/_attention_injectors.py:9:25: F401 [*] `typing.Callable` imported but unused
neuralops/core/_attention_surgery.py:9:20: F401 [*] `typing.Any` imported but unused
neuralops/core/cuda_graph.py:3:30: F401 [*] `typing.Any` imported but unused
neuralops/core/replace_attention.py:16:8: F401 [*] `torch` imported but unused
neuralops/core/replace_attention.py:28:5: F401 [*] `neuralops.core._attention_helpers._get_attn_submodule` imported but unused
neuralops/core/replace_attention.py:33:43: F401 [*] `neuralops.core.attention_mask.get_token_valid_mask` imported but unused
neuralops/core/triton_kernels/flash_decoding.py:31:9: F841 Local variable `q` is assigned to but never used
neuralops/core/triton_kernels/flash_decoding.py:33:9: F841 Local variable `start_n` is assigned to but never used
neuralops/core/triton_kernels/fms_scan.py:22:51: F401 [*] `neuralops.core.triton_kernels._common.HAS_TRITON` imported but unused
neuralops/core/triton_kernels/gdn_scan.py:19:51: F401 [*] `neuralops.core.triton_kernels._common.HAS_TRITON` imported but unused
neuralops/core/triton_kernels/gla_scan.py:21:51: F401 [*] `neuralops.core.triton_kernels._common.HAS_TRITON` imported but unused
neuralops/core/triton_kernels/hgrn2_scan.py:19:51: F401 [*] `neuralops.core.triton_kernels._common.HAS_TRITON` imported but unused
neuralops/core/triton_kernels/paged_attention.py:30:9: F841 Local variable `q` is assigned to but never used
neuralops/core/triton_kernels/paged_attention.py:32:9: F841 Local variable `m_i` is assigned to but never used
neuralops/core/triton_kernels/paged_attention.py:33:9: F841 Local variable `l_i` is assigned to but never used
neuralops/core/triton_kernels/paged_attention.py:36:9: F841 Local variable `scale` is assigned to but never used
neuralops/core/triton_kernels/paged_attention.py:39:13: F841 Local variable `physical_block` is assigned to but never used
neuralops/core/triton_kernels/selective_scan.py:18:51: F401 [*] `neuralops.core.triton_kernels._common.HAS_TRITON` imported but unused
neuralops/evaluation/eval_all_models_datasets.py:17:20: F401 [*] `typing.Any` imported but unused
neuralops/inference/engine.py:3:36: F401 [*] `typing.List` imported but unused
neuralops/inference/engine.py:3:42: F401 [*] `typing.Optional` imported but unused
neuralops/inference/engine.py:174:5: F841 Local variable `B` is assigned to but never used
neuralops/inference/paged_cache.py:3:20: F401 [*] `typing.Dict` imported but unused
neuralops/inference/paged_cache.py:3:26: F401 [*] `typing.List` imported but unused
neuralops/inference/paged_cache.py:3:32: F401 [*] `typing.Optional` imported but unused
neuralops/inference/paged_cache.py:3:42: F401 [*] `typing.Tuple` imported but unused
neuralops/integrations/__init__.py:7:5: F401 `neuralops.integrations.distributed_compile.get_deepspeed_config` imported but unused; consider removing, adding to `__all__`, or using a redundant alias
neuralops/integrations/__init__.py:8:5: F401 `neuralops.integrations.distributed_compile.apply_fsdp_wrapping` imported but unused; consider removing, adding to `__all__`, or using a redundant alias
neuralops/integrations/distributed_compile.py:3:20: F401 [*] `typing.Callable` imported but unused
neuralops/integrations/hf_automodel.py:58:16: F401 `transformers` imported but unused; consider using `importlib.util.find_spec` to test for availability
neuralops/integrations/serving.py:4:20: F401 [*] `torch.nn` imported but unused
neuralops/trainer/__init__.py:31:5: F401 `.optimizers.OPTIMIZER_REGISTRY` imported but unused; consider removing, adding to `__all__`, or using a redundant alias
neuralops/training/trainer.py:3:35: F401 [*] `typing.Iterator` imported but unused
neuralops/training/trainer.py:6:30: F401 [*] `torch.utils.data.DataLoader` imported but unused
neuralops/utils/__init__.py:13:5: F401 `.results_saver` imported but unused; consider removing, adding to `__all__`, or using a redundant alias
scripts/gen_api_docs.py:3:8: F401 [*] `sys` imported but unused
scripts/gen_api_docs.py:4:21: F401 [*] `pathlib.Path` imported but unused
scripts/gen_api_docs.py:6:8: F401 [*] `neuralops` imported but unused
scripts/normalize_models.py:4:25: F401 [*] `collections.defaultdict` imported but unused
scripts/normalize_results.py:6:20: F401 [*] `typing.Any` imported but unused
scripts/normalize_results.py:20:34: F401 [*] `neuralops.compression_engine` imported but unused
Found 93 errors.
[*] 52 fixable with the `--fix` option (12 hidden fixes can be enabled with the `--unsafe-fixes` option).
```
