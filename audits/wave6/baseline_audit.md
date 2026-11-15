# NeuralOps Wave 6 — Baseline Audit Report

**Scan Timestamp:** 2026-11-15 00:21:01 UTC
**Total Examples Scanned:** 20

## 1. Summary Status Table

| # | File | Exit Code | Runtime (s) | `torch.randn` Lines | Hardcoded `pass_fail True` | Silent Skips | Status |
|---|---|:---:|:---:|---|---|---|:---:|
| 01 | `01_attention_surgery.py` | 0 | 18.42s | None | 155 | None | **FLAGGED** |
| 02 | `02_model_compression.py` | 1 | 7.59s | None | 168 | None | **FLAGGED** |
| 03 | `03_lora_finetuning.py` | 0 | 8.22s | None | 171 | None | **FLAGGED** |
| 04 | `04_knowledge_distillation.py` | 1 | 18.32s | 182, 187, 182, 187 | 259, 264, 266, 282 | None | **FLAGGED** |
| 05 | `05_fla_style_benchmarking.py` | 0 | 15.23s | None | 187, 193, 209 | None | **FLAGGED** |
| 06 | `06_export_and_push.py` | 0 | 14.13s | None | 225 | None | **FLAGGED** |
| 07 | `07_custom_attention.py` | 0 | 13.64s | 166 | 250 | None | **FLAGGED** |
| 08 | `08_grpo_reasoning.py` | 0 | 15.77s | None | 176 | None | **FLAGGED** |
| 09 | `09_multi_stage_model_surgery.py` | 0 | 12.83s | None | 206 | None | **FLAGGED** |
| 10 | `10_advanced_compression_zoo.py` | 0 | 6.4s | 221, 226 | 326 | None | **FLAGGED** |
| 11 | `11_quantization_fp8_int4_mx.py` | 0 | 5.49s | 193 | None | None | **CLEAN** |
| 12 | `12_structured_pruning_and_token_shedding.py` | 0 | 4.43s | 182, 211, 239, 242 | 269 | None | **FLAGGED** |
| 13 | `13_six_pillar_knowledge_distillation.py` | 0 | 8.88s | 138 | None | None | **FLAGGED** |
| 14 | `14_advanced_optimizers_muon_stiefel.py` | 0 | 6.26s | 79, 98, 145, 97, 137 | 185 | None | **FLAGGED** |
| 15 | `15_serving_vllm_sglang_trtllm.py` | 0 | 4.61s | None | 238, 105, 153, 201 | None | **FLAGGED** |
| 16 | `16_distributed_parallelism.py` | 0 | 4.57s | 206, 207, 208, 226, 227, 228 | 270 | None | **FLAGGED** |
| 17 | `17_recipe_dsl_and_snapshot_rollback.py` | 1 | 6.56s | None | 119, 210 | None | **FLAGGED** |
| 18 | `18_three_level_kernel_dispatch.py` | 0 | 4.73s | 73, 74, 75 | 179, 147, 92, 120 | None | **FLAGGED** |
| 19 | `19_spectral_and_hessian_diagnostics.py` | 1 | 5.06s | 199, 84 | 237 | None | **FLAGGED** |
| 20 | `20_long_context_niah_eval.py` | 0 | 5.12s | 107, 74 | 207 | None | **FLAGGED** |

## 2. Issues Requiring Remediation
- Examples with hardcoded pass_fail True must be refactored to compute boolean status.
- Examples with inline reimplemented pillars must use library APIs or raise NotImplementedError.
- Duplicate and timing-out examples (such as Example 13) must be deleted.
