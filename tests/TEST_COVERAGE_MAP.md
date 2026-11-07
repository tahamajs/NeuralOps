# NeuralOps Test Coverage Map

Inventory of matching flat-root test files mapped to their canonical neuralops subsystems.
Classification method: filename match, content import grep (from/import neuralops.<subsystem>), or both.

## Subsystem: `neuralops.cli`
| Test File | Confirmation Method | Target Path |
| :--- | :--- | :--- |
| `test_cli_canonical.py` | Both (Filename + Import) | `tests/cli/test_cli_canonical.py` |
| `test_cli_safety_contracts.py` | Both (Filename + Import) | `tests/cli/test_cli_safety_contracts.py` |
| `test_exhaustive_dx_cli_and_evaluation.py` | Filename Pattern | `tests/cli/test_exhaustive_dx_cli_and_evaluation.py` |
| `test_installation_and_doctor.py` | Content Import Grep | `tests/cli/test_installation_and_doctor.py` |
| `test_master_exhaustive_ecosystem_integration.py` | Content Import Grep | `tests/cli/test_master_exhaustive_ecosystem_integration.py` |
| `test_packaging_wheel_cli.py` | Filename Pattern | `tests/cli/test_packaging_wheel_cli.py` |
| `test_serve_security.py` | Content Import Grep | `tests/cli/test_serve_security.py` |
| `test_train_cmd_fallback.py` | Content Import Grep | `tests/cli/test_train_cmd_fallback.py` |

## Subsystem: `neuralops.compression`
| Test File | Confirmation Method | Target Path |
| :--- | :--- | :--- |
| `test_aasvd_exhaustive.py` | Content Import Grep | `tests/compression/test_aasvd_exhaustive.py` |
| `test_adaptive_ranks_extra.py` | Content Import Grep | `tests/compression/test_adaptive_ranks_extra.py` |
| `test_adaptive_ranks_jg.py` | Content Import Grep | `tests/compression/test_adaptive_ranks_jg.py` |
| `test_adaptive_svd_verification.py` | Content Import Grep | `tests/compression/test_adaptive_svd_verification.py` |
| `test_adasvd_hardened.py` | Content Import Grep | `tests/compression/test_adasvd_hardened.py` |
| `test_admm_hardened.py` | Content Import Grep | `tests/compression/test_admm_hardened.py` |
| `test_admm_pruner_distill.py` | Content Import Grep | `tests/compression/test_admm_pruner_distill.py` |
| `test_all_203_modules_hardware_kernel_coverage.py` | Content Import Grep | `tests/compression/test_all_203_modules_hardware_kernel_coverage.py` |
| `test_all_aliases.py` | Content Import Grep | `tests/compression/test_all_aliases.py` |
| `test_all_compression_engines.py` | Both (Filename + Import) | `tests/compression/test_all_compression_engines.py` |
| `test_all_compressor_modules.py` | Content Import Grep | `tests/compression/test_all_compressor_modules.py` |
| `test_armor_hardened.py` | Content Import Grep | `tests/compression/test_armor_hardened.py` |
| `test_armor_pruner_hardened.py` | Content Import Grep | `tests/compression/test_armor_pruner_hardened.py` |
| `test_astro_hardened.py` | Content Import Grep | `tests/compression/test_astro_hardened.py` |
| `test_auto_svd_and_nas_search.py` | Content Import Grep | `tests/compression/test_auto_svd_and_nas_search.py` |
| `test_awq_real.py` | Content Import Grep | `tests/compression/test_awq_real.py` |
| `test_benchmark_bugfixes.py` | Content Import Grep | `tests/compression/test_benchmark_bugfixes.py` |
| `test_boundary_and_low_score_modules_hardening.py` | Content Import Grep | `tests/compression/test_boundary_and_low_score_modules_hardening.py` |
| `test_cachenotes_hardened.py` | Content Import Grep | `tests/compression/test_cachenotes_hardened.py` |
| `test_cli_safety_contracts.py` | Content Import Grep | `tests/compression/test_cli_safety_contracts.py` |
| `test_compression_engine_debugging.py` | Both (Filename + Import) | `tests/compression/test_compression_engine_debugging.py` |
| `test_compression_enhancements.py` | Both (Filename + Import) | `tests/compression/test_compression_enhancements.py` |
| `test_core_modules.py` | Content Import Grep | `tests/compression/test_core_modules.py` |
| `test_cur_compressor_comprehensive.py` | Content Import Grep | `tests/compression/test_cur_compressor_comprehensive.py` |
| `test_dimension_alignment.py` | Content Import Grep | `tests/compression/test_dimension_alignment.py` |
| `test_dobi_svd_comprehensive.py` | Content Import Grep | `tests/compression/test_dobi_svd_comprehensive.py` |
| `test_enlighten.py` | Content Import Grep | `tests/compression/test_enlighten.py` |
| `test_exhaustive_72_compressors_matrix.py` | Content Import Grep | `tests/compression/test_exhaustive_72_compressors_matrix.py` |
| `test_exhaustive_autosvd_quarot_w4a16.py` | Content Import Grep | `tests/compression/test_exhaustive_autosvd_quarot_w4a16.py` |
| `test_framework_serving_adapters.py` | Content Import Grep | `tests/compression/test_framework_serving_adapters.py` |
| `test_full_framework_end_to_end.py` | Content Import Grep | `tests/compression/test_full_framework_end_to_end.py` |
| `test_fused_svd_kernel.py` | Content Import Grep | `tests/compression/test_fused_svd_kernel.py` |
| `test_fwsvd_and_curvature_conditioning.py` | Content Import Grep | `tests/compression/test_fwsvd_and_curvature_conditioning.py` |
| `test_gptq_real.py` | Content Import Grep | `tests/compression/test_gptq_real.py` |
| `test_hnc_svd_exhaustive.py` | Content Import Grep | `tests/compression/test_hnc_svd_exhaustive.py` |
| `test_huffman_coder.py` | Content Import Grep | `tests/compression/test_huffman_coder.py` |
| `test_imports.py` | Content Import Grep | `tests/compression/test_imports.py` |
| `test_inference_acceleration.py` | Content Import Grep | `tests/compression/test_inference_acceleration.py` |
| `test_integration_all_families.py` | Content Import Grep | `tests/compression/test_integration_all_families.py` |
| `test_integration_end_to_end.py` | Content Import Grep | `tests/compression/test_integration_end_to_end.py` |
| `test_integration_pipeline.py` | Content Import Grep | `tests/compression/test_integration_pipeline.py` |
| `test_integration_regression.py` | Content Import Grep | `tests/compression/test_integration_regression.py` |
| `test_integration_smoke.py` | Content Import Grep | `tests/compression/test_integration_smoke.py` |
| `test_kvtc_hardened.py` | Content Import Grep | `tests/compression/test_kvtc_hardened.py` |
| `test_lazy_architecture.py` | Content Import Grep | `tests/compression/test_lazy_architecture.py` |
| `test_lazy_family_imports.py` | Content Import Grep | `tests/compression/test_lazy_family_imports.py` |
| `test_lora_qlora_dora_real.py` | Content Import Grep | `tests/compression/test_lora_qlora_dora_real.py` |
| `test_models_and_datasets_matrix.py` | Content Import Grep | `tests/compression/test_models_and_datasets_matrix.py` |
| `test_quantization_and_pruning_systems.py` | Content Import Grep | `tests/compression/test_quantization_and_pruning_systems.py` |
| `test_qwen3_5_compressor_awareness.py` | Content Import Grep | `tests/compression/test_qwen3_5_compressor_awareness.py` |
| `test_smoke_modules.py` | Content Import Grep | `tests/compression/test_smoke_modules.py` |
| `test_smoothquant_real.py` | Content Import Grep | `tests/compression/test_smoothquant_real.py` |
| `test_sota_compression.py` | Both (Filename + Import) | `tests/compression/test_sota_compression.py` |
| `test_surgery_manifest_roundtrip.py` | Content Import Grep | `tests/compression/test_surgery_manifest_roundtrip.py` |
| `test_svd_audit_suite.py` | Content Import Grep | `tests/compression/test_svd_audit_suite.py` |
| `test_svd_lowrank_comprehensive.py` | Content Import Grep | `tests/compression/test_svd_lowrank_comprehensive.py` |
| `test_track2_spectral_ssm_audit.py` | Content Import Grep | `tests/compression/test_track2_spectral_ssm_audit.py` |
| `test_v135_flagship_subsystems.py` | Content Import Grep | `tests/compression/test_v135_flagship_subsystems.py` |
| `test_w4a16_and_hadamard_svd.py` | Content Import Grep | `tests/compression/test_w4a16_and_hadamard_svd.py` |
| `test_wanda_real.py` | Content Import Grep | `tests/compression/test_wanda_real.py` |

## Subsystem: `neuralops.evaluation`
| Test File | Confirmation Method | Target Path |
| :--- | :--- | :--- |
| `test_all_compression_engines.py` | Content Import Grep | `tests/evaluation/test_all_compression_engines.py` |
| `test_eval_data.py` | Content Import Grep | `tests/evaluation/test_eval_data.py` |
| `test_eval_patcher.py` | Content Import Grep | `tests/evaluation/test_eval_patcher.py` |
| `test_eval_pipeline.py` | Content Import Grep | `tests/evaluation/test_eval_pipeline.py` |
| `test_eval_types.py` | Content Import Grep | `tests/evaluation/test_eval_types.py` |
| `test_eval_utils.py` | Content Import Grep | `tests/evaluation/test_eval_utils.py` |
| `test_exhaustive_dx_cli_and_evaluation.py` | Filename Pattern | `tests/evaluation/test_exhaustive_dx_cli_and_evaluation.py` |
| `test_fla_evaluation_and_benchmarking.py` | Both (Filename + Import) | `tests/evaluation/test_fla_evaluation_and_benchmarking.py` |
| `test_niah_and_lm_eval_adapter.py` | Content Import Grep | `tests/evaluation/test_niah_and_lm_eval_adapter.py` |
| `test_regression_suite.py` | Content Import Grep | `tests/evaluation/test_regression_suite.py` |
| `test_training_tracking_evaluation_regressions.py` | Both (Filename + Import) | `tests/evaluation/test_training_tracking_evaluation_regressions.py` |

## Subsystem: `neuralops.export`
| Test File | Confirmation Method | Target Path |
| :--- | :--- | :--- |
| `test_benchmark_pipeline.py` | Content Import Grep | `tests/export/test_benchmark_pipeline.py` |
| `test_cli_canonical.py` | Content Import Grep | `tests/export/test_cli_canonical.py` |
| `test_compile_and_export.py` | Both (Filename + Import) | `tests/export/test_compile_and_export.py` |
| `test_exhaustive_end_to_end_lifecycle.py` | Content Import Grep | `tests/export/test_exhaustive_end_to_end_lifecycle.py` |
| `test_exhaustive_export_and_serialization.py` | Both (Filename + Import) | `tests/export/test_exhaustive_export_and_serialization.py` |
| `test_framework_serving_adapters.py` | Content Import Grep | `tests/export/test_framework_serving_adapters.py` |
| `test_gguf_edge_export.py` | Both (Filename + Import) | `tests/export/test_gguf_edge_export.py` |
| `test_huggingface_export_roundtrip.py` | Filename Pattern | `tests/export/test_huggingface_export_roundtrip.py` |
| `test_integrations.py` | Content Import Grep | `tests/export/test_integrations.py` |
| `test_no_silent_swallow.py` | Content Import Grep | `tests/export/test_no_silent_swallow.py` |
| `test_serving_matrix_all_nine.py` | Content Import Grep | `tests/export/test_serving_matrix_all_nine.py` |
| `test_silent_failures.py` | Content Import Grep | `tests/export/test_silent_failures.py` |
| `test_surgery_manifest_roundtrip.py` | Content Import Grep | `tests/export/test_surgery_manifest_roundtrip.py` |
| `test_universal_surgery_director.py` | Content Import Grep | `tests/export/test_universal_surgery_director.py` |
| `test_v13_complete_features.py` | Content Import Grep | `tests/export/test_v13_complete_features.py` |

## Subsystem: `neuralops.utils`
| Test File | Confirmation Method | Target Path |
| :--- | :--- | :--- |
| `test_adasvd_hardened.py` | Content Import Grep | `tests/utils/test_adasvd_hardened.py` |
| `test_admm_hardened.py` | Content Import Grep | `tests/utils/test_admm_hardened.py` |
| `test_admm_pruner_distill.py` | Content Import Grep | `tests/utils/test_admm_pruner_distill.py` |
| `test_amp_safety.py` | Content Import Grep | `tests/utils/test_amp_safety.py` |
| `test_armor_hardened.py` | Content Import Grep | `tests/utils/test_armor_hardened.py` |
| `test_attention_linear_family.py` | Content Import Grep | `tests/utils/test_attention_linear_family.py` |
| `test_attention_modules_compat.py` | Content Import Grep | `tests/utils/test_attention_modules_compat.py` |
| `test_attention_profiler.py` | Content Import Grep | `tests/utils/test_attention_profiler.py` |
| `test_attention_ssm_family.py` | Content Import Grep | `tests/utils/test_attention_ssm_family.py` |
| `test_attention_standard_family.py` | Content Import Grep | `tests/utils/test_attention_standard_family.py` |
| `test_attention_svd_family.py` | Content Import Grep | `tests/utils/test_attention_svd_family.py` |
| `test_attention_utils.py` | Filename Pattern | `tests/utils/test_attention_utils.py` |
| `test_compat.py` | Content Import Grep | `tests/utils/test_compat.py` |
| `test_data_utils.py` | Filename Pattern | `tests/utils/test_data_utils.py` |
| `test_eval_utils.py` | Filename Pattern | `tests/utils/test_eval_utils.py` |
| `test_full_framework_end_to_end.py` | Content Import Grep | `tests/utils/test_full_framework_end_to_end.py` |
| `test_gqa_utils.py` | Filename Pattern | `tests/utils/test_gqa_utils.py` |
| `test_hybrid_freeze.py` | Content Import Grep | `tests/utils/test_hybrid_freeze.py` |
| `test_integration_end_to_end.py` | Content Import Grep | `tests/utils/test_integration_end_to_end.py` |
| `test_lora.py` | Content Import Grep | `tests/utils/test_lora.py` |
| `test_model_family_compat.py` | Content Import Grep | `tests/utils/test_model_family_compat.py` |
| `test_multiarch_modules.py` | Content Import Grep | `tests/utils/test_multiarch_modules.py` |
| `test_part4_dynamic_lora_alias.py` | Content Import Grep | `tests/utils/test_part4_dynamic_lora_alias.py` |
| `test_part5_compressor_lora_unmerged.py` | Content Import Grep | `tests/utils/test_part5_compressor_lora_unmerged.py` |
| `test_part7_numerical_safety.py` | Content Import Grep | `tests/utils/test_part7_numerical_safety.py` |
| `test_part8_recovery_targets.py` | Content Import Grep | `tests/utils/test_part8_recovery_targets.py` |
| `test_part9_output_tuple.py` | Content Import Grep | `tests/utils/test_part9_output_tuple.py` |
| `test_public_api.py` | Content Import Grep | `tests/utils/test_public_api.py` |
| `test_qwen3_5_comprehensive.py` | Content Import Grep | `tests/utils/test_qwen3_5_comprehensive.py` |
| `test_qwen3_5_compressor_awareness.py` | Content Import Grep | `tests/utils/test_qwen3_5_compressor_awareness.py` |
| `test_sweep_orchestration.py` | Content Import Grep | `tests/utils/test_sweep_orchestration.py` |
| `test_t5_all_attention_compat.py` | Content Import Grep | `tests/utils/test_t5_all_attention_compat.py` |
| `test_v17_hardening.py` | Content Import Grep | `tests/utils/test_v17_hardening.py` |
| `test_v18_dispatch_cache.py` | Content Import Grep | `tests/utils/test_v18_dispatch_cache.py` |
| `test_weight_copy.py` | Content Import Grep | `tests/utils/test_weight_copy.py` |

## Subsystem: `neuralops.adapters`
| Test File | Confirmation Method | Target Path |
| :--- | :--- | :--- |
| `test_framework_serving_adapters.py` | Filename Pattern | `tests/adapters/test_framework_serving_adapters.py` |

## Subsystem: `neuralops.kernels`
| Test File | Confirmation Method | Target Path |
| :--- | :--- | :--- |
| `test_13_flagship_triton_kernels.py` | Filename Pattern | `tests/kernels/test_13_flagship_triton_kernels.py` |
| `test_all_cuda_and_triton_kernels_exhaustive.py` | Filename Pattern | `tests/kernels/test_all_cuda_and_triton_kernels_exhaustive.py` |
| `test_all_cuda_kernels_comprehensive.py` | Filename Pattern | `tests/kernels/test_all_cuda_kernels_comprehensive.py` |
| `test_cuda_catalog.py` | Content Import Grep | `tests/kernels/test_cuda_catalog.py` |
| `test_cuda_kernels_lazy.py` | Filename Pattern | `tests/kernels/test_cuda_kernels_lazy.py` |
| `test_elementwise_kernels.py` | Filename Pattern | `tests/kernels/test_elementwise_kernels.py` |
| `test_exhaustive_26_triton_kernels_matrix.py` | Filename Pattern | `tests/kernels/test_exhaustive_26_triton_kernels_matrix.py` |
| `test_extended_20_kernels_matrix.py` | Both (Filename + Import) | `tests/kernels/test_extended_20_kernels_matrix.py` |
| `test_fp8_and_new_triton_kernels.py` | Filename Pattern | `tests/kernels/test_fp8_and_new_triton_kernels.py` |
| `test_kernels_unified.py` | Both (Filename + Import) | `tests/kernels/test_kernels_unified.py` |
| `test_triton_all_kernels.py` | Both (Filename + Import) | `tests/kernels/test_triton_all_kernels.py` |
| `test_triton_kernels_suite.py` | Filename Pattern | `tests/kernels/test_triton_kernels_suite.py` |

## Subsystem: `neuralops.layers`
| Test File | Confirmation Method | Target Path |
| :--- | :--- | :--- |
| `test_lazy_architecture.py` | Content Import Grep | `tests/layers/test_lazy_architecture.py` |
| `test_phase2_migration.py` | Content Import Grep | `tests/layers/test_phase2_migration.py` |
| `test_selective_ssm_zoh_discretization.py` | Content Import Grep | `tests/layers/test_selective_ssm_zoh_discretization.py` |
| `test_silent_failures.py` | Content Import Grep | `tests/layers/test_silent_failures.py` |

## Subsystem: `neuralops.ops`
| Test File | Confirmation Method | Target Path |
| :--- | :--- | :--- |
| `test_alibi.py` | Content Import Grep | `tests/ops/test_alibi.py` |
| `test_backend_forcing_phase11.py` | Content Import Grep | `tests/ops/test_backend_forcing_phase11.py` |
| `test_cuda_catalog.py` | Content Import Grep | `tests/ops/test_cuda_catalog.py` |
| `test_cuda_triton_3level_complete.py` | Content Import Grep | `tests/ops/test_cuda_triton_3level_complete.py` |
| `test_dimension_alignment.py` | Content Import Grep | `tests/ops/test_dimension_alignment.py` |
| `test_dispatch_forcing_phase11.py` | Content Import Grep | `tests/ops/test_dispatch_forcing_phase11.py` |
| `test_flash_gqa_cuda_and_autograd.py` | Content Import Grep | `tests/ops/test_flash_gqa_cuda_and_autograd.py` |
| `test_lazy_architecture.py` | Content Import Grep | `tests/ops/test_lazy_architecture.py` |
| `test_library_health.py` | Content Import Grep | `tests/ops/test_library_health.py` |
| `test_moe_routing_cuda_and_autograd.py` | Content Import Grep | `tests/ops/test_moe_routing_cuda_and_autograd.py` |
| `test_ops_dispatch.py` | Both (Filename + Import) | `tests/ops/test_ops_dispatch.py` |
| `test_ops_triton_imports.py` | Both (Filename + Import) | `tests/ops/test_ops_triton_imports.py` |
| `test_paged_linear_attention_vllm.py` | Content Import Grep | `tests/ops/test_paged_linear_attention_vllm.py` |
| `test_serving_matrix_all_nine.py` | Content Import Grep | `tests/ops/test_serving_matrix_all_nine.py` |
| `test_triton_fms_scan.py` | Content Import Grep | `tests/ops/test_triton_fms_scan.py` |
| `test_triton_gdn_scan.py` | Content Import Grep | `tests/ops/test_triton_gdn_scan.py` |
| `test_triton_gla_scan.py` | Content Import Grep | `tests/ops/test_triton_gla_scan.py` |
| `test_universal_gpu_compatibility.py` | Content Import Grep | `tests/ops/test_universal_gpu_compatibility.py` |
| `test_v17_hardening.py` | Content Import Grep | `tests/ops/test_v17_hardening.py` |

## Subsystem: `neuralops.recipe`
| Test File | Confirmation Method | Target Path |
| :--- | :--- | :--- |
| `test_exhaustive_dx_cli_and_evaluation.py` | Content Import Grep | `tests/recipe/test_exhaustive_dx_cli_and_evaluation.py` |
| `test_v13_complete_features.py` | Content Import Grep | `tests/recipe/test_v13_complete_features.py` |

## Subsystem: `neuralops.results_saver`
| Test File | Confirmation Method | Target Path |
| :--- | :--- | :--- |
| `test_parquet_and_storage.py` | Content Import Grep | `tests/results_saver/test_parquet_and_storage.py` |
| `test_results_saver.py` | Both (Filename + Import) | `tests/results_saver/test_results_saver.py` |

## Subsystem: `neuralops.runners`
| Test File | Confirmation Method | Target Path |
| :--- | :--- | :--- |
| `test_monitor_server.py` | Content Import Grep | `tests/runners/test_monitor_server.py` |
| `test_runners.py` | Both (Filename + Import) | `tests/runners/test_runners.py` |
| `test_runners_imports.py` | Filename Pattern | `tests/runners/test_runners_imports.py` |

## Subsystem: `neuralops.runtime`
| Test File | Confirmation Method | Target Path |
| :--- | :--- | :--- |
| `test_circuit_breaker.py` | Content Import Grep | `tests/runtime/test_circuit_breaker.py` |
