#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Core Modules.
"""

from __future__ import annotations

import importlib
import sys

def _load_evaluation_package() -> Any:
    """Load the canonical framework evaluation package."""
    return importlib.import_module("neuralops.evaluation")
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from neuralops.api import (
        AutoNeuralOpsForCausalLM,
        AutoNeuralOpsForSequenceClassification,
        AutoNeuralOpsModel,
        available_backends,
        benchmark,
        benchmark_all,
        collect_surgery_distillation_loss,
        compare,
        compare_attentions_ablation,  # noqa: F401
        compile_to_so,
        compress,
        distill,
        evaluate,
        export_model,
        finalize_surgery,
        get_attention_class,
        get_compressor_class,
        get_version,
        has_triton,
        info,
        list_attentions,
        list_compressors,
        list_families,
        load_exported_model,
        load_model,
        merge_for_inference,
        oneshot,
        patch_transformers,
        push_to_hub,
        recommend,
        replace_attention,
        set_surgery_alpha,
        set_surgery_stage,
        summary,
        surgical_transplant_and_heal_pipeline,  # noqa: F401
        train,
        transplant_attention,
        validate,
    )
    from neuralops.core.exceptions import (
        BackendError,
        ConfigError,
        NeuralOpsError,
        NumericalError,
        WeightCopyError,
    )
    from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY  # noqa: F401
    from neuralops.kernels import list_cuda_kernels, list_triton_kernels  # noqa: F401

__version__ = "1.5.0"

import importlib.machinery
from importlib.abc import MetaPathFinder


class _AliasLoader:
    def __init__(self, mod: Any) -> None:
        """Execute __init__ operation."""
        self.mod = mod

    def create_module(self, _spec: Any) -> Any:
        """Execute create_module operation."""
        return self.mod

    def exec_module(self, module: Any) -> None:
        """Execute exec_module operation."""


class _ModuleAliasFinder(MetaPathFinder):
    _ALIASES = {
        "neuralops.attentions.gqa": "neuralops.attentions.gqa_attention",
        "neuralops.attentions.galerkin": "neuralops.attentions.galerkin_attention",
        "neuralops.attentions.galerkin_inject": "neuralops.attentions.galerkin_attention",
        "neuralops.attentions.adaptive_span_inject": "neuralops.attentions.adaptive_span_attention",
        "neuralops.attentions.adaptive_sparse_inject": "neuralops.attentions.adaptive_sparse_attention",
        "neuralops.attentions.adasplash_inject": "neuralops.attentions.adasplash_attention",
        "neuralops.attentions.aegis_hybrid_attention": "neuralops.attentions.aegis_hybrid",
        "neuralops.attentions.aegis_v1": "neuralops.attentions.aegis_v2",
        "neuralops.attentions.aegis_v3": "neuralops.attentions.aegis_v2",
        "neuralops.attentions.aegis_v4": "neuralops.attentions.aegis_v2",
        "neuralops.attentions.minimal_aurora": "neuralops.attentions.minimal_svd",
        "neuralops.attentions.minimal_prism": "neuralops.attentions.minimal_svd",
        "neuralops.attentions.minimal_svd_v2": "neuralops.attentions.minimal_svd",
        "neuralops.attentions.cluster_inject": "neuralops.attentions.cluster_attention",
        "neuralops.attentions.compressed_mha": "neuralops.attentions.compressed_mha_attention",
        "neuralops.attentions.cosformer_inject": "neuralops.attentions.cosformer",
        "neuralops.attentions.hla_inject": "neuralops.attentions.hla_attention",
        "neuralops.attentions.kda_inject": "neuralops.attentions.kda_attention",
        "neuralops.attentions.leka_v1_inject": "neuralops.attentions.leka_v1",
        "neuralops.attentions.leka_v2_inject": "neuralops.attentions.leka_v2",
        "neuralops.attentions.leka_v3_inject": "neuralops.attentions.leka_v3",
        "neuralops.attentions.Mamba": "neuralops.attentions.mamba_svd_flash",
        "neuralops.attentions.SVD-Mamba": "neuralops.attentions.svd_mamba",
        "neuralops.attentions.mamba_attention": "neuralops.attentions.mamba_block",
        "neuralops.attentions.moba": "neuralops.attentions.moba_attention",
        "neuralops.attentions.nha_compressor": "neuralops.attentions.nha_attention",
        "neuralops.attentions.rpc": "neuralops.attentions.rpc_attention",
        "neuralops.attentions.salt": "neuralops.attentions.salt_attention",
        "neuralops.attentions.soa_fused_module": "neuralops.attentions.soa_attn",
        "neuralops.attentions.SOAFusedAttention": "neuralops.attentions.soa_attn",
        "neuralops.attentions.soa_module": "neuralops.attentions.soa_attn",
        "neuralops.attentions.spectra_v2_attention": "neuralops.attentions.spectra_v2",
        "neuralops.attentions.svd_compressed_attention": "neuralops.attentions.svd_compressed",
        "neuralops.attentions.svd_llm_v2": "neuralops.attentions.svd_llm_v2_attention",
        "neuralops.attentions.svd_proj_attn": "neuralops.attentions.svd_projection_attention",
        "neuralops.attentions.token_compressed": "neuralops.attentions.token_compressed_attention",
        "neuralops.attentions.tt": "neuralops.attentions.token_compressed_attention",
        "neuralops.attentions.upi_mamba": "neuralops.attentions.upi_stuffed_mamba",
        "neuralops.attentions.mamba": "neuralops.attentions.mamba_block",
        "neuralops.attentions.mamba_inject": "neuralops.attentions.mamba_block",
        "neuralops.attentions.simple_ssm": "neuralops.attentions.selective_ssm",
        "neuralops.attentions.smart_weighted_ensemble": "neuralops.attentions.hybrid_attention",
        "neuralops.attentions.weighted_ensemble": "neuralops.attentions.hybrid_attention",
        "neuralops.attentions.spectron_attention": "neuralops.attentions.spectron",
        "neuralops.attentions.squeezing_heads": "neuralops.attentions.squeezing_heads_distill",
        "neuralops.attentions.squeezing_heads_distiller": "neuralops.attentions.squeezing_heads_distill",
        "neuralops.attentions.spectral_mamba_attention": "neuralops.attentions.spectral_mamba",
        "neuralops.attentions.svd_sequence_attention": "neuralops.attentions.svd_attention",
        "neuralops.attentions.rala_compressor": "neuralops.attentions.rala_attention",
        "neuralops.attentions.selective_scan": "neuralops.core.selective_scan",
        "neuralops.attentions.ssm_scan": "neuralops.core.selective_scan",
        "neuralops.attentions.legacy_ensemble": "neuralops.core.legacy_ensemble",
        "neuralops.compression_engine.aasvd": "neuralops.compression_engine.aa_svd_compressor",
        "neuralops.compression_engine.adasvd": "neuralops.compression_engine.adasvd_compressor",
        "neuralops.compression_engine.cur_moe": "neuralops.compression_engine.cur_moe_compressor",
        "neuralops.compression_engine.frandi": "neuralops.compression_engine.frandi_compressor",
        "neuralops.compression_engine.grecodec": "neuralops.compression_engine.grecodec_compressor",
        "neuralops.compression_engine.hybridkv": "neuralops.compression_engine.hybridkv_compressor",
        "neuralops.compression_engine.kvtc": "neuralops.compression_engine.kvtc_compressor",
        "neuralops.compression_engine.modegpt": "neuralops.compression_engine.modegpt_compressor",
        "neuralops.compression_engine.oats": "neuralops.compression_engine.oats_compressor",
        "neuralops.compression_engine.proxsparse_pruner": "neuralops.compression_engine.proxsparse",
        "neuralops.compression_engine.rpca": "neuralops.compression_engine.rpca_compressor",
        "neuralops.compression_engine.saes_svd": "neuralops.compression_engine.saes_svd_v2",
        "neuralops.compression_engine.saes_svd_compressor": "neuralops.compression_engine.saes_svd_v2",
        "neuralops.compression_engine.sdrm": "neuralops.compression_engine.sdrm_compressor",
        "neuralops.compression_engine.slicegpt_compressor": "neuralops.compression_engine.slicegpt",
        "neuralops.compression_engine.soka": "neuralops.compression_engine.soka_compressor",
        "neuralops.compression_engine.swift_svd_compressor": "neuralops.compression_engine.swift_svd_act",
        "neuralops.compression_engine.zerokv_neo": "neuralops.compression_engine.zerokv_neo_compressor",
        "neuralops.compression_engine.cure": "neuralops.compression_engine.cur_moe_compressor",
        "neuralops.compression_engine.dobi": "neuralops.compression_engine.dobi_svd",
        "neuralops.compression_engine.dynamic_entropy": "neuralops.compression_engine.dynamic_rank",
        "neuralops.compression_engine.tomoe": "neuralops.compression_engine.tomoe_compressor",
        "neuralops.compression_engine.adaptive_compressor": "neuralops.compression_engine.adaptive_ranks",
        "neuralops.compression_engine.padre_attention": "neuralops.attentions.padre_attention",
        "neuralops.compression_engine.spectraformer_attention": "neuralops.attentions.spectraformer_attention",
        "neuralops.compression_engine.tlinformer_attention": "neuralops.attentions.tlinformer_attention",
        "neuralops.compression_engine.pruning_utils": "neuralops.core.pruning_utils",
        "neuralops.compression_engine.cur_linear": "neuralops.compression_engine.cur_compressor",
        "neuralops.compression_engine.kronecker_linear": "neuralops.compression_engine.kronecker_compressor",
        "neuralops.compression_engine.tt_linear": "neuralops.compression_engine.tt_compressor",
        "neuralops.compression_engine.hashed_linear": "neuralops.compression_engine.hashed_net_compressor",
        "neuralops.compression_engine.come": "neuralops.compression_engine.come_compressor",
        "neuralops.compression_engine.rala_compressor": "neuralops.attentions.rala_attention",
        "neuralops.run_quality_sweep": "neuralops.benchmarks.run_quality_sweep",
        "neuralops.run_rank_sweeper": "neuralops.benchmarks.run_rank_sweeper",
        "neuralops.run_all_benchmarks": "neuralops.benchmarks.run_all_benchmarks",
        "neuralops.run_gpt2_sweep": "neuralops.benchmarks.run_gpt2_sweep",
        "neuralops.run_long_context_sweep": "neuralops.benchmarks.run_long_context_sweep",
        "neuralops.run_seqlen_scaling": "neuralops.benchmarks.run_seqlen_scaling",
        "neuralops.run_seqlen_scaling_bench": "neuralops.benchmarks.run_seqlen_scaling_bench",
        "neuralops.run_multiseed": "neuralops.benchmarks.run_multiseed",
        "neuralops.run_multiseed_rigor": "neuralops.benchmarks.run_multiseed_rigor",
        "neuralops.run_one_module_isolated": "neuralops.benchmarks.run_one_module_isolated",
        "neuralops.run_ruler_benchmark": "neuralops.benchmarks.run_ruler_benchmark",
        "neuralops.run_whitened_svd": "neuralops.benchmarks.run_whitened_svd",
        "neuralops.launch_parallel_sweep": "neuralops.benchmarks.launch_parallel_sweep",
        "neuralops.run_all_modules_scheduler": "neuralops.benchmarks.run_all_modules_scheduler",
        "neuralops.kd_quick_test": "neuralops.benchmarks.kd_quick_test",
        "neuralops.eval_all_models_datasets": "neuralops.evaluation.eval_all_models_datasets",
        "neuralops.eval_perplexity": "neuralops.evaluation.eval_perplexity",
        "neuralops.full_mnli_validation": "neuralops.evaluation.full_mnli_validation",
        "neuralops.run_lm_eval_harness": "neuralops.evaluation.run_lm_eval_harness",
        "neuralops.run_baseline_comparison": "neuralops.evaluation.run_baseline_comparison",
        "neuralops.train_lra": "neuralops.trainer.train_lra",
        "neuralops.generate_all_modules_analysis": "neuralops.analysis.generate_all_modules_analysis",
        "neuralops.core.triton_kernels.block_sparse_streaming_attn": "neuralops.kernels.triton.block_sparse_streaming_attn",
        "neuralops.core.triton_kernels.chunked_delta_gate_scan": "neuralops.kernels.triton.chunked_delta_gate_scan",
        "neuralops.core.triton_kernels.cosformer_linear_attention": "neuralops.kernels.triton.cosformer_linear_attention",
        "neuralops.core.triton_kernels.fused_act_scale_quant_gemm": "neuralops.kernels.triton.fused_act_scale_quant_gemm",
        "neuralops.core.triton_kernels.fused_hadamard_w4a16_gemm": "neuralops.kernels.triton.fused_hadamard_w4a16_gemm",
        "neuralops.core.triton_kernels.fused_kronecker_matmul": "neuralops.kernels.triton.fused_kronecker_matmul",
        "neuralops.core.triton_kernels.fused_latent_kv_cache_attention": "neuralops.kernels.triton.fused_latent_kv_cache_attention",
        "neuralops.core.triton_kernels.fused_lowrank_gemm": "neuralops.kernels.triton.fused_lowrank_gemm",
        "neuralops.core.triton_kernels.hgrn2_fused_bwd_scan": "neuralops.kernels.triton.hgrn2_fused_bwd_scan",
        "neuralops.core.triton_kernels.lut_codebook_dequant_gemm": "neuralops.kernels.triton.lut_codebook_dequant_gemm",
        "neuralops.core.triton_kernels.rpca_sparse_gemm": "neuralops.kernels.triton.rpca_sparse_gemm",
        "neuralops.core.triton_kernels.spectral_entropy_svd_gate": "neuralops.kernels.triton.spectral_entropy_svd_gate",
        "neuralops.core.triton_kernels.strided_tt": "neuralops.kernels.triton.strided_tt",
        "neuralops.kernels.triton.gla_scan": "neuralops.core.triton_kernels.gla_scan",
        "neuralops.kernels.triton.selective_scan": "neuralops.core.triton_kernels.selective_scan",
        "neuralops.kernels.triton.fms_scan": "neuralops.core.triton_kernels.fms_scan",
        "neuralops.kernels.triton.paged_attention": "neuralops.core.triton_kernels.paged_attention",
        "neuralops.kernels.triton.flash_decoding": "neuralops.core.triton_kernels.flash_decoding",
        "neuralops.kernels.triton.sliding_window": "neuralops.core.triton_kernels.sliding_window",
        "neuralops.kernels.triton.fused_ce": "neuralops.core.triton_kernels.fused_ce",
        "neuralops.kernels.triton.fused_kd": "neuralops.core.triton_kernels.fused_kd",
        "neuralops.kernels.triton.fused_norm_linear": "neuralops.core.triton_kernels.fused_norm_linear",
        "neuralops.kernels.triton.fused_rope": "neuralops.core.triton_kernels.fused_rope",
        "neuralops.kernels.triton.low_rank": "neuralops.core.triton_kernels.low_rank",
        "neuralops.kernels.triton.elementwise": "neuralops.core.triton_kernels.elementwise",
        "neuralops.kernels.triton.linear_attn": "neuralops.core.triton_kernels.linear_attn",
        "neuralops.kernels.triton.cosformer_triton_kernel": "neuralops.core.triton_kernels.cosformer_triton_kernel",
        "neuralops.kernels.triton.block_sparse": "neuralops.core.triton_kernels.block_sparse",
        "neuralops.kernels.triton.sparse_attn": "neuralops.core.triton_kernels.sparse_attn",
        "neuralops.kernels.triton.gqa": "neuralops.core.triton_kernels.gqa",
        "neuralops.kernels.triton.varlen_attention": "neuralops.core.triton_kernels.varlen_attention",
        "neuralops.kernels.triton.state_update": "neuralops.core.triton_kernels.state_update",
        "neuralops.kernels.triton.softplus": "neuralops.core.triton_kernels.softplus",
        "neuralops.kernels.triton.autotuner": "neuralops.core.triton_kernels.autotuner",
        "neuralops.kernels.triton.hardware_profile": "neuralops.core.triton_kernels.hardware_profile",
        "neuralops.kernels.triton.op_registry": "neuralops.core.triton_kernels.op_registry",
        "neuralops.kernels.triton._common": "neuralops.core.triton_kernels._common",
        "neuralops.kernels.triton._types": "neuralops.core.triton_kernels._types",
        "neuralops.kernels.triton.differential_attention": "neuralops.core.triton_kernels.differential_attention",
        "neuralops.kernels.triton.hydra_attention": "neuralops.core.triton_kernels.hydra_attention",
        "neuralops.kernels.triton.retnet_retention": "neuralops.core.triton_kernels.retnet_retention",
        "neuralops.kernels.triton.dynamic_sparse_attention": "neuralops.core.triton_kernels.dynamic_sparse_attention",
        "neuralops.kernels.triton.cur_linear": "neuralops.core.triton_kernels.cur_linear",
    }

    def find_spec(
        self, fullname: str, path: Any, target: Any = None
    ) -> Any | None:
        """Execute find_spec operation."""
        if fullname.startswith("neuralops.kernels.cuda."):
            sub = fullname[len("neuralops.kernels.cuda.") :]
            canonical = f"neuralops.core.cuda_kernels.{sub}"
            try:
                mod = importlib.import_module(canonical)
                sys.modules[fullname] = mod
                return getattr(mod, "__spec__", None)
            except Exception:
                pass
        if fullname in self._ALIASES:
            canonical = self._ALIASES[fullname]
            try:
                mod = importlib.import_module(canonical)
                sys.modules[fullname] = mod
                return getattr(mod, "__spec__", None)
            except Exception:
                pass
        return None


if not any(isinstance(finder, _ModuleAliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _ModuleAliasFinder())

def _patch_pascal_svd_if_needed():
    if "torch" in sys.modules:
        try:
            import torch
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()
                if cap < (7, 0) and not getattr(torch.linalg, "_pascal_patched", False):
                    _orig_torch_linalg_svd = torch.linalg.svd
                    def _safe_pascal_linalg_svd(A, full_matrices=True, *, driver=None, out=None):
                        if isinstance(A, torch.Tensor) and A.is_cuda:
                            dev = A.device
                            A_cpu = A.detach().cpu().float()
                            U_cpu, S_cpu, Vh_cpu = _orig_torch_linalg_svd(A_cpu, full_matrices=full_matrices)
                            U = U_cpu.to(device=dev, dtype=A.dtype if A.dtype != torch.float16 else torch.float32)
                            S = S_cpu.to(device=dev, dtype=A.dtype if A.dtype != torch.float16 else torch.float32)
                            Vh = Vh_cpu.to(device=dev, dtype=A.dtype if A.dtype != torch.float16 else torch.float32)
                            if out is not None:
                                out[0].copy_(U)
                                out[1].copy_(S)
                                out[2].copy_(Vh)
                                return out
                            return (U, S, Vh)
                        return _orig_torch_linalg_svd(A, full_matrices=full_matrices, driver=driver, out=out)
                    torch.linalg.svd = _safe_pascal_linalg_svd
                    torch.linalg._pascal_patched = True
        except Exception:
            pass

_patch_pascal_svd_if_needed()


_LAZY_EXPORTS = {
    "ABTestResult": ("neuralops.evaluation.ab_testing", "ABTestResult"),
    "ABTester": ("neuralops.evaluation.ab_testing", "ABTester"),
    "ATTENTION_REGISTRY": ("neuralops.core.registry", "ATTENTION_REGISTRY"),
    "BaseAttention": ("neuralops.core.base", "BaseAttention"),
    "BaseCompressor": ("neuralops.core.base", "BaseCompressor"),
    "ArchitectureSpec": ("neuralops.core.inspector", "ArchitectureSpec"),
    "inspect_architecture": ("neuralops.core.inspector", "inspect_architecture"),
    "render_architecture_card": ("neuralops.core.inspector", "render_architecture_card"),
    "ComposableAttention": ("neuralops.core.composable", "ComposableAttention"),
    "compose_attention": ("neuralops.core.composable", "compose_attention"),
    "register_composable_projection": ("neuralops.core.composable", "register_composable_projection"),
    "register_composable_kernel": ("neuralops.core.composable", "register_composable_kernel"),
    "register_composable_mask": ("neuralops.core.composable", "register_composable_mask"),
    "register_composable_engine": ("neuralops.core.composable", "register_composable_engine"),
    "register_custom_attention": ("neuralops.core.extensibility", "register_custom_attention"),
    "register_custom_compressor": ("neuralops.core.extensibility", "register_custom_compressor"),
    "unregister_custom_module": ("neuralops.core.extensibility", "unregister_custom_module"),
    "validate_module_contract": ("neuralops.core.extensibility", "validate_module_contract"),
    "scaffold_module": ("neuralops.core.extensibility", "scaffold_module"),
    "calibrate_attention_subspace": ("neuralops.core._attention_surgery", "calibrate_attention_subspace"),
    "heal_replaced_model": ("neuralops.core._attention_surgery", "heal_replaced_model"),
    "RooflineReport": ("neuralops.core.roofline", "RooflineReport"),
    "profile_roofline": ("neuralops.core.roofline", "profile_roofline"),
    "render_roofline_terminal": ("neuralops.core.roofline", "render_roofline_terminal"),
    "extract_surgical_manifest": ("neuralops.models.surgical_io", "extract_surgical_manifest"),
    "save_surgical_checkpoint": ("neuralops.models.surgical_io", "save_surgical_checkpoint"),
    "load_surgical_checkpoint": ("neuralops.models.surgical_io", "load_surgical_checkpoint"),
    "NeuralOpsStateCache": ("neuralops.core.state_cache", "NeuralOpsStateCache"),
    "benchmark_autoregressive_decoding": ("neuralops.core.state_cache", "benchmark_autoregressive_decoding"),
    "select_optimal_triton_config": ("neuralops.core.triton_kernels.autotuner", "select_optimal_triton_config"),
    "compute_pareto_frontier": ("neuralops.core.paper_export", "compute_pareto_frontier"),
    "export_benchmark_table": ("neuralops.core.paper_export", "export_benchmark_table"),
    "ResultRecord": ("neuralops.core.results_ledger", "ResultRecord"),
    "ResultsLedger": ("neuralops.core.results_ledger", "ResultsLedger"),
    "get_default_ledger": ("neuralops.core.results_ledger", "get_default_ledger"),
    "log_experiment_result": ("neuralops.core.results_ledger", "log_experiment_result"),
    "BaseTrainer": ("neuralops.trainer.base_trainer", "BaseTrainer"),
    "COMPRESSOR_REGISTRY": ("neuralops.core.registry", "COMPRESSOR_REGISTRY"),
    "CompressionPipeline": ("neuralops.core.compression_pipeline", "CompressionPipeline"),
    "CustomLoopTrainer": ("neuralops.trainer.custom_loop_trainer", "CustomLoopTrainer"),
    "FairnessEvaluator": ("neuralops.evaluation.fairness", "FairnessEvaluator"),
    "FairnessReport": ("neuralops.evaluation.fairness", "FairnessReport"),
    "HuggingFaceTrainer": ("neuralops.trainer.hf_trainer", "HuggingFaceTrainer"),
    "NeuralOpsTrainer": ("neuralops.trainer.neuralops_trainer", "NeuralOpsTrainer"),
    "save_attention_only": ("neuralops.utils.hub_integration", "save_attention_only"),
    "load_attention_from_hub": ("neuralops.utils.hub_integration", "load_attention_from_hub"),
    "LabelQualityDetector": ("neuralops.evaluation.label_quality", "LabelQualityDetector"),
    "LabelQualityReport": ("neuralops.evaluation.label_quality", "LabelQualityReport"),
    "MultiModeBenchmarkRunner": ("neuralops.benchmarks.harness", "MultiModeBenchmarkRunner"),
    "ResultWriter": ("neuralops.results_saver.pipeline.result_writer", "ResultWriter"),
    "ResultsSaver": ("neuralops.results_saver", "ResultsSaver"),
    "RetrainDecision": ("neuralops.evaluation.retraining", "RetrainDecision"),
    "RetrainPolicy": ("neuralops.evaluation.retraining", "RetrainPolicy"),
    "RetrainingOrchestrator": ("neuralops.evaluation.retraining", "RetrainingOrchestrator"),
    "Trainer": ("neuralops.trainer", "Trainer"),
    "TrainerRegistry": ("neuralops.trainer.registry", "TrainerRegistry"),
    "UnifiedTransformerConfig": ("neuralops.core.model_wrapper", "UnifiedTransformerConfig"),
    "UnifiedTransformerModel": ("neuralops.core.model_wrapper", "UnifiedTransformerModel"),
    "aggregate_results": ("neuralops.trainer.metrics", "aggregate_results"),
    "compress_model": ("neuralops.core.compression_pipeline", "compress_model"),
    "detect_architecture": ("neuralops.core.architecture", "detect_architecture"),
    "fms_scan_triton": ("neuralops.core.triton_kernels", "fms_scan_triton"),
    "get_task_metrics": ("neuralops.trainer.metrics", "get_task_metrics"),
    "gla_scan_triton": ("neuralops.core.triton_kernels", "gla_scan_triton"),
    "inject_attention": ("neuralops.core.attention_utils", "inject_attention"),
    "linear_attn": ("neuralops.core.triton_kernels", "linear_attn"),
    "selective_scan_fn": ("neuralops.core.triton_kernels", "selective_scan_fn"),
    "triton_dispatch": ("neuralops.core.triton_kernels", "triton_dispatch"),
}

_API_EXPORTS = {
    "load_model": "neuralops.api",
    "replace_attention": "neuralops.api",
    "transplant_attention": "neuralops.api",
    "compress": "neuralops.api",
    "export_model": "neuralops.api",
    "compile_to_so": "neuralops.api",
    "push_to_hub": "neuralops.api",
    "load_exported_model": "neuralops.api",
    "compare_attentions_ablation": "neuralops.api",
    "surgical_transplant_and_heal_pipeline": "neuralops.api",

    "collect_surgery_distillation_loss": "neuralops.api",
    "set_surgery_alpha": "neuralops.api",
    "set_surgery_stage": "neuralops.api",
    "finalize_surgery": "neuralops.api",
    "list_attentions": "neuralops.api",
    "list_compressors": "neuralops.api",
    "list_triton_kernels": "neuralops.kernels",
    "list_cuda_kernels": "neuralops.kernels",
    "get_attention_class": "neuralops.api",
    "get_compressor_class": "neuralops.api",
    "list_families": "neuralops.api",
    "train": "neuralops.api",
    "distill": "neuralops.api",
    "evaluate": "neuralops.api",
    "load_and_prepare_dataset": "neuralops.api",
    "train_grpo": "neuralops.api",
    "benchmark": "neuralops.api",
    "benchmark_all": "neuralops.api",
    "info": "neuralops.api",
    "available_backends": "neuralops.api",
    "has_triton": "neuralops.api",
    "validate": "neuralops.api",
    "get_version": "neuralops.api",
    "oneshot": "neuralops.api",
    "recommend": "neuralops.api",
    "summary": "neuralops.api",
    "compare": "neuralops.api",
    "patch_transformers": "neuralops.api",
    "AutoNeuralOpsModel": "neuralops.api",
    "AutoNeuralOpsForCausalLM": "neuralops.api",
    "AutoNeuralOpsForSequenceClassification": "neuralops.api",
    "merge_for_inference": "neuralops.api",
    "GLAttention": "neuralops.attentions.linear.gla_attention",
    "CosformerAttention": "neuralops.attentions.linear.cosformer",
    "SVDLinearAttention": "neuralops.attentions.svd.svd_linear_attention",
    "DecayLinearAttention": "neuralops.attentions.gated.decay_linear_attention",
    "Muon": "neuralops.trainer.optimizers.muon",
    "StiefelOptimizer": "neuralops.trainer.optimizers",
    "RiemannianAdam": "neuralops.trainer.optimizers",
    "HybridAttentionSpec": "neuralops.models.hybrid_spec",
    "probe_optimal_batch_size": "neuralops.api",
    "NeuralOpsError": "neuralops.core.exceptions",
    "ConfigError": "neuralops.core.exceptions",
    "WeightCopyError": "neuralops.core.exceptions",
    "BackendError": "neuralops.core.exceptions",
    "NumericalError": "neuralops.core.exceptions",
    "BitPackedW4A16Linear": "neuralops.core.triton_kernels.low_rank",
    "BitPackedLowRankSVDLinear": "neuralops.core.triton_kernels.low_rank",
    "pack_int4_weights": "neuralops.core.triton_kernels.low_rank",
    "unpack_int4_weights": "neuralops.core.triton_kernels.low_rank",
    "fused_w4a16_gemm_triton": "neuralops.core.triton_kernels.low_rank",
    "fast_hadamard_transform": "neuralops.core.triton_kernels.elementwise",
    "HadamardIncoherentSVDCompressor": "neuralops.compression_engine.svd.hadamard_incoherent_svd",
    "SelfSpeculativeDecoder": "neuralops.inference.speculative_decoding",
    "SpeculativeDecodeStats": "neuralops.inference.speculative_decoding",
    "PagedKVCache": "neuralops.inference.paged_cache",
    "PagedStateTable": "neuralops.models.paged_state",
    "ATTENTION_REGISTRY": "neuralops.core.registry",
    "COMPRESSOR_REGISTRY": "neuralops.core.registry",
    "NeedleInAHaystackEvaluator": "neuralops.evaluation.niah_passkey",
    "plot_niah_heatmap": "neuralops.evaluation.niah_passkey",
    "NeuralOpsLMEvalAdapter": "neuralops.evaluation.lm_eval_adapter",
    "evaluate_zero_shot_tasks": "neuralops.evaluation.lm_eval_adapter",
}

_SUBPACKAGES = {
    "analysis", "attentions", "benchmarks", "cli", "compression_engine",
    "config", "core", "distributed", "experiment_logging",
    "inference", "integrations", "kernels", "models",
    "results_saver", "runners", "tests", "tracking", "trainer",
    "training", "utils", "main", "surgery", "export",
}


def inject_gla(model, **kwargs):
    """Replace all standard attention layers with GLA. Convenience wrapper."""
    from neuralops.attentions.linear.gla_attention import GLAttention as _GLA
    from neuralops.core._attention_surgery import replace_attention_in_model
    return replace_attention_in_model(model, _GLA, **kwargs)

def create_attention(
    name: str,
    config: Any = None,
    dim: int | None = None,
    num_heads: int | None = None,
    embed_dim: int | None = None,
    rank: int | None = None,
    **kwargs: Any,
) -> Any:
    """Universal attention factory instantiating any of the 131 registered attention mechanisms.

    Normalizes architectural hyperparameter aliases (embed_dim, dim, d_model, hidden_size,
    num_heads, n_heads, heads, rank, r) and builds standard compliant configs automatically.
    """
    from neuralops.core.registry import ATTENTION_REGISTRY
    cls = ATTENTION_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown attention type '{name}'. Registered: {list(ATTENTION_REGISTRY.keys())}")

    resolved_dim = (
        embed_dim
        or dim
        or kwargs.pop("d_model", None)
        or kwargs.pop("hidden_size", None)
        or getattr(config, "hidden_size", None)
        or getattr(config, "n_embd", None)
        or getattr(config, "d_model", None)
    )
    resolved_heads = (
        num_heads
        or kwargs.pop("n_heads", None)
        or kwargs.pop("heads", None)
        or kwargs.pop("num_attention_heads", None)
        or getattr(config, "num_attention_heads", None)
        or getattr(config, "num_heads", None)
        or getattr(config, "n_head", None)
    )
    resolved_rank = rank or kwargs.pop("r", None)

    call_kwargs = dict(kwargs)
    if resolved_dim is not None:
        call_kwargs["dim"] = resolved_dim
        call_kwargs["embed_dim"] = resolved_dim
        call_kwargs["hidden_size"] = resolved_dim
        call_kwargs["d_model"] = resolved_dim
    if resolved_heads is not None:
        call_kwargs["num_heads"] = resolved_heads
        call_kwargs["n_heads"] = resolved_heads
        call_kwargs["num_attention_heads"] = resolved_heads
    if resolved_rank is not None:
        call_kwargs["rank"] = resolved_rank
        call_kwargs["r"] = resolved_rank

    if config is None and resolved_dim is not None:
        from transformers import BertConfig
        h = resolved_heads or max(1, resolved_dim // 64)
        is_dec = kwargs.get("is_decoder", kwargs.get("is_causal", True))
        config = BertConfig(hidden_size=resolved_dim, num_attention_heads=h, is_decoder=is_dec)
        config.is_causal = is_dec
        config._is_synthetic = True
        call_kwargs["config"] = config

    init_attempts = []
    if config is not None:
        init_attempts.append(lambda: cls(config=config, **{k: v for k, v in call_kwargs.items() if k != "config"}))
        init_attempts.append(lambda: cls(config, **{k: v for k, v in call_kwargs.items() if k != "config"}))
    if resolved_dim is not None and resolved_heads is not None:
        init_attempts.append(lambda: cls(embed_dim=resolved_dim, num_heads=resolved_heads, **{k: v for k, v in call_kwargs.items() if k not in ("embed_dim", "num_heads")}))
        init_attempts.append(lambda: cls(dim=resolved_dim, num_heads=resolved_heads, **{k: v for k, v in call_kwargs.items() if k not in ("dim", "num_heads")}))
        init_attempts.append(lambda: cls(resolved_dim, num_heads=resolved_heads, **{k: v for k, v in call_kwargs.items() if k not in ("embed_dim", "dim", "num_heads")}))
    init_attempts.append(lambda: cls(**call_kwargs))
    init_attempts.append(lambda: cls())

    last_err = None
    for attempt in init_attempts:
        try:
            return attempt()
        except TypeError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    return cls(**call_kwargs)


def create_compressor(name: str, model: Any = None, rank: int | None = None, **kwargs: Any) -> Any:
    """Universal compressor factory instantiating any of the 72 registered compression engines.

    Normalizes hyperparameter aliases (rank, r, method, model) and handles optional dummy hosts.
    """
    from neuralops.core.registry import COMPRESSOR_REGISTRY
    cls = COMPRESSOR_REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Compressor '{name}' not found in COMPRESSOR_REGISTRY.")

    call_kwargs = dict(kwargs)
    resolved_rank = rank or call_kwargs.pop("r", None)
    if resolved_rank is not None:
        call_kwargs["rank"] = resolved_rank

    if model is not None:
        call_kwargs["model"] = model

    attempts = [
        lambda: cls(**call_kwargs),
    ]
    if model is not None:
        attempts.append(lambda: cls(model, **{k: v for k, v in call_kwargs.items() if k != "model"}))
    else:
        import torch.nn as nn
        dummy_host = nn.Sequential(nn.Linear(64, 64))
        attempts.append(lambda: cls(model=dummy_host, **call_kwargs))
        attempts.append(lambda: cls(dummy_host, **call_kwargs))

    last_err = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    return cls(**call_kwargs)


__all__ = [
    "__version__",
    "load_model",
    "replace_attention",
    "transplant_attention",
    "compress",
    "create_compressor",
    "export_model",
    "compile_to_so",
    "push_to_hub",
    "load_exported_model",

    "collect_surgery_distillation_loss",
    "set_surgery_alpha",
    "set_surgery_stage",
    "finalize_surgery",
    "list_attentions",
    "list_compressors",
    "get_attention_class",
    "get_compressor_class",
    "list_families",
    "train",
    "distill",
    "evaluate",
    "benchmark",
    "benchmark_all",
    "info",
    "available_backends",
    "has_triton",
    "validate",
    "get_version",
    "GLAttention",
    "CosformerAttention",
    "SVDLinearAttention",
    "DecayLinearAttention",
    "inject_gla",
    "create_attention",
    "ATTENTION_REGISTRY",
    "NeuralOpsError",
    "ConfigError",
    "WeightCopyError",
    "BackendError",
    "NumericalError",
    "Muon",
    "StiefelOptimizer",
    "RiemannianAdam",
    "HybridAttentionSpec",
    "probe_optimal_batch_size",
    "oneshot",
    "recommend",
    "summary",
    "compare",
    "patch_transformers",
    "AutoNeuralOpsModel",
    "AutoNeuralOpsForCausalLM",
    "AutoNeuralOpsForSequenceClassification",
    "merge_for_inference",
    "ComposableAttention",
    "compose_attention",
] + sorted(list(_LAZY_EXPORTS.keys()))


def __getattr__(name: str) -> Any:
    """PEP 562 dynamic lazy loader."""
    if name == "ATTENTION_REGISTRY":
        from neuralops.core.registry import ATTENTION_REGISTRY as reg
        globals()[name] = reg
        return reg

    if name == "ALL_ATTENTION_CLASSES":
        from neuralops.core.registry import ATTENTION_REGISTRY as reg
        classes = list(reg.values())
        globals()[name] = classes
        return classes

    if name in _API_EXPORTS:
        mod_name = _API_EXPORTS[name]
        mod = importlib.import_module(mod_name)
        val = getattr(mod, name)
        globals()[name] = val
        return val

    if name in _LAZY_EXPORTS:
        module_path, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_path)
        val = getattr(module, attr_name)
        globals()[name] = val
        return val

    if name == "evaluation":
        try:
            evaluation_package = importlib.import_module("neuralops.evaluation")
            globals()[name] = evaluation_package
            return evaluation_package
        except ImportError:
            pass

    if name in _SUBPACKAGES:
        submod = importlib.import_module(f"neuralops.{name}")
        globals()[name] = submod
        return submod

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Execute __dir__ operation."""
    return sorted(list(globals().keys()) + __all__ + list(_SUBPACKAGES))
