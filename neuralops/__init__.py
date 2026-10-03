"""
NeuralOps Core ML Framework & Efficient Transformer Toolkit.

A unified, modular, and high-performance toolkit for custom attention mechanisms,
model compression (SVD, quantization, pruning, distillation), distributed training,
hyperparameter optimization, federated learning, data loading, benchmarking,
and interactive visualization.
"""

import importlib
from typing import Any, List

__version__ = "0.1.0"

import importlib.machinery
import importlib.util

# Transparent Import Hook for Cleaned Submodule Aliases
import sys
from importlib.abc import MetaPathFinder


class _AliasLoader:
    def __init__(self, mod):
        self.mod = mod

    def create_module(self, spec):
        return self.mod

    def exec_module(self, module):
        pass


class _ModuleAliasFinder(MetaPathFinder):
    _ALIASES = {
        # Attentions
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
        # Compressors
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
        # Reorganized Benchmarks & Sweepers
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
        # Reorganized Evaluation & Verification
        "neuralops.eval_all_models_datasets": "neuralops.evaluation.eval_all_models_datasets",
        "neuralops.eval_perplexity": "neuralops.evaluation.eval_perplexity",
        "neuralops.full_mnli_validation": "neuralops.evaluation.full_mnli_validation",
        "neuralops.run_lm_eval_harness": "neuralops.evaluation.run_lm_eval_harness",
        "neuralops.run_baseline_comparison": "neuralops.evaluation.run_baseline_comparison",
        # Reorganized Trainer & Analysis
        "neuralops.train_lra": "neuralops.trainer.train_lra",
        "neuralops.generate_all_modules_analysis": "neuralops.analysis.generate_all_modules_analysis",
    }

    def find_spec(self, fullname, path, target=None):
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


# Symbol to (module_path, attr_name) mapping for fast, lazy resolution
_LAZY_EXPORTS = {
    "ABTestResult": ("neuralops.evaluation.ab_testing", "ABTestResult"),
    "ABTester": ("neuralops.evaluation.ab_testing", "ABTester"),
    "ATTENTION_REGISTRY": ("neuralops.core.registry", "ATTENTION_REGISTRY"),
    "BaseAttention": ("neuralops.core.base", "BaseAttention"),
    "BaseCompressor": ("neuralops.core.base", "BaseCompressor"),
    "BaseTrainer": ("neuralops.trainer.base_trainer", "BaseTrainer"),
    "BenchmarkDatasetInfo": ("neuralops.data.benchmark_datasets", "BenchmarkDatasetInfo"),
    "BenchmarkDatasetRegistry": ("neuralops.data.benchmark_datasets", "BenchmarkDatasetRegistry"),
    "COMPRESSOR_REGISTRY": ("neuralops.core.registry", "COMPRESSOR_REGISTRY"),
    "CompressionPipeline": ("neuralops.core.compression_engine", "CompressionPipeline"),
    "CustomLoopTrainer": ("neuralops.trainer.custom_loop_trainer", "CustomLoopTrainer"),
    "DataConfig": ("neuralops.data.base", "DataConfig"),
    "ExperimentConfig": ("neuralops.data.dataset_loader", "ExperimentConfig"),
    "FairnessEvaluator": ("neuralops.evaluation.fairness", "FairnessEvaluator"),
    "FairnessReport": ("neuralops.evaluation.fairness", "FairnessReport"),
    "GLUE_TASK_KEYS": ("neuralops.data.glue_loader", "GLUE_TASK_KEYS"),
    "HuggingFaceTrainer": ("neuralops.trainer.hf_trainer", "HuggingFaceTrainer"),
    "LabelQualityDetector": ("neuralops.evaluation.label_quality", "LabelQualityDetector"),
    "LabelQualityReport": ("neuralops.evaluation.label_quality", "LabelQualityReport"),
    "MultiModeBenchmarkRunner": ("neuralops.core.multi_mode_benchmark", "MultiModeBenchmarkRunner"),
    "ResultWriter": ("neuralops.pipeline.result_writer", "ResultWriter"),
    "ResultsSaver": ("neuralops.results_saver", "ResultsSaver"),
    "RetrainDecision": ("neuralops.evaluation.retraining", "RetrainDecision"),
    "RetrainPolicy": ("neuralops.evaluation.retraining", "RetrainPolicy"),
    "RetrainingOrchestrator": ("neuralops.evaluation.retraining", "RetrainingOrchestrator"),
    "TaskType": ("neuralops.data.base", "TaskType"),
    "TrainerRegistry": ("neuralops.trainer.registry", "TrainerRegistry"),
    "UnifiedTransformerConfig": ("neuralops.core.model_wrapper", "UnifiedTransformerConfig"),
    "UnifiedTransformerModel": ("neuralops.core.model_wrapper", "UnifiedTransformerModel"),
    "aggregate_results": ("neuralops.trainer.metrics", "aggregate_results"),
    "compress_model": ("neuralops.core.compression_engine", "compress_model"),
    "create_dataloader": ("neuralops.data.base", "create_dataloader"),
    "detect_architecture": ("neuralops.core.architecture", "detect_architecture"),
    "fms_scan_triton": ("neuralops.core.triton_kernels", "fms_scan_triton"),
    "generate_synthetic_data": ("neuralops.data.synthetic_generator", "generate_synthetic_data"),
    "get_calibration_dataloader": ("neuralops.data", "get_calibration_dataloader"),
    "get_task_metrics": ("neuralops.trainer.metrics", "get_task_metrics"),
    "gla_scan_triton": ("neuralops.core.triton_kernels", "gla_scan_triton"),
    "inject_attention": ("neuralops.core.attention_utils", "inject_attention"),
    "linear_attn": ("neuralops.core.triton_kernels", "linear_attn"),
    "load_benchmark_dataset": ("neuralops.data.benchmark_datasets", "load_benchmark_dataset"),
    "load_dataset": ("neuralops.data.dataset_loader", "load_dataset"),
    "load_glue_local": ("neuralops.data.glue_loader", "load_glue_local"),
    "preprocess_function": ("neuralops.data.base", "preprocess_function"),
    "run_universal_generation": ("neuralops.data.synthetic_generator", "run_universal_generation"),
    "selective_scan_fn": ("neuralops.core.triton_kernels", "selective_scan_fn"),
    "triton_dispatch": ("neuralops.core.triton_kernels", "triton_dispatch"),
}

__all__: list[str] = ["__version__"] + sorted(list(_LAZY_EXPORTS.keys()))


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_path, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_path)
        val = getattr(module, attr_name)
        globals()[name] = val  # cache attribute on module
        return val
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
