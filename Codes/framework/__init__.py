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

# Transparent Import Hook for Cleaned Submodule Aliases
import sys
import importlib
import importlib.util
import importlib.machinery
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
        "framework.attentions.gqa": "framework.attentions.gqa_attention",
        "framework.attentions.galerkin": "framework.attentions.galerkin_attention",
        "framework.attentions.galerkin_inject": "framework.attentions.galerkin_attention",
        "framework.attentions.adaptive_span_inject": "framework.attentions.adaptive_span_attention",
        "framework.attentions.adaptive_sparse_inject": "framework.attentions.adaptive_sparse_attention",
        "framework.attentions.adasplash_inject": "framework.attentions.adasplash_attention",
        "framework.attentions.aegis_hybrid_attention": "framework.attentions.aegis_hybrid",
        "framework.attentions.cluster_inject": "framework.attentions.cluster_attention",
        "framework.attentions.compressed_mha": "framework.attentions.compressed_mha_attention",
        "framework.attentions.cosformer_inject": "framework.attentions.cosformer",
        "framework.attentions.hla_inject": "framework.attentions.hla_attention",
        "framework.attentions.kda_inject": "framework.attentions.kda_attention",
        "framework.attentions.leka_v1_inject": "framework.attentions.leka_v1",
        "framework.attentions.leka_v2_inject": "framework.attentions.leka_v2",
        "framework.attentions.leka_v3_inject": "framework.attentions.leka_v3",
        "framework.attentions.mamba_attention": "framework.attentions.mamba_block",
        "framework.attentions.moba": "framework.attentions.moba_attention",
        "framework.attentions.nha_compressor": "framework.attentions.nha_attention",
        "framework.attentions.rpc": "framework.attentions.rpc_attention",
        "framework.attentions.salt": "framework.attentions.salt_attention",
        "framework.attentions.soa_fused_module": "framework.attentions.soa_attn",
        "framework.attentions.SOAFusedAttention": "framework.attentions.soa_attn",
        "framework.attentions.soa_module": "framework.attentions.soa_attn",
        "framework.attentions.spectra_v2_attention": "framework.attentions.spectra_v2",
        "framework.attentions.svd_compressed_attention": "framework.attentions.svd_compressed",
        "framework.attentions.svd_llm_v2": "framework.attentions.svd_llm_v2_attention",
        "framework.attentions.svd_proj_attn": "framework.attentions.svd_projection_attention",
        "framework.attentions.token_compressed": "framework.attentions.token_compressed_attention",
        "framework.attentions.tt": "framework.attentions.token_compressed_attention",
        "framework.attentions.upi_mamba": "framework.attentions.upi_stuffed_mamba",
        "framework.attentions.mamba": "framework.attentions.mamba_block",
        "framework.attentions.mamba_inject": "framework.attentions.mamba_block",
        "framework.attentions.simple_ssm": "framework.attentions.selective_ssm",
        "framework.attentions.smart_weighted_ensemble": "framework.attentions.hybrid_attention",
        "framework.attentions.weighted_ensemble": "framework.attentions.hybrid_attention",
        "framework.attentions.spectron_attention": "framework.attentions.spectron",
        "framework.attentions.squeezing_heads": "framework.attentions.squeezing_heads_distill",
        "framework.attentions.squeezing_heads_distiller": "framework.attentions.squeezing_heads_distill",
        "framework.attentions.spectral_mamba_attention": "framework.attentions.spectral_mamba",
        "framework.attentions.svd_sequence_attention": "framework.attentions.svd_attention",
        "framework.attentions.rala_compressor": "framework.attentions.rala_attention",
        "framework.attentions.selective_scan": "framework.core.selective_scan",
        "framework.attentions.ssm_scan": "framework.core.selective_scan",
        "framework.attentions.legacy_ensemble": "framework.core.legacy_ensemble",
        # Compressors
        "framework.compression_engine.aasvd": "framework.compression_engine.aa_svd_compressor",
        "framework.compression_engine.adasvd": "framework.compression_engine.adasvd_compressor",
        "framework.compression_engine.cur_moe": "framework.compression_engine.cur_moe_compressor",
        "framework.compression_engine.frandi": "framework.compression_engine.frandi_compressor",
        "framework.compression_engine.grecodec": "framework.compression_engine.grecodec_compressor",
        "framework.compression_engine.hybridkv": "framework.compression_engine.hybridkv_compressor",
        "framework.compression_engine.kvtc": "framework.compression_engine.kvtc_compressor",
        "framework.compression_engine.modegpt": "framework.compression_engine.modegpt_compressor",
        "framework.compression_engine.oats": "framework.compression_engine.oats_compressor",
        "framework.compression_engine.proxsparse_pruner": "framework.compression_engine.proxsparse",
        "framework.compression_engine.rpca": "framework.compression_engine.rpca_compressor",
        "framework.compression_engine.saes_svd": "framework.compression_engine.saes_svd_v2",
        "framework.compression_engine.saes_svd_compressor": "framework.compression_engine.saes_svd_v2",
        "framework.compression_engine.sdrm": "framework.compression_engine.sdrm_compressor",
        "framework.compression_engine.slicegpt_compressor": "framework.compression_engine.slicegpt",
        "framework.compression_engine.soka": "framework.compression_engine.soka_compressor",
        "framework.compression_engine.swift_svd_compressor": "framework.compression_engine.swift_svd_act",
        "framework.compression_engine.zerokv_neo": "framework.compression_engine.zerokv_neo_compressor",
        "framework.compression_engine.cure": "framework.compression_engine.cur_moe_compressor",
        "framework.compression_engine.dobi": "framework.compression_engine.dobi_svd",
        "framework.compression_engine.dynamic_entropy": "framework.compression_engine.dynamic_rank",
        "framework.compression_engine.tomoe": "framework.compression_engine.tomoe_compressor",
        "framework.compression_engine.adaptive_compressor": "framework.compression_engine.adaptive_ranks",
        "framework.compression_engine.padre_attention": "framework.attentions.padre_attention",
        "framework.compression_engine.spectraformer_attention": "framework.attentions.spectraformer_attention",
        "framework.compression_engine.tlinformer_attention": "framework.attentions.tlinformer_attention",
        "framework.compression_engine.pruning_utils": "framework.core.pruning_utils",
        "framework.compression_engine.cur_linear": "framework.compression_engine.cur_compressor",
        "framework.compression_engine.kronecker_linear": "framework.compression_engine.kronecker_compressor",
        "framework.compression_engine.tt_linear": "framework.compression_engine.tt_compressor",
        "framework.compression_engine.hashed_linear": "framework.compression_engine.hashed_net_compressor",
        "framework.compression_engine.come": "framework.compression_engine.come_compressor",
        "framework.compression_engine.rala_compressor": "framework.attentions.rala_attention",
        # Reorganized Benchmarks & Sweepers
        "framework.run_quality_sweep": "framework.benchmarks.run_quality_sweep",
        "framework.run_rank_sweeper": "framework.benchmarks.run_rank_sweeper",
        "framework.run_all_benchmarks": "framework.benchmarks.run_all_benchmarks",
        "framework.run_gpt2_sweep": "framework.benchmarks.run_gpt2_sweep",
        "framework.run_long_context_sweep": "framework.benchmarks.run_long_context_sweep",
        "framework.run_seqlen_scaling": "framework.benchmarks.run_seqlen_scaling",
        "framework.run_seqlen_scaling_bench": "framework.benchmarks.run_seqlen_scaling_bench",
        "framework.run_multiseed": "framework.benchmarks.run_multiseed",
        "framework.run_multiseed_rigor": "framework.benchmarks.run_multiseed_rigor",
        "framework.run_one_module_isolated": "framework.benchmarks.run_one_module_isolated",
        "framework.run_ruler_benchmark": "framework.benchmarks.run_ruler_benchmark",
        "framework.run_whitened_svd": "framework.benchmarks.run_whitened_svd",
        "framework.launch_parallel_sweep": "framework.benchmarks.launch_parallel_sweep",
        "framework.run_all_modules_scheduler": "framework.benchmarks.run_all_modules_scheduler",
        "framework.kd_quick_test": "framework.benchmarks.kd_quick_test",
        # Reorganized Evaluation & Verification
        "framework.eval_all_models_datasets": "framework.evaluation.eval_all_models_datasets",
        "framework.eval_perplexity": "framework.evaluation.eval_perplexity",
        "framework.full_mnli_validation": "framework.evaluation.full_mnli_validation",
        "framework.run_lm_eval_harness": "framework.evaluation.run_lm_eval_harness",
        "framework.run_baseline_comparison": "framework.evaluation.run_baseline_comparison",
        # Reorganized Trainer & Analysis
        "framework.train_lra": "framework.trainer.train_lra",
        "framework.generate_all_modules_analysis": "framework.analysis.generate_all_modules_analysis",
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
    # Core Interfaces & Registries
    "BaseAttention": ("framework.core.base", "BaseAttention"),
    "BaseCompressor": ("framework.core.base", "BaseCompressor"),
    "ATTENTION_REGISTRY": ("framework.core.registry", "ATTENTION_REGISTRY"),
    "COMPRESSOR_REGISTRY": ("framework.core.registry", "COMPRESSOR_REGISTRY"),
    "available_attention_types": ("framework.core.registry", "available_attention_types"),
    "register_attention": ("framework.core.registry", "register_attention"),
    "register_compressor": ("framework.core.registry", "register_compressor"),
    "inject_attention": ("framework.core.attention_utils", "inject_attention"),
    "compress_model": ("framework.core.compression_engine", "compress_model"),
    "CompressionPipeline": ("framework.core.compression_engine", "CompressionPipeline"),
    "detect_architecture": ("framework.core.architecture", "detect_architecture"),
    "UnifiedTransformerConfig": ("framework.core.model_wrapper", "UnifiedTransformerConfig"),
    "UnifiedTransformerModel": ("framework.core.model_wrapper", "UnifiedTransformerModel"),
    "MultiModeBenchmarkRunner": ("framework.core.multi_mode_benchmark", "MultiModeBenchmarkRunner"),
    
    # Models
    "BaseModel": ("framework.models", "BaseModel"),
    
    # Trainers & Metrics
    "BaseTrainer": ("framework.trainer.base_trainer", "BaseTrainer"),
    "HuggingFaceTrainer": ("framework.trainer.hf_trainer", "HuggingFaceTrainer"),
    "CustomLoopTrainer": ("framework.trainer.custom_loop_trainer", "CustomLoopTrainer"),
    "TrainerRegistry": ("framework.trainer.registry", "TrainerRegistry"),
    "get_task_metrics": ("framework.trainer.metrics", "get_task_metrics"),
    "aggregate_results": ("framework.trainer.metrics", "aggregate_results"),
    
    # Data & Benchmark Datasets
    "DataConfig": ("framework.data.base", "DataConfig"),
    "TaskType": ("framework.data.base", "TaskType"),
    "create_dataloader": ("framework.data.base", "create_dataloader"),
    "preprocess_function": ("framework.data.base", "preprocess_function"),
    "ExperimentConfig": ("framework.data.dataset_loader", "ExperimentConfig"),
    "load_dataset": ("framework.data.dataset_loader", "load_dataset"),
    "load_glue_local": ("framework.data.glue_loader", "load_glue_local"),
    "GLUE_TASK_KEYS": ("framework.data.glue_loader", "GLUE_TASK_KEYS"),
    "BenchmarkDatasetInfo": ("framework.data.benchmark_datasets", "BenchmarkDatasetInfo"),
    "BenchmarkDatasetRegistry": ("framework.data.benchmark_datasets", "BenchmarkDatasetRegistry"),
    "load_benchmark_dataset": ("framework.data.benchmark_datasets", "load_benchmark_dataset"),
    "generate_synthetic_data": ("framework.data.synthetic_generator", "generate_synthetic_data"),
    "run_universal_generation": ("framework.data.synthetic_generator", "run_universal_generation"),
    
    # Benchmark & Pipeline
    "build_attention_module": ("framework.benchmark_attention", "build_attention_module"),
    "benchmark_single_module": ("framework.benchmark_attention", "benchmark_single_module"),
    "run_benchmark_suite": ("framework.benchmark_attention", "run_benchmark_suite"),
    "ResultWriter": ("framework.pipeline.result_writer", "ResultWriter"),
    
    # Evaluation & Robustness
    "ABTester": ("framework.evaluation.ab_testing", "ABTester"),
    "ABTestResult": ("framework.evaluation.ab_testing", "ABTestResult"),
    "FairnessEvaluator": ("framework.evaluation.fairness", "FairnessEvaluator"),
    "FairnessReport": ("framework.evaluation.fairness", "FairnessReport"),
    "LabelQualityDetector": ("framework.evaluation.label_quality", "LabelQualityDetector"),
    "LabelQualityReport": ("framework.evaluation.label_quality", "LabelQualityReport"),
    "RetrainDecision": ("framework.evaluation.retraining", "RetrainDecision"),
    "RetrainPolicy": ("framework.evaluation.retraining", "RetrainPolicy"),
    "RetrainingOrchestrator": ("framework.evaluation.retraining", "RetrainingOrchestrator"),
    "RobustnessReport": ("framework.evaluation.robustness", "RobustnessReport"),
    "TextAdversarialTester": ("framework.evaluation.robustness", "TextAdversarialTester"),
    
    # Visualization & Dashboards
    "ChartGenerator": ("framework.visualization.chart_generator", "ChartGenerator"),
    "BaseVisualizer": ("framework.visualization.base", "BaseVisualizer"),
    "VISUALIZER_REGISTRY": ("framework.visualization.base", "VISUALIZER_REGISTRY"),
    "PLOT_REGISTRY": ("framework.visualization.plot_registry", "PLOT_REGISTRY"),
    "register_default_plots": ("framework.visualization.plot_registry", "register_default_plots"),
    "generate_csv_summary": ("framework.visualization.report_generator", "generate_csv_summary"),
    "generate_html_report": ("framework.visualization.report_generator", "generate_html_report"),
    "generate_latex_table": ("framework.visualization.report_generator", "generate_latex_table"),
    "plot_accuracy_vs_rank": ("framework.visualization.report_generator", "plot_accuracy_vs_rank"),
    "plot_radar_chart": ("framework.visualization.report_generator", "plot_radar_chart"),
    "plot_attention_heatmap": ("framework.visualization.benchmark_visualizer", "plot_attention_heatmap"),
    "plot_singular_value_decay": ("framework.visualization.benchmark_visualizer", "plot_singular_value_decay"),
    "plot_kv_cache_scaling": ("framework.visualization.benchmark_visualizer", "plot_kv_cache_scaling"),
    "plot_pareto_frontier": ("framework.visualization.benchmark_visualizer", "plot_pareto_frontier"),
    "plot_dataset_radar_chart": ("framework.visualization.benchmark_visualizer", "plot_dataset_radar_chart"),
    "plot_layerwise_sparsity_rank": ("framework.visualization.benchmark_visualizer", "plot_layerwise_sparsity_rank"),
    "generate_dashboard_html": ("framework.visualization.dashboard_generator", "generate_dashboard_html"),
    "generate_summary_json": ("framework.visualization.dashboard_generator", "generate_summary_json"),
    "set_plot_style": ("framework.visualization.utils", "set_plot_style"),
    "save_figure": ("framework.visualization.utils", "save_figure"),
    "get_color_palette": ("framework.visualization.utils", "get_color_palette"),
    "create_custom_colormap": ("framework.visualization.utils", "create_custom_colormap"),
}

__all__: List[str] = ["__version__"] + sorted(list(_LAZY_EXPORTS.keys()))


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_path, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_path)
        val = getattr(module, attr_name)
        globals()[name] = val  # cache attribute on module
        return val
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> List[str]:
    return sorted(list(globals().keys()) + __all__)
