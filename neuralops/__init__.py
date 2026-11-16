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

# Compatibility shim for PyTorch < 2.0 with Setuptools >= 70 (pkg_resources removal)
import logging as _logging
import sys as _sys

_logger = _logging.getLogger(__name__)

if "pkg_resources" not in _sys.modules:
    try:
        import warnings as _warnings
        with _warnings.catch_warnings():
            _warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")
            import pkg_resources as _pkg_resources  # noqa: F401
    except ImportError:
        try:
            from typing import Any as _Any
            import packaging as _packaging
            import types as _types

            _pr: _Any = _types.ModuleType("pkg_resources")
            _pr.packaging = _packaging
            _sys.modules["pkg_resources"] = _pr
        except Exception as exc:
            _logger.debug("pkg_resources polyfill initialization failed: %s", exc)

# Zero-dependency polyfill for pydantic if missing in runtime environment
if "pydantic" not in _sys.modules:
    try:
        import pydantic as _pydantic  # noqa: F401
    except ImportError:
        import types as _types

        class _PydanticBaseModelMeta(type):
            def __new__(cls, name, bases, dct):
                annotations = dct.get("__annotations__", {})
                defaults = {}
                for k in annotations:
                    if k in dct:
                        val = dct[k]
                        if isinstance(val, _FieldInfo):
                            if callable(val.default_factory):
                                defaults[k] = val.default_factory()
                            else:
                                defaults[k] = val.default
                        else:
                            defaults[k] = val
                dct["_defaults"] = defaults
                return super().__new__(cls, name, bases, dct)

        class _FieldInfo:
            def __init__(self, default=None, default_factory=None, **kwargs):
                self.default = default
                self.default_factory = default_factory
                self._is_field = True
                for k, v in kwargs.items():
                    setattr(self, k, v)

        class _BaseModel(metaclass=_PydanticBaseModelMeta):
            def __init__(self, **kwargs):
                for k, v in getattr(self, "_defaults", {}).items():
                    setattr(self, k, v)
                for k, v in kwargs.items():
                    setattr(self, k, v)

            def model_dump(self, *args, **kwargs):
                return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

            def dict(self, *args, **kwargs):
                return self.model_dump(*args, **kwargs)

        def _field_func(default=None, default_factory=None, **kwargs):
            return _FieldInfo(default=default, default_factory=default_factory, **kwargs)

        def _config_dict(**kwargs):
            return kwargs

        def _validator_func(*fields, **kwargs):
            return lambda fn: fn

        from typing import Any as _Any

        _pydantic_mock: _Any = _types.ModuleType("pydantic")
        _pydantic_mock.BaseModel = _BaseModel
        _pydantic_mock.Field = _field_func
        _pydantic_mock.ConfigDict = _config_dict
        _pydantic_mock.field_validator = _validator_func
        _pydantic_mock.validator = _validator_func
        _sys.modules["pydantic"] = _pydantic_mock

import importlib
import importlib.util
import os
from pathlib import Path
import sys
import warnings

from neuralops.core._compat_importer import install_compatibility_interceptor

install_compatibility_interceptor()

from typing import TYPE_CHECKING, Any

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
        get_attention_module,
        quick_benchmark,
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
        pipeline,
        push_to_hub,
        recommend,
        replace_attention,
        set_surgery_alpha,
        set_surgery_stage,
        summary,
        surgical_transplant_and_heal_pipeline,  # noqa: F401
        train,
        transplant_attention,
        transplant,
        export,
        validate,
    )
    from neuralops.integrations.fla import list_fla_attentions
    from neuralops.core.exceptions import (
        BackendError,
        ConfigError,
        NeuralOpsError,
        NumericalError,
        WeightCopyError,
    )
    from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY  # noqa: F401
    from neuralops.kernels import list_cuda_kernels, list_triton_kernels  # noqa: F401

import importlib.metadata

try:
    from neuralops._version import __version__
except ImportError:
    try:
        import importlib.metadata

        __version__ = importlib.metadata.version("neuralops")
    except Exception:
        __version__ = "2.1.0"

_TORCH_FREE_SYMBOLS = frozenset({"__version__", "__doc__", "__all__"})
try:
    _TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
except (ImportError, ValueError):
    _TORCH_AVAILABLE = False

from neuralops.core.exceptions import torch_install_error as _torch_install_error


def _require_torch(feature: str = "this feature") -> None:
    """Raise a clear installation error when PyTorch is absent."""
    if not _TORCH_AVAILABLE:
        raise _torch_install_error(feature)


import importlib.machinery
from importlib.abc import MetaPathFinder


from neuralops.core._module_alias_finder import install_module_alias_finder
from neuralops.core._torch_compat import patch_pascal_svd_if_needed, patch_torch_compat

install_module_alias_finder()
patch_torch_compat()
patch_pascal_svd_if_needed()


_LAZY_EXPORTS = {
    "RuntimeConfig": ("neuralops.runtime", "RuntimeConfig"),
    "init": ("neuralops.runtime", "init"),
    "use_runtime": ("neuralops.runtime", "use_runtime"),
    "get_runtime_config": ("neuralops.runtime", "get_runtime_config"),
    "ABTestResult": ("neuralops.evaluation.ab_testing", "ABTestResult"),
    "ABTester": ("neuralops.evaluation.ab_testing", "ABTester"),
    "ATTENTION_REGISTRY": ("neuralops.core.registry", "ATTENTION_REGISTRY"),
    "BaseAttention": ("neuralops.core.base", "BaseAttention"),
    "BaseCompressor": ("neuralops.core.base", "BaseCompressor"),
    "ArchitectureSpec": ("neuralops.analysis.inspector", "ArchitectureSpec"),
    "inspect_architecture": ("neuralops.analysis.inspector", "inspect_architecture"),
    "render_architecture_card": ("neuralops.analysis.inspector", "render_architecture_card"),
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
    "calibrate_attention_subspace": ("neuralops.models.surgery", "calibrate_attention_subspace"),
    "heal_replaced_model": ("neuralops.models.surgery", "heal_replaced_model"),
    "RooflineReport": ("neuralops.analysis.roofline", "RooflineReport"),
    "profile_roofline": ("neuralops.analysis.roofline", "profile_roofline"),
    "render_roofline_terminal": ("neuralops.analysis.roofline", "render_roofline_terminal"),
    "extract_surgical_manifest": ("neuralops.models.surgical_io", "extract_surgical_manifest"),
    "save_surgical_checkpoint": ("neuralops.models.surgical_io", "save_surgical_checkpoint"),
    "load_surgical_checkpoint": ("neuralops.models.surgical_io", "load_surgical_checkpoint"),
    "NeuralOpsStateCache": ("neuralops.core.state_cache", "NeuralOpsStateCache"),
    "benchmark_autoregressive_decoding": ("neuralops.core.state_cache", "benchmark_autoregressive_decoding"),
    "select_optimal_triton_config": ("neuralops.core.triton_kernels.autotuner", "select_optimal_triton_config"),
    "compute_pareto_frontier": ("neuralops.analysis.paper_export", "compute_pareto_frontier"),
    "export_benchmark_table": ("neuralops.analysis.paper_export", "export_benchmark_table"),
    "paper_export": ("neuralops.analysis", "paper_export"),
    "ResultRecord": ("neuralops.analysis.results_ledger", "ResultRecord"),
    "ResultsLedger": ("neuralops.analysis.results_ledger", "ResultsLedger"),
    "get_default_ledger": ("neuralops.analysis.results_ledger", "get_default_ledger"),
    "log_experiment_result": ("neuralops.analysis.results_ledger", "log_experiment_result"),
    "BaseTrainer": ("neuralops.trainer.base_trainer", "BaseTrainer"),
    "COMPRESSOR_REGISTRY": ("neuralops.core.registry", "COMPRESSOR_REGISTRY"),
    "CompressionPipeline": ("neuralops.core.compression_pipeline", "CompressionPipeline"),
    "CustomLoopTrainer": ("neuralops.trainer.custom_loop_trainer", "CustomLoopTrainer"),
    "FairnessEvaluator": ("neuralops.evaluation.fairness", "FairnessEvaluator"),
    "Recipe": ("neuralops.recipe.schema", "Recipe"),
    "load_recipe": ("neuralops.recipe.loader", "load_recipe"),
    "loads_recipe": ("neuralops.recipe.loader", "loads_recipe"),
    "save_recipe": ("neuralops.recipe.serializer", "save_recipe"),
    "recipe_hash": ("neuralops.recipe.canonical", "recipe_hash"),
    "plan_recipe": ("neuralops.pipeline_transform.planner", "plan_recipe"),
    "ExecutionPlan": ("neuralops.pipeline_transform.plan", "ExecutionPlan"),
    "execute_plan": ("neuralops.pipeline_transform.executor", "execute_plan"),
    "PipelineResult": ("neuralops.pipeline_transform.result", "PipelineResult"),
    "PipelineError": ("neuralops.pipeline_transform.result", "PipelineError"),
    "ValidationFailure": ("neuralops.pipeline_transform.result", "ValidationFailure"),
    "analyze_recipe": ("neuralops.pipeline_transform.dependencies", "analyze_recipe"),
    "estimate_resources": ("neuralops.pipeline_transform.hardware_planner", "estimate_resources"),
    "suggest_config": ("neuralops.pipeline_transform.auto_config", "suggest_config"),
    "export_to_hf": ("neuralops.export.hf_compat", "export_to_hf"),
    "check_compatibility": ("neuralops.export.runtime_compat", "check_compatibility"),
    "ModelSurgery": ("neuralops.surgery.model_surgery", "ModelSurgery"),
    "get_operation": ("neuralops.operations.registry", "get_operation"),
    "list_operations": ("neuralops.operations.registry", "list_operations"),
    "find_adapter": ("neuralops.adapters.registry", "find_adapter"),
    "list_adapters": ("neuralops.adapters.registry", "list_adapters"),
    "EvidenceRecord": ("neuralops.reporting.evidence", "EvidenceRecord"),
    "build_evidence": ("neuralops.reporting.evidence", "build_evidence"),
    "render_report": ("neuralops.reporting.report", "render_report"),
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


def replace_attention(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
    """Surgically replace attention mechanisms in a model (requires PyTorch)."""
    from neuralops._optional import _require_torch

    _require_torch("replace_attention")
    from neuralops.api import replace_attention as _impl

    return _impl(*args, **kwargs)


def compress(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
    """Compress a neural network model (requires PyTorch)."""
    from neuralops._optional import _require_torch

    _require_torch("compress")
    from neuralops.api import compress as _impl

    return _impl(*args, **kwargs)


_STATIC_API_NAMES = (
    "load_model",
    "replace_attention",
    "transplant_attention",
    "transplant",
    "serve",
    "compress",
    "get_attention_module",
    "quick_benchmark",
    "export_model",
    "export",
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
    "create_attention",
    "create_compressor",
    "list_families",
    "register_custom_attention",
    "register_custom_compressor",
    "unregister_custom_module",
    "validate_module_contract",
    "scaffold_module",
    "compare_attentions_ablation",
    "surgical_transplant_and_heal_pipeline",
    "heal_distributed",
    "train",
    "distill",
    "evaluate",
    "load_and_prepare_dataset",
    "train_grpo",
    "benchmark",
    "benchmark_all",
    "info",
    "available_backends",
    "has_triton",
    "validate",
    "certify",
    "get_version",
    "NeuralOpsStateCache",
    "benchmark_autoregressive_decoding",
    "select_optimal_triton_config",
    "oneshot",
    "recommend",
    "summary",
    "compare",
    "patch_transformers",
    "AutoNeuralOpsModel",
    "AutoNeuralOpsForCausalLM",
    "AutoNeuralOpsForSequenceClassification",
    "merge_for_inference",
    "NeuralOpsTrainer",
    "save_attention_only",
    "load_attention_from_hub",
    "pipeline",
    "TextGenerationPipeline",
    "probe_optimal_batch_size",
)

_PUBLIC_API = {name: "neuralops.api" for name in _STATIC_API_NAMES}

_API_EXPORTS = {
    **_PUBLIC_API,
    "compile_to_so": "neuralops.api",
    "list_fla_attentions": "neuralops.integrations.fla",
    "list_surgeries": "neuralops.surgery",
    "SurgeryRecipe": "neuralops.surgery.pipeline",
    "force_kernel_backend": "neuralops.ops.cuda._fallback",
    "list_triton_kernels": "neuralops.kernels",
    "list_cuda_kernels": "neuralops.kernels",
    "GLAttention": "neuralops.layers.linear.gla_attention",
    "CosformerAttention": "neuralops.layers.linear.cosformer",
    "SVDLinearAttention": "neuralops.layers.svd.svd_linear_attention",
    "DecayLinearAttention": "neuralops.layers.gated.decay_linear_attention",
    "Muon": "neuralops.trainer.optimizers.muon",
    "StiefelOptimizer": "neuralops.trainer.optimizers",
    "RiemannianAdam": "neuralops.trainer.optimizers",
    "HybridAttentionSpec": "neuralops.models.hybrid_spec",
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
    "HadamardIncoherentSVDCompressor": "neuralops.compression.svd.hadamard_incoherent_svd",
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
    "load_pretrained_bert": ("neuralops.hub", "load_pretrained_bert"),
    "list_pretrained_bert_models": ("neuralops.hub", "list_pretrained_bert_models"),
}

_PKG_DIR = Path(__file__).resolve().parent
_SUBPACKAGES: frozenset[str] = frozenset(
    d.name
    for d in _PKG_DIR.iterdir()
    if d.is_dir() and (d / "__init__.py").exists() and not d.name.startswith((".", "_"))
)

# Compatibility-only redirects for legacy package namespaces (never in _SUBPACKAGES)
_COMPAT_SUBPACKAGES: dict[str, str] = {
    "attentions": "neuralops.layers",
    "compression_engine": "neuralops.compression",
    "training": "neuralops.trainer",
}


def inject_gla(model, **kwargs):
    """Replace all standard attention layers with GLA. Convenience wrapper."""
    from neuralops.layers.linear.gla_attention import GLAttention as _GLA
    from neuralops.models.surgery import replace_attention_in_model

    return replace_attention_in_model(model, _GLA, **kwargs)


# Universal factories (F-80 single implementation in core.registry)
from neuralops.core.registry import create_attention, create_compressor, list_attentions, list_compressors

__all__ = sorted(
    list(
        set(
            [
                "__version__",
                "create_attention",
                "create_compressor",
                "list_attentions",
                "list_compressors",
                "inject_gla",
                "ComposableAttention",
                "compose_attention",
                "compile_to_so",
                "list_surgeries",
                "SurgeryRecipe",
                "surgery",
                "force_kernel_backend",
                "list_triton_kernels",
                "list_cuda_kernels",
                "list_fla_attentions",
                "GLAttention",
                "CosformerAttention",
                "SVDLinearAttention",
                "DecayLinearAttention",
                "Muon",
                "StiefelOptimizer",
                "RiemannianAdam",
                "HybridAttentionSpec",
                "NeuralOpsError",
                "ConfigError",
                "WeightCopyError",
                "BackendError",
                "NumericalError",
                "BitPackedW4A16Linear",
                "BitPackedLowRankSVDLinear",
                "pack_int4_weights",
                "unpack_int4_weights",
                "fused_w4a16_gemm_triton",
                "fast_hadamard_transform",
                "HadamardIncoherentSVDCompressor",
                "SelfSpeculativeDecoder",
                "SpeculativeDecodeStats",
                "PagedKVCache",
                "PagedStateTable",
                "ATTENTION_REGISTRY",
                "COMPRESSOR_REGISTRY",
                "NeedleInAHaystackEvaluator",
                "plot_niah_heatmap",
                "NeuralOpsLMEvalAdapter",
                "evaluate_zero_shot_tasks",
            ]
            + list(_PUBLIC_API.keys())
            + list(_LAZY_EXPORTS.keys())
        )
    )
)


def __getattr__(name: str) -> Any:
    """PEP 562 dynamic lazy loader."""
    if name not in _TORCH_FREE_SYMBOLS and (
        name in __all__
        or name in _API_EXPORTS
        or name in _LAZY_EXPORTS
        or name in _SUBPACKAGES
        or name in {"ATTENTION_REGISTRY", "ALL_ATTENTION_CLASSES", "evaluation"}
    ):
        _require_torch(f"neuralops.{name}")

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
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            if exc.name == "torch" or (exc.name and exc.name.startswith("torch.")):
                raise _torch_install_error(f"neuralops.{name}", exc) from exc
            raise
        val = getattr(mod, name)
        globals()[name] = val
        return val

    if name in _LAZY_EXPORTS:
        module_path, attr_name = _LAZY_EXPORTS[name]
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            if exc.name == "torch" or (exc.name and exc.name.startswith("torch.")):
                raise _torch_install_error(f"neuralops.{name}", exc) from exc
            raise
        val = getattr(module, attr_name)
        globals()[name] = val
        return val

    if name == "evaluation":
        try:
            evaluation_package = importlib.import_module("neuralops.evaluation")
            globals()[name] = evaluation_package
            return evaluation_package
        except ImportError as exc:
            _logger.debug("Optional submodule evaluation not importable: %s", exc)

    if name in _COMPAT_SUBPACKAGES:
        target = _COMPAT_SUBPACKAGES[name]
        try:
            if not os.environ.get("NEURALOPS_NO_DEPRECATION_WARNINGS"):
                warnings.warn(
                    f"neuralops.{name} is deprecated and will be removed in v3.0.0; use {target} instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            submod = importlib.import_module(target)
            globals()[name] = submod
            return submod
        except (ModuleNotFoundError, ImportError) as exc:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc

    if name in _SUBPACKAGES:
        try:
            submod = importlib.import_module(f"neuralops.{name}")
            globals()[name] = submod
            return submod
        except (ModuleNotFoundError, ImportError) as exc:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Execute __dir__ operation."""
    return sorted(list(globals().keys()) + __all__ + list(_SUBPACKAGES))
