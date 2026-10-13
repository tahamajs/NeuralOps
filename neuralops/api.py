"""
NeuralOps Public Clean API.

Enables drop-in replacement, compression, benchmarking, and export of attention layers
in any PyTorch or HuggingFace Transformer model with a clean 19-function interface:

    import neuralops

    # Drop-in replace attention mechanism
    model = neuralops.replace_attention(model, "cosformer")

    # Compress model using SVD or low-rank adapters
    model = neuralops.compress(model, method="svd", rank=32)

    # Inspect available modules and backends
    attentions = neuralops.list_attentions()
    compressors = neuralops.list_compressors()
    backends = neuralops.available_backends()
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import torch
import torch.nn as nn

from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
import neuralops.attentions  # noqa: F401 - ensure registrations
import neuralops.compression_engine  # noqa: F401 - ensure registrations
from neuralops.core.attention_utils import inject_attention
from neuralops.export.exporter import export_model as _export_model, load_model as _load_model, load_exported_model

logger = logging.getLogger("NeuralOps.API")


# =====================================================================
# 1. Model / Surgery / Export Operations
# =====================================================================

def replace_attention(
    model: nn.Module,
    attention: Optional[str] = None,
    rank: Optional[int] = None,
    copy_weights: bool = True,
    layers: Optional[List[int]] = None,
    attention_type: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module:
    """Drop-in replace attention layers in any PyTorch / HuggingFace model.

    Records the operation in `model._neuralops_metadata`.

    Args:
        model: HuggingFace or PyTorch Transformer model (GPT-2, LLaMA, Mistral, BERT, etc.).
        attention: Attention mechanism name from `neuralops.list_attentions()`
                   (e.g., "cosformer", "mamba_block", "gqa_attention", "svd_linear_attention").
        rank: Rank parameter for low-rank or SVD-based attention variants (default: None).
        copy_weights: Whether to transplant weights from the original model (default: True).
        layers: Optional list of layer indices to replace (default: None replaces all layers).
        attention_type: Alias for attention.
        **kwargs: Additional parameters forwarded to the attention module constructor.

    Returns:
        The model with attention layers replaced in-place.
    """
    attn_name = attention or attention_type
    if not attn_name:
        raise ValueError("Must specify attention or attention_type.")
    if attn_name not in ATTENTION_REGISTRY:
        raise ValueError(
            f"Unknown attention module '{attn_name}'. "
            f"Use neuralops.list_attentions() to see all {len(ATTENTION_REGISTRY)} available modules."
        )

    call_kwargs = dict(kwargs)
    if layers is not None:
        call_kwargs["layer_indices"] = layers

    out_model = inject_attention(
        model=model,
        attention_type=attn_name,
        rank=rank,
        copy_weights=copy_weights,
        **call_kwargs,
    )

    # Attach surgery metadata onto model instance
    meta_entry: Dict[str, Any] = {
        "op": "replace",
        "type": attn_name,
        "kwargs": {
            "rank": rank,
            "copy_weights": copy_weights,
            "layers": layers,
            **kwargs,
        },
    }
    if not hasattr(out_model, "_neuralops_metadata") or out_model._neuralops_metadata is None:
        out_model._neuralops_metadata = []
    out_model._neuralops_metadata.append(meta_entry)

    if hasattr(out_model, "config") and hasattr(out_model.config, "to_dict"):
        if not hasattr(out_model.config, "neuralops_history") or out_model.config.neuralops_history is None:
            out_model.config.neuralops_history = []
        out_model.config.neuralops_history.append(meta_entry)

    return out_model


def compress(
    model: nn.Module,
    method: str = "svd",
    rank: int = 32,
    layers: Optional[List[int]] = None,
    **kwargs: Any,
) -> nn.Module:
    """Compress model weights using SVD, low-rank factorization, or pruning.

    Records the operation in `model._neuralops_metadata`.

    Args:
        model: PyTorch or HuggingFace model to compress.
        method: Compression engine name from `neuralops.list_compressors()`.
        rank: Target rank for low-rank decomposition (default: 32).
        layers: Optional list of layer indices to compress (default: None compresses all).
        **kwargs: Extra parameters passed to the compressor.

    Returns:
        The compressed model.
    """
    from neuralops.core.compression_pipeline import compress_model

    comp_step: Dict[str, Any] = {
        "method": method,
        "rank": rank,
        **kwargs,
    }
    if layers is not None:
        comp_step["layer_indices"] = layers

    config = {
        "compressors": [comp_step],
    }

    out_model = compress_model(model=model, config=config)

    # Attach compression metadata onto model instance
    meta_entry: Dict[str, Any] = {
        "op": "compress",
        "type": method,
        "kwargs": {
            "rank": rank,
            "layers": layers,
            **kwargs,
        },
    }
    if not hasattr(out_model, "_neuralops_metadata") or out_model._neuralops_metadata is None:
        out_model._neuralops_metadata = []
    out_model._neuralops_metadata.append(meta_entry)

    if hasattr(out_model, "config") and hasattr(out_model.config, "to_dict"):
        if not hasattr(out_model.config, "neuralops_history") or out_model.config.neuralops_history is None:
            out_model.config.neuralops_history = []
        out_model.config.neuralops_history.append(meta_entry)

    return out_model


def export_model(
    model: nn.Module,
    output_dir: Union[str, Path],
    format: str = "safetensors",
    **kwargs: Any,
) -> Path:
    """Export a NeuralOps-modified or compressed model to a production artifact format.

    Persists `model._neuralops_metadata` into `neuralops_config.json` alongside weights.

    Args:
        model: The PyTorch/HuggingFace model to export.
        output_dir: Destination folder path.
        format: Export format ('safetensors', 'onnx', or 'torchscript').
        **kwargs: Additional exporter arguments.

    Returns:
        Path to output directory or exported artifact file.
    """
    return _export_model(model=model, output_dir=output_dir, format=format, **kwargs)


def load_model(
    export_dir: Union[str, Path],
    base_model: Optional[nn.Module] = None,
) -> nn.Module:
    """Load an exported NeuralOps model and replay surgery metadata.

    Reads `neuralops_config.json` and replays all recorded `_neuralops_metadata`
    operations sequentially before loading weights.

    Args:
        export_dir: Directory containing `neuralops_config.json` and weight files.
        base_model: Optional pre-instantiated base model.

    Returns:
        The reconstructed neuralops model.
    """
    return _load_model(export_dir=export_dir, base_model=base_model)


# =====================================================================
# 2. Discovery Operations
# =====================================================================

def list_attentions() -> List[str]:
    """Return a sorted list of all available canonical attention module names."""
    return sorted(list(ATTENTION_REGISTRY.keys()))


def list_compressors() -> List[str]:
    """Return a sorted list of all available canonical compressor names."""
    return sorted(list(COMPRESSOR_REGISTRY.keys()))


def get_attention_class(name: str) -> Type[Any]:
    """Retrieve the class or constructor for a registered attention module.

    Args:
        name: Attention module identifier from `list_attentions()`.

    Returns:
        Registered attention class.
    """
    if name not in ATTENTION_REGISTRY:
        raise KeyError(f"Unknown attention '{name}'. Available: {list_attentions()}")
    return ATTENTION_REGISTRY[name]


def get_compressor_class(name: str) -> Type[Any]:
    """Retrieve the class or constructor for a registered compressor engine.

    Args:
        name: Compressor engine identifier from `list_compressors()`.

    Returns:
        Registered compressor class.
    """
    if name not in COMPRESSOR_REGISTRY:
        raise KeyError(f"Unknown compressor '{name}'. Available: {list_compressors()}")
    return COMPRESSOR_REGISTRY[name]


def list_families() -> Dict[str, List[str]]:
    """Return grouped categorization of all registered attention modules and compressors.

    Returns:
        Dictionary mapping family names (e.g., 'attention_linear', 'compression_svd') to module lists.
    """
    families: Dict[str, List[str]] = {}

    for name, cls in ATTENTION_REGISTRY.items():
        mod = getattr(cls, "__module__", "")
        subfam = mod.split(".")[2] if "neuralops.attentions." in mod else "other"
        group_key = f"attention_{subfam}"
        families.setdefault(group_key, []).append(name)

    for name, cls in COMPRESSOR_REGISTRY.items():
        mod = getattr(cls, "__module__", "")
        subfam = mod.split(".")[2] if "neuralops.compression_engine." in mod else "other"
        group_key = f"compression_{subfam}"
        families.setdefault(group_key, []).append(name)

    # Sort modules within each family
    for k in families:
        families[k] = sorted(families[k])

    return dict(sorted(families.items()))


# =====================================================================
# 3. Runners (Train / Distill / Evaluate / Benchmark)
# =====================================================================

def train(
    model: nn.Module,
    dataset: Any,
    epochs: int = 3,
    lr: float = 2e-5,
    eval_dataset: Optional[Any] = None,
    device: Union[str, torch.device] = "cpu",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train or fine-tune a model on a dataset.

    Args:
        model: PyTorch model to train.
        dataset: Training dataset or DataLoader.
        epochs: Number of training epochs (default: 3).
        lr: Learning rate (default: 2e-5).
        eval_dataset: Optional evaluation dataset or DataLoader.
        device: Target execution device.
        **kwargs: Additional training arguments.

    Returns:
        Dictionary containing training telemetry and metrics.
    """
    from neuralops.benchmarks.core.training import finetune_classification
    from torch.utils.data import DataLoader

    train_loader = dataset if isinstance(dataset, DataLoader) else DataLoader(dataset, batch_size=kwargs.pop("batch_size", 32), shuffle=True)
    eval_loader = eval_dataset if (eval_dataset is None or isinstance(eval_dataset, DataLoader)) else DataLoader(eval_dataset, batch_size=kwargs.pop("eval_batch_size", 32))

    return finetune_classification(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        **kwargs,
    )


def distill(
    student: nn.Module,
    teacher: nn.Module,
    dataset: Any,
    epochs: int = 3,
    lr: float = 2e-5,
    device: Union[str, torch.device] = "cpu",
    temperature: float = 2.0,
    alpha: float = 0.5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute knowledge distillation from teacher to student.

    Args:
        student: Student model to train.
        teacher: Teacher model providing soft target distributions.
        dataset: Training dataset or DataLoader.
        epochs: Number of distillation epochs (default: 3).
        lr: Learning rate (default: 2e-5).
        device: Target device.
        temperature: Softmax temperature for KD loss (default: 2.0).
        alpha: Weight for KD loss vs task loss (default: 0.5).
        **kwargs: Additional training arguments.

    Returns:
        Dictionary containing distillation metrics.
    """
    from neuralops.benchmarks.core.training import train_kd
    from torch.utils.data import DataLoader

    train_loader = dataset if isinstance(dataset, DataLoader) else DataLoader(dataset, batch_size=kwargs.pop("batch_size", 32), shuffle=True)

    return train_kd(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        temperature=temperature,
        alpha=alpha,
        **kwargs,
    )


def evaluate(
    model: nn.Module,
    dataset: str = "sst2",
    device: Optional[Union[str, torch.device]] = None,
    batch_size: int = 16,
    seq_len: int = 128,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate model performance, accuracy, latency, and throughput.

    Args:
        model: Model to evaluate.
        dataset: Dataset identifier (e.g., 'sst2', 'wikitext2').
        device: Execution device.
        batch_size: Evaluation batch size.
        seq_len: Sequence length.
        **kwargs: Extra evaluation parameters.

    Returns:
        Dictionary containing evaluation metrics (accuracy, latency, memory, etc.).
    """
    from neuralops.evaluation.eval_all_models_datasets import evaluate_single_combination_opt
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    result = evaluate_single_combination_opt(
        track=kwargs.pop("track", "track1_svd"),
        module_name=kwargs.pop("module_name", "standard"),
        module_type=kwargs.pop("module_type", "attention"),
        model_family=kwargs.pop("model_family", "distilbert"),
        dataset=dataset,
        batch_size=batch_size,
        seq_len=seq_len,
        device=torch.device(dev),
        **kwargs,
    )
    return result.to_dict() if hasattr(result, "to_dict") else dict(result.__dict__)


def benchmark(
    module: Optional[str] = None,
    model: str = "distilbert-base-uncased",
    dataset: str = "sst2",
    rank: Optional[int] = 64,
    device: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    seq_len: int = 128,
    eval_only: bool = False,
    timeout_s: int = 1800,
    **kwargs: Any,
) -> Any:
    """Run an isolated single benchmark execution.

    Args:
        module: Module name to benchmark.
        model: Base model identifier.
        dataset: Dataset identifier.
        rank: Target rank parameter.
        device: Execution device.
        epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        seq_len: Sequence length.
        eval_only: Skip fine-tuning and measure evaluation metrics only.
        timeout_s: Timeout in seconds.
        **kwargs: Additional benchmark options.

    Returns:
        BenchmarkResult object.
    """
    from neuralops.benchmarks.api import benchmark as _bm
    return _bm(
        module=module,
        model=model,
        dataset=dataset,
        rank=rank,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seq_len=seq_len,
        eval_only=eval_only,
        timeout_s=timeout_s,
        **kwargs,
    )


def benchmark_all(
    modules: Union[str, List[str]] = "all",
    models: Union[str, List[str]] = "distilbert-base-uncased",
    datasets: Union[str, List[str]] = "sst2",
    ranks: Union[int, List[int]] = 64,
    parallel: int = 1,
    output_dir: Union[Path, str] = "results",
    **kwargs: Any,
) -> Any:
    """Orchestrate multi-module and multi-model benchmark sweeps.

    Args:
        modules: List of module names or 'all'.
        models: List of model identifiers.
        datasets: List of dataset identifiers.
        ranks: List of ranks.
        parallel: Number of parallel jobs.
        output_dir: Destination results folder.
        **kwargs: Additional Runner configurations.

    Returns:
        Aggregated summary dataframe or results list.
    """
    from neuralops.benchmarks.api import Runner
    runner = Runner(
        modules=modules,
        models=models,
        datasets=datasets,
        ranks=ranks,
        parallel=parallel,
        output_dir=output_dir,
        **kwargs,
    )
    return runner.run()


# =====================================================================
# 4. Utilities & Diagnostics
# =====================================================================

def get_version() -> str:
    """Return the current NeuralOps release version."""
    return "1.3.0"


def has_triton() -> bool:
    """Check whether Triton kernel compiler and GPU backend are available."""
    try:
        from neuralops.core.triton_kernels import HAS_TRITON
        return bool(HAS_TRITON)
    except Exception:
        return False


def available_backends() -> List[str]:
    """Return a list of available hardware acceleration backends.

    Returns:
        List of backend identifiers (e.g., ['cpu', 'cuda', 'mps', 'triton']).
    """
    backends = ["cpu"]
    if torch.cuda.is_available():
        backends.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        backends.append("mps")
    if has_triton():
        backends.append("triton")
    return backends


def validate(
    model: Optional[nn.Module] = None,
    input_shape: Tuple[int, int] = (2, 32),
    device: str = "cpu",
) -> bool:
    """Validate model construction, device movement, forward pass, and backward gradient flow.

    Args:
        model: nn.Module instance to validate.
        input_shape: Shape tuple (batch_size, seq_len) for dummy tokens.
        device: Device to validate on.

    Returns:
        True if all checks pass without NaNs or exceptions, False otherwise.
    """
    if model is None:
        raise ValueError("Must provide a model instance to validate.")
    from neuralops.validation.model_validator import ModelValidator
    validator = ModelValidator(model=model, input_shape=input_shape, device=device)
    return bool(validator.validate())


def info() -> Dict[str, Any]:
    """Return diagnostic overview of NeuralOps installation and system capabilities."""
    return {
        "version": get_version(),
        "attentions_count": len(ATTENTION_REGISTRY),
        "compressors_count": len(COMPRESSOR_REGISTRY),
        "backends": available_backends(),
        "has_triton": has_triton(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


__all__ = [
    # Model/Export
    "load_model",
    "replace_attention",
    "compress",
    "export_model",
    "load_exported_model",
    # Discovery
    "list_attentions",
    "list_compressors",
    "get_attention_class",
    "get_compressor_class",
    "list_families",
    # Runners
    "train",
    "distill",
    "evaluate",
    "benchmark",
    "benchmark_all",
    # Utils
    "info",
    "available_backends",
    "has_triton",
    "validate",
    "get_version",
]
