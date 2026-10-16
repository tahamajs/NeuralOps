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
    # Allow passing target_module or attention as an instantiated nn.Module or string alias
    target_mod = kwargs.pop("target_module", None)
    if isinstance(target_mod, str) and not attention:
        attention = target_mod
        target_mod = None
    elif isinstance(attention, nn.Module):
        target_mod = attention
        attention = None

    if target_mod is not None:
        from neuralops.core.model_utils import replace_module
        from neuralops.core.architecture import detect_architecture, get_attention_modules
        arch = detect_architecture(model)
        modules_list = list(get_attention_modules(model))
        for layer_i, (name, old_attn) in enumerate(modules_list):
            if layers is not None and layer_i not in layers:
                continue
            import copy
            mod_copy = copy.deepcopy(target_mod)
            old_param = next(old_attn.parameters(), None)
            if old_param is not None:
                mod_copy = mod_copy.to(device=old_param.device, dtype=old_param.dtype)
            from neuralops.utils.replace_attention import UniversalAttentionWrapper
            wrapped = UniversalAttentionWrapper(mod_copy, layer_name=name, return_tuple=True)
            replace_module(model, name, wrapped)
        return model

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

    hybrid_spec = kwargs.get("hybrid_spec", None)
    if hybrid_spec is not None:
        from neuralops.models.hybrid_spec import normalize_hybrid_spec
        norm_spec = normalize_hybrid_spec(hybrid_spec)
        call_kwargs["hybrid_spec"] = norm_spec

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

    from neuralops.export.surgery_manifest import record_attention_surgery
    record_attention_surgery(
        out_model,
        layer_path="all_attention",
        attention_type=attn_name,
        kwargs=meta_entry["kwargs"],
    )

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


# =====================================================================
# Model Surgery & Transplantation Aliases & Functions
# =====================================================================

def transplant_attention(
    model: nn.Module,
    attention: Optional[str] = None,
    rank: Optional[int] = None,
    copy_weights: bool = True,
    layers: Optional[List[int]] = None,
    attention_type: Optional[str] = None,
    target_attention: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module:
    """Canonical alias for replace_attention."""
    attn = attention or attention_type or target_attention
    return replace_attention(
        model=model,
        attention=attn,
        rank=rank,
        copy_weights=copy_weights,
        layers=layers,
        **kwargs,
    )


def collect_surgery_distillation_loss(model: nn.Module) -> torch.Tensor:
    """Collect shadow twin distillation loss across all transplanted attention layers."""
    from neuralops.surgery.shadow_twin import collect_surgery_distillation_loss as _collect
    return _collect(model)


def set_surgery_alpha(model: nn.Module, alpha: float) -> None:
    """Adjust distillation blending weight alpha in [0, 1] across all transplanted layers."""
    from neuralops.surgery.shadow_twin import set_surgery_alpha as _set_alpha
    _set_alpha(model, alpha)


def set_surgery_stage(model: nn.Module, stage: int) -> None:
    """Set multi-stage recovery training curriculum stage (1: Shadow, 2: Joint, 3: Finalized)."""
    from neuralops.surgery.shadow_twin import set_surgery_stage as _set_stage
    _set_stage(model, stage)


def finalize_surgery(model: nn.Module) -> nn.Module:
    """Sever teacher branches and permanently finalize student attention transplantation."""
    from neuralops.surgery.shadow_twin import finalize_surgery as _finalize
    return _finalize(model)



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


def push_to_hub(
    model: nn.Module,
    repo_id: str,
    tokenizer: Any = None,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload NeuralOps model",
    **kwargs: Any,
) -> str:
    """Export a self-contained NeuralOps model and push directly to Hugging Face Hub.

    Configures auto_map so the model can be loaded by anyone via:
        AutoModel.from_pretrained(repo_id, trust_remote_code=True)

    Args:
        model: PyTorch/HuggingFace model to upload.
        repo_id: Target HuggingFace repository ID (e.g., 'username/my-model').
        tokenizer: Optional tokenizer to upload alongside model.
        token: HuggingFace API token (or read from HF_TOKEN env var).
        private: Whether the repository should be private.
        commit_message: Git commit message for upload.

    Returns:
        URL string of the uploaded HuggingFace Hub repository.
    """
    from neuralops.models.hub import push_model_to_hub as _push
    return _push(
        model=model,
        repo_id=repo_id,
        tokenizer=tokenizer,
        token=token,
        private=private,
        commit_message=commit_message,
    )


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


def compile_to_so(
    model: nn.Module,
    sample_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    output_path: Union[str, Path] = "model_aot.so",
    dynamic_batch: bool = True,
    dynamic_seq_len: bool = True,
    min_seq_len: int = 1,
    max_seq_len: int = 32768,
    device: Optional[Union[str, torch.device]] = None,
) -> Any:
    """Compile a model to a standalone C++ AOTInductor shared library (.so or .pt2)."""
    from neuralops.export.exporter import compile_to_so as _compile_to_so
    return _compile_to_so(
        model=model,
        sample_inputs=sample_inputs,
        output_path=output_path,
        dynamic_batch=dynamic_batch,
        dynamic_seq_len=dynamic_seq_len,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        device=device,
    )



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

    return families


def create_attention(
    name: str,
    config: Any = None,
    dim: int | None = None,
    num_heads: int | None = None,
    embed_dim: int | None = None,
    rank: int | None = None,
    **kwargs: Any,
) -> Any:
    """Universal attention factory instantiating any of the 131 registered attention mechanisms."""
    import neuralops
    return neuralops.create_attention(
        name=name,
        config=config,
        dim=dim,
        num_heads=num_heads,
        embed_dim=embed_dim,
        rank=rank,
        **kwargs,
    )


def create_compressor(name: str, model: Any = None, rank: int | None = None, **kwargs: Any) -> Any:
    """Universal compressor factory instantiating any of the 72 registered compression engines."""
    import neuralops
    return neuralops.create_compressor(name=name, model=model, rank=rank, **kwargs)


def register_custom_attention(*args: Any, **kwargs: Any) -> Any:
    """Register a custom attention mechanism with full framework compatibility."""
    from neuralops.core.extensibility import register_custom_attention as _reg
    return _reg(*args, **kwargs)


def register_custom_compressor(*args: Any, **kwargs: Any) -> Any:
    """Register a custom compression engine with full framework compatibility."""
    from neuralops.core.extensibility import register_custom_compressor as _reg
    return _reg(*args, **kwargs)


def unregister_custom_module(name: str, category: Optional[str] = None) -> bool:
    """Cleanly unregister a custom module to restore the canonical registry counts."""
    from neuralops.core.extensibility import unregister_custom_module as _unreg
    return _unreg(name, category)


def validate_module_contract(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Automated 6-point contract diagnostic check for any module."""
    from neuralops.core.extensibility import validate_module_contract as _val
    return _val(*args, **kwargs)


def scaffold_module(name: str, category: str = "attention", family: str = "linear-scan", output_file: Optional[str] = None) -> str:
    """Generate boilerplate code for a new custom module."""
    from neuralops.core.extensibility import scaffold_module as _scaff
    return _scaff(name, category=category, family=family, output_file=output_file)



def compare_attentions_ablation(
    attention_names: Sequence[str],
    batch_size: int = 2,
    seq_len: int = 1024,
    embed_dim: int = 768,
    num_heads: int = 12,
    dtype: str = "float16",
    device: Optional[Union[str, torch.device]] = None,
    export_format: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Ablation benchmark workflow across multiple attention candidate architectures.

    Profiles empirical hardware performance via `profile_roofline`, computes multi-objective
    Pareto frontiers via `compute_pareto_frontier`, and generates publication tables.
    """
    from neuralops.core.roofline import profile_roofline
    from neuralops.core.paper_export import compute_pareto_frontier, export_benchmark_table

    target_dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    records = []

    for name in attention_names:
        try:
            report = profile_roofline(
                module_or_name=name,
                batch_size=batch_size,
                seq_len=seq_len,
                embed_dim=embed_dim,
                num_heads=num_heads,
                dtype=dtype,
                device=target_dev,
                **kwargs,
            )
            rec = {
                "module": name,
                "latency_triton_ms": round(report.latency_triton_ms, 3),
                "speedup_x": round(report.speedup_x, 2),
                "achieved_tflops": round(report.achieved_tflops, 2),
                "sram_bytes_saved_mb": round(report.sram_bytes_saved_mb, 2),
                "peak_vram_mb": round(report.peak_vram_mb, 1),
                "bottleneck": report.bottleneck_regime,
            }
            records.append(rec)
        except Exception as e:
            logger.warning("Failed to profile '%s' in ablation study: %s", name, e)

    pareto_mask = compute_pareto_frontier(records)
    for rec, is_p in zip(records, pareto_mask):
        rec["pareto_optimal"] = is_p

    formatted_table = None
    if export_format in ("latex", "markdown", "csv"):
        formatted_table = export_benchmark_table(
            records=records,
            format=export_format,
            caption=f"Empirical Hardware Ablation (seq_len={seq_len}, dim={embed_dim})",
        )

    return {
        "records": records,
        "pareto_optimal_count": sum(pareto_mask),
        "table": formatted_table,
    }


def surgical_transplant_and_heal_pipeline(
    teacher_model: nn.Module,
    student_base_model: Optional[nn.Module] = None,
    target_attention: str = "gated_delta_net",
    calibration_inputs: Any = None,
    heal_steps: int = 25,
    heal_lr: float = 2e-3,
    save_dir: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """End-to-end surgical transplant, closed-form Ridge alignment, and micro-healing pipeline.

    Executes:
      1. replace_attention on student model (raw replacement baseline)
      2. calibrate_attention_subspace (closed-form analytical Ridge projection alignment)
      3. heal_replaced_model (layerwise local gradient micro-healing with best-state rollback)
      4. Optional save_surgical_checkpoint serialization
    """
    import copy
    from neuralops.core._attention_surgery import calibrate_attention_subspace, heal_replaced_model

    if student_base_model is None:
        student = copy.deepcopy(teacher_model)
    else:
        student = student_base_model

    # Detect dimension from teacher model or calibration inputs if not passed
    call_kwargs = dict(kwargs)
    if "embed_dim" not in call_kwargs and "d_model" not in call_kwargs and "hidden_size" not in call_kwargs and "dim" not in call_kwargs:
        t_dim = getattr(teacher_model, "dim", getattr(teacher_model, "hidden_size", getattr(teacher_model, "d_model", None)))
        if t_dim is None and hasattr(teacher_model, "config"):
            t_dim = getattr(teacher_model.config, "hidden_size", getattr(teacher_model.config, "d_model", None))
        if t_dim is None:
            # Check calibration inputs shape: (B, T, D)
            if calibration_inputs is not None and isinstance(calibration_inputs, torch.Tensor) and calibration_inputs.ndim >= 2:
                t_dim = calibration_inputs.shape[-1]
            elif isinstance(calibration_inputs, (list, tuple)) and len(calibration_inputs) > 0 and isinstance(calibration_inputs[0], torch.Tensor):
                t_dim = calibration_inputs[0].shape[-1]
        if t_dim is not None:
            call_kwargs["embed_dim"] = t_dim

    # 1. Raw Surgical Replacement
    student = replace_attention(student, attention=target_attention, **call_kwargs)

    # Measure raw student error if inputs provided
    initial_metrics: Dict[str, float] = {}
    if calibration_inputs is not None:
        try:
            student.eval()
            teacher_model.eval()
            with torch.no_grad():
                inp = calibration_inputs if not isinstance(calibration_inputs, (list, tuple)) else calibration_inputs[0]
                y_t = teacher_model(inp) if not isinstance(inp, dict) else teacher_model(**inp)
                y_s = student(inp) if not isinstance(inp, dict) else student(**inp)
                if isinstance(y_t, tuple): y_t = y_t[0]
                if isinstance(y_s, tuple): y_s = y_s[0]
                mse_raw = F.mse_loss(y_s.float(), y_t.float()).item()
                cos_raw = F.cosine_similarity(y_s.float().flatten(), y_t.float().flatten(), dim=0).item()
                initial_metrics = {"mse_raw": mse_raw, "cos_sim_raw": cos_raw}
        except Exception as e_metric:
            logger.debug("Raw metric check skipped: %s", e_metric)

    # 2. Closed-Form Ridge Alignment & 3. Micro-Healing
    heal_stats: Dict[str, Any] = {}
    if calibration_inputs is not None:
        heal_stats = heal_replaced_model(
            teacher_model=teacher_model,
            student_model=student,
            calibration_inputs=calibration_inputs,
            steps=heal_steps,
            lr=heal_lr,
            also_ridge_init=True,
        )

    # 4. Optional Persistence
    saved_path = None
    if save_dir is not None:
        from neuralops.models.surgical_io import save_surgical_checkpoint
        saved_path = save_surgical_checkpoint(
            model=student,
            save_dir=save_dir,
            metadata={"target_attention": target_attention, "heal_stats": heal_stats},
        )

    return {
        "model": student,
        "initial_metrics": initial_metrics,
        "heal_stats": heal_stats,
        "saved_path": str(saved_path) if saved_path is not None else None,
    }
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
    return "1.3.7"


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
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }



__all__ = [
    # Model/Export
    "load_model",
    "replace_attention",
    "transplant_attention",
    "compress",
    "export_model",
    "push_to_hub",
    "load_exported_model",
    # Surgery operations
    "collect_surgery_distillation_loss",
    "set_surgery_alpha",
    "set_surgery_stage",
    "finalize_surgery",
    # Discovery & Factories
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
    # Workflows & Pipelines
    "compare_attentions_ablation",
    "surgical_transplant_and_heal_pipeline",
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
    # State Cache & Decoding
    "NeuralOpsStateCache",
    "benchmark_autoregressive_decoding",
    "select_optimal_triton_config",
    # DX Ergonomics
    "oneshot",
    "recommend",
    "summary",
    "compare",
    "patch_transformers",
    "AutoNeuralOpsModel",
    "AutoNeuralOpsForCausalLM",
    "AutoNeuralOpsForSequenceClassification",
    "merge_for_inference",
]

from neuralops.core.state_cache import NeuralOpsStateCache, benchmark_autoregressive_decoding
from neuralops.core.triton_kernels.autotuner import select_optimal_triton_config

from neuralops.recipes import oneshot, recommend
from neuralops.analysis.inspector import summary, compare
from neuralops.transformers import (
    AutoNeuralOpsModel,
    AutoNeuralOpsForCausalLM,
    AutoNeuralOpsForSequenceClassification,
    patch_transformers,
)
from neuralops.export.surgery_manifest import merge_for_inference
