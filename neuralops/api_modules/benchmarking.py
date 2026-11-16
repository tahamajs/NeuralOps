# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Benchmarking, ablation, validation, system inspection, and profiling:
compare_attentions_ablation, benchmark, benchmark_all, get_version, has_triton,
available_backends, validate, certify, info, quick_benchmark.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

logger = logging.getLogger("NeuralOps.API")


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
    from neuralops.core.paper_export import compute_pareto_frontier, export_benchmark_table
    from neuralops.core.roofline import profile_roofline

    target_dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
        except (RuntimeError, ValueError, KeyError, AttributeError) as e:
            logger.warning("Failed to profile '%s' in ablation study: %s", name, e, exc_info=True)

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
    seed: Optional[int] = 42,
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
        seed: Random seed for reproducible benchmarking (default: 42).
        **kwargs: Additional benchmark options.

    Returns:
        BenchmarkResult object.
    """
    from neuralops.core.utils.general import set_seed

    resolved_seed = 42 if seed is None else seed
    set_seed(resolved_seed)
    logger.info("Executing benchmark for module='%s' on model='%s' (dataset='%s')", module, model, dataset)
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
        seed=resolved_seed,
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


def get_version() -> str:
    """Return the current NeuralOps release version."""
    import neuralops

    if not hasattr(neuralops, "__version__"):
        raise RuntimeError("neuralops.__version__ is not defined")
    return str(neuralops.__version__)


def has_triton() -> bool:
    """Check whether Triton kernel compiler and GPU backend are available."""
    try:
        from neuralops.core.triton_kernels import HAS_TRITON

        return bool(HAS_TRITON)
    except (ImportError, AttributeError):
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


def certify(
    model: nn.Module,
    strict: bool = True,
) -> Dict[str, Any]:
    """Certify that a model satisfies Universal Surgery invariants.

    Executes all 10 invariant certification laws (QK-norm preservation, logit
    soft-capping temperature folding, padding mask sanitization, RoPE alignment, etc.)
    via UniversalSurgeryDirector.

    Args:
        model: PyTorch model or surgeried transformer model to certify.
        strict: Whether to raise InvariantViolationError if any mandatory invariant fails.

    Returns:
        Dict[str, Any]: Audit report containing structured per-law verdicts.
    """
    if model is None:
        raise ValueError("Must provide a model instance to certify.")
    from neuralops.surgery.director import UniversalSurgeryDirector

    director = UniversalSurgeryDirector(model)
    return director.certify_model(strict=strict)


def info() -> Dict[str, Any]:
    """Return diagnostic overview of NeuralOps installation and system capabilities."""
    from neuralops.api_modules.registry_ops import _ensure_registries_loaded

    _ensure_registries_loaded()
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


def quick_benchmark(
    model: nn.Module,
    task: str = "synthetic_sst2",
    device: Optional[Union[str, torch.device]] = None,
    iterations: int = 5,
    **kwargs: Any,
) -> Any:
    """Quickly benchmark a model on sample inputs and return a latency summary DataFrame."""
    import time

    import pandas as pd

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(dev, str):
        dev = torch.device(dev)
    model = model.to(dev).eval()

    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if vocab_size is None and hasattr(model, "vocab_size"):
        vocab_size = model.vocab_size
    if vocab_size is None:
        for m in model.modules():
            if isinstance(m, nn.Embedding):
                vocab_size = m.num_embeddings
                break
    if vocab_size is None or vocab_size <= 0:
        vocab_size = 1000

    sample = torch.randint(0, vocab_size, (2, 64), device=dev)
    with torch.no_grad():
        for _ in range(2):
            try:
                _ = model(sample)
            except TypeError:
                _ = model(input_ids=sample)
        if dev.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            for _ in range(max(1, iterations)):
                try:
                    out = model(sample)
                except TypeError:
                    out = model(input_ids=sample)
            end_event.record()
            torch.cuda.synchronize()
            latency_ms = start_event.elapsed_time(end_event) / max(1, iterations)
        else:
            t0 = time.perf_counter()
            for _ in range(max(1, iterations)):
                try:
                    out = model(sample)
                except TypeError:
                    out = model(input_ids=sample)
            t1 = time.perf_counter()
            latency_ms = ((t1 - t0) / max(1, iterations)) * 1000.0
    is_finite = True
    if hasattr(out, "logits"):
        is_finite = bool(torch.isfinite(out.logits).all())
    elif isinstance(out, torch.Tensor):
        is_finite = bool(torch.isfinite(out).all())

    return pd.DataFrame([{"task": task, "device": dev.type, "latency_ms": latency_ms, "finite": is_finite}])
