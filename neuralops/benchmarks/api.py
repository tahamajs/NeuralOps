"""High-level Python API for running and orchestrating benchmarks."""
from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pandas as pd

from neuralops.benchmarks.config import BenchmarkConfig
from neuralops.benchmarks.executor import execute_single
from neuralops.benchmarks.orchestrator import BenchmarkOrchestrator
from neuralops.benchmarks.result import BenchmarkResult


def benchmark(
    module: str | None = None,
    model: str = "distilbert-base-uncased",
    dataset: str = "sst2",
    rank: int | None = 64,
    device: str | None = None,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    seq_len: int = 128,
    eval_only: bool = False,
    timeout_s: int = 1800,
    **kwargs: Any,
) -> BenchmarkResult:
    """Run a single benchmark evaluation for a module on model & dataset."""
    return execute_single(
        module=module,
        model=model,
        dataset=dataset,
        rank=rank,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        max_length=seq_len,
        eval_only=eval_only,
        timeout_s=timeout_s,
        **kwargs,
    )


class Runner:
    """Orchestrates multi-module sweeps, grids, and resumable executions."""

    def __init__(
        self,
        modules: str | list[str] = "all",
        models: str | list[str] = "distilbert-base-uncased",
        datasets: str | list[str] = "sst2",
        ranks: int | list[int] = 64,
        parallel: int = 1,
        output_dir: Path | str = "results",
        resume: bool = True,
        epochs: int = 3,
        batch_size: int = 32,
        lr: float = 2e-5,
        seq_len: int = 128,
        device: str = "cuda",
        timeout_s: int = 1800,
        **kwargs: Any,
    ) -> None:
        shortcut_keys = {
            "delta", "all_delta", "gated", "all_gated", "svd", "all_svd",
            "ssm", "all_ssm", "linear", "all_linear", "sparse", "all_sparse",
            "hybrid", "all_hybrid", "all_attn", "all_attention", "all_comp",
            "all_compressors", "all_modules", "fail_delta", "errors_only",
        }
        config_kwargs = {k: v for k, v in kwargs.items() if k not in shortcut_keys}
        self.config = BenchmarkConfig(
            model=models[0] if isinstance(models, list) and models else (models if isinstance(models, str) else "distilbert-base-uncased"),
            dataset=datasets[0] if isinstance(datasets, list) and datasets else (datasets if isinstance(datasets, str) else "sst2"),
            output_dir=Path(output_dir),
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seq_len=seq_len,
            max_length=seq_len,
            timeout_s=timeout_s,
            **config_kwargs,
        )
        self.orchestrator = BenchmarkOrchestrator(
            config=self.config,
            output_dir=output_dir,
            modules=modules,
            models=models,
            datasets=datasets,
            ranks=ranks,
            parallel=parallel,
            resume=resume,
            **kwargs,
        )

    def run(self) -> list[BenchmarkResult]:
        """Run all benchmark tasks in the grid."""
        return self.orchestrator.run()

    def run_async(self) -> None:
        """Launch asynchronous background run."""
        self.orchestrator.run_async()

    def status(self) -> dict[str, Any]:
        """Get live execution progress status."""
        return self.orchestrator.status()

    def stream(self) -> Generator[BenchmarkResult, None, None]:
        """Stream benchmark results as they finish."""
        yield from self.orchestrator.stream()

    def summary(self) -> pd.DataFrame:
        """Get aggregated benchmark results as DataFrame."""
        return self.orchestrator.summary()

    def resume(self) -> list[BenchmarkResult]:
        """Resume execution."""
        return self.orchestrator.resume()


__all__ = [
    "benchmark",
    "Runner",
]
