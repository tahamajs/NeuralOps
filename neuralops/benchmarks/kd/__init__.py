"""Knowledge Distillation (KD) benchmarking framework for attention mechanisms.

This package provides high-performance, crash-safe, and resume-capable knowledge
distillation benchmarking across various sequence classification tasks.

Exports
-------
run_kd_benchmark : function
    Primary functional API to execute a KD benchmark run.
BenchmarkConfig : class
    Dataclass for configuring benchmark settings and datasets.
KDRunner : class
    Low-level orchestrator class managing training loops and state.
KDReporter : class
    Streaming reporter for atomic CSV, JSON, and Markdown generation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from neuralops.benchmarks.kd.config import BenchmarkConfig
from neuralops.benchmarks.kd.reporter import KDReporter
from neuralops.benchmarks.kd.runner import KDRunner


def run_kd_benchmark(config: Optional[BenchmarkConfig] = None, **kwargs: Any) -> List[Dict[str, Any]]:
    """Execute knowledge distillation benchmark across attention modules.

    Parameters
    ----------
    config : Optional[BenchmarkConfig], default=None
        Pre-instantiated configuration object.
    **kwargs : Any
        Keyword arguments passed to :class:`BenchmarkConfig` if ``config`` is None.

    Returns
    -------
    list of dict
        Complete benchmark results for each evaluated attention module.

    Examples
    --------
    >>> from neuralops.benchmarks.kd import run_kd_benchmark, BenchmarkConfig
    >>> cfg = BenchmarkConfig(task="glue:sst2", kd_steps=20)
    >>> results = run_kd_benchmark(cfg)
    """
    if config is None:
        config = BenchmarkConfig(**kwargs)
    runner = KDRunner(config)
    return runner.run()


__all__ = [
    "run_kd_benchmark",
    "BenchmarkConfig",
    "KDRunner",
    "KDReporter",
]
