#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Benchmarks Knowledge Distillation Subpackage.
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
