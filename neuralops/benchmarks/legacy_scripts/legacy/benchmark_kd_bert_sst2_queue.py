"""Deprecation forwarding shim for legacy/benchmark_kd_bert_sst2_queue."""

from __future__ import annotations

import warnings

warnings.warn(
    "neuralops.benchmarks.legacy_scripts.legacy.benchmark_kd_bert_sst2_queue is deprecated and has been moved to "
    "neuralops.benchmarks._legacy.legacy.benchmark_kd_bert_sst2_queue. Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.benchmarks._legacy.legacy.benchmark_kd_bert_sst2_queue import *  # noqa: F403, F401
