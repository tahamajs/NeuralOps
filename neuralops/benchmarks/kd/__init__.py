#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Benchmarks Knowledge Distillation Subpackage.

Exports:
  - BenchmarkConfig, KDRunner, KDReporter, run_kd_benchmark
  - Offline teacher caching (teacher_cache)
  - Two-stage distillation engine (distill_engine)
  - Multi-module zoo execution orchestrator (zoo_orchestrator)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from neuralops.benchmarks.kd.config import BenchmarkConfig
from neuralops.benchmarks.kd.distill_engine import (
    TwoStageDistillationEngine,
    compute_topk_sparse_kd_loss,
    evaluate_causal_ppl,
    evaluate_sst2_accuracy,
    execute_with_oom_retry,
    fast_gram_covariance_loss,
)
from neuralops.benchmarks.kd.reporter import KDReporter
from neuralops.benchmarks.kd.runner import KDRunner
from neuralops.benchmarks.kd.teacher_cache import (
    HPC_DATA_CACHE_ROOT,
    VLM_MAX_CACHE_BYTES,
    CachedCausalDataset,
    atomic_torch_save,
    build_or_load_causal_teacher_cache,
    build_or_load_vlm_teacher_cache,
    compute_topk_sparse_kd_loss_from_cache,
    enable_student_gradient_checkpointing,
    has_sufficient_space,
    resolve_teacher_cache_path,
    select_strided_layer_indices,
    select_vlm_layer_indices,
    try_load_causal_teacher_cache,
    try_load_vlm_teacher_cache,
)
from neuralops.benchmarks.kd.zoo_orchestrator import (
    CausalTextDataset,
    StratifiedQwenDataset,
    ZooOrchestrator,
    apply_qwen_attention_surgery,
    build_cache_wikitext,
    build_llama_readme,
    build_qwen_readme,
    build_vlm_readme,
    collect_artifact_sizes,
    extract_attention_metadata,
    format_size,
    load_tokenized_qwen_corpus,
    normalize_base_model_id,
    summarize_attention_doc,
)


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
    """
    if config is None:
        config = BenchmarkConfig(**kwargs)
    runner = KDRunner(config)
    return runner.run()


__all__ = [
    # Core benchmark
    "run_kd_benchmark",
    "BenchmarkConfig",
    "KDRunner",
    "KDReporter",
    # Teacher caching
    "HPC_DATA_CACHE_ROOT",
    "VLM_MAX_CACHE_BYTES",
    "atomic_torch_save",
    "resolve_teacher_cache_path",
    "has_sufficient_space",
    "select_strided_layer_indices",
    "select_vlm_layer_indices",
    "enable_student_gradient_checkpointing",
    "compute_topk_sparse_kd_loss_from_cache",
    "CachedCausalDataset",
    "try_load_causal_teacher_cache",
    "build_or_load_causal_teacher_cache",
    "try_load_vlm_teacher_cache",
    "build_or_load_vlm_teacher_cache",
    # Distillation engine
    "TwoStageDistillationEngine",
    "fast_gram_covariance_loss",
    "compute_topk_sparse_kd_loss",
    "evaluate_causal_ppl",
    "evaluate_sst2_accuracy",
    "execute_with_oom_retry",
    # Zoo orchestrator
    "StratifiedQwenDataset",
    "CausalTextDataset",
    "load_tokenized_qwen_corpus",
    "build_cache_wikitext",
    "apply_qwen_attention_surgery",
    "normalize_base_model_id",
    "summarize_attention_doc",
    "format_size",
    "extract_attention_metadata",
    "collect_artifact_sizes",
    "build_qwen_readme",
    "build_llama_readme",
    "build_vlm_readme",
    "ZooOrchestrator",
]
