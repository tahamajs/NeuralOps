#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Backward-compatibility adapter for Offline Teacher Representation Caching.

Re-exports canonical utilities from `neuralops.benchmarks.kd.teacher_cache`.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    _cache_file_lock,
    _infer_num_layers,
    _is_valid_causal_cache,
    _is_valid_vlm_cache,
    _safe_torch_load,
)

__all__ = [
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
    "_cache_file_lock",
    "_infer_num_layers",
    "_is_valid_causal_cache",
    "_is_valid_vlm_cache",
    "_safe_torch_load",
]
