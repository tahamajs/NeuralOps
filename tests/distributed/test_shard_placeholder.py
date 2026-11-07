#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests verifying removal of placeholder shard_model_tensor_parallel from public API (F-62).
"""

from __future__ import annotations

import neuralops.distributed as dist_pkg
from neuralops.distributed.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    _shard_model_tensor_parallel,
)


def test_shard_not_exported() -> None:
    """Verify shard_model_tensor_parallel is NOT exported in distributed.__all__."""
    assert "shard_model_tensor_parallel" not in dist_pkg.__all__

    # Function is retained privately with explicit warning/unsupported docstring
    assert hasattr(_shard_model_tensor_parallel, "__doc__")
    assert "unsupported" in (_shard_model_tensor_parallel.__doc__ or "").lower()

    # Importing from distributed package directly raises ImportError
    import pytest

    with pytest.raises(ImportError):
        from neuralops.distributed import shard_model_tensor_parallel  # type: ignore[attr-defined] # noqa: F401


def test_column_row_parallel_real() -> None:
    """Verify real ColumnParallelLinear and RowParallelLinear layers are constructible."""
    col = ColumnParallelLinear(in_features=64, out_features=128, tp_size=2, tp_rank=0)
    assert col.in_features == 64
    assert col.out_features == 128
    assert col.local_out_features == 64

    row = RowParallelLinear(in_features=128, out_features=64, tp_size=2, tp_rank=0)
    assert row.in_features == 128
    assert row.out_features == 64
    assert row.local_in_features == 64
