#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Columnar Parquet Serialization Backend

Provides high-throughput Apache Parquet serialization for benchmark telemetry
and granular step iterations with fallback safety.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def is_pyarrow_available() -> bool:
    """Check if pyarrow is installed and functional."""
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401

        return True
    except ImportError:
        return False


def write_metrics_parquet(
    records: List[Dict[str, Any]],
    target_path: Path | str,
) -> Optional[Path]:
    """Write metric records to a Parquet file atomically using pyarrow.

    Args:
        records: List of iteration metric records.
        target_path: Path where the parquet file should be written.

    Returns:
        Path to the written parquet file, or None if pyarrow is unavailable.
    """
    if not records:
        return None

    if not is_pyarrow_available():
        logger.warning("pyarrow is not installed; skipping Parquet export.")
        return None

    import pyarrow as pa
    import pyarrow.parquet as pq

    dest = Path(target_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(f".tmp.{os.getpid()}.parquet")

    try:
        # Normalize records into uniform column dictionary
        keys = list({k for r in records for k in r.keys()})
        col_data: Dict[str, List[Any]] = {k: [] for k in keys}
        for r in records:
            for k in keys:
                val = r.get(k)
                if isinstance(val, (dict, list)):
                    import json

                    val = json.dumps(val)
                col_data[k].append(val)

        table = pa.Table.from_pydict(col_data)
        pq.write_table(table, str(temp_dest), compression="snappy")
        os.replace(temp_dest, dest)
        return dest
    except Exception as exc:
        logger.warning("Failed writing Parquet metrics to %s: %s", dest, exc)
        return None
    finally:
        if temp_dest.exists():
            try:
                temp_dest.unlink()
            except OSError:
                pass


def read_metrics_parquet(parquet_path: Path | str) -> List[Dict[str, Any]]:
    """Read metric records from a Parquet file.

    Args:
        parquet_path: Path to the parquet file.

    Returns:
        List of dictionaries containing row records.
    """
    if not is_pyarrow_available():
        raise RuntimeError("pyarrow is required to read Parquet files.")

    import pyarrow.parquet as pq

    table = pq.read_table(str(parquet_path))
    return table.to_pylist()
