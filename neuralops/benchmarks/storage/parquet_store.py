#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Idempotent Parquet storage store for benchmark CanonicalRecords."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from neuralops.benchmarks.storage.schema import CanonicalRecord

LOG = logging.getLogger(__name__)


def record_to_pyarrow_dict(record: CanonicalRecord) -> Dict[str, Any]:
    """Convert CanonicalRecord to flat dictionary suitable for pyarrow."""
    import json

    return {
        "run_id": str(record.run_id),
        "timestamp": str(record.timestamp),
        "suite": str(record.suite),
        "module": str(record.module),
        "model": str(record.model),
        "device": str(record.device),
        "status": str(record.status),
        "metrics": json.dumps(record.metrics, sort_keys=True),
        "config": json.dumps(record.config, sort_keys=True),
        "telemetry": json.dumps(record.telemetry, sort_keys=True),
        "git_commit": str(record.git_commit),
        "git_dirty": bool(record.git_dirty),
        "checksum": str(record.checksum),
    }


class ParquetLedgerStore:
    """Stores benchmark canonical records in Parquet datasets with deduplication."""

    def __init__(
        self,
        base_dir: Path | str = "results/ledger",
        partition_by: Optional[Sequence[str]] = None,
        compression: str = "snappy",
    ) -> None:
        self.base_dir = Path(base_dir)
        self.partition_by = list(partition_by) if partition_by else ["suite"]
        self.compression = compression
        self._lock = threading.Lock()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_records(self, records: Sequence[CanonicalRecord]) -> List[CanonicalRecord]:
        """Write canonical records deduplicated by run_id in the current batch."""
        if not records:
            return []

        import pyarrow as pa
        import pyarrow.parquet as pq

        seen = set()
        unique: List[CanonicalRecord] = []
        for r in records:
            if r.run_id in seen:
                LOG.warning("Duplicate run_id in batch: %s", r.run_id)
                continue
            seen.add(r.run_id)
            unique.append(r)

        schema = pa.schema(
            [
                ("run_id", pa.string()),
                ("timestamp", pa.string()),
                ("suite", pa.string()),
                ("module", pa.string()),
                ("model", pa.string()),
                ("device", pa.string()),
                ("status", pa.string()),
                ("metrics", pa.string()),
                ("config", pa.string()),
                ("telemetry", pa.string()),
                ("git_commit", pa.string()),
                ("git_dirty", pa.bool_()),
                ("checksum", pa.string()),
            ]
        )

        rows = [record_to_pyarrow_dict(r) for r in unique]
        table = pa.Table.from_pylist(rows, schema=schema)

        with self._lock:
            pq.write_to_dataset(
                table,
                root_path=str(self.base_dir),
                partition_cols=self.partition_by,
                compression=self.compression,
                use_dictionary=True,
                existing_data_behavior="overwrite_or_ignore",
            )

        return unique
