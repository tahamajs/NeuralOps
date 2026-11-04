#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Legacy Results Migration Utility

Scans un-indexed historical CSV and JSON results files in results/
and organizes them into hermetic per-run directories:
results/benchmarks/<suite>/runs/<legacy_run_id>/
Synchronizes metadata and metrics into the SQLite master ledger.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from neuralops.benchmarks.core.paths import RunDirectoryContext, StoragePathManager
from neuralops.results_saver.engine import DualCommitPersistenceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def infer_suite_name(filename: str) -> str:
    """Infer the canonical benchmark suite from a legacy filename."""
    lower = filename.lower()
    if "triton" in lower:
        return "triton_ops"
    if "cuda" in lower:
        return "cuda_ops"
    if "kd" in lower or "compression" in lower or "ablation" in lower or "qwen" in lower or "sst2" in lower:
        return "compression"
    if "attention" in lower or "attn" in lower:
        return "attention"
    if "serving" in lower or "vllm" in lower:
        return "e2e_serving"
    if "needle" in lower or "niah" in lower:
        return "niah"
    return "legacy_benchmarks"


def parse_legacy_file(file_path: Path) -> List[Dict[str, Any]]:
    """Parse CSV or JSON legacy benchmark output into standardized iteration dictionaries."""
    records: List[Dict[str, Any]] = []
    if file_path.suffix.lower() == ".csv":
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Clean empty/none values
                    cleaned = {k.strip(): v.strip() for k, v in row.items() if k}
                    # Map standard latency/memory fields if present
                    if "latency_ms" in cleaned:
                        try:
                            cleaned["latency_ms"] = float(cleaned["latency_ms"])
                        except ValueError:
                            pass
                    if "vram_mb" in cleaned:
                        try:
                            cleaned["vram_mb"] = float(cleaned["vram_mb"])
                        except ValueError:
                            pass
                    records.append(cleaned)
        except Exception as exc:
            logger.warning("Could not parse CSV %s: %s", file_path, exc)
    elif file_path.suffix.lower() == ".json":
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
                if isinstance(data, list):
                    records = [r for r in data if isinstance(r, dict)]
                elif isinstance(data, dict):
                    if "results" in data and isinstance(data["results"], list):
                        records = [r for r in data["results"] if isinstance(r, dict)]
                    elif "iterations" in data and isinstance(data["iterations"], list):
                        records = [r for r in data["iterations"] if isinstance(r, dict)]
                    else:
                        records = [data]
        except Exception as exc:
            logger.warning("Could not parse JSON %s: %s", file_path, exc)

    return records


def migrate_legacy_results(
    results_dir: Path | str = "results",
    dest_root: Path | str = "results/benchmarks",
    ledger_path: Path | str = "results/master_ledger.sqlite",
    dry_run: bool = False,
) -> List[Tuple[str, str, int]]:
    """Migrate legacy loose result files into hermetic run folders and master ledger."""
    src = Path(results_dir)
    dest = Path(dest_root)
    engine = DualCommitPersistenceEngine(ledger_path=ledger_path)

    migrated_runs: List[Tuple[str, str, int]] = []
    target_suffixes = {".csv", ".json"}

    for item in src.iterdir():
        if not item.is_file() or item.suffix.lower() not in target_suffixes:
            continue
        if item.name.startswith(".") or "ledger" in item.name or "summary" in item.name.lower():
            continue

        suite_name = infer_suite_name(item.name)
        mtime = int(item.stat().st_mtime)
        mtime_str = datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.sha256(item.name.encode("utf-8")).hexdigest()[:8]
        run_id = f"legacy_{mtime_str}_{name_hash}"

        records = parse_legacy_file(item)
        if not records:
            continue

        logger.info(
            "Found legacy candidate: %s -> suite: %s, run_id: %s (%d records)",
            item.name,
            suite_name,
            run_id,
            len(records),
        )

        if dry_run:
            migrated_runs.append((suite_name, run_id, len(records)))
            continue

        context: RunDirectoryContext = StoragePathManager.resolve_run_dir(
            benchmark_name=suite_name,
            run_id=run_id,
            output_root=dest,
        )

        manifest = {
            "run_id": run_id,
            "benchmark_name": suite_name,
            "source_legacy_file": item.name,
            "timestamp_start_utc": datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc).isoformat(),
            "timestamp_end_utc": datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc).isoformat(),
            "git_provenance": {"commit_sha": "legacy_historical", "branch": "main", "is_dirty": False},
            "device_name": "Historical Benchmark Device",
            "hostname": "legacy_cluster_runner",
            "status": "PASSED",
        }

        # Normalize records for persistence
        normalized_records = []
        for i, r in enumerate(records):
            entry = dict(r)
            entry["step"] = i + 1
            if "mechanism_or_kernel" not in entry:
                entry["mechanism_or_kernel"] = entry.get("module") or entry.get("kernel") or entry.get("name") or "legacy_kernel"
            normalized_records.append(entry)

        engine.finalize_run(
            context=context,
            manifest=manifest,
            iterations=normalized_records,
            duration_seconds=0.0,
            enable_parquet=True,
        )

        migrated_runs.append((suite_name, run_id, len(records)))

    return migrated_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy results into isolated per-run benchmark directories.")
    parser.add_argument("--results-dir", default="results", help="Directory containing loose legacy files.")
    parser.add_argument("--dest-dir", default="results/benchmarks", help="Root directory for unified benchmark runs.")
    parser.add_argument("--db-path", default="results/master_ledger.sqlite", help="Master SQLite ledger database path.")
    parser.add_argument("--dry-run", action="store_true", help="Simulate migration without modifying disk or database.")
    args = parser.parse_args()

    results = migrate_legacy_results(
        results_dir=args.results_dir,
        dest_root=args.dest_dir,
        ledger_path=args.db_path,
        dry_run=args.dry_run,
    )

    logger.info("Migration complete: %d legacy benchmark runs cataloged.", len(results))


if __name__ == "__main__":
    main()
