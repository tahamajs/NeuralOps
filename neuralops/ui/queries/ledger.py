#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""SQLite query operations for master ledger and system status."""

from __future__ import annotations

import csv
import io
import os
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import neuralops
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY


def get_db_path() -> Path:
    """Resolve the path to the master SQLite ledger (F-59).

    Precedence:
      1. Environment variable NEURALOPS_LEDGER_PATH
      2. Local working directory results: `Path.cwd() / 'results' / 'master_ledger.sqlite'`
      3. User cache directory: `~/.cache/neuralops/master_ledger.sqlite`

    Returns:
        Path to master_ledger.sqlite

    Raises:
        FileNotFoundError: If the ledger file does not exist.
    """
    env_path = os.environ.get("NEURALOPS_LEDGER_PATH")
    if env_path:
        db_path = Path(env_path).expanduser().resolve()
    else:
        cwd_path = Path.cwd() / "results" / "master_ledger.sqlite"
        if cwd_path.is_file():
            db_path = cwd_path
        else:
            db_path = Path.home() / ".cache" / "neuralops" / "master_ledger.sqlite"

    if not db_path.is_file():
        raise FileNotFoundError(
            f"Master ledger SQLite database not found at: {db_path}. "
            "Configure NEURALOPS_LEDGER_PATH or generate results using `neuralops benchmark`."
        )
    return db_path


def _get_connection() -> sqlite3.Connection:
    """Open a read-only connection with dictionary row access."""
    db_path = get_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def list_tables() -> list[str]:
    """Return all table names present in the SQLite master ledger."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name ASC")
        return [row["name"] for row in cursor.fetchall()]
    finally:
        conn.close()


def get_doctor_status() -> dict[str, Any]:
    """Collect repository health and invariant metrics."""
    pkg_root = Path(neuralops.__file__).resolve().parent
    repo_root = pkg_root.parent

    # Invariant counts matching orchestrator Gate D rules (maxdepth 1)
    triton_dir = pkg_root / "ops" / "triton"
    cuda_dir = pkg_root / "ops" / "cuda"
    notebooks_dir = repo_root / "notebooks"

    triton_count = len([p for p in triton_dir.glob("*.py") if p.name != "__init__.py"])
    cuda_count = len([p for p in cuda_dir.iterdir() if p.is_dir()])
    notebook_count = len([p for p in notebooks_dir.glob("*.ipynb")])

    # Git status
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        branch = "unknown"

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = "unknown"

    try:
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        git_dirty = len(porcelain) > 0
    except Exception:
        git_dirty = False

    tables = list_tables()

    return {
        "version": neuralops.__version__,
        "attentions": len(ATTENTION_REGISTRY),
        "compressors": len(COMPRESSOR_REGISTRY),
        "triton_modules": triton_count,
        "cuda_packages": cuda_count,
        "notebooks": notebook_count,
        "sqlite_tables": len(tables),
        "git_branch": branch,
        "git_commit": commit,
        "git_dirty": git_dirty,
    }


def get_recent_runs(limit: int = 5) -> list[dict[str, Any]]:
    """Retrieve the most recent benchmark run records."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT module_name, status, latency_triton_ms, peak_vram_mb, timestamp_utc
            FROM results
            ORDER BY record_id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


_ALLOWED_FILTER_COLUMNS = {
    "module_name",
    "status",
    "min_latency",
    "max_latency",
    "family",
}


def get_results(
    filters: dict[str, Any] | None = None, limit: int = 50, offset: int = 0
) -> tuple[list[dict[str, Any]], int]:
    """Retrieve benchmark results with whitelisted parameterized filters and pagination."""
    filters = filters or {}
    where_clauses: list[str] = []
    params: list[Any] = []

    for key, value in filters.items():
        if key not in _ALLOWED_FILTER_COLUMNS or value is None:
            continue
        if key == "module_name":
            where_clauses.append("module_name = ?")
            params.append(str(value))
        elif key == "status":
            where_clauses.append("status = ?")
            params.append(str(value))
        elif key == "family":
            where_clauses.append("family = ?")
            params.append(str(value))
        elif key == "min_latency":
            where_clauses.append("latency_triton_ms >= ?")
            params.append(float(value))
        elif key == "max_latency":
            where_clauses.append("latency_triton_ms <= ?")
            params.append(float(value))

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    count_query = f"SELECT COUNT(*) AS total FROM results {where_sql}"
    data_query = f"""
        SELECT record_id, module_name, category, family, status, timestamp_utc,
               latency_triton_ms, latency_pytorch_ms, speedup_x, peak_vram_mb,
               throughput_tok_s, cosine_sim
        FROM results
        {where_sql}
        ORDER BY record_id DESC
        LIMIT ? OFFSET ?
    """

    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(count_query, params)
        total_row = cursor.fetchone()
        total_count = total_row["total"] if total_row else 0

        cursor.execute(data_query, params + [limit, offset])
        rows = [dict(row) for row in cursor.fetchall()]
        return rows, total_count
    finally:
        conn.close()


def get_attention_results(name: str) -> list[dict[str, Any]]:
    """Retrieve all execution results for a given attention mechanism."""
    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT record_id, module_name, category, family, status, timestamp_utc,
                   latency_triton_ms, latency_pytorch_ms, speedup_x, peak_vram_mb,
                   throughput_tok_s, cosine_sim
            FROM results
            WHERE module_name = ?
            ORDER BY record_id DESC
            """,
            (name,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_compressor_results(name: str) -> list[dict[str, Any]]:
    """Retrieve all execution results for a given compressor."""
    return get_attention_results(name)


def get_benchmark_comparison(module_names: list[str]) -> list[dict[str, Any]]:
    """Compute aggregated performance metrics for a list of module names."""
    if not module_names:
        return []

    placeholders = ",".join("?" for _ in module_names)
    query = f"""
        SELECT module_name,
               AVG(speedup_x) AS mean_speedup,
               MAX(speedup_x) AS max_speedup,
               AVG(latency_triton_ms) AS mean_latency,
               COUNT(*) AS sample_count
        FROM results
        WHERE module_name IN ({placeholders})
        GROUP BY module_name
        ORDER BY module_name ASC
    """

    conn = _get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(query, module_names)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def export_results_csv(filters: dict[str, Any] | None = None) -> str:
    """Export filtered benchmark results as a CSV string."""
    rows, _ = get_results(filters=filters, limit=100000, offset=0)
    if not rows:
        return ""

    output = io.StringIO()
    headers = list(rows[0].keys())
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()
