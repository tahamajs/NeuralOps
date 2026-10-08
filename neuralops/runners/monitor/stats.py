"""Aggregated statistical calculation and benchmark record loading for monitor server."""

from __future__ import annotations

import datetime
import glob
import json
import os
import time
from typing import Any

import pandas as pd

from neuralops.runners.monitor.telemetry import get_live_gpu_telemetry

_CACHE_DATA: dict[str, Any] | None = None
_CACHE_TIME: float = 0.0
_ALL_MODULE_RUNS_MAP: dict[str, list[dict[str, Any]]] = {}


def load_harvested_benchmark_records(
    project_dir: str, harvested_json_path: str
) -> list[dict[str, Any]]:
    """Reads all active CSV and historical benchmark records with code-accurate metrics.

    Args:
        project_dir: Base project root.
        harvested_json_path: Path to harvested benchmark data JSON.

    Returns:
        List of benchmark record dictionaries.
    """
    records: list[dict[str, Any]] = []

    local_csvs = glob.glob(
        os.path.join(project_dir, "results", "**", "*summary*.csv"), recursive=True
    )
    local_csvs.sort(key=os.path.getmtime, reverse=True)

    for cf in local_csvs:
        try:
            mtime_dt = datetime.datetime.fromtimestamp(os.path.getmtime(cf)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            df = pd.read_csv(cf)
            df.columns = [str(c).strip().lower() for c in df.columns]
            for _, row in df.iterrows():
                d = row.to_dict()
                d["_source_path"] = cf.replace("\\", "/")
                if not d.get("timestamp") or str(d.get("timestamp")).lower() == "nan":
                    d["timestamp"] = mtime_dt
                else:
                    ts_str = str(d.get("timestamp")).replace("T", " ")[:19]
                    d["timestamp"] = ts_str
                records.append(d)
        except Exception:
            continue

    if os.path.exists(harvested_json_path):
        try:
            with open(harvested_json_path, "r", encoding="utf-8") as f:
                records.extend(json.load(f))
        except Exception as e:
            print(f"Error loading {harvested_json_path}: {e}")

    return records


def get_module_doc_index(docs_dir: str) -> dict[str, str]:
    """Build an index of available markdown documentation across docs_dir.

    Args:
        docs_dir: Directory containing Markdown documentation.

    Returns:
        Mapping of module names/aliases to file paths.
    """
    doc_index: dict[str, str] = {}
    if os.path.exists(docs_dir):
        for md_file in glob.glob(os.path.join(docs_dir, "**", "*.md"), recursive=True):
            base_name = os.path.splitext(os.path.basename(md_file))[0].lower()
            clean_name = (
                base_name.replace("unified_", "")
                .replace("_attention", "")
                .replace("_compressor", "")
                .replace("-", "_")
                .strip()
            )
            doc_index[clean_name] = md_file
            doc_index[base_name] = md_file
    return doc_index


def build_aggregated_stats(
    project_dir: str,
    harvested_json_path: str,
    docs_dir: str,
    is_local_gpu: bool = False,
) -> dict[str, Any]:
    """Calculate aggregated leaderboard, memory profiles, and sweep metrics.

    Args:
        project_dir: Base project root.
        harvested_json_path: Path to harvested json.
        docs_dir: Path to documentation dir.
        is_local_gpu: True if local machine has GPU runner.

    Returns:
        Structured statistics payload.
    """
    global _CACHE_DATA, _CACHE_TIME, _ALL_MODULE_RUNS_MAP
    now = time.time()
    if _CACHE_DATA and (now - _CACHE_TIME < 2.0):
        gpu = get_live_gpu_telemetry(project_dir, is_local_gpu)
        _CACHE_DATA["gpu"] = {
            "gpu_util": gpu["gpu_util"],
            "mem_used_mb": gpu["mem_used_mb"],
            "mem_total_mb": gpu["mem_total_mb"],
            "mem_pct": round((gpu["mem_used_mb"] / gpu["mem_total_mb"]) * 100, 1),
            "power_w": gpu["power_w"],
            "temp_c": gpu["temp_c"],
        }
        _CACHE_DATA["active_module"] = gpu.get("active_module", "hgrn2_gated_deltanet")
        _CACHE_DATA["logs"] = gpu.get("logs", "")
        return _CACHE_DATA

    gpu_data = get_live_gpu_telemetry(project_dir, is_local_gpu)
    raw_records = load_harvested_benchmark_records(project_dir, harvested_json_path)

    module_runs: dict[str, list[dict[str, Any]]] = {}
    pass_cnt = 0
    fail_cnt = 0
    timeout_cnt = 0

    for r in raw_records:
        m_name = str(r.get("module_name", r.get("module", "unknown"))).strip().lower()
        if not m_name or m_name in ("none", "nan"):
            continue
        st = str(r.get("status", "pass")).lower()
        if "pass" in st:
            pass_cnt += 1
        elif "fail" in st:
            fail_cnt += 1
        else:
            timeout_cnt += 1

        if m_name not in module_runs:
            module_runs[m_name] = []
        module_runs[m_name].append(r)

    _ALL_MODULE_RUNS_MAP = module_runs

    leaderboard = []
    for mod, runs in module_runs.items():
        accs = [
            float(x.get("accuracy", x.get("acc", 0.0)))
            for x in runs
            if x.get("accuracy") is not None or x.get("acc") is not None
        ]
        avg_acc = round(sum(accs) / max(1, len(accs)), 2) if accs else 0.0
        leaderboard.append({"module": mod, "avg_accuracy": avg_acc, "total_runs": len(runs)})

    leaderboard.sort(key=lambda x: x["avg_accuracy"], reverse=True)

    _CACHE_DATA = {
        "is_running": True,
        "last_updated": time.strftime("%H:%M:%S"),
        "gpu": {
            "name": gpu_data.get("gpu_name", "NVIDIA GeForce RTX 4070 (8GB)"),
            "gpu_name": gpu_data.get("gpu_name", "NVIDIA GeForce RTX 4070 (8GB)"),
            "gpu_util": gpu_data["gpu_util"],
            "mem_used_mb": gpu_data["mem_used_mb"],
            "mem_total_mb": gpu_data["mem_total_mb"],
            "mem_pct": round((gpu_data["mem_used_mb"] / gpu_data["mem_total_mb"]) * 100, 1),
            "power_w": gpu_data["power_w"],
            "temp_c": gpu_data["temp_c"],
        },
        "counts": {"pass": pass_cnt, "fail": fail_cnt, "timeout": timeout_cnt},
        "analytics": {"modules": leaderboard},
        "results": raw_records[:100],
        "active_module": gpu_data.get("active_module", "idle"),
        "logs": gpu_data.get("logs", ""),
    }
    _CACHE_TIME = now
    return _CACHE_DATA


__all__ = [
    "load_harvested_benchmark_records",
    "get_module_doc_index",
    "build_aggregated_stats",
]
