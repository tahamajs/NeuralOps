"""Multi-format saver (JSON, JSONL, CSV, Markdown, Parquet) for benchmark results."""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid
import pandas as pd

from neuralops.benchmarks.results.schema import BenchmarkResult

LOG = logging.getLogger(__name__)

CATEGORY_MAP = {
    "unified": "01_unified_runs",
    "sweeps": "02_rank_sweeps",
    "zeroshot": "03_zeroshot_evals",
    "baseline": "04_baselines",
    "glue": "05_glue_finetuning",
    "adhoc": "06_adhoc_experiments",
    "misc": "07_misc_tests",
}


def resolve_category(module_type: str, dataset_name: str, task: str = "") -> str:
    """Determine structured results directory category."""
    ds = dataset_name.lower()
    if module_type == "baseline" or "baseline" in ds:
        return "04_baselines"
    if any(k in ds for k in ("arc", "hellaswag", "mmlu", "gsm8k")):
        return "03_zeroshot_evals"
    if any(k in ds for k in ("sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "cola", "stsb", "wnli")):
        return "05_glue_finetuning"
    return "01_unified_runs"


def compute_result_path(result: BenchmarkResult, base_dir: Path | str = "results") -> Path:
    """
    Compute structured canonical path:
    results/<category>/<module>/<model>/<dataset>/rank_<rank>/<run_id>.json
    """
    mod_name = result.module.get("name", "baseline") if isinstance(result.module, dict) else str(result.module)
    mod_type = result.module.get("type", "attention") if isinstance(result.module, dict) else "attention"
    mdl_name = result.model.get("name", "distilbert-base-uncased") if isinstance(result.model, dict) else str(result.model)
    # Sanitize model name if it contains slashes (e.g. meta-llama/Llama-3.2-1B)
    mdl_safe = mdl_name.replace("/", "_")
    ds_name = result.dataset.get("name", "sst2") if isinstance(result.dataset, dict) else str(result.dataset)
    rk = result.config.get("rank", 0) if isinstance(result.config, dict) else 0

    cat = resolve_category(mod_type, ds_name)
    target_dir = Path(base_dir) / cat / mod_name / mdl_safe / ds_name / f"rank_{rk}"
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{result.run_id}.json"


def save_json(result: BenchmarkResult, target_path: Path | str) -> Path:
    """Save canonical pretty-printed self-contained JSON with atomic write."""
    p = Path(target_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = p.with_suffix(f".tmp.{os.getpid()}_{uuid.uuid4().hex[:6]}")
    with open(tmp_path, "w", encoding="utf-8") as fp:
        json.dump(result.to_dict(), fp, indent=2, default=str)
    tmp_path.replace(p)
    return p


def save_jsonl_index(result: BenchmarkResult, index_path: Path | str) -> Path:
    """Append key summary row to rolling JSONL index concurrently and safely."""
    p = Path(index_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    summary_entry = {
        "run_id": result.run_id,
        "run_key": result.compute_run_key(),
        "timestamp": result.timestamp_end or result.timestamp_start,
        "status": result.status,
        "module": result.module.get("name", "") if isinstance(result.module, dict) else str(result.module),
        "module_type": result.module.get("type", "") if isinstance(result.module, dict) else "",
        "model": result.model.get("name", "") if isinstance(result.model, dict) else str(result.model),
        "dataset": result.dataset.get("name", "") if isinstance(result.dataset, dict) else str(result.dataset),
        "rank": result.config.get("rank", 0) if isinstance(result.config, dict) else 0,
        "training_type": getattr(result, "training_type", "finetune"),
        "accuracy": (result.metrics or {}).get("accuracy"),
        "f1": (result.metrics or {}).get("f1"),
        "loss": (result.metrics or {}).get("loss"),
        "perplexity": (result.metrics or {}).get("post_ppl") or (result.metrics or {}).get("perplexity"),
        "kd_gain": (result.metrics or {}).get("kd_gain"),
        "recovery_pct": (result.metrics or {}).get("recovery_pct"),
        "duration_s": (result.resources or {}).get("duration_s", 0.0),
        "peak_vram_mb": (result.resources or {}).get("peak_vram_mb", 0.0),
        "checksum": getattr(result, "checksum", ""),
    }
    with open(p, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(summary_entry) + "\n")
    return p


def save_csv_index(result: BenchmarkResult, csv_path: Path | str) -> Path:
    """Append key summary row to rolling CSV index."""
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists() or p.stat().st_size == 0

    row = {
        "run_id": result.run_id,
        "run_key": result.compute_run_key(),
        "timestamp": result.timestamp_end or result.timestamp_start,
        "status": result.status,
        "module": result.module.get("name", "") if isinstance(result.module, dict) else str(result.module),
        "module_type": result.module.get("type", "") if isinstance(result.module, dict) else "",
        "model": result.model.get("name", "") if isinstance(result.model, dict) else str(result.model),
        "dataset": result.dataset.get("name", "") if isinstance(result.dataset, dict) else str(result.dataset),
        "rank": result.config.get("rank", 0) if isinstance(result.config, dict) else 0,
        "training_type": getattr(result, "training_type", "finetune"),
        "accuracy": (result.metrics or {}).get("accuracy", ""),
        "f1": (result.metrics or {}).get("f1", ""),
        "loss": (result.metrics or {}).get("loss", ""),
        "perplexity": (result.metrics or {}).get("post_ppl") or (result.metrics or {}).get("perplexity", ""),
        "kd_gain": (result.metrics or {}).get("kd_gain", ""),
        "recovery_pct": (result.metrics or {}).get("recovery_pct", ""),
        "duration_s": (result.resources or {}).get("duration_s", ""),
        "peak_vram_mb": (result.resources or {}).get("peak_vram_mb", ""),
        "checksum": getattr(result, "checksum", ""),
    }

    with open(p, "a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return p


def save_markdown_summary(result: BenchmarkResult, md_path: Path | str) -> Path:
    """Generate and write human-readable publication summary in Markdown."""
    p = Path(md_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    mod_name = result.module.get("name", "baseline") if isinstance(result.module, dict) else str(result.module)
    mdl_name = result.model.get("name", "") if isinstance(result.model, dict) else str(result.model)
    ds_name = result.dataset.get("name", "") if isinstance(result.dataset, dict) else str(result.dataset)
    rk = result.config.get("rank", 0) if isinstance(result.config, dict) else 0

    acc = (result.metrics or {}).get("accuracy")
    acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else "N/A"
    f1 = (result.metrics or {}).get("f1")
    f1_str = f"{f1:.4f}" if isinstance(f1, (int, float)) else "N/A"
    loss = (result.metrics or {}).get("loss")
    loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else "N/A"
    dur = (result.resources or {}).get("duration_s", 0.0)
    vram = (result.resources or {}).get("peak_vram_mb", 0.0)

    content = f"""# Benchmark Run Summary: {mod_name}

- **Run ID**: `{result.run_id}`
- **Status**: **{result.status.upper()}**
- **Timestamp**: {result.timestamp_start}
- **Git Commit**: `{result.git_commit[:10]}` (dirty={result.git_dirty})

## Architecture & Configuration
| Field | Value |
|---|---|
| **Module** | `{mod_name}` |
| **Model** | `{mdl_name}` |
| **Dataset** | `{ds_name}` |
| **Rank** | `{rk}` |
| **Epochs** | `{result.config.get('epochs', 1)}` |
| **Batch Size** | `{result.config.get('batch_size', 32)}` |

## Primary Metrics
| Metric | Value |
|---|---|
| **Accuracy** | {acc_str} |
| **F1 Score** | {f1_str} |
| **Loss** | {loss_str} |
| **Duration (s)** | {dur}s |
| **Peak VRAM** | {vram} MB |

## Reproduction Command
```bash
{result.reproducibility.reproduction_command or f'python -m neuralops.benchmarks run --module {mod_name} --model {mdl_name} --dataset {ds_name} --rank {rk}'}
```
"""
    with open(p, "w", encoding="utf-8") as fp:
        fp.write(content)
    return p


def save_parquet_index(result: BenchmarkResult, parquet_path: Path | str) -> Path:
    """Update columnar Parquet storage index for fast querying with pandas/dask."""
    p = Path(parquet_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    new_row = {
        "run_id": str(result.run_id),
        "run_key": str(result.compute_run_key()),
        "timestamp": str(result.timestamp_end or result.timestamp_start),
        "status": str(result.status),
        "module": str(result.module.get("name", "") if isinstance(result.module, dict) else result.module),
        "module_type": str(result.module.get("type", "") if isinstance(result.module, dict) else ""),
        "model": str(result.model.get("name", "") if isinstance(result.model, dict) else result.model),
        "dataset": str(result.dataset.get("name", "") if isinstance(result.dataset, dict) else result.dataset),
        "rank": int(result.config.get("rank", 0) if isinstance(result.config, dict) else 0),
        "accuracy": float((result.metrics or {}).get("accuracy") or 0.0),
        "f1": float((result.metrics or {}).get("f1") or 0.0),
        "loss": float((result.metrics or {}).get("loss") or 0.0),
        "duration_s": float((result.resources or {}).get("duration_s") or 0.0),
        "peak_vram_mb": float((result.resources or {}).get("peak_vram_mb") or 0.0),
    }

    df_new = pd.DataFrame([new_row])
    if p.exists() and p.stat().st_size > 0:
        try:
            df_existing = pd.read_parquet(p)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_parquet(p, index=False)
            return p
        except Exception as e:
            LOG.warning("Could not append to existing parquet index, creating new: %s", e)

    df_new.to_parquet(p, index=False)
    return p


def save_all_formats(
    result: BenchmarkResult,
    output_dir: Path | str = "results",
) -> Dict[str, Path]:
    """
    Save the same benchmark run across all 5 canonical formats:
    1. JSON (canonical self-contained)
    2. JSONL (append-only index)
    3. CSV (rolling spreadsheet index)
    4. Markdown (human-readable summary)
    5. Parquet (columnar storage for queries)
    """
    out_dir = Path(output_dir)
    json_path = compute_result_path(result, base_dir=out_dir)
    md_path = json_path.with_suffix(".md")

    save_json(result, json_path)
    save_markdown_summary(result, md_path)

    jsonl_path = out_dir / "results.jsonl"
    save_jsonl_index(result, jsonl_path)

    csv_path = out_dir / "results.csv"
    save_csv_index(result, csv_path)

    parquet_path = out_dir / "results.parquet"
    save_parquet_index(result, parquet_path)

    # Record saved paths into result artifact metadata
    result.artifacts.json_file = str(json_path.relative_to(out_dir) if json_path.is_relative_to(out_dir) else json_path)

    return {
        "json": json_path,
        "markdown": md_path,
        "jsonl": jsonl_path,
        "csv": csv_path,
        "parquet": parquet_path,
    }


__all__ = [
    "CATEGORY_MAP",
    "resolve_category",
    "compute_result_path",
    "save_json",
    "save_jsonl_index",
    "save_csv_index",
    "save_markdown_summary",
    "save_parquet_index",
    "save_all_formats",
]
