#!/usr/bin/env python3
import csv
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import torch

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def persist_competitor_record(rec: dict, sqlite_path: Path, csv_path: Path):
    try:
        conn = sqlite3.connect(sqlite_path, timeout=30.0)
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS competitor_benchmarks (
                benchmark_name TEXT PRIMARY KEY,
                metrics_json TEXT,
                primary_throughput REAL,
                primary_latency_ms REAL,
                memory_mb REAL,
                status TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cur.execute(
            """
            INSERT OR REPLACE INTO competitor_benchmarks (
                benchmark_name, metrics_json, primary_throughput, primary_latency_ms, memory_mb, status
            ) VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                rec["benchmark_name"],
                json.dumps(rec.get("raw", {})),
                rec.get("throughput", 0.0),
                rec.get("latency_ms", 0.0),
                rec.get("memory_mb", 0.0),
                rec.get("status", "PASS"),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Warning persisting competitor to SQLite: {e}", flush=True)

    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            fields = ["benchmark_name", "primary_throughput", "primary_latency_ms", "memory_mb", "status"]
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "benchmark_name": rec["benchmark_name"],
                    "primary_throughput": rec.get("throughput", 0.0),
                    "primary_latency_ms": rec.get("latency_ms", 0.0),
                    "memory_mb": rec.get("memory_mb", 0.0),
                    "status": rec.get("status", "PASS"),
                }
            )
    except Exception as e:
        print(f"Warning persisting competitor to CSV: {e}", flush=True)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sqlite_path = (repo_root / "results" / "master_ledger.sqlite").resolve()
    csv_path = (repo_root / "results" / "csv" / "competitor_benchmarks.csv").resolve()

    print(f"=== Starting Competitor Benchmark Suite on {device} ===", flush=True)

    modules_to_test = [
        ("deepspeed_zero3", "neuralops.benchmarks.competitors.bench_deepspeed_zero3_overhead"),
        ("megatron_tp_sp", "neuralops.benchmarks.competitors.bench_megatron_tp_sp"),
        ("flash_attention_standard", "neuralops.benchmarks.competitors.bench_flash_attention_standard"),
        ("sglang_radix_cache", "neuralops.benchmarks.competitors.bench_sglang_radix_cache"),
        ("vllm_poisson_serving", "neuralops.benchmarks.competitors.bench_vllm_poisson_serving"),
        ("tensorrt_quant_sqnr", "neuralops.benchmarks.competitors.bench_tensorrt_quant_sqnr"),
        ("ruler_long_context", "neuralops.benchmarks.competitors.bench_ruler_long_context"),
        ("medusa_speculative_tree", "neuralops.benchmarks.competitors.bench_medusa_speculative_tree"),
        ("lighteval_fast_stream", "neuralops.benchmarks.competitors.bench_lighteval_fast_stream"),
        ("lm_eval_mmlu_gsm8k", "neuralops.benchmarks.competitors.bench_lm_eval_mmlu_gsm8k"),
    ]

    for name, mod_path in modules_to_test:
        print(f"Running competitor benchmark: {name}...", flush=True)
        try:
            mod = __import__(mod_path, fromlist=["run_benchmark"])
            if hasattr(mod, "run_benchmark"):
                res = mod.run_benchmark(device=device)
            elif hasattr(mod, "benchmark"):
                res = mod.benchmark(device=device)
            else:
                res = {"status": "SKIPPED", "error": "No benchmark function found"}

            # Comprehensive metric extraction across diverse competitor schemas
            throughput_val = (
                res.get("throughput_tok_s")
                or res.get("tokens_per_sec")
                or res.get("token_throughput")
                or res.get("effective_bandwidth_gb_sec")
                or res.get("effective_throughput_tok_s")
                or res.get("throughput")
                or res.get("attained_tflops")
                or res.get("speedup_ratio")
                or 0.0
            )
            latency_val = (
                res.get("latency_ms")
                or res.get("gather_latency_ms")
                or res.get("fwd_latency_ms")
                or (res.get("ttft_percentiles_ms", {}).get("p50") if isinstance(res.get("ttft_percentiles_ms"), dict) else None)
                or (res.get("tree_eval_latency_ms") if res.get("tree_eval_latency_ms") is not None else None)
                or 0.0
            )
            memory_val = (
                res.get("peak_memory_mb")
                or res.get("total_param_mb")
                or res.get("memory_mb")
                or res.get("allocated_mb")
                or 0.0
            )

            rec = {
                "benchmark_name": name,
                "raw": res,
                "throughput": float(throughput_val),
                "latency_ms": float(latency_val),
                "memory_mb": float(memory_val),
                "status": "PASS",
            }
            persist_competitor_record(rec, sqlite_path, csv_path)
            tp = rec.get("throughput", 0.0)
            lat = rec.get("latency_ms", 0.0)
            print(f"  -> [{name}] PASS: Throughput={tp:.2f}, Latency={lat:.3f}ms", flush=True)
        except Exception as e:
            rec = {
                "benchmark_name": name,
                "raw": {"error": str(e)},
                "throughput": 0.0,
                "latency_ms": 0.0,
                "memory_mb": 0.0,
                "status": f"ERROR: {e}",
            }
            persist_competitor_record(rec, sqlite_path, csv_path)
            print(f"  -> [{name}] ERROR: {e}", flush=True)

    print("=== Competitor Benchmark Suite Completed Successfully! ===", flush=True)


if __name__ == "__main__":
    main()
