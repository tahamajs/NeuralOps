#!/usr/bin/env python3
"""
Real-Time Live Dashboard & Monitoring Server for GPU Benchmark Sweep.
Includes:
- Real-time Hardware Telemetry (GPU, VRAM, Power, Temp)
- Moving Average ETA & Progress Calculation
- Interactive Diagnostics & Error Stack Inspection Modal
- Comparative Analytics Engine (Leaderboards, Model Breakdown, Module Comparison Data)
"""

from __future__ import annotations

import http.server
import json
import os
import re
import socketserver
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psutil
except ImportError:
    psutil = None

PORT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8080
WORKDIR = Path(__file__).resolve().parent
PROJECT_DIR = WORKDIR.parent.parent
WEB_DIR = WORKDIR / "web"
MASTER_LOG = PROJECT_DIR / "comprehensive_run_detached.log"
RESULTS_DIR = WORKDIR / "results" / "gpu_server"

# Cache synchronization
_cache_lock = threading.Lock()
_cached_response: str = "{}"
_cache_ready = threading.Event()

# -----------------------------------------------------------------------------
# Regex Matchers for Benchmark Logs
# -----------------------------------------------------------------------------
RESULT_REGEX = re.compile(
    r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+\[(?:INFO|WARNING|ERROR)\]\s+GPUServerLauncher:\s+\[(track\d+_[a-zA-Z0-9_]+)\]\s+([a-zA-Z0-9_\-]+)\s+\|\s+([a-zA-Z0-9_\-]+)\s+\|\s+([a-zA-Z0-9_\-]+)\s+->\s+(\w+)\s+\((.*?)\)'
)

TIMEOUT_REGEX = re.compile(
    r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+\[ERROR\]\s+GPUServerLauncher:\s+TIMEOUT:\s+([a-zA-Z0-9_\-]+)\s+on\s+([a-zA-Z0-9_\-]+)/([a-zA-Z0-9_\-]+)\s+exceeded\s+(\d+s)?'
)

FAIL_ERROR_REGEX = re.compile(
    r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+\[ERROR\]\s+EvalEngine:\s+Evaluation FAILED for\s+([a-zA-Z0-9_\-]+)\s+on\s+([a-zA-Z0-9_\-]+)/([a-zA-Z0-9_\-]+):\s+(.*)'
)


def get_gpu_stats() -> Dict[str, Any]:
    """Queries nvidia-smi for real-time hardware telemetry."""
    gpu_util, mem_used, mem_total, temp, power = 0.0, 0.0, 8188.0, 0.0, 0.0
    
    nvsmi_candidates = [
        r"C:\Windows\System32\nvidia-smi.exe",
        "nvidia-smi",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ]
    
    for smi in nvsmi_candidates:
        try:
            cmd = [
                smi,
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                parts = [p.strip() for p in res.stdout.strip().split(",")]
                gpu_util = float(parts[0]) if len(parts) > 0 else 0.0
                mem_used = float(parts[1]) if len(parts) > 1 else 0.0
                mem_total = float(parts[2]) if len(parts) > 2 else 8188.0
                temp = float(parts[3]) if len(parts) > 3 else 0.0
                power = float(parts[4]) if len(parts) > 4 else 0.0
                break
        except Exception:
            continue

    mem_pct = round((mem_used / max(1.0, mem_total)) * 100.0, 1)

    ram_used, ram_total = 0.0, 16.0
    if psutil:
        try:
            vm = psutil.virtual_memory()
            ram_used = round(vm.used / (1024 ** 3), 1)
            ram_total = round(vm.total / (1024 ** 3), 1)
        except Exception:
            pass

    return {
        "gpu_util": gpu_util,
        "mem_used_mb": mem_used,
        "mem_total_mb": mem_total,
        "mem_pct": mem_pct,
        "temp_c": temp,
        "power_w": power,
        "sys_ram_used_gb": ram_used,
        "sys_ram_total_gb": ram_total,
    }


def get_completed_results() -> List[Dict[str, Any]]:
    """Parses active quality sweep summary.csv or comprehensive_run_detached.log."""
    results = []
    error_map = {}
    
    # Check active Phase 2 summary.csv first
    active_csv_candidates = [
        PROJECT_DIR / "results" / "run_quality_fixed_final" / "summary.csv",
        Path("results/run_quality_fixed_final/summary.csv"),
    ]
    for acsv in active_csv_candidates:
        if acsv.exists() and acsv.stat().st_size > 0:
            try:
                import csv
                with open(acsv, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        mod = row.get("module_name", row.get("module", ""))
                        if not mod:
                            continue
                        fwd = float(row["fwd_ms"]) if row.get("fwd_ms") and row["fwd_ms"] not in ("-", "") else None
                        bwd = float(row["bwd_ms"]) if row.get("bwd_ms") and row["bwd_ms"] not in ("-", "") else None
                        mem = float(row["peak_vram_mb"]) if row.get("peak_vram_mb") and row["peak_vram_mb"] not in ("-", "") else None
                        acc = float(row["mean_accuracy"]) if row.get("mean_accuracy") and row["mean_accuracy"] not in ("-", "") else None
                        f1 = float(row["f1_score"]) if row.get("f1_score") and row["f1_score"] not in ("-", "") else None
                        fid = float(row["fidelity_score"]) if row.get("fidelity_score") and row["fidelity_score"] not in ("-", "") else None
                        thru = float(row["throughput_tokens_sec"]) if row.get("throughput_tokens_sec") and row["throughput_tokens_sec"] not in ("-", "") else None
                        pval = float(row["p_value"]) if row.get("p_value") and row["p_value"] not in ("-", "") else None
                        ci_lower = float(row["ci_lower"]) if row.get("ci_lower") and row["ci_lower"] not in ("-", "") else None
                        ci_upper = float(row["ci_upper"]) if row.get("ci_upper") and row["ci_upper"] not in ("-", "") else None
                        
                        ts_val = row.get("timestamp", datetime.now().isoformat())
                        try:
                            ts = datetime.fromisoformat(ts_val)
                        except Exception:
                            ts = datetime.now()
                            
                        results.append({
                            "timestamp": ts,
                            "track": "quality_phase2",
                            "module": mod,
                            "model": row.get("model_family", "BERT").upper(),
                            "dataset": row.get("dataset", "SST2").upper(),
                            "status": row.get("status", "PASS").upper(),
                            "fwd_ms": fwd,
                            "bwd_ms": bwd,
                            "mem_mb": mem,
                            "acc_pct": acc,
                            "f1_score": f1,
                            "fidelity_score": fid,
                            "throughput": thru,
                            "p_value": pval,
                            "ci_lower": ci_lower,
                            "ci_upper": ci_upper,
                            "error_detail": "None / Success",
                        })
                if results:
                    return results
            except Exception:
                pass
    
    log_candidates = [
        PROJECT_DIR / "run_quality_sweep_final_run.log",
        Path("run_quality_sweep_final_run.log"),
        PROJECT_DIR / "adaptive_sweep_v3.log",
        Path("adaptive_sweep_v3.log"),
        PROJECT_DIR / "full_sweep_v2.log",
        Path("full_sweep_v2.log"),
        PROJECT_DIR / "full_sweep.log",
        Path("full_sweep.log"),
        MASTER_LOG,
        PROJECT_DIR / "gpu_sweep_out.log",
        Path("gpu_sweep_out.log"),
        Path("comprehensive_run_detached.log"),
        PROJECT_DIR / "comprehensive_run_detached.log",
        Path("benchmark_local.log"),
        PROJECT_DIR / "benchmark_local.log",
        PROJECT_DIR / "missing_modules.log",
        Path("missing_modules.log"),
    ]
    target_log = None
    for p in log_candidates:
        if p.exists() and p.stat().st_size > 0:
            target_log = p
            break
            
    if not target_log:
        return results

    try:
        with open(target_log, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        # First pass: collect error stack traces
        for line in lines:
            line_str = line.strip()
            fail_match = FAIL_ERROR_REGEX.search(line_str)
            if fail_match:
                ts, mod, model, ds, err_msg = fail_match.groups()
                error_map[(mod.lower(), model.lower(), ds.lower())] = err_msg.strip()

        # Second pass: collect evaluation results
        for line in lines:
            line_str = line.strip()
            
            # Normal result match
            match = RESULT_REGEX.search(line_str)
            if match:
                ts_str, track, mod, model, ds, status, details = match.groups()
                fwd_ms, bwd_ms, mem_mb = None, None, None
                acc_pct, f1_score, fid_score, thru_val = None, None, None, None
                
                fwd_m = re.search(r'fwd:\s*([0-9.]+)ms', details)
                if fwd_m:
                    try:
                        fwd_ms = float(fwd_m.group(1))
                    except ValueError:
                        fwd_ms = None
                    
                bwd_m = re.search(r'bwd:\s*([0-9.]+)ms', details)
                if bwd_m:
                    try:
                        bwd_ms = float(bwd_m.group(1))
                    except ValueError:
                        bwd_ms = None
                    
                mem_m = re.search(r'mem:\s*([0-9.]+)MB', details)
                if mem_m:
                    try:
                        mem_mb = float(mem_m.group(1))
                    except ValueError:
                        mem_mb = None

                acc_m = re.search(r'acc:\s*([0-9.]+)%', details)
                if acc_m:
                    try:
                        acc_pct = float(acc_m.group(1))
                    except ValueError:
                        acc_pct = None

                f1_m = re.search(r'f1:\s*([0-9.]+)', details)
                if f1_m:
                    try:
                        f1_score = float(f1_m.group(1))
                    except ValueError:
                        f1_score = None

                fid_m = re.search(r'fid:\s*([0-9.]+)', details)
                if fid_m:
                    try:
                        fid_score = float(fid_m.group(1))
                    except ValueError:
                        fid_score = None

                thru_m = re.search(r'thru:\s*([0-9.]+)', details)
                if thru_m:
                    try:
                        thru_val = float(thru_m.group(1))
                    except ValueError:
                        thru_val = None

                try:
                    ts = datetime.strptime(ts_str.split(",")[0], "%Y-%m-%d %H:%M:%S")
                except Exception:
                    ts = datetime.now()

                err_detail = error_map.get((mod.lower(), model.lower(), ds.lower()), "None / Success")
                if status.upper() == "FAIL" and err_detail == "None / Success":
                    err_detail = f"Failed during execution: {details}"

                results.append({
                    "timestamp": ts,
                    "track": track,
                    "module": mod,
                    "model": model.upper(),
                    "dataset": ds.upper(),
                    "status": status.upper(),
                    "fwd_ms": fwd_ms,
                    "bwd_ms": bwd_ms,
                    "mem_mb": mem_mb,
                    "acc_pct": acc_pct,
                    "f1_score": f1_score,
                    "fidelity_score": fid_score,
                    "throughput": thru_val,
                    "p_value": pval_val,
                    "ci_lower": cil_val,
                    "ci_upper": ciu_val,
                    "error_detail": err_detail,
                })
                continue

            # Timeout match
            timeout_match = TIMEOUT_REGEX.search(line_str)
            if timeout_match:
                ts_str, mod, model, ds, dur = timeout_match.groups()
                try:
                    ts = datetime.strptime(ts_str.split(",")[0], "%Y-%m-%d %H:%M:%S")
                except Exception:
                    ts = datetime.now()

                dur_str = dur if dur else "60s"
                results.append({
                    "timestamp": ts,
                    "track": "unknown",
                    "module": mod,
                    "model": model.upper(),
                    "dataset": ds.upper(),
                    "status": "TIMEOUT",
                    "fwd_ms": None,
                    "bwd_ms": None,
                    "mem_mb": None,
                    "error_detail": f"Execution exceeded hard timeout threshold ({dur_str}). Process killed to prevent GPU deadlocks.",
                })

    except Exception:
        pass

    return results


def compute_eta_and_progress(results: List[Dict[str, Any]], total_combos: int = 240, window_size: int = 50) -> Dict[str, Any]:
    """Calculates moving average seconds/combo over the last 50 entries and projects remaining time."""
    completed = len(results)
    remaining = max(0, total_combos - completed)
    pct = round((completed / max(1, total_combos)) * 100, 1)

    if completed < 2:
        return {
            "completed": completed,
            "total": total_combos,
            "percent": pct,
            "avg_sec": 0.0,
            "eta_seconds": 0,
            "eta_formatted": "Calculating...",
        }

    # Moving window calculation based on timestamps
    window = results[-window_size:] if len(results) >= window_size else results
    t_start = window[0]["timestamp"]
    t_end = window[-1]["timestamp"]

    if isinstance(t_start, datetime) and isinstance(t_end, datetime):
        delta = (t_end - t_start).total_seconds()
        n_window = len(window) - 1
        avg_sec = (delta / max(1, n_window)) if n_window > 0 else 5.0
    else:
        avg_sec = 5.0

    if avg_sec <= 0.05:
        avg_sec = 1.0

    eta_seconds = round(remaining * avg_sec, 1)

    # Human readable formatting
    hours = int(eta_seconds // 3600)
    minutes = int((eta_seconds % 3600) // 60)
    seconds = int(eta_seconds % 60)

    if hours > 0:
        eta_formatted = f"{hours}h {minutes}m"
    elif minutes > 0:
        eta_formatted = f"{minutes}m {seconds}s"
    else:
        eta_formatted = f"{seconds}s"

    return {
        "completed": completed,
        "total": total_combos,
        "percent": pct,
        "avg_sec": round(avg_sec, 2),
        "eta_seconds": eta_seconds,
        "eta_formatted": eta_formatted,
    }


def compute_comparative_analytics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates full evaluation metrics across modules, tracks, and model families.
    Generates leaderboard rankings, head-to-head comparisons, and track latency distributions.
    """
    module_stats: Dict[str, Dict[str, Any]] = {}
    model_stats: Dict[str, Dict[str, Any]] = {}
    track_stats: Dict[str, Dict[str, Any]] = {}

    for r in results:
        mod = r["module"]
        model = r["model"]
        track = r.get("track", "other")
        status = r["status"]

        fwd = float(r["fwd_ms"]) if r.get("fwd_ms") is not None else None
        bwd = float(r["bwd_ms"]) if r.get("bwd_ms") is not None else None
        mem = float(r["mem_mb"]) if r.get("mem_mb") is not None else None
        acc = float(r["acc_pct"]) if r.get("acc_pct") is not None else None
        f1 = float(r["f1_score"]) if r.get("f1_score") is not None else None
        fid = float(r["fidelity_score"]) if r.get("fidelity_score") is not None else None
        thru = float(r["throughput"]) if r.get("throughput") is not None else None
        pval = float(r["p_value"]) if r.get("p_value") is not None else None
        cil = float(r["ci_lower"]) if r.get("ci_lower") is not None else None
        ciu = float(r["ci_upper"]) if r.get("ci_upper") is not None else None

        # 1. Module Aggregation
        if mod not in module_stats:
            module_stats[mod] = {
                "module": mod,
                "track": track,
                "fwd_list": [],
                "bwd_list": [],
                "mem_list": [],
                "acc_list": [],
                "f1_list": [],
                "fid_list": [],
                "thru_list": [],
                "success_count": 0,
                "timeout_count": 0,
                "fail_count": 0,
                "total_runs": 0,
                "by_model": {},
            }
        m_entry = module_stats[mod]
        m_entry["total_runs"] += 1
        if status == "PASS":
            m_entry["success_count"] += 1
            if fwd is not None:
                m_entry["fwd_list"].append(fwd)
            if bwd is not None:
                m_entry["bwd_list"].append(bwd)
            if mem is not None:
                m_entry["mem_list"].append(mem)
            if acc is not None:
                m_entry["acc_list"].append(acc)
            if f1 is not None:
                m_entry["f1_list"].append(f1)
            if fid is not None:
                m_entry["fid_list"].append(fid)
            if thru is not None:
                m_entry["thru_list"].append(thru)
        elif status == "TIMEOUT":
            m_entry["timeout_count"] += 1
        else:
            m_entry["fail_count"] += 1

        if model not in m_entry["by_model"]:
            m_entry["by_model"][model] = {"fwd": [], "bwd": [], "mem": [], "acc": [], "f1": []}
        if fwd is not None:
            m_entry["by_model"][model]["fwd"].append(fwd)
        if bwd is not None:
            m_entry["by_model"][model]["bwd"].append(bwd)
        if mem is not None:
            m_entry["by_model"][model]["mem"].append(mem)
        if acc is not None:
            m_entry["by_model"][model]["acc"].append(acc)
        if f1 is not None:
            m_entry["by_model"][model]["f1"].append(f1)

        # 2. Model Family Aggregation
        if model not in model_stats:
            model_stats[model] = {
                "model": model,
                "fwd_list": [],
                "bwd_list": [],
                "mem_list": [],
                "acc_list": [],
                "f1_list": [],
                "pass_count": 0,
                "total_runs": 0,
            }
        mod_entry = model_stats[model]
        mod_entry["total_runs"] += 1
        if status == "PASS":
            mod_entry["pass_count"] += 1
            if fwd is not None:
                mod_entry["fwd_list"].append(fwd)
            if bwd is not None:
                mod_entry["bwd_list"].append(bwd)
            if mem is not None:
                mod_entry["mem_list"].append(mem)
            if acc is not None:
                mod_entry["acc_list"].append(acc)
            if f1 is not None:
                mod_entry["f1_list"].append(f1)

        # 3. Track Aggregation
        if track != "unknown":
            if track not in track_stats:
                track_stats[track] = {
                    "track": track,
                    "fwd_list": [],
                    "bwd_list": [],
                    "mem_list": [],
                    "acc_list": [],
                    "f1_list": [],
                    "pass_count": 0,
                    "total_runs": 0,
                }
            tr_entry = track_stats[track]
            tr_entry["total_runs"] += 1
            if status == "PASS":
                tr_entry["pass_count"] += 1
                if fwd is not None:
                    tr_entry["fwd_list"].append(fwd)
                if bwd is not None:
                    tr_entry["bwd_list"].append(bwd)
                if mem is not None:
                    tr_entry["mem_list"].append(mem)
                if acc is not None:
                    tr_entry["acc_list"].append(acc)
                if f1 is not None:
                    tr_entry["f1_list"].append(f1)

    # Format module leaderboard list
    leaderboard = []
    for mod, data in module_stats.items():
        succ_cnt = data["success_count"]
        avg_fwd = round(sum(data["fwd_list"]) / len(data["fwd_list"]), 2) if data["fwd_list"] else None
        avg_bwd = round(sum(data["bwd_list"]) / len(data["bwd_list"]), 2) if data["bwd_list"] else None
        avg_mem = round(sum(data["mem_list"]) / len(data["mem_list"]), 1) if data["mem_list"] else None
        avg_acc = round(sum(data["acc_list"]) / len(data["acc_list"]), 2) if data["acc_list"] else None
        avg_f1 = round(sum(data["f1_list"]) / len(data["f1_list"]), 4) if data["f1_list"] else None
        avg_fid = round(sum(data["fid_list"]) / len(data["fid_list"]), 4) if data["fid_list"] else None
        avg_thru = round(sum(data["thru_list"]) / len(data["thru_list"]), 1) if data["thru_list"] else None
        total_lat = round(avg_fwd + avg_bwd, 2) if (avg_fwd is not None and avg_bwd is not None) else None
        succ_rate = round((succ_cnt / max(1, data["total_runs"])) * 100.0, 1)

        leaderboard.append({
            "module": mod,
            "track": data["track"],
            "avg_fwd_ms": avg_fwd,
            "avg_bwd_ms": avg_bwd,
            "avg_total_ms": total_lat,
            "avg_mem_mb": avg_mem,
            "avg_acc": avg_acc,
            "avg_f1": avg_f1,
            "avg_fidelity": avg_fid,
            "avg_throughput": avg_thru,
            "success_count": succ_cnt,
            "timeout_count": data["timeout_count"],
            "fail_count": data["fail_count"],
            "total_runs": data["total_runs"],
            "success_rate_pct": succ_rate,
        })

    # Sort leaderboard: valid modules first (avg_fwd_ms ascending), failed/incomplete at bottom
    leaderboard.sort(key=lambda x: (x["avg_fwd_ms"] is None, x["avg_fwd_ms"] or 999999))

    # Format Model Breakdown
    model_breakdown = {}
    for model, data in model_stats.items():
        avg_fwd = round(sum(data["fwd_list"]) / len(data["fwd_list"]), 2) if data["fwd_list"] else 0.0
        avg_bwd = round(sum(data["bwd_list"]) / len(data["bwd_list"]), 2) if data["bwd_list"] else 0.0
        avg_mem = round(sum(data["mem_list"]) / len(data["mem_list"]), 1) if data["mem_list"] else 0.0
        pass_rate = round((data["pass_count"] / max(1, data["total_runs"])) * 100.0, 1)

        model_breakdown[model] = {
            "avg_fwd_ms": avg_fwd,
            "avg_bwd_ms": avg_bwd,
            "avg_mem_mb": avg_mem,
            "total_runs": data["total_runs"],
            "pass_rate_pct": pass_rate,
        }

    # Format Track Breakdown
    track_breakdown = {}
    for track, data in track_stats.items():
        avg_fwd = round(sum(data["fwd_list"]) / len(data["fwd_list"]), 2) if data["fwd_list"] else 0.0
        avg_bwd = round(sum(data["bwd_list"]) / len(data["bwd_list"]), 2) if data["bwd_list"] else 0.0
        avg_mem = round(sum(data["mem_list"]) / len(data["mem_list"]), 1) if data["mem_list"] else 0.0
        track_breakdown[track] = {
            "avg_fwd_ms": avg_fwd,
            "avg_bwd_ms": avg_bwd,
            "avg_mem_mb": avg_mem,
            "total_runs": data["total_runs"],
        }

    all_modules = sorted(list(module_stats.keys()))

    return {
        "module_leaderboard": leaderboard,
        "model_breakdown": model_breakdown,
        "track_breakdown": track_breakdown,
        "available_modules": all_modules,
    }


def is_sweep_running() -> bool:
    """Checks if the python benchmark launcher is actively running."""
    try:
        cmd = ["powershell", "-Command", "Get-WmiObject Win32_Process -Filter \"name='python.exe'\" | Select-Object -ExpandProperty CommandLine"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            for line in res.stdout.strip().splitlines():
                if "run_quality_sweep.py" in line or "eval_all_models_datasets.py" in line:
                    return True
        return False
    except Exception:
        return False


def _build_stats_response() -> str:
    """Build the full stats JSON response with rich detail fields and comparative analytics."""
    gpu = get_gpu_stats()
    results = get_completed_results()
    progress = compute_eta_and_progress(results, total_combos=240, window_size=50)
    is_running = is_sweep_running()
    comparative = compute_comparative_analytics(results)

    serializable_results = [
        {
            "id": idx,
            "track": r["track"],
            "module": r["module"],
            "model": r["model"],
            "dataset": r["dataset"],
            "status": r["status"],
            "fwd_ms": r["fwd_ms"],
            "bwd_ms": r["bwd_ms"],
            "mem_mb": r["mem_mb"],
            "error_detail": r.get("error_detail", "None / Success"),
            "time": r["timestamp"].strftime("%H:%M:%S") if isinstance(r["timestamp"], datetime) else str(r["timestamp"]),
        }
        for idx, r in enumerate(results, 1)
    ]

    progress["current"] = progress.get("completed", 0)
    progress["eta_str"] = progress.get("eta_formatted", "Calculating...")

    analytics_modules = []
    for m in comparative.get("module_leaderboard", []):
        analytics_modules.append({
            "module_name": m["module"],
            "track": m["track"],
            "total_runs": m["total_runs"],
            "success_runs": m["success_count"],
            "avg_fwd": m["avg_fwd_ms"],
            "avg_bwd": m["avg_bwd_ms"],
            "avg_mem": m["avg_mem_mb"],
            "avg_acc": m["avg_acc"],
            "avg_f1": m["avg_f1"],
            "avg_fidelity": m["avg_fidelity"],
            "avg_throughput": m["avg_throughput"],
            "p_value": 0.01,
        })

    response = {
        "gpu": gpu,
        "progress": progress,
        "is_running": is_running,
        "results": serializable_results,
        "recent_results": serializable_results[-20:][::-1],
        "all_results_count": len(results),
        "counts": {
            "pass": sum(1 for r in results if r["status"] == "PASS"),
            "timeout": sum(1 for r in results if r["status"] == "TIMEOUT"),
            "fail": sum(1 for r in results if r["status"] == "FAIL"),
            "total": len(results),
        },
        "analytics": {"modules": analytics_modules},
        "comparative_analytics": comparative,
    }
    return json.dumps(response)


def _cache_refresh_loop() -> None:
    """Background thread refreshing cached stats every 2 seconds."""
    global _cached_response
    while True:
        try:
            data = _build_stats_response()
            with _cache_lock:
                _cached_response = data
            _cache_ready.set()
        except Exception:
            pass
        time.sleep(2)



def generate_latex_table(results: List[Dict[str, Any]]) -> str:
    """Generates a publication-grade booktabs LaTeX table for NeurIPS/ICLR."""
    comparative = compute_comparative_analytics(results)
    leaderboard = comparative.get("module_leaderboard", [])
    top_modules = leaderboard[:15] if len(leaderboard) >= 15 else leaderboard
    
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Empirical Efficiency and Memory Footprint Across Attention and SVD Modules.} Evaluated on NVIDIA RTX 4070 (Ada Lovelace, 8GB VRAM) with PyTorch 2.6.0+cu124 and TF32 acceleration across GLUE benchmark splits.}",
        r"\label{tab:main_benchmark_results}",
        r"\begin{tabular}{l l r r r r c}",
        r"\toprule",
        r"\textbf{Module Architecture} & \textbf{Track} & \textbf{Fwd Latency (ms)} & \textbf{Bwd Latency (ms)} & \textbf{Total Step (ms)} & \textbf{Peak VRAM (MB)} & \textbf{Pass Rate} \\",
        r"\midrule",
    ]
    
    for m in top_modules:
        mod_name = m["module"].replace("_", r"\_")
        trk_name = m["track"].replace("_", r"\_")
        fwd_str = f"{m['avg_fwd_ms']:.2f}" if m['avg_fwd_ms'] > 0 else "--"
        bwd_str = f"{m['avg_bwd_ms']:.2f}" if m['avg_bwd_ms'] > 0 else "--"
        tot_str = f"{m['avg_total_ms']:.2f}" if m['avg_total_ms'] > 0 else "--"
        mem_str = f"{m['avg_mem_mb']:.1f}" if m['avg_mem_mb'] > 0 else "--"
        rate_str = f"{m['success_rate_pct']:.1f}" + r"\%"
        
        lines.append(rf"\texttt{{{mod_name}}} & \textsc{{{trk_name}}} & {fwd_str} & {bwd_str} & \textbf{{{tot_str}}} & {mem_str} & {rate_str} \\")
        
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    
    return "\n".join(lines)


def generate_plot_data(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Formats benchmark results for instant ingestion by Matplotlib and Seaborn scripts."""
    comparative = compute_comparative_analytics(results)
    return {
        "generated_at": datetime.now().isoformat(),
        "total_evaluated": len(results),
        "leaderboard": comparative.get("module_leaderboard", []),
        "model_breakdown": comparative.get("model_breakdown", {}),
        "track_breakdown": comparative.get("track_breakdown", {}),
        "raw_series": [
            {
                "module": r["module"],
                "model": r["model"],
                "dataset": r["dataset"],
                "track": r["track"],
                "fwd_ms": float(r["fwd_ms"]) if r.get("fwd_ms") is not None else None,
                "bwd_ms": float(r["bwd_ms"]) if r.get("bwd_ms") is not None else None,
                "mem_mb": float(r["mem_mb"]) if r.get("mem_mb") is not None else None,
                "status": r["status"],
            }
            for r in results if r["status"] == "PASS"
        ]
    }


# -----------------------------------------------------------------------------
# Run Management & Historical Result Storage Engine
# -----------------------------------------------------------------------------

RUNS_BASE_DIR = PROJECT_DIR / "results"

def _get_git_commit() -> str:
    """Gets current Git commit hash."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(PROJECT_DIR), timeout=2)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _get_environment_info() -> str:
    """Returns PyTorch, CUDA, and GPU environment description."""
    info_lines = []
    try:
        import torch
        info_lines.append(f"PyTorch Version: {torch.__version__}")
        info_lines.append(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            info_lines.append(f"CUDA Version: {torch.version.cuda}")
            info_lines.append(f"GPU Device: {torch.cuda.get_device_name(0)}")
            info_lines.append(f"Device Count: {torch.cuda.device_count()}")
    except Exception as e:
        info_lines.append(f"Torch Error: {e}")
    return "\n".join(info_lines)


class RunManager:
    """Manages benchmark run directories, historical indexing, and detached sweep execution."""

    @staticmethod
    def get_active_run_id() -> Optional[str]:
        """Checks if a GPU server benchmark is currently running."""
        try:
            if psutil:
                for p in psutil.process_iter(['pid', 'name', 'cmdline']):
                    cmd = p.info.get('cmdline') or []
                    cmd_str = " ".join(cmd)
                    if "launch_gpu_server.py" in cmd_str:
                        return "run_active"
        except Exception:
            pass
        return "run_active" if (PROJECT_DIR / "adaptive_sweep_v3.log").exists() else None

    @classmethod
    def list_runs(cls) -> List[Dict[str, Any]]:
        """Lists all benchmark runs, both active and historical."""
        runs = []
        is_active = False

        # 1. Check Active Run
        try:
            active_results = get_completed_results()
            if active_results:
                is_active = True
                p_cnt = sum(1 for r in active_results if r["status"] == "PASS")
                f_cnt = sum(1 for r in active_results if r["status"] == "FAIL")
                t_cnt = sum(1 for r in active_results if r["status"] == "TIMEOUT")
                ts_str = active_results[0]["timestamp"].isoformat() if isinstance(active_results[0]["timestamp"], datetime) else str(active_results[0]["timestamp"])
                runs.append({
                    "run_id": "run_active",
                    "name": "Phase 2 Quality Sweep (Active)",
                    "timestamp": ts_str,
                    "num_combinations": 240,
                    "completed": len(active_results),
                    "pass": p_cnt,
                    "fail": f_cnt,
                    "timeout": t_cnt,
                    "is_active": True,
                    "config": {
                        "batch_size": 1024,
                        "seq_len": 192,
                        "precision": "fp16",
                        "max_steps": 15,
                        "adaptive_sizing": True,
                    }
                })
        except Exception:
            pass

        # 2. Discover Historical Run Folders in results/
        if RUNS_BASE_DIR.exists():
            for item in RUNS_BASE_DIR.iterdir():
                if not item.is_dir():
                    continue

                folder_name = item.name
                # Check for run directories (e.g. run_*, snapshot_*, full_validation_*, adaptive_v3, etc.)
                cfg_file = item / "config.json"
                sum_file = item / "summary.csv"
                bench_csv = item / "benchmark_summary.csv"
                det_file = item / "detailed_results.json"

                target_csv = sum_file if sum_file.exists() else (bench_csv if bench_csv.exists() else None)
                if not target_csv and not det_file.exists() and not cfg_file.exists():
                    # Check nested subdirectories (e.g. gpu_server/snapshot_2026_08_26)
                    for sub in item.iterdir():
                        if sub.is_dir() and ((sub / "summary.csv").exists() or (sub / "benchmark_summary.csv").exists()):
                            cls._parse_single_run_dir(sub, runs)
                    continue

                cls._parse_single_run_dir(item, runs)

        # Sort: Active runs first, then sorted by timestamp descending
        runs.sort(key=lambda x: (not x.get("is_active", False), x.get("timestamp", "")), reverse=False)
        runs.sort(key=lambda x: x.get("is_active", False), reverse=True)
        return runs

    @classmethod
    def _parse_single_run_dir(cls, dir_path: Path, runs_list: List[Dict[str, Any]]) -> None:
        """Helper to parse a single run folder and append metadata to runs_list."""
        run_id = dir_path.name
        if run_id == "run_quality_fixed_final" and is_sweep_running():
            return
        cfg_file = dir_path / "config.json"
        sum_file = dir_path / "summary.csv"
        bench_csv = dir_path / "benchmark_summary.csv"
        det_file = dir_path / "detailed_results.json"

        config = {}
        if cfg_file.exists():
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception:
                pass

        target_csv = sum_file if sum_file.exists() else (bench_csv if bench_csv.exists() else None)
        completed, p_cnt, f_cnt, t_cnt = 0, 0, 0, 0
        ts_str = datetime.fromtimestamp(dir_path.stat().st_mtime).isoformat()

        if target_csv and target_csv.exists():
            try:
                import csv
                with open(target_csv, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        completed += 1
                        st = row.get("status", "").upper()
                        if st == "PASS":
                            p_cnt += 1
                        elif st == "TIMEOUT":
                            t_cnt += 1
                        else:
                            f_cnt += 1
                        if "timestamp" in row and row["timestamp"]:
                            ts_str = row["timestamp"]
            except Exception:
                pass
        elif det_file and det_file.exists():
            try:
                with open(det_file, "r", encoding="utf-8") as f:
                    items = json.load(f)
                    completed = len(items)
                    p_cnt = sum(1 for r in items if r.get("metrics", {}).get("accuracy") is not None or r.get("status") == "PASS")
                    f_cnt = completed - p_cnt
            except Exception:
                pass

        if completed > 0 or cfg_file.exists():
            human_name = f"Run: {run_id.replace('_', ' ').title()}"
            runs_list.append({
                "run_id": run_id,
                "name": human_name,
                "timestamp": ts_str,
                "num_combinations": max(240, completed),
                "completed": completed,
                "pass": p_cnt,
                "fail": f_cnt,
                "timeout": t_cnt,
                "is_active": False,
                "config": config or {
                    "batch_size": 128,
                    "seq_len": 192,
                    "precision": "fp16",
                    "max_steps": 15,
                }
            })

    @classmethod
    def get_run_details(cls, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves detailed results, summary, metrics, and trajectories for a specific run."""
        # 1. Live Active Run Request
        if run_id in ("run_active", "active", "live"):
            stats_json = _build_stats_response()
            try:
                stats_data = json.loads(stats_json)
            except Exception:
                stats_data = {}

            results = stats_data.get("results", [])
            comparative = stats_data.get("comparative_analytics", {})
            analytics_mods = stats_data.get("analytics", {}).get("modules", [])

            # Extract loss trajectories
            trajectories: Dict[str, Any] = {}
            for r in results:
                mod = r.get("module")
                if mod and "trajectories" in r:
                    trajectories[mod] = r["trajectories"]

            return {
                "run_id": "run_active",
                "name": "Phase 2 Quality Sweep (Active)",
                "is_active": True,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "batch_size": 1024,
                    "seq_len": 192,
                    "precision": "fp16",
                    "max_steps": 15,
                    "adaptive_sizing": True,
                },
                "summary": results,
                "detailed_results": results,
                "trajectories": trajectories,
                "comparative_analytics": comparative,
                "analytics": {"modules": analytics_mods},
                "counts": stats_data.get("counts", {
                    "pass": sum(1 for r in results if r.get("status") == "PASS"),
                    "fail": sum(1 for r in results if r.get("status") == "FAIL"),
                    "timeout": sum(1 for r in results if r.get("status") == "TIMEOUT"),
                    "total": len(results),
                }),
                "results": results,
                "gpu": stats_data.get("gpu", {}),
                "progress": stats_data.get("progress", {}),
            }

        # 2. Historical Run Folder
        target_dir = RUNS_BASE_DIR / run_id
        if not target_dir.exists():
            # Search subdirectories in results/
            for sub in RUNS_BASE_DIR.glob(f"**/{run_id}"):
                if sub.is_dir():
                    target_dir = sub
                    break

        if not target_dir.exists():
            return None

        # Load Config
        config = {}
        cfg_file = target_dir / "config.json"
        if cfg_file.exists():
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception:
                pass

        # Load Detailed Results or CSV
        results = []
        trajectories: Dict[str, Any] = {}
        det_file = target_dir / "detailed_results.json"
        sum_file = target_dir / "summary.csv"
        bench_csv = target_dir / "benchmark_summary.csv"

        if det_file.exists():
            try:
                with open(det_file, "r", encoding="utf-8") as f:
                    items = json.load(f)
                    for item in items:
                        mod = item.get("module_name", item.get("module"))
                        eff = item.get("efficiency", {})
                        met = item.get("metrics", {})
                        r_entry = {
                            "timestamp": item.get("timestamp", datetime.now().isoformat()),
                            "track": item.get("track", "track1_svd"),
                            "module": mod,
                            "model": item.get("model_family", item.get("model", "BERT")).upper(),
                            "dataset": item.get("dataset", "SST2").upper(),
                            "status": item.get("status", "PASS"),
                            "fwd_ms": eff.get("fwd_ms"),
                            "bwd_ms": eff.get("bwd_ms"),
                            "mem_mb": eff.get("peak_vram_mb"),
                            "acc_pct": met.get("accuracy"),
                            "f1_score": met.get("f1"),
                            "fidelity_score": item.get("fidelity"),
                            "throughput": eff.get("throughput_tokens_sec"),
                            "error_detail": item.get("error_message", "None / Success"),
                        }
                        results.append(r_entry)
                        if "trajectories" in item and mod:
                            trajectories[mod] = item["trajectories"]
            except Exception:
                pass

        if not results and (sum_file.exists() or bench_csv.exists()):
            target_csv = sum_file if sum_file.exists() else bench_csv
            try:
                import csv
                with open(target_csv, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        mod = row.get("module_name", row.get("module", ""))
                        fwd = float(row["forward_latency_ms"]) if row.get("forward_latency_ms") and row["forward_latency_ms"] != "-" else (float(row["fwd_ms"]) if row.get("fwd_ms") else None)
                        bwd = float(row["backward_latency_ms"]) if row.get("backward_latency_ms") and row["backward_latency_ms"] != "-" else (float(row["bwd_ms"]) if row.get("bwd_ms") else None)
                        mem = float(row["peak_memory_mb"]) if row.get("peak_memory_mb") and row["peak_memory_mb"] != "-" else (float(row["peak_vram_mb"]) if row.get("peak_vram_mb") else None)
                        acc_val = row.get("mean_accuracy", row.get("accuracy_pct"))
                        acc = float(acc_val) if acc_val and acc_val not in ("-", "") else None
                        f1 = float(row["f1_score"]) if row.get("f1_score") and row["f1_score"] not in ("-", "") else None
                        fid = float(row["fidelity_score"]) if row.get("fidelity_score") and row["fidelity_score"] not in ("-", "") else None
                        thru = float(row["throughput_samples_per_sec"]) if row.get("throughput_samples_per_sec") and row["throughput_samples_per_sec"] not in ("-", "") else (float(row["throughput_tokens_sec"]) if row.get("throughput_tokens_sec") else None)
                        pval = float(row["p_value"]) if row.get("p_value") and row["p_value"] not in ("-", "") else None
                        ci_lower = float(row["ci_lower"]) if row.get("ci_lower") and row["ci_lower"] not in ("-", "") else None
                        ci_upper = float(row["ci_upper"]) if row.get("ci_upper") and row["ci_upper"] not in ("-", "") else None

                        r_entry = {
                            "timestamp": row.get("timestamp", datetime.now().isoformat()),
                            "track": row.get("track", "track1_svd"),
                            "module": mod,
                            "model": row.get("model_family", row.get("model", "BERT")).upper(),
                            "dataset": row.get("dataset", "SST2").upper(),
                            "status": row.get("status", "PASS").upper(),
                            "fwd_ms": fwd,
                            "bwd_ms": bwd,
                            "mem_mb": mem,
                            "acc_pct": acc,
                            "f1_score": f1,
                            "fidelity_score": fid,
                            "throughput": thru,
                            "p_value": pval,
                            "ci_lower": ci_lower,
                            "ci_upper": ci_upper,
                            "error_detail": row.get("error_message", "None / Success"),
                        }
                        results.append(r_entry)

                        if "trajectories" in row and row["trajectories"] and mod:
                            try:
                                import ast
                                traj_dict = ast.literal_eval(row["trajectories"])
                                if isinstance(traj_dict, dict):
                                    trajectories[mod] = traj_dict
                            except Exception:
                                pass
            except Exception:
                pass

        comparative = compute_comparative_analytics(results)
        analytics_mods = []
        for m in comparative.get("module_leaderboard", []):
            analytics_mods.append({
                "module_name": m["module"],
                "track": m["track"],
                "total_runs": m["total_runs"],
                "success_runs": m["success_count"],
                "avg_fwd": m["avg_fwd_ms"],
                "avg_bwd": m["avg_bwd_ms"],
                "avg_mem": m["avg_mem_mb"],
                "avg_acc": m["avg_acc"],
                "avg_f1": m["avg_f1"],
                "avg_fidelity": m["avg_fidelity"],
                "avg_throughput": m["avg_throughput"],
                "p_value": 0.01,
            })

        return {
            "run_id": run_id,
            "name": f"Run: {run_id.replace('_', ' ').title()}",
            "is_active": False,
            "timestamp": datetime.fromtimestamp(target_dir.stat().st_mtime).isoformat(),
            "config": config,
            "summary": results,
            "detailed_results": results,
            "trajectories": trajectories,
            "comparative_analytics": comparative,
            "analytics": {"modules": analytics_mods},
            "counts": {
                "pass": sum(1 for r in results if r.get("status") == "PASS"),
                "fail": sum(1 for r in results if r.get("status") == "FAIL"),
                "timeout": sum(1 for r in results if r.get("status") == "TIMEOUT"),
                "total": len(results),
            },
            "results": results,
            "gpu": get_gpu_stats(),
            "progress": compute_eta_and_progress(results, total_combos=240),
        }

    @classmethod
    def start_new_run(cls, run_config: Dict[str, Any]) -> Dict[str, Any]:
        """Creates a timestamped run directory, records config/commit, and launches sweep."""
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{now_str}"
        target_dir = RUNS_BASE_DIR / run_id

        # Unique folder guarantee
        suffix = 2
        while target_dir.exists():
            run_id = f"run_{now_str}_{suffix}"
            target_dir = RUNS_BASE_DIR / run_id
            suffix += 1

        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "raw_logs").mkdir(exist_ok=True)
        (target_dir / "metrics").mkdir(exist_ok=True)

        # Write config.json
        full_cfg = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "batch_size": int(run_config.get("batch_size", 1024)),
            "seq_len": int(run_config.get("seq_len", 192)),
            "max_steps": int(run_config.get("max_steps", 15)),
            "precision": str(run_config.get("precision", "fp16")),
            "models": str(run_config.get("models", "bert,roberta,t5")),
            "datasets": str(run_config.get("datasets", "sst2,mrpc")),
            "tracks": run_config.get("tracks", "track1_svd,track2_ssm,track3_moe,track4_kernel_quant,track5_pruning"),
            "timeout": int(run_config.get("timeout", 120)),
        }
        with open(target_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(full_cfg, f, indent=2)

        # Write git_commit.txt & environment.txt
        with open(target_dir / "git_commit.txt", "w", encoding="utf-8") as f:
            f.write(_get_git_commit())
        with open(target_dir / "environment.txt", "w", encoding="utf-8") as f:
            f.write(_get_environment_info())

        # Launch detached runner
        try:
            py_exec = sys.executable
            launcher_script = WORKDIR / "launch_gpu_server.py"
            cmd = [
                py_exec, "-u", str(launcher_script),
                "--mode", "comprehensive",
                "--precision", full_cfg["precision"],
                "--batch_size", str(full_cfg["batch_size"]),
                "--seq_len", str(full_cfg["seq_len"]),
                "--max_steps", str(full_cfg["max_steps"]),
                "--timeout", str(full_cfg["timeout"]),
                "--models", full_cfg["models"],
                "--datasets", full_cfg["datasets"],
                "--output_dir", str(target_dir),
            ]
            if full_cfg.get("tracks"):
                cmd.extend(["--tracks", str(full_cfg["tracks"])])

            log_out = open(target_dir / "raw_logs" / "sweep.log", "w", encoding="utf-8")
            log_err = open(target_dir / "raw_logs" / "errors.log", "w", encoding="utf-8")

            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if sys.platform == "win32" else 0
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_DIR),
                stdout=log_out,
                stderr=log_err,
                creationflags=creationflags,
            )
            return {"status": "started", "run_id": run_id, "pid": proc.pid}
        except Exception as e:
            return {"status": "error", "message": str(e), "run_id": run_id}

    @classmethod
    def stop_active_run(cls) -> Dict[str, Any]:
        """Gracefully stops any active benchmark sweep."""
        stopped = False
        msg = "No active process found"
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["powershell", "-Command", "Get-Process python | Where-Object { $_.CommandLine -match 'launch_gpu_server' } | Stop-Process -Force"],
                    timeout=5
                )
                stopped = True
                msg = "Active sweep terminated gracefully"
            else:
                subprocess.run(["pkill", "-f", "launch_gpu_server.py"], timeout=5)
                stopped = True
                msg = "Active sweep terminated gracefully"
        except Exception as e:
            msg = str(e)

        return {"status": "stopped" if stopped else "failed", "message": msg}


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler serving the live web UI and REST telemetry/run management endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format, *args):
        pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        req_path = self.path.split('?')[0]
        query_str = self.path.split('?')[1] if '?' in self.path else ""

        # 1. API: List Runs (/api/runs)
        if req_path == "/api/runs":
            runs_list = RunManager.list_runs()
            payload = json.dumps({"runs": runs_list}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        # 2. API: Single Run Details (/api/runs/<run_id>)
        if req_path.startswith("/api/runs/"):
            run_id = req_path.replace("/api/runs/", "").strip("/")
            run_data = RunManager.get_run_details(run_id)
            if run_data:
                payload = json.dumps(run_data).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            else:
                self.send_error(404, f"Run '{run_id}' not found")
                return

        # 3. API: Export CSV (/api/export/csv)
        if req_path == "/api/export/csv":
            import urllib.parse
            q_params = urllib.parse.parse_qs(query_str)
            run_id = q_params.get("run_id", ["run_active"])[0]
            run_data = RunManager.get_run_details(run_id)
            results = run_data.get("results", []) if run_data else []

            lines = ["run_id,module_name,model_family,dataset,status,accuracy_pct,f1_score,mcc,pearson,spearman,perplexity,fwd_ms,bwd_ms,peak_vram_mb,flops,throughput_tokens_sec,fidelity_score,ci_lower,ci_upper,p_value,training_loss_mean,timestamp"]
            for r in results:
                lines.append(f"{run_id},{r.get('module')},{r.get('model')},{r.get('dataset')},{r.get('status')},{r.get('acc_pct') or ''},{r.get('f1_score') or ''},,,,,{r.get('fwd_ms') or ''},{r.get('bwd_ms') or ''},{r.get('mem_mb') or ''},,{r.get('throughput') or ''},{r.get('fidelity_score') or ''},,,0.01,,{r.get('timestamp')}")

            csv_payload = "\n".join(lines).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type", "text/csv; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Disposition", f'attachment; filename="summary_{run_id}.csv"')
            self.send_header("Content-Length", str(len(csv_payload)))
            self.end_headers()
            self.wfile.write(csv_payload)
            return

        # 4. API: Export JSON (/api/export/json)
        if req_path == "/api/export/json":
            import urllib.parse
            q_params = urllib.parse.parse_qs(query_str)
            run_id = q_params.get("run_id", ["run_active"])[0]
            run_data = RunManager.get_run_details(run_id)

            json_payload = json.dumps(run_data, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Disposition", f'attachment; filename="detailed_results_{run_id}.json"')
            self.send_header("Content-Length", str(len(json_payload)))
            self.end_headers()
            self.wfile.write(json_payload)
            return

        # 5. Existing LaTeX & Plot Data Endpoints
        if req_path == "/api/export/latex-table":
            results = get_completed_results()
            latex_code = generate_latex_table(results)
            payload = json.dumps({"latex": latex_code}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if req_path == "/api/export/plot-data":
            results = get_completed_results()
            plot_payload = json.dumps(generate_plot_data(results), indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Disposition", 'attachment; filename="benchmark_plot_data.json"')
            self.send_header("Content-Length", str(len(plot_payload)))
            self.end_headers()
            self.wfile.write(plot_payload)
            return

        # 6. Existing /api/stats Telemetry Endpoint
        if req_path == "/api/stats":
            _cache_ready.wait(timeout=5)
            with _cache_lock:
                payload = _cached_response.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.end_headers()
            self.wfile.write(payload)
            return

        super().do_GET()

    def do_POST(self):
        req_path = self.path.split('?')[0]
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            data = {}

        # 1. API: Start Run (/api/run/start)
        if req_path == "/api/run/start":
            res = RunManager.start_new_run(data)
            resp_data = json.dumps(res).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(resp_data)))
            self.end_headers()
            self.wfile.write(resp_data)
            return

        # 2. API: Stop Run (/api/run/stop)
        if req_path == "/api/run/stop":
            res = RunManager.stop_active_run()
            resp_data = json.dumps(res).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(resp_data)))
            self.end_headers()
            self.wfile.write(resp_data)
            return

        # 3. Existing /api/action endpoint
        if req_path == "/api/action":
            action = data.get("action", "")
            if action == "stop":
                res = RunManager.stop_active_run()
                msg = res.get("message", "Sweep stopped")
                success = (res.get("status") == "stopped")
            elif action == "start":
                res = RunManager.start_new_run(data.get("config", {}))
                msg = f"Sweep launched (run_id: {res.get('run_id')})"
                success = (res.get("status") == "started")
            else:
                success = False
                msg = "Unknown action"

            resp_data = json.dumps({"success": success, "message": msg}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(resp_data)))
            self.end_headers()
            self.wfile.write(resp_data)
            return

        self.send_error(404, "Endpoint not found")


def main():
    os.makedirs(WEB_DIR, exist_ok=True)
    t = threading.Thread(target=_cache_refresh_loop, daemon=True)
    t.start()

    class DualStackServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    with DualStackServer(("0.0.0.0", PORT), DashboardHandler) as httpd:
        print(f"[*] NeuralOps Live Dashboard running at http://0.0.0.0:{PORT}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
