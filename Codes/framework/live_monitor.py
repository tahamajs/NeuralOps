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

PORT = 8080
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
        "nvidia-smi",
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ]
    
    for smi in nvsmi_candidates:
        try:
            cmd = [
                smi,
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
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
    """Parses comprehensive_run_detached.log and returns structured results with error details."""
    results = []
    error_map = {}
    
    log_candidates = [MASTER_LOG, Path("comprehensive_run_detached.log"), PROJECT_DIR / "comprehensive_run_detached.log"]
    target_log = None
    for p in log_candidates:
        if p.exists():
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
                
                fwd_m = re.search(r'fwd:\s*([0-9.]+)ms', details)
                if fwd_m:
                    fwd_ms = fwd_m.group(1)
                    
                bwd_m = re.search(r'bwd:\s*([0-9.]+)ms', details)
                if bwd_m:
                    bwd_ms = bwd_m.group(1)
                    
                mem_m = re.search(r'mem:\s*([0-9.]+)MB', details)
                if mem_m:
                    mem_mb = mem_m.group(1)

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


def compute_eta_and_progress(results: List[Dict[str, Any]], total_combos: int = 720, window_size: int = 50) -> Dict[str, Any]:
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

        # 1. Module Aggregation
        if mod not in module_stats:
            module_stats[mod] = {
                "module": mod,
                "track": track,
                "fwd_list": [],
                "bwd_list": [],
                "mem_list": [],
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
        elif status == "TIMEOUT":
            m_entry["timeout_count"] += 1
        else:
            m_entry["fail_count"] += 1

        if model not in m_entry["by_model"]:
            m_entry["by_model"][model] = {"fwd": [], "bwd": [], "mem": []}
        if fwd is not None:
            m_entry["by_model"][model]["fwd"].append(fwd)
        if bwd is not None:
            m_entry["by_model"][model]["bwd"].append(bwd)
        if mem is not None:
            m_entry["by_model"][model]["mem"].append(mem)

        # 2. Model Family Aggregation
        if model not in model_stats:
            model_stats[model] = {
                "model": model,
                "fwd_list": [],
                "bwd_list": [],
                "mem_list": [],
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

        # 3. Track Aggregation
        if track != "unknown":
            if track not in track_stats:
                track_stats[track] = {
                    "track": track,
                    "fwd_list": [],
                    "bwd_list": [],
                    "mem_list": [],
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

    # Format module leaderboard list
    leaderboard = []
    for mod, data in module_stats.items():
        avg_fwd = round(sum(data["fwd_list"]) / len(data["fwd_list"]), 2) if data["fwd_list"] else 0.0
        avg_bwd = round(sum(data["bwd_list"]) / len(data["bwd_list"]), 2) if data["bwd_list"] else 0.0
        avg_mem = round(sum(data["mem_list"]) / len(data["mem_list"]), 1) if data["mem_list"] else 0.0
        total_lat = round(avg_fwd + avg_bwd, 2)
        succ_rate = round((data["success_count"] / max(1, data["total_runs"])) * 100.0, 1)

        leaderboard.append({
            "module": mod,
            "track": data["track"],
            "avg_fwd_ms": avg_fwd,
            "avg_bwd_ms": avg_bwd,
            "avg_total_ms": total_lat,
            "avg_mem_mb": avg_mem,
            "success_count": data["success_count"],
            "timeout_count": data["timeout_count"],
            "fail_count": data["fail_count"],
            "total_runs": data["total_runs"],
            "success_rate_pct": succ_rate,
        })

    # Sort leaderboard: primary = avg_fwd_ms ascending (put 0/incomplete at end)
    leaderboard.sort(key=lambda x: (x["avg_fwd_ms"] == 0, x["avg_fwd_ms"]))

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
        cmd = ["powershell", "-Command", "Get-Process python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        return bool(res.stdout.strip())
    except Exception:
        return False


def _build_stats_response() -> str:
    """Build the full stats JSON response with rich detail fields and comparative analytics."""
    gpu = get_gpu_stats()
    results = get_completed_results()
    progress = compute_eta_and_progress(results, total_combos=720, window_size=50)
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

    response = {
        "gpu": gpu,
        "progress": progress,
        "is_running": is_running,
        "recent_results": serializable_results[-20:][::-1],
        "all_results_count": len(results),
        "counts": {
            "pass": sum(1 for r in results if r["status"] == "PASS"),
            "timeout": sum(1 for r in results if r["status"] == "TIMEOUT"),
            "fail": sum(1 for r in results if r["status"] == "FAIL"),
            "total": len(results),
        },
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


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler serving the live web UI and JSON telemetry endpoints."""

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
        if req_path == "/api/action":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length > 0 else b"{}"
            try:
                data = json.loads(body.decode("utf-8"))
            except Exception:
                data = {}

            action = data.get("action", "")
            success = False
            msg = "Unknown action"

            if action == "stop":
                try:
                    subprocess.run(
                        ["powershell", "-Command", "Get-Process python | Where-Object { $_.CommandLine -match 'launch_gpu_server' } | Stop-Process -Force"],
                        timeout=5
                    )
                    success = True
                    msg = "Sweep process terminated"
                except Exception as e:
                    msg = str(e)

            elif action == "start":
                try:
                    bat_path = PROJECT_DIR / "run_comprehensive.bat"
                    if bat_path.exists():
                        subprocess.Popen(
                            ["cmd.exe", "/c", str(bat_path)],
                            cwd=str(PROJECT_DIR),
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if sys.platform == "win32" else 0
                        )
                        success = True
                        msg = "Sweep launched in background"
                    else:
                        msg = "run_comprehensive.bat not found"
                except Exception as e:
                    msg = str(e)

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
