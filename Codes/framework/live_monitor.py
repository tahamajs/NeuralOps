#!/usr/bin/env python3
"""
Real-Time Live Dashboard & Monitoring Server for GPU Benchmark Sweep.
Supports interactive error inspection, detail inspection modal, and real-time telemetry.
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

    ram_used, ram_total = 0.0, 0.0
    if psutil:
        try:
            ram_used = round(psutil.virtual_memory().used / (1024**3), 1)
            ram_total = round(psutil.virtual_memory().total / (1024**3), 1)
        except Exception:
            pass

    return {
        "gpu_util": gpu_util,
        "mem_used_mb": mem_used,
        "mem_total_mb": mem_total,
        "mem_pct": round((mem_used / max(1.0, mem_total)) * 100, 1),
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
        lines = target_log.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines:
            line_str = line.strip()

            # Catch explicit failure error messages
            fe_match = FAIL_ERROR_REGEX.match(line_str)
            if fe_match:
                _, mod, model, dataset, err_msg = fe_match.groups()
                error_map[(mod.lower(), model.lower(), dataset.lower())] = err_msg
                continue
            
            # Standard result match
            m = RESULT_REGEX.match(line_str)
            if m:
                ts_str, track, module, model, dataset, status, details = m.groups()
                try:
                    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")
                except Exception:
                    dt = datetime.now()
                
                fwd, bwd, mem = "-", "-", "-"
                fwd_m = re.search(r'fwd:\s*([\d\.]+)ms', details)
                bwd_m = re.search(r'bwd:\s*([\d\.]+)ms', details)
                mem_m = re.search(r'mem:\s*([\d\.]+)MB', details)
                
                if fwd_m: fwd = fwd_m.group(1)
                if bwd_m: bwd = bwd_m.group(1)
                if mem_m: mem = mem_m.group(1)

                key = (module.lower(), model.lower(), dataset.lower())
                err_detail = "None / Success"
                if status == "FAIL":
                    err_detail = error_map.get(key, "Evaluation failed during forward/backward pass (Check log).")
                elif status == "TIMEOUT":
                    err_detail = "Execution exceeded hard time limit (>120s)."

                results.append({
                    "timestamp": dt,
                    "track": track,
                    "module": module,
                    "model": model.upper(),
                    "dataset": dataset.upper(),
                    "status": status,
                    "fwd_ms": fwd,
                    "bwd_ms": bwd,
                    "mem_mb": mem,
                    "error_detail": err_detail,
                })
                continue

            # Timeout match
            tm = TIMEOUT_REGEX.match(line_str)
            if tm:
                ts_str, module, model, dataset, timeout_len = tm.groups()
                try:
                    dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S,%f")
                except Exception:
                    dt = datetime.now()

                to_str = timeout_len or "120s"
                results.append({
                    "timestamp": dt,
                    "track": "track1_svd",
                    "module": module,
                    "model": model.upper(),
                    "dataset": dataset.upper(),
                    "status": "TIMEOUT",
                    "fwd_ms": f">{to_str}",
                    "bwd_ms": "-",
                    "mem_mb": "-",
                    "error_detail": f"Execution exceeded hard timeout ({to_str}). Task was killed safely to prevent sweep stall.",
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
            "eta_formatted": "Calculating...",
            "eta_seconds": 0
        }

    window = min(window_size, completed - 1)
    recent = results[-(window + 1):]
    
    t_start = recent[0]["timestamp"]
    t_end = recent[-1]["timestamp"]
    elapsed_sec = (t_end - t_start).total_seconds()

    if elapsed_sec <= 0 or window == 0:
        return {
            "completed": completed,
            "total": total_combos,
            "percent": pct,
            "avg_sec": 0.0,
            "eta_formatted": "Calculating...",
            "eta_seconds": 0
        }

    avg_sec = elapsed_sec / window
    eta_sec = remaining * avg_sec

    hrs = int(eta_sec // 3600)
    mins = int((eta_sec % 3600) // 60)
    secs = int(eta_sec % 60)

    if hrs > 0:
        eta_str = f"{hrs}h {mins}m"
    elif mins > 0:
        eta_str = f"{mins}m {secs}s"
    else:
        eta_str = f"{secs}s"

    return {
        "completed": completed,
        "total": total_combos,
        "percent": pct,
        "avg_sec": round(avg_sec, 2),
        "eta_formatted": eta_str,
        "eta_seconds": round(eta_sec, 1)
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
    """Build the full stats JSON response with rich detail fields."""
    gpu = get_gpu_stats()
    results = get_completed_results()
    progress = compute_eta_and_progress(results, total_combos=720, window_size=50)
    is_running = is_sweep_running()

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
        "recent_results": serializable_results[-15:][::-1],
        "all_results_count": len(results),
        "counts": {
            "pass": sum(1 for r in results if r["status"] == "PASS"),
            "timeout": sum(1 for r in results if r["status"] == "TIMEOUT"),
            "fail": sum(1 for r in results if r["status"] == "FAIL"),
            "total": len(results),
        }
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
            self.end_headers()
            self.wfile.write(payload)
            return

        if req_path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        return super().do_GET()

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
            msg = ""

            if action == "stop":
                try:
                    cmd = 'powershell -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match \\"run_comprehensive\\" -or $_.CommandLine -match \\"launch_gpu_server\\" } | Invoke-CimMethod -MethodName Terminate; Stop-Process -Name python -Force -ErrorAction SilentlyContinue"'
                    subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
                    success = True
                    msg = "Sweep terminated."
                except Exception as e:
                    msg = f"Failed to stop: {e}"

            elif action == "start":
                try:
                    bat_path = PROJECT_DIR / "run_comprehensive.bat"
                    cmd = f'wmic process call create "cmd.exe /c \\"{bat_path}\\""'
                    subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
                    success = True
                    msg = "Sweep started in detached mode."
                except Exception as e:
                    msg = f"Failed to start: {e}"

            resp = json.dumps({"success": success, "message": msg}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return

        self.send_response(404)
        self.end_headers()


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
