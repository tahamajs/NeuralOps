#!/usr/bin/env python3
"""
Zero-Bloat Modular Live Experiment Monitor & Dashboard
Serves modular static files (web/index.html, web/app.js, web/styles.css)
and provides real-time JSON API /api/stats.
Background cache thread: stats refreshed every 3s so HTTP requests are instant.
"""
import http.server
import json
import os
import re
import socketserver
import subprocess
import threading
import time
from pathlib import Path

try:
    import psutil
except ImportError:
    psutil = None

# ── Global response cache ─────────────────────────────────────────────────────
_cache_lock = threading.Lock()
_cached_response: str = "{}"
_cache_ready = threading.Event()

PORT = 8080
WORKDIR = Path(__file__).resolve().parent
PROJECT_DIR = WORKDIR.parent.parent
WEB_DIR = WORKDIR / "web"
MASTER_LOG = PROJECT_DIR / "trajectory_master.log"
RESULTS_DIR = WORKDIR / "experiments"
WORKER_STATE_FILE = WORKDIR / "worker_status.json"

try:
    import sys
    if str(PROJECT_DIR / "Codes") not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR / "Codes"))
    if str(WORKDIR) not in sys.path:
        sys.path.insert(0, str(WORKDIR))
    from framework.attentions import ATTENTION_REGISTRY
    ATTN_MODELS = [(k, k.replace('_', ' ').title()) for k in sorted(ATTENTION_REGISTRY.keys())]
except Exception:
    ATTN_MODELS = [
        ("svd_linear", "SVD Linear Attention"),
        ("svd", "SVD Base Attention"),
        ("full", "Standard Multi-Head Attention"),
        ("lora_attn", "LoRA Low-Rank Attention"),
        ("performer", "Performer (FAVOR+)"),
        ("nystrom", "Nyströmformer"),
        ("linformer", "Linformer"),
        ("flash3", "FlashAttention-3"),
        ("mamba", "Mamba SSM Attention"),
        ("gated_delta_net", "Gated DeltaNet"),
        ("adasplash", "AdaSplash Attention"),
        ("aegis_v3", "Aegis-v3 Attention"),
        ("griffin_ssd", "Griffin SSD"),
        ("spectra_v2", "Spectra-v2 Attention"),
        ("samba_m3", "Samba-M3"),
    ]

TASKS = ["sst2"]


def get_gpu_stats():
    gpu_util, mem_used, mem_total, temp, power = 0.0, 0.0, 8188.0, 0.0, 0.0
    try:
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            gpu_util = float(parts[0]) if len(parts) > 0 else 0.0
            mem_used = float(parts[1]) if len(parts) > 1 else 0.0
            mem_total = float(parts[2]) if len(parts) > 2 else 8188.0
            temp = float(parts[3]) if len(parts) > 3 else 0.0
            power = float(parts[4]) if len(parts) > 4 else 0.0
    except Exception:
        pass

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
        "temp_c": temp,
        "power_w": power,
        "sys_ram_used_gb": ram_used,
        "sys_ram_total_gb": ram_total,
    }


def get_storage_stats():
    total_bytes = 0
    file_count = 0
    if RESULTS_DIR.exists():
        for p in RESULTS_DIR.rglob("*"):
            if p.is_file():
                file_count += 1
                total_bytes += p.stat().st_size
    return {
        "used_mb": round(total_bytes / (1024 * 1024), 2),
        "file_count": file_count,
        "policy": "Zero-Bloat (Metrics Only)",
    }


def get_active_workers():
    workers = {}
    if WORKER_STATE_FILE.exists():
        try:
            with open(WORKER_STATE_FILE, "r", encoding="utf-8", errors="ignore") as f:
                workers = json.load(f)
        except Exception:
            pass
    if not workers:
        for i in (1, 2, 3):
            workers[str(i)] = {"status": "idle", "attn": "-", "task": "-", "idx": "--"}
    return workers


def get_completed_results():
    results = {}
    if MASTER_LOG.exists():
        try:
            lines = MASTER_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in lines:
                if "<< SUCCESS:" in line and "attn=" in line and "task=" in line:
                    m = re.search(r"attn=(\w+) \| task=(\w+) \| Acc: ([\d\.]+)% \| F1: ([\d\.]+)% \| Loss: ([\d\.]+) \| Time: ([\d\.]+)s", line)
                    if m:
                        k = f"{m.group(1)}_{m.group(2)}_s42"
                        results[k] = {
                            "attn": m.group(1),
                            "task": m.group(2),
                            "acc": float(m.group(3)),
                            "f1": float(m.group(4)),
                            "loss": float(m.group(5)),
                            "time_s": float(m.group(6)),
                            "status": "completed"
                        }
        except Exception:
            pass
    return results


def get_trajectory_matrix(completed_results, active_workers):
    running_keys = set()
    for w in active_workers.values():
        if w.get("status") == "running" and w.get("attn") != "-" and w.get("task") != "-":
            running_keys.add(f"{w['attn']}_{w['task']}_s42")

    matrix = []
    for task in TASKS:
        runs = []
        for attn_id, attn_name in ATTN_MODELS:
            key = f"{attn_id}_{task}_s42"
            status = "queued"
            metrics = None
            if key in completed_results:
                status = "completed"
                metrics = completed_results[key]
            elif key in running_keys:
                status = "running"
            runs.append({
                "attn_id": attn_id,
                "attn_name": attn_name,
                "task": task,
                "key": key,
                "status": status,
                "metrics": metrics,
            })
        matrix.append({"task": task, "runs": runs})
    return matrix


def get_all_active_loss_histories(workers):
    histories = []
    
    for wid, w in workers.items():
        if w.get("status") == "running":
            attn = w.get("attn", "")
            task = w.get("task", "")
            import glob
            target_pattern = str(RESULTS_DIR / f"{attn}_{task}_*" / "history" / "log_history.json")
            matches = glob.glob(target_pattern)
            
            if matches:
                # Sort by mtime to get the most recent one if multiple exist
                matches.sort(key=os.path.getmtime, reverse=True)
                hist_file = Path(matches[0])
                try:
                    data = json.loads(hist_file.read_text(encoding="utf-8", errors="ignore"))
                    histories.append({
                        "worker_id": wid,
                        "active_model": attn,
                        "active_task": task,
                        "loss_history": data[-30:] if len(data) > 30 else data
                    })
                except Exception:
                    pass
    return histories


def _build_stats_response() -> str:
    """Build the full stats JSON response. Called by background thread only."""
    gpu = get_gpu_stats()
    storage = get_storage_stats()
    workers = get_active_workers()
    results = get_completed_results()
    matrix = get_trajectory_matrix(results, workers)
    train_multi = get_all_active_loss_histories(workers)

    logs = []
    if MASTER_LOG.exists():
        try:
            logs = MASTER_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()[-25:]
        except Exception:
            pass

    gpu["0"] = gpu.get("gpu_util", 0.0)
    gpu["1"] = gpu.get("mem_used_mb", 0.0)
    gpu["2"] = gpu.get("mem_total_mb", 8188.0)
    gpu["3"] = gpu.get("temp_c", 0.0)
    gpu["4"] = gpu.get("power_w", 0.0)
    gpu["5"] = gpu.get("sys_ram_used_gb", 0.0)

    total_completed = len(results)
    first_active = train_multi[0] if (train_multi and len(train_multi) > 0) else {}
    train_obj = {
        "active_model": first_active.get("active_model", "SVD"),
        "active_task": first_active.get("active_task", "SST2"),
        "active_index": f"{total_completed}/1008",
        "loss_history": first_active.get("loss_history", [])
    }

    response = {
        "gpu": gpu,
        "train": train_obj,
        "storage": storage,
        "workers": workers,
        "results": results,
        "matrix": matrix,
        "train_multi": train_multi,
        "logs": logs,
    }
    return json.dumps(response).replace("NaN", "null").replace("Infinity", "null")


def _cache_refresh_loop():
    """Background thread: refresh cached stats every 3 seconds."""
    global _cached_response
    while True:
        try:
            data = _build_stats_response()
            with _cache_lock:
                _cached_response = data
            _cache_ready.set()
        except Exception:
            pass
        time.sleep(3)


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress per-request stdout noise

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        req_path = self.path.split('?')[0]

        if self.path == "/api/stats":
            _cache_ready.wait(timeout=10)
            with _cache_lock:
                payload = _cached_response.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if req_path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return

        if req_path in ("/", "/index.html"):
            html_candidates = [
                WEB_DIR / "index.html",
                WORKDIR / "web" / "index.html",
                PROJECT_DIR / "Codes" / "framework" / "web" / "index.html",
                Path("Codes/framework/web/index.html").resolve(),
                Path("web/index.html").resolve(),
                Path("index.html").resolve(),
            ]
            content = None
            for cand in html_candidates:
                if cand.exists():
                    content = cand.read_bytes()
                    break
            if content is None:
                content = b"<!DOCTYPE html><html><body><h1>Dashboard Initializing...</h1><p>Please refresh in 3 seconds.</p></body></html>"
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if req_path in ("/styles.css", "/style.css"):
            for candidate in [WEB_DIR / "style.css", WEB_DIR / "styles.css", PROJECT_DIR / "Docs" / "style.css"]:
                if candidate.exists():
                    self.send_response(200)
                    self.send_header("Content-type", "text/css; charset=utf-8")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(candidate.read_bytes())
                    return

        # Universal static files lookup across WEB_DIR, Docs, and PROJECT_DIR
        for base in (WEB_DIR, PROJECT_DIR / "Docs", PROJECT_DIR):
            target = base / req_path.lstrip('/')
            if target.is_file():
                self.send_response(200)
                ct = "text/plain; charset=utf-8"
                if req_path.endswith(".md"): ct = "text/markdown; charset=utf-8"
                elif req_path.endswith(".json"): ct = "application/json; charset=utf-8"
                elif req_path.endswith(".html"): ct = "text/html; charset=utf-8"
                elif req_path.endswith(".js"): ct = "application/javascript; charset=utf-8"
                elif req_path.endswith(".css"): ct = "text/css; charset=utf-8"
                elif req_path.endswith(".svg"): ct = "image/svg+xml"
                elif req_path.endswith(".png"): ct = "image/png"
                elif req_path.endswith(".ico"): ct = "image/x-icon"
                self.send_header("Content-type", ct)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(target.read_bytes())
                return

        super().do_GET()

    def do_POST(self):
        if self.path == "/api/action":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            try:
                req = json.loads(post_data.decode("utf-8"))
                action = req.get("action")
                msg = "Unknown action"
                if action == "stop":
                    # Kill only the orchestrator and its workers (avoid killing this monitor)
                    kill_cmd = "Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -match 'run_suite_multiprocess.py' } | ForEach-Object { $_.Terminate() }"
                    subprocess.run(["powershell", "-Command", kill_cmd])
                    # Update worker status to stopped
                    with open(WORKER_STATE_FILE, "w", encoding="utf-8") as f:
                        json.dump({str(i): {"status": "stopped", "attn": "-", "task": "-", "idx": "--"} for i in (1, 2, 3)}, f)
                    msg = "Orchestrator successfully stopped. All GPU workers terminated."
                elif action == "start":
                    # Run the scheduled task to restart the orchestrator
                    subprocess.run(["schtasks", "/run", "/tn", "RunOrchestrator"])
                    msg = "Orchestrator triggered successfully via schtasks."
                
                self.send_response(200)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "message": msg}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
            return

        if self.path == "/api/settings":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            try:
                req = json.loads(post_data.decode("utf-8"))
                with open(PROJECT_DIR / "orchestrator_settings.json", "w", encoding="utf-8") as f:
                    json.dump(req, f)
                self.send_response(200)
                self.send_header("Content-type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "message": "Settings updated!"}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
            return
        
        self.send_response(404)
        self.end_headers()


def main():
    # Start background cache thread so /api/stats responds instantly
    t = threading.Thread(target=_cache_refresh_loop, daemon=True)
    t.start()

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", PORT), DashboardHandler) as httpd:
        print(f"Modular Live Analytics Hub active on http://0.0.0.0:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
