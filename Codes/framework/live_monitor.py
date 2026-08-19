#!/usr/bin/env python3
"""
Self-Contained Live GPU Experiment & Telemetry Hub
Zero CDN dependencies, Built-in Canvas charting, Full CORS & OPTIONS handling.
"""
import http.server
import json
import os
import re
import socketserver
import subprocess
import sys
from pathlib import Path

PORT = 8080
PROJECT_DIR = Path("C:/Users/Hadi Schumacher/SVD_Project")
WORKDIR = PROJECT_DIR / "Codes/framework"
RESULTS_DIR = WORKDIR / "results"
MASTER_LOG = PROJECT_DIR / "trajectory_master.log"


def get_gpu_stats():
    try:
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            return {
                "gpu_util": float(parts[0]) if len(parts) > 0 else 0,
                "mem_used_mb": float(parts[1]) if len(parts) > 1 else 0,
                "mem_total_mb": float(parts[2]) if len(parts) > 2 else 8188,
                "temp_c": float(parts[3]) if len(parts) > 3 else 0,
            }
    except Exception:
        pass
    return {"gpu_util": 0, "mem_used_mb": 0, "mem_total_mb": 8188, "temp_c": 0}


def get_active_training_info():
    active_model = "Active / Queued"
    active_task = "SST-2"
    step_info = {"step": 0, "total_steps": 2105, "loss": 0.0, "epoch": 0.0}
    loss_history = []
    
    if MASTER_LOG.exists():
        try:
            lines = MASTER_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in reversed(lines):
                m = re.search(r"\[(\d+/\d+)\] >> Launching: attn=(\w+) \| task=(\w+)", line)
                if m:
                    active_model = f"{m.group(2)} ({m.group(1)})"
                    active_task = m.group(3)
                    break
        except Exception:
            pass

    if RESULTS_DIR.exists():
        for d in sorted(RESULTS_DIR.iterdir(), key=os.path.getmtime, reverse=True):
            if d.is_dir():
                hist_file = d / "train_history.json"
                if hist_file.exists():
                    try:
                        hist = json.loads(hist_file.read_text(encoding="utf-8"))
                        for h in hist:
                            if "loss" in h:
                                loss_history.append({
                                    "epoch": h.get("epoch", 0),
                                    "step": h.get("step", 0),
                                    "loss": h.get("loss", 0),
                                    "learning_rate": h.get("learning_rate", 0),
                                    "gpu_mem": h.get("gpu_memory_allocated", 0),
                                })
                        if hist:
                            last = hist[-1]
                            step_info["loss"] = last.get("loss", 0)
                            step_info["epoch"] = last.get("epoch", 0)
                            step_info["step"] = last.get("step", 0)
                    except Exception:
                        pass
                break

    return {
        "active_model": active_model,
        "active_task": active_task,
        "step_info": step_info,
        "loss_history": loss_history[-60:],
    }


def get_completed_results():
    results = []
    if RESULTS_DIR.exists():
        for d in RESULTS_DIR.iterdir():
            if d.is_dir():
                eval_file = d / "eval_final.json"
                sys_file = d / "system_info.json"
                if eval_file.exists():
                    try:
                        ev = json.loads(eval_file.read_text(encoding="utf-8"))
                        sys_info = json.loads(sys_file.read_text(encoding="utf-8")) if sys_file.exists() else {}
                        results.append({
                            "name": d.name,
                            "accuracy": ev.get("eval_accuracy", ev.get("accuracy", 0)),
                            "f1": ev.get("eval_f1_macro", ev.get("f1", 0)),
                            "loss": ev.get("eval_loss", 0),
                            "runtime_s": sys_info.get("train_runtime_s", ev.get("eval_runtime", 0)),
                            "params": sys_info.get("n_params", 0),
                        })
                    except Exception:
                        pass
    return sorted(results, key=lambda x: x.get("accuracy", 0), reverse=True)


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVD Linear Attention — Live GPU Experiment Hub</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: #030712;
            color: #f3f4f6;
            padding: 24px;
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; }
        .glass {
            background: rgba(17, 24, 39, 0.85);
            border: 1px solid rgba(55, 65, 81, 0.6);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #06b6d4;
        }
        .title { font-size: 24px; font-weight: 800; color: #38bdf8; }
        .badge {
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.4);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
        }
        .grid-4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }
        .stat-card {
            background: rgba(31, 41, 55, 0.6);
            border: 1px solid rgba(75, 85, 99, 0.4);
            border-radius: 12px;
            padding: 16px;
        }
        .stat-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: #9ca3af; font-weight: 600; }
        .stat-val { font-size: 28px; font-weight: 800; font-family: monospace; margin: 8px 0; color: #06b6d4; }
        .progress-bg { width: 100%; height: 8px; background: #374151; border-radius: 4px; overflow: hidden; margin-top: 8px; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #06b6d4, #3b82f6); width: 0%; transition: width 0.4s; }
        .progress-fill-purple { height: 100%; background: linear-gradient(90deg, #a855f7, #ec4899); width: 0%; transition: width 0.4s; }
        .stat-sub { font-size: 11px; color: #9ca3af; display: flex; justify-content: space-between; margin-top: 6px; }
        
        .main-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
        @media(max-width: 850px) { .main-grid { grid-template-columns: 1fr; } }
        
        .chart-box { height: 260px; position: relative; width: 100%; }
        canvas { width: 100%; height: 100%; display: block; }
        .logs-box {
            background: #000;
            border: 1px solid #374151;
            border-radius: 10px;
            padding: 12px;
            font-family: monospace;
            font-size: 11px;
            color: #d1d5db;
            height: 260px;
            overflow-y: auto;
            line-height: 1.5;
        }
        
        table { width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; margin-top: 10px; }
        th { font-size: 11px; text-transform: uppercase; color: #9ca3af; padding: 10px 14px; border-bottom: 1px solid #374151; font-family: monospace; }
        td { padding: 10px 14px; border-bottom: 1px solid rgba(55, 65, 81, 0.4); font-family: monospace; }
        tr:hover { background: rgba(55, 65, 81, 0.2); }
        .acc { color: #34d399; font-weight: 700; }
        .f1 { color: #818cf8; font-weight: 700; }
        .loss { color: #fbbf24; }
    </style>
</head>
<body>
    <div class="container">
        <!-- Top Banner -->
        <div class="glass header">
            <div>
                <div class="title">SVD Linear Attention — Live GPU Hub</div>
                <div style="font-size: 12px; color: #9ca3af; margin-top: 4px;">
                    NVIDIA RTX 4070 Laptop GPU (8.6GB VRAM) • FP16 CUDA Engine • 45 Trajectory Matrix
                </div>
            </div>
            <div class="badge" id="live-badge">TRAINING ACTIVE</div>
        </div>

        <!-- Telemetry Cards -->
        <div class="glass">
            <div class="grid-4">
                <div class="stat-card">
                    <div class="stat-label">GPU Core Utilization</div>
                    <div class="stat-val" id="gpu-util">0%</div>
                    <div class="progress-bg"><div id="gpu-bar" class="progress-fill"></div></div>
                    <div class="stat-sub"><span>Temp: <b id="gpu-temp" style="color:#fff">0°C</b></span><span>Target: 100%</span></div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">VRAM Allocation</div>
                    <div class="stat-val" id="vram-val" style="color: #c084fc;">0 / 8.2 GB</div>
                    <div class="progress-bg"><div id="vram-bar" class="progress-fill-purple"></div></div>
                    <div class="stat-sub"><span>FP16 CUDA</span><span id="vram-pct" style="color:#c084fc">0%</span></div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">Active Architecture</div>
                    <div class="stat-val" id="active-model" style="color: #fbbf24; font-size: 20px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Loading...</div>
                    <div class="stat-sub"><span>Task: <b id="active-task" style="color:#fbbf24">-</b></span><span>Epoch: <b id="active-epoch" style="color:#fff">-</b></span></div>
                </div>

                <div class="stat-card">
                    <div class="stat-label">Step Training Loss</div>
                    <div class="stat-val" id="curr-loss" style="color: #34d399;">0.0000</div>
                    <div class="stat-sub"><span>Step: <b id="curr-step" style="color:#fff">-</b></span><span>Storage: Zero Bloat</span></div>
                </div>
            </div>
        </div>

        <!-- Chart & Log Row -->
        <div class="main-grid">
            <div class="glass">
                <div style="font-weight: 700; font-size: 14px; margin-bottom: 12px; display: flex; justify-content: space-between;">
                    <span>Live Loss Trajectory</span>
                    <span style="font-size: 11px; color: #9ca3af; font-family: monospace;">Auto-refresh 1.5s</span>
                </div>
                <div class="chart-box">
                    <canvas id="lossCanvas"></canvas>
                </div>
            </div>

            <div class="glass">
                <div style="font-weight: 700; font-size: 14px; margin-bottom: 12px;">Live Trajectory Stream</div>
                <div class="logs-box" id="log-box">
                    <div style="color: #06b6d4;">> Live telemetry connection initialized...</div>
                </div>
            </div>
        </div>

        <!-- Results Table -->
        <div class="glass">
            <div style="font-weight: 700; font-size: 14px; display: flex; justify-content: space-between;">
                <span>Completed Experiments & Metrics</span>
                <span style="font-size: 11px; color: #9ca3af; font-family: monospace;" id="completed-count">0 runs completed</span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Experiment Name</th>
                        <th>Accuracy</th>
                        <th>F1 Macro</th>
                        <th>Final Loss</th>
                        <th>Runtime</th>
                    </tr>
                </thead>
                <tbody id="results-body">
                    <tr><td colspan="5" style="text-align: center; color: #6b7280; padding: 20px;">Waiting for completed trajectories...</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('lossCanvas');
        const ctx = canvas.getContext('2d');
        
        function resizeCanvas() {
            canvas.width = canvas.parentElement.clientWidth;
            canvas.height = canvas.parentElement.clientHeight;
        }
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        function drawLossChart(points) {
            resizeCanvas();
            const w = canvas.width;
            const h = canvas.height;
            ctx.clearRect(0, 0, w, h);

            if (!points || points.length < 2) {
                ctx.fillStyle = '#6b7280';
                ctx.font = '12px monospace';
                ctx.fillText('Streaming live training loss points...', 20, h / 2);
                return;
            }

            const losses = points.map(p => p.loss);
            const minL = Math.max(0, Math.min(...losses) * 0.9);
            const maxL = Math.max(...losses) * 1.1 || 1.0;

            // Grid lines
            ctx.strokeStyle = 'rgba(55, 65, 81, 0.3)';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {
                const y = h - 20 - (i / 4) * (h - 40);
                ctx.beginPath();
                ctx.moveTo(40, y);
                ctx.lineTo(w - 10, y);
                ctx.stroke();

                const val = minL + (i / 4) * (maxL - minL);
                ctx.fillStyle = '#9ca3af';
                ctx.font = '10px monospace';
                ctx.fillText(val.toFixed(2), 5, y + 3);
            }

            // Draw line
            ctx.strokeStyle = '#06b6d4';
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            points.forEach((p, i) => {
                const x = 40 + (i / (points.length - 1)) * (w - 50);
                const y = h - 20 - ((p.loss - minL) / (maxL - minL)) * (h - 40);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();

            // Gradient Fill
            const grad = ctx.createLinearGradient(0, 0, 0, h);
            grad.addColorStop(0, 'rgba(6, 182, 212, 0.25)');
            grad.addColorStop(1, 'rgba(6, 182, 212, 0.0)');
            ctx.fillStyle = grad;
            ctx.lineTo(w - 10, h - 20);
            ctx.lineTo(40, h - 20);
            ctx.closePath();
            ctx.fill();
        }

        async function updateDashboard() {
            try {
                const res = await fetch('/api/stats');
                if (!res.ok) return;
                const data = await res.json();

                // GPU
                document.getElementById('gpu-util').innerText = data.gpu.gpu_util + '%';
                document.getElementById('gpu-bar').style.width = data.gpu.gpu_util + '%';
                document.getElementById('gpu-temp').innerText = data.gpu.temp_c + '°C';

                const memUsedGb = (data.gpu.mem_used_mb / 1024).toFixed(1);
                const memTotalGb = (data.gpu.mem_total_mb / 1024).toFixed(1);
                const vramPct = Math.round((data.gpu.mem_used_mb / data.gpu.mem_total_mb) * 100);
                document.getElementById('vram-val').innerText = `${memUsedGb} / ${memTotalGb} GB`;
                document.getElementById('vram-bar').style.width = vramPct + '%';
                document.getElementById('vram-pct').innerText = vramPct + '%';

                // Training
                document.getElementById('active-model').innerText = data.train.active_model;
                document.getElementById('active-task').innerText = data.train.active_task.toUpperCase();
                document.getElementById('active-epoch').innerText = typeof data.train.step_info.epoch === 'number' ? data.train.step_info.epoch.toFixed(2) : '-';
                document.getElementById('curr-loss').innerText = typeof data.train.step_info.loss === 'number' ? data.train.step_info.loss.toFixed(4) : '0.0000';
                document.getElementById('curr-step').innerText = data.train.step_info.step;

                // Chart
                if (data.train.loss_history) {
                    drawLossChart(data.train.loss_history);
                }

                // Leaderboard
                if (data.results && data.results.length > 0) {
                    document.getElementById('completed-count').innerText = data.results.length + ' runs completed';
                    document.getElementById('results-body').innerHTML = data.results.map(r => `
                        <tr>
                            <td style="color:#38bdf8; font-weight:600;">${r.name}</td>
                            <td class="acc">${(r.accuracy * 100).toFixed(2)}%</td>
                            <td class="f1">${(r.f1 * 100).toFixed(2)}%</td>
                            <td class="loss">${r.loss.toFixed(4)}</td>
                            <td style="color:#9ca3af;">${Math.round(r.runtime_s)}s</td>
                        </tr>
                    `).join('');
                }

                // Logs
                if (data.logs && data.logs.length > 0) {
                    document.getElementById('log-box').innerHTML = data.logs.slice(-15).map(l => `<div>${l}</div>`).join('');
                }
            } catch (e) {
                console.error("Fetch error:", e);
            }
        }

        setInterval(updateDashboard, 1500);
        updateDashboard();
    </script>
</body>
</html>
"""


class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif self.path == "/api/stats":
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()
            
            gpu = get_gpu_stats()
            train = get_active_training_info()
            results = get_completed_results()
            
            logs = []
            if MASTER_LOG.exists():
                try:
                    logs = MASTER_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()[-20:]
                except Exception:
                    pass

            payload = {
                "gpu": gpu,
                "train": train,
                "results": results,
                "logs": logs,
            }
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", PORT), DashboardHandler) as httpd:
        sys.stdout.write(f"Live Dashboard running on http://0.0.0.0:{PORT}\n")
        sys.stdout.flush()
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
