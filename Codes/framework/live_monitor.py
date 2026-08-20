#!/usr/bin/env python3
"""
SVD Linear Attention — Ultimate Live Experiment, Telemetry & Analytics Hub
100% Self-Contained, Zero CDN dependencies, Built-in Canvas charting,
Complete 45-Trajectory Matrix, Live GPU Gauges, Leaderboard, and Terminal Stream.
"""
import http.server
import json
import os
import re
import socketserver
import subprocess
import sys
import time
from pathlib import Path

PORT = 8080
PROJECT_DIR = Path("C:/Users/Hadi Schumacher/SVD_Project")
WORKDIR = PROJECT_DIR / "Codes/framework"
RESULTS_DIR = WORKDIR / "results"
MASTER_LOG = PROJECT_DIR / "trajectory_master.log"

ATTN_ARCHITECTURES = [
    {"id": "svd_linear", "name": "SVD Linear Attention", "type": "Proposed / Primary", "complexity": "O(N) Linear", "params_m": 66.4, "desc": "Truncated SVD low-rank projection with dynamic gating and zero quadratic memory."},
    {"id": "svd", "name": "SVD Base Attention", "type": "Low-Rank SVD", "complexity": "O(N) Linear", "params_m": 66.4, "desc": "Standard singular value decomposition attention decomposition."},
    {"id": "full", "name": "Standard Multi-Head Attention", "type": "Softmax Baseline", "complexity": "O(N^2) Quadratic", "params_m": 66.4, "desc": "Exact full quadratic multi-head self-attention baseline."},
    {"id": "lora_attn", "name": "LoRA Low-Rank Attention", "type": "Low-Rank Baseline", "complexity": "O(N^2) Rank Adapt", "params_m": 66.8, "desc": "Low-Rank Adaptation matrices injected into query/key projections."},
    {"id": "performer", "name": "Performer (FAVOR+)", "type": "Kernel Linear", "complexity": "O(N) Linear", "params_m": 66.4, "desc": "Fast Attention Via positive Orthogonal Random features."},
    {"id": "nystrom", "name": "Nyströmformer", "type": "Landmark Approx", "complexity": "O(N) Linear", "params_m": 66.4, "desc": "Nystrom matrix approximation with landmark segment clustering."},
    {"id": "linformer", "name": "Linformer", "type": "Projected Linear", "complexity": "O(N) Linear", "params_m": 66.4, "desc": "Fixed-dimension projection matrix for sequence dimension compression."},
    {"id": "flash3", "name": "FlashAttention-3", "type": "Fused Kernel", "complexity": "O(N^2) Fused HW", "params_m": 66.4, "desc": "Hardware-accelerated tiled asynchronous matrix multiplication."},
    {"id": "mamba", "name": "Mamba SSM Attention", "type": "State-Space Hybrid", "complexity": "O(N) Linear SSM", "params_m": 67.1, "desc": "Selective structured state space model with input-dependent parameters."},
    {"id": "gated_delta_net", "name": "Gated DeltaNet", "type": "Recurrent Delta", "complexity": "O(N) Linear", "params_m": 66.5, "desc": "Delta rule associative memory with input-dependent gating mechanisms."},
    {"id": "adasplash", "name": "AdaSplash Attention", "type": "Adaptive Sparse", "complexity": "O(N log N) Sparse", "params_m": 66.4, "desc": "Dynamic adaptive sparsity with learned flash decay envelopes."},
    {"id": "aegis_v3", "name": "Aegis-v3 Attention", "type": "Robust Shielded", "complexity": "O(N) Linear", "params_m": 66.6, "desc": "Riemannian manifold curvature bounds with gradient variance control."},
    {"id": "griffin_ssd", "name": "Griffin SSD", "type": "Gated Recurrent", "complexity": "O(N) Linear", "params_m": 66.5, "desc": "Structured state space with local multi-query recurrence."},
    {"id": "spectra_v2", "name": "Spectra-v2 Attention", "type": "Spectral SVD", "complexity": "O(N) Spectral", "params_m": 66.4, "desc": "Orthogonal Fourier-SVD hybrid compression with phase modulation."},
    {"id": "samba_m3", "name": "Samba-M3", "type": "SSM + Attention", "complexity": "O(N) Hybrid", "params_m": 67.2, "desc": "Hybrid selective Mamba SSM interleaved with SVD Linear attention."},
]

TASKS = ["sst2", "mrpc", "qnli"]


def get_gpu_stats():
    try:
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            return {
                "gpu_util": float(parts[0]) if len(parts) > 0 else 0,
                "mem_used_mb": float(parts[1]) if len(parts) > 1 else 0,
                "mem_total_mb": float(parts[2]) if len(parts) > 2 else 8188,
                "temp_c": float(parts[3]) if len(parts) > 3 else 0,
                "power_w": float(parts[4]) if len(parts) > 4 else 0,
            }
    except Exception:
        pass
    return {"gpu_util": 0, "mem_used_mb": 0, "mem_total_mb": 8188, "temp_c": 0, "power_w": 0}


def get_storage_stats():
    total_bytes = 0
    file_count = 0
    if RESULTS_DIR.exists():
        try:
            for root, _, files in os.walk(RESULTS_DIR):
                for f in files:
                    file_count += 1
                    total_bytes += os.path.getsize(os.path.join(root, f))
        except Exception:
            pass
    return {
        "used_mb": round(total_bytes / (1024 * 1024), 2),
        "file_count": file_count,
        "policy": "Zero-Bloat (Metrics Only)",
    }


def get_trajectory_matrix():
    completed_keys = set()
    if RESULTS_DIR.exists():
        for d in RESULTS_DIR.iterdir():
            if d.is_dir() and (d / "eval_final.json").exists():
                completed_keys.add(d.name)

    matrix = []
    for t in TASKS:
        task_runs = []
        for m in ATTN_ARCHITECTURES:
            run_key = f"{m['id']}_{t}_s42"
            status = "completed" if run_key in completed_keys else "queued"
            task_runs.append({
                "attn_id": m["id"],
                "attn_name": m["name"],
                "task": t,
                "key": run_key,
                "status": status,
            })
        matrix.append({"task": t, "runs": task_runs})
    return matrix


def get_active_training_info():
    active_model = "Idle / Complete"
    active_task = "None"
    active_index = "0/45"
    step_info = {"step": 0, "total_steps": 2105, "loss": 0.0, "epoch": 0.0, "lr": 0.0}
    loss_history = []
    
    if MASTER_LOG.exists():
        try:
            lines = MASTER_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in reversed(lines):
                m = re.search(r"\[(\d+/\d+)\] >> Launching: attn=(\w+) \| task=(\w+)", line)
                if m:
                    active_index = m.group(1)
                    active_model = m.group(2)
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
                                    "lr": h.get("learning_rate", 0),
                                    "gpu_mem": h.get("gpu_memory_allocated", 0),
                                })
                        if hist:
                            last = hist[-1]
                            step_info["loss"] = last.get("loss", 0)
                            step_info["epoch"] = last.get("epoch", 0)
                            step_info["step"] = last.get("step", 0)
                            step_info["lr"] = last.get("learning_rate", 0)
                    except Exception:
                        pass
                break

    return {
        "active_model": active_model,
        "active_task": active_task,
        "active_index": active_index,
        "step_info": step_info,
        "loss_history": loss_history[-80:],
    }


def get_completed_results():
    results = []
    if RESULTS_DIR.exists():
        for d in RESULTS_DIR.iterdir():
            if d.is_dir():
                eval_file = d / "eval_final.json"
                sys_file = d / "system_info.json"
                cfg_file = d / "config.json"
                if eval_file.exists():
                    try:
                        ev = json.loads(eval_file.read_text(encoding="utf-8"))
                        sys_info = json.loads(sys_file.read_text(encoding="utf-8")) if sys_file.exists() else {}
                        cfg_info = json.loads(cfg_file.read_text(encoding="utf-8")) if cfg_file.exists() else {}
                        
                        acc = ev.get("eval_accuracy", ev.get("accuracy", 0))
                        f1 = ev.get("eval_f1_macro", ev.get("f1", 0))
                        loss = ev.get("eval_loss", 0)
                        runtime = sys_info.get("train_runtime_s", ev.get("eval_runtime", 0))
                        samples_per_sec = ev.get("eval_samples_per_second", 0)
                        
                        parts = d.name.split("_")
                        task = cfg_info.get("task", parts[-2] if len(parts) >= 2 else "sst2")
                        attn = cfg_info.get("attn_type", "_".join(parts[:-2]) if len(parts) >= 3 else parts[0])

                        results.append({
                            "key": d.name,
                            "attn": attn,
                            "task": task.upper(),
                            "accuracy": acc,
                            "f1": f1,
                            "loss": loss,
                            "runtime_s": runtime,
                            "throughput": samples_per_sec,
                            "params": sys_info.get("n_params", 66362880),
                        })
                    except Exception:
                        pass
    return sorted(results, key=lambda x: (x.get("task", ""), x.get("accuracy", 0)), reverse=True)


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVD Linear Attention — Full GPU Experiment & Analytics Hub</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #030712;
            color: #f3f4f6;
            padding: 24px;
            min-height: 100vh;
        }
        .container { max-width: 1380px; margin: 0 auto; display: flex; flex-direction: column; gap: 24px; }
        .glass {
            background: rgba(17, 24, 39, 0.9);
            border: 1px solid rgba(55, 65, 81, 0.7);
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 10px 30px -5px rgba(0,0,0,0.6);
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #06b6d4;
            padding-bottom: 18px;
        }
        .title { font-size: 26px; font-weight: 800; color: #38bdf8; letter-spacing: -0.02em; }
        .subtitle { font-size: 13px; color: #9ca3af; margin-top: 4px; }
        .badge {
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.4);
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .grid-4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }
        .stat-card {
            background: rgba(31, 41, 55, 0.7);
            border: 1px solid rgba(75, 85, 99, 0.5);
            border-radius: 12px;
            padding: 16px;
        }
        .stat-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #9ca3af; font-weight: 700; }
        .stat-val { font-size: 28px; font-weight: 800; font-family: ui-monospace, monospace; margin: 8px 0; color: #06b6d4; }
        .progress-bg { width: 100%; height: 8px; background: #374151; border-radius: 4px; overflow: hidden; margin-top: 8px; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #06b6d4, #3b82f6); width: 0%; transition: width 0.4s; }
        .progress-fill-purple { height: 100%; background: linear-gradient(90deg, #a855f7, #ec4899); width: 0%; transition: width 0.4s; }
        .stat-sub { font-size: 11px; color: #9ca3af; display: flex; justify-content: space-between; margin-top: 6px; }
        
        .main-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
        @media(max-width: 960px) { .main-grid { grid-template-columns: 1fr; } }
        
        .chart-box { height: 280px; position: relative; width: 100%; }
        canvas { width: 100%; height: 100%; display: block; }
        .logs-box {
            background: #000;
            border: 1px solid #374151;
            border-radius: 10px;
            padding: 12px;
            font-family: ui-monospace, monospace;
            font-size: 11px;
            color: #d1d5db;
            height: 280px;
            overflow-y: auto;
            line-height: 1.6;
        }

        .matrix-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; margin-top: 14px; }
        .matrix-card {
            background: rgba(31, 41, 55, 0.5);
            border: 1px solid rgba(75, 85, 99, 0.4);
            border-radius: 8px;
            padding: 10px 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
        }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
        .dot-done { background: #34d399; box-shadow: 0 0 8px #34d399; }
        .dot-active { background: #38bdf8; box-shadow: 0 0 8px #38bdf8; animation: pulse 1.5s infinite; }
        .dot-queued { background: #6b7280; }
        
        table { width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; margin-top: 14px; }
        th { font-size: 11px; text-transform: uppercase; color: #9ca3af; padding: 12px 14px; border-bottom: 1px solid #374151; font-family: ui-monospace, monospace; }
        td { padding: 12px 14px; border-bottom: 1px solid rgba(55, 65, 81, 0.4); font-family: ui-monospace, monospace; }
        tr:hover { background: rgba(55, 65, 81, 0.25); }
        .acc { color: #34d399; font-weight: 700; }
        .f1 { color: #818cf8; font-weight: 700; }
        .loss { color: #fbbf24; }
        .tag-model { background: rgba(6, 182, 212, 0.15); color: #38bdf8; border: 1px solid rgba(6, 182, 212, 0.3); padding: 3px 8px; border-radius: 6px; font-weight: 600; }
        
        @keyframes pulse { 0% { opacity: 0.4; } 50% { opacity: 1; } 100% { opacity: 0.4; } }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header Banner -->
        <div class="glass header">
            <div>
                <div class="title">SVD Linear Attention — Complete Experiment & Analytics Hub</div>
                <div class="subtitle">
                    NVIDIA GeForce RTX 4070 Laptop GPU (8.6GB VRAM) • CUDA 12.4 • FP16 Mixed Precision • Zero-Bloat Storage
                </div>
            </div>
            <div class="badge" id="live-badge">EXPERIMENTS ACTIVE</div>
        </div>

        <!-- Real-Time Hardware & Training Telemetry -->
        <div class="glass">
            <div class="grid-4">
                <!-- GPU Core -->
                <div class="stat-card">
                    <div class="stat-label">GPU Core Utilization</div>
                    <div class="stat-val" id="gpu-util">0%</div>
                    <div class="progress-bg"><div id="gpu-bar" class="progress-fill"></div></div>
                    <div class="stat-sub"><span>Temp: <b id="gpu-temp" style="color:#fff">0°C</b></span><span>Power: <b id="gpu-power" style="color:#fff">0 W</b></span></div>
                </div>

                <!-- VRAM Allocation -->
                <div class="stat-card">
                    <div class="stat-label">VRAM Allocation</div>
                    <div class="stat-val" id="vram-val" style="color: #c084fc;">0 / 8.2 GB</div>
                    <div class="progress-bg"><div id="vram-bar" class="progress-fill-purple"></div></div>
                    <div class="stat-sub"><span>Precision: <b>FP16 CUDA</b></span><span id="vram-pct" style="color:#c084fc">0%</span></div>
                </div>

                <!-- Active Model -->
                <div class="stat-card">
                    <div class="stat-label">Active Trajectory</div>
                    <div class="stat-val" id="active-model" style="color: #fbbf24; font-size: 20px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Loading...</div>
                    <div class="stat-sub"><span>Task: <b id="active-task" style="color:#fbbf24">-</b></span><span>Progress: <b id="active-idx" style="color:#fff">0/45</b></span></div>
                </div>

                <!-- Current Step Loss -->
                <div class="stat-card">
                    <div class="stat-label">Latest Step Loss</div>
                    <div class="stat-val" id="curr-loss" style="color: #34d399;">0.0000</div>
                    <div class="stat-sub"><span>Epoch: <b id="active-epoch" style="color:#fff">-</b></span><span>Disk Used: <b id="storage-val" style="color:#34d399">0 MB</b></span></div>
                </div>
            </div>
        </div>

        <!-- Loss Curve & Stream Logs -->
        <div class="main-grid">
            <div class="glass">
                <div style="font-weight: 700; font-size: 15px; margin-bottom: 12px; display: flex; justify-content: space-between;">
                    <span>Live Loss Trajectory Curve</span>
                    <span style="font-size: 11px; color: #9ca3af; font-family: ui-monospace, monospace;">Auto-refresh 1.5s</span>
                </div>
                <div class="chart-box">
                    <canvas id="lossCanvas"></canvas>
                </div>
            </div>

            <div class="glass">
                <div style="font-weight: 700; font-size: 15px; margin-bottom: 12px;">Live Trajectory Execution Log</div>
                <div class="logs-box" id="log-box">
                    <div style="color: #06b6d4;">> Initializing live stream telemetry...</div>
                </div>
            </div>
        </div>

        <!-- 45-Trajectory Matrix Grid -->
        <div class="glass">
            <div style="font-weight: 700; font-size: 15px; display: flex; justify-content: space-between;">
                <span>45-Trajectory Experiment Matrix Status</span>
                <span style="font-size: 11px; color: #9ca3af; font-family: ui-monospace, monospace;" id="matrix-summary">15 Models x 3 Tasks</span>
            </div>
            <div class="matrix-grid" id="matrix-container">
                <!-- Dynamically populated -->
            </div>
        </div>

        <!-- Completed Experiments Leaderboard -->
        <div class="glass">
            <div style="font-weight: 700; font-size: 15px; display: flex; justify-content: space-between;">
                <span>Benchmark Leaderboard & Results</span>
                <span style="font-size: 11px; color: #9ca3af; font-family: ui-monospace, monospace;" id="completed-count">0 runs completed</span>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Model Architecture</th>
                        <th>Task</th>
                        <th>Accuracy</th>
                        <th>Macro F1</th>
                        <th>Final Loss</th>
                        <th>Throughput</th>
                        <th>Runtime</th>
                    </tr>
                </thead>
                <tbody id="results-body">
                    <tr><td colspan="7" style="text-align: center; color: #6b7280; padding: 20px;">Waiting for completed trajectories...</td></tr>
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
                ctx.font = '12px ui-monospace, monospace';
                ctx.fillText('Streaming live training loss steps...', 20, h / 2);
                return;
            }

            const losses = points.map(p => p.loss);
            const minL = Math.max(0, Math.min(...losses) * 0.9);
            const maxL = Math.max(...losses) * 1.1 || 1.0;

            // Grid lines
            ctx.strokeStyle = 'rgba(55, 65, 81, 0.3)';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {
                const y = h - 25 - (i / 4) * (h - 45);
                ctx.beginPath();
                ctx.moveTo(45, y);
                ctx.lineTo(w - 10, y);
                ctx.stroke();

                const val = minL + (i / 4) * (maxL - minL);
                ctx.fillStyle = '#9ca3af';
                ctx.font = '10px ui-monospace, monospace';
                ctx.fillText(val.toFixed(2), 5, y + 3);
            }

            // Draw line
            ctx.strokeStyle = '#06b6d4';
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            points.forEach((p, i) => {
                const x = 45 + (i / (points.length - 1)) * (w - 55);
                const y = h - 25 - ((p.loss - minL) / (maxL - minL)) * (h - 45);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();

            // Gradient Fill
            const grad = ctx.createLinearGradient(0, 0, 0, h);
            grad.addColorStop(0, 'rgba(6, 182, 212, 0.25)');
            grad.addColorStop(1, 'rgba(6, 182, 212, 0.0)');
            ctx.fillStyle = grad;
            ctx.lineTo(w - 10, h - 25);
            ctx.lineTo(45, h - 25);
            ctx.closePath();
            ctx.fill();
        }

        async function updateDashboard() {
            try {
                const res = await fetch('/api/stats');
                if (!res.ok) return;
                const data = await res.json();

                // GPU Telemetry
                document.getElementById('gpu-util').innerText = data.gpu.gpu_util + '%';
                document.getElementById('gpu-bar').style.width = data.gpu.gpu_util + '%';
                document.getElementById('gpu-temp').innerText = data.gpu.temp_c + '°C';
                document.getElementById('gpu-power').innerText = (data.gpu.power_w || 0) + ' W';

                const memUsedGb = (data.gpu.mem_used_mb / 1024).toFixed(1);
                const memTotalGb = (data.gpu.mem_total_mb / 1024).toFixed(1);
                const vramPct = Math.round((data.gpu.mem_used_mb / data.gpu.mem_total_mb) * 100);
                document.getElementById('vram-val').innerText = `${memUsedGb} / ${memTotalGb} GB`;
                document.getElementById('vram-bar').style.width = vramPct + '%';
                document.getElementById('vram-pct').innerText = vramPct + '%';

                // Training State
                document.getElementById('active-model').innerText = data.train.active_model;
                document.getElementById('active-task').innerText = data.train.active_task.toUpperCase();
                document.getElementById('active-idx').innerText = data.train.active_index || '1/45';
                document.getElementById('active-epoch').innerText = typeof data.train.step_info.epoch === 'number' ? data.train.step_info.epoch.toFixed(2) : '-';
                document.getElementById('curr-loss').innerText = typeof data.train.step_info.loss === 'number' ? data.train.step_info.loss.toFixed(4) : '0.0000';
                document.getElementById('storage-val').innerText = (data.storage.used_mb || 0) + ' MB';

                // Loss Chart
                if (data.train.loss_history) {
                    drawLossChart(data.train.loss_history);
                }

                // Trajectory Matrix
                if (data.matrix && data.matrix.length > 0) {
                    let matrixHtml = '';
                    let completedCount = 0;
                    data.matrix.forEach(taskGroup => {
                        taskGroup.runs.forEach(run => {
                            let dotClass = 'dot-queued';
                            let statusText = 'Queued';
                            if (run.status === 'completed') {
                                dotClass = 'dot-done';
                                statusText = 'Done';
                                completedCount++;
                            } else if (data.train.active_model.includes(run.attn_id) && data.train.active_task.toLowerCase() === run.task.toLowerCase()) {
                                dotClass = 'dot-active';
                                statusText = 'Training';
                            }
                            matrixHtml += `
                                <div class="matrix-card">
                                    <div>
                                        <span class="status-dot ${dotClass}"></span>
                                        <b>${run.attn_id}</b>
                                    </div>
                                    <span style="font-size: 10px; text-transform: uppercase; color: #9ca3af;">${run.task}</span>
                                </div>
                            `;
                        });
                    });
                    document.getElementById('matrix-container').innerHTML = matrixHtml;
                    document.getElementById('matrix-summary').innerText = `${completedCount}/45 Completed`;
                }

                // Leaderboard Table
                if (data.results && data.results.length > 0) {
                    document.getElementById('completed-count').innerText = data.results.length + ' runs completed';
                    document.getElementById('results-body').innerHTML = data.results.map(r => `
                        <tr>
                            <td><span class="tag-model">${r.attn}</span></td>
                            <td style="color:#fbbf24; font-weight:700;">${r.task}</td>
                            <td class="acc">${(r.accuracy * 100).toFixed(2)}%</td>
                            <td class="f1">${(r.f1 * 100).toFixed(2)}%</td>
                            <td class="loss">${r.loss.toFixed(4)}</td>
                            <td style="color:#a78bfa;">${Math.round(r.throughput || 0)}/s</td>
                            <td style="color:#9ca3af;">${Math.round(r.runtime_s)}s</td>
                            <td>
                                <button onclick="showPlots('${r.key}', '${r.attn}', '${r.task}')" style="background:rgba(6,182,212,0.2); color:#38bdf8; border:1px solid rgba(6,182,212,0.4); padding:4px 10px; border-radius:6px; font-size:11px; cursor:pointer; font-weight:600;">📊 Plots</button>
                            </td>
                        </tr>
                    `).join('');
                }

                // Logs Stream
                if (data.logs && data.logs.length > 0) {
                    document.getElementById('log-box').innerHTML = data.logs.slice(-20).map(l => `<div>${l}</div>`).join('');
                }
            } catch (e) {
                console.error("Dashboard update error:", e);
            }
        }

        function showPlots(key, attn, task) {
            const modal = document.getElementById('plotModal');
            document.getElementById('modalTitle').innerText = `${attn.toUpperCase()} on ${task.toUpperCase()} — Saved Plots`;
            document.getElementById('lossImg').src = `/plots/${key}/loss_curve.png?t=${Date.now()}`;
            document.getElementById('metricImg').src = `/plots/${key}/metric_curve.png?t=${Date.now()}`;
            modal.style.display = 'flex';
        }

        function closePlots() {
            document.getElementById('plotModal').style.display = 'none';
        }

        setInterval(updateDashboard, 1500);
        updateDashboard();
    </script>

    <!-- Modal for Saved Plot Images -->
    <div id="plotModal" style="display:none; position:fixed; top:0; left:0; width:100vw; height:100vh; background:rgba(0,0,0,0.85); z-index:9999; justify-content:center; align-items:center; padding:20px;">
        <div style="background:#111827; border:1px solid #374151; border-radius:16px; max-width:900px; width:100%; max-height:90vh; overflow-y:auto; padding:24px; position:relative; display:flex; flex-direction:column; gap:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #374151; padding-bottom:12px;">
                <h3 id="modalTitle" style="color:#38bdf8; font-size:18px; font-weight:700;">Saved Experiment Plots</h3>
                <button onclick="closePlots()" style="background:#ef4444; color:#fff; border:none; padding:4px 12px; border-radius:8px; cursor:pointer; font-weight:700;">✕ Close</button>
            </div>
            <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(350px, 1fr)); gap:16px;">
                <div style="background:#1f2937; padding:12px; border-radius:10px; border:1px solid #374151;">
                    <div style="color:#9ca3af; font-size:12px; margin-bottom:8px; font-weight:600;">LOSS TRAJECTORY CURVE</div>
                    <img id="lossImg" style="width:100%; border-radius:8px;" onerror="this.src=''; this.alt='Plot generating or unavailable';" />
                </div>
                <div style="background:#1f2937; padding:12px; border-radius:10px; border:1px solid #374151;">
                    <div style="color:#9ca3af; font-size:12px; margin-bottom:8px; font-weight:600;">VALIDATION ACCURACY / METRIC CURVE</div>
                    <img id="metricImg" style="width:100%; border-radius:8px;" onerror="this.src=''; this.alt='Plot generating or unavailable';" />
                </div>
            </div>
        </div>
    </div>
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
            storage = get_storage_stats()
            train = get_active_training_info()
            results = get_completed_results()
            matrix = get_trajectory_matrix()
            
            logs = []
            if MASTER_LOG.exists():
                try:
                    logs = MASTER_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()[-25:]
                except Exception:
                    pass

            payload = {
                "gpu": gpu,
                "storage": storage,
                "train": train,
                "results": results,
                "matrix": matrix,
                "logs": logs,
            }
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        elif self.path.startswith("/plots/"):
            # Format: /plots/<exp_name>/<filename.png>
            parts = self.path.strip("/").split("/")
            if len(parts) >= 3:
                exp_name = parts[1]
                filename = parts[2]
                img_path = RESULTS_DIR / exp_name / "plots" / filename
                if not img_path.exists():
                    img_path = RESULTS_DIR / exp_name / filename
                if img_path.exists() and img_path.is_file():
                    self.send_response(200)
                    self.send_header("Content-type", "image/png")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(img_path.read_bytes())
                    return
            self.send_response(404)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", PORT), DashboardHandler) as httpd:
        sys.stdout.write(f"Live Analytics Hub active on http://0.0.0.0:{PORT}\n")
        sys.stdout.flush()
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
