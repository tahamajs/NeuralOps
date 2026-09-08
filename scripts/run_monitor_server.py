import http.server
import socketserver
import subprocess
import json
import os
import glob
import time
import pandas as pd

PORT = 8080

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hoosha AI NeuralOps - Mission Control Center</title>
    <style>
        :root {
            --bg: #090d13;
            --card-bg: #121820;
            --card-header: #1a222d;
            --border: #232f3e;
            --text: #c9d1d9;
            --text-muted: #8b949e;
            --accent: #58a6ff;
            --green: #3fb950;
            --yellow: #d29922;
            --red: #f85149;
            --purple: #bc8cff;
        }
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 18px 24px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 14px;
            margin-bottom: 20px;
        }
        .header h1 { margin: 0; color: #fff; font-size: 22px; display: flex; align-items: center; gap: 10px; }
        .grid-kpi {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 14px;
            margin-bottom: 20px;
        }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }
        .card-body { padding: 14px; }
        .card-header {
            background: var(--card-header);
            padding: 10px 14px;
            font-size: 13px;
            font-weight: 600;
            color: var(--accent);
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .metric-val { font-size: 24px; font-weight: bold; color: #fff; margin: 4px 0; }
        .metric-sub { font-size: 12px; color: var(--text-muted); }
        .badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }
        .badge-green { background: rgba(63, 185, 80, 0.15); color: var(--green); border: 1px solid var(--green); }
        .badge-blue { background: rgba(88, 166, 255, 0.15); color: var(--accent); border: 1px solid var(--accent); }
        .badge-purple { background: rgba(188, 140, 255, 0.15); color: var(--purple); border: 1px solid var(--purple); }
        .badge-red { background: rgba(248, 81, 73, 0.15); color: var(--red); border: 1px solid var(--red); }
        
        .main-layout {
            display: grid;
            grid-template-columns: 1.1fr 1.1fr;
            gap: 16px;
            margin-bottom: 20px;
        }
        @media (max-width: 1100px) {
            .main-layout { grid-template-columns: 1fr; }
        }

        .log-box {
            background: #06090e;
            padding: 12px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 12px;
            height: 420px;
            overflow-y: auto;
            white-space: pre-wrap;
            color: #79c0ff;
            line-height: 1.45;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            text-align: left;
        }
        th, td {
            padding: 9px 12px;
            border-bottom: 1px solid var(--border);
        }
        th {
            background: var(--card-header);
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 11px;
        }
        tr:hover { background: rgba(255, 255, 255, 0.02); }
        .progress-bar-bg {
            background: #21262d;
            border-radius: 4px;
            height: 6px;
            overflow: hidden;
            margin-top: 6px;
        }
        .progress-bar-fill {
            background: var(--accent);
            height: 100%;
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>⚡ Hoosha AI NeuralOps — Mission Control</h1>
            <div style="font-size: 13px; color: var(--text-muted); margin-top: 3px;">
                End-to-End Real-Time Telemetry, GPU Execution Stream & Empirical Benchmark Database
            </div>
        </div>
        <div style="display: flex; gap: 10px; align-items: center;">
            <span class="badge badge-purple" id="runner-badge">SINGLE-PROCESS MUTEX LOCK</span>
            <span class="badge badge-green" id="status-badge">● LIVE (2s)</span>
        </div>
    </div>

    <!-- KPI ROW -->
    <div class="grid-kpi">
        <div class="card">
            <div class="card-body">
                <div class="metric-sub">GPU VRAM ALLOCATION</div>
                <div class="metric-val" id="gpu-mem">--</div>
                <div class="progress-bar-bg"><div class="progress-bar-fill" id="vram-bar" style="width: 0%;"></div></div>
                <div class="metric-sub" style="margin-top: 4px;" id="gpu-util">Load: -- | Temp: --</div>
            </div>
        </div>
        <div class="card">
            <div class="card-body">
                <div class="metric-sub">STORAGE MARGIN (D:\)</div>
                <div class="metric-val" id="disk-free">--</div>
                <div class="progress-bar-bg"><div class="progress-bar-fill" id="disk-bar" style="width: 25%; background: var(--green);"></div></div>
                <div class="metric-sub" style="margin-top: 4px;" id="disk-sub">Healthy Storage Margin</div>
            </div>
        </div>
        <div class="card">
            <div class="card-body">
                <div class="metric-sub">ACTIVE HARDENED OPERATOR</div>
                <div class="metric-val" id="active-mod" style="color: var(--accent); font-size: 21px;">--</div>
                <div class="metric-sub" id="active-status">Benchmarking across 4 Architectures</div>
            </div>
        </div>
        <div class="card">
            <div class="card-body">
                <div class="metric-sub">DEPLOYED CHECKPOINTS (HF)</div>
                <div class="metric-val" id="hf-count" style="color: var(--purple);">299 Repos</div>
                <div class="metric-sub">Hooshaai (Private + Enriched)</div>
            </div>
        </div>
        <div class="card">
            <div class="card-body">
                <div class="metric-sub">BENCHMARK RUNS INDEXED</div>
                <div class="metric-val" id="sweeps-count" style="color: var(--green);">--</div>
                <div class="metric-sub">SST-2 Standard GLUE Tasks</div>
            </div>
        </div>
    </div>

    <!-- MAIN DUAL SPLIT -->
    <div class="main-layout">
        <!-- LEFT: LOG STREAM -->
        <div class="card">
            <div class="card-header">
                <span>🖥️ LIVE SERVER EXECUTION STREAM</span>
                <span style="font-size: 11px; color: var(--text-muted);">RTX 4070 8GB Server</span>
            </div>
            <div class="log-box" id="logs">Connecting to GPU telemetry stream...</div>
        </div>

        <!-- RIGHT: RECENT VERIFIED BENCHMARKS -->
        <div class="card">
            <div class="card-header">
                <span>📊 EMPIRICAL BENCHMARK DATABASE (ACCURATELY PARSED)</span>
                <span class="badge badge-blue">SST-2 Quality Gate</span>
            </div>
            <div style="max-height: 420px; overflow-y: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Module</th>
                            <th>Architecture</th>
                            <th>Accuracy</th>
                            <th>F1 Score</th>
                            <th>Delta</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="results-table">
                        <tr><td colspan="6" style="text-align:center; color: var(--text-muted);">Loading benchmark database...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        async function updateData() {
            try {
                const res = await fetch('/api/telemetry');
                const data = await res.json();
                
                // Hardware KPIs
                document.getElementById('gpu-mem').textContent = data.gpu_mem;
                document.getElementById('gpu-util').textContent = `Compute Load: ${data.gpu_util} | Temp: ${data.gpu_temp}`;
                
                // VRAM calculation
                if (data.gpu_mem && data.gpu_mem.includes('MiB')) {
                    const used = parseInt(data.gpu_mem);
                    const pct = Math.min(100, Math.round((used / 8192) * 100));
                    document.getElementById('vram-bar').style.width = pct + "%";
                }

                document.getElementById('disk-free').textContent = data.disk_free + " GB Free";
                document.getElementById('active-mod').textContent = data.active_module;
                document.getElementById('sweeps-count').textContent = data.total_sweeps + " Evaluated";
                document.getElementById('logs').textContent = data.logs;
                
                const logBox = document.getElementById('logs');
                logBox.scrollTop = logBox.scrollHeight;

                // Render Table
                if (data.recent_results && data.recent_results.length > 0) {
                    const tbody = document.getElementById('results-table');
                    tbody.innerHTML = data.recent_results.map(r => {
                        const isPass = r.status === 'PASS';
                        const isLow = r.status.includes('LOW') || r.acc < 56.0;
                        const badgeClass = isPass ? 'badge-green' : (isLow ? 'badge-red' : 'badge-purple');
                        const accColor = r.acc >= 70 ? 'var(--green)' : (r.acc >= 56 ? 'var(--accent)' : 'var(--red)');
                        return `
                        <tr>
                            <td style="font-weight: 600; color: #fff;">${r.module}</td>
                            <td><span class="badge badge-blue">${r.arch}</span></td>
                            <td style="color: ${accColor}; font-weight:600;">${r.acc.toFixed(2)}%</td>
                            <td>${r.f1}</td>
                            <td style="color: ${r.delta.startsWith('+') || r.delta === '0.00%' ? 'var(--green)' : 'var(--yellow)'};">${r.delta}</td>
                            <td><span class="badge ${badgeClass}">${r.status}</span></td>
                        </tr>
                        `;
                    }).join('');
                }
            } catch (e) {
                console.error("Telemetry fetch error", e);
            }
        }
        setInterval(updateData, 2000);
        updateData();
    </script>
</body>
</html>
"""

class MissionControlHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif self.path == "/api/telemetry":
            gpu_mem = "N/A"
            gpu_util = "N/A"
            gpu_temp = "N/A"
            disk_free = "N/A"
            active_module = "Idle"
            logs = "No log data available."

            # 1. Fetch GPU hardware stats & queue log
            try:
                g_res = subprocess.run(
                    ["ssh", "-o", "ConnectTimeout=3", "gpu", "nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader"],
                    capture_output=True, text=True, timeout=5
                )
                if g_res.returncode == 0:
                    parts = [p.strip() for p in g_res.stdout.strip().split("\n")[-1].split(",")]
                    if len(parts) >= 3:
                        gpu_mem = parts[0]
                        gpu_util = parts[1]
                        gpu_temp = parts[2] + "°C"

                d_res = subprocess.run(
                    ["ssh", "-o", "ConnectTimeout=3", "gpu", 'powershell -Command "[math]::Round((Get-PSDrive D).Free/1GB, 2); Write-Output \\"---LOG---\\"; Get-Content D:/SVD_Project/queue_launcher.log -Tail 22"'],
                    capture_output=True, text=True, timeout=5
                )
                if d_res.returncode == 0:
                    lines = d_res.stdout.replace("\r", "").split("\n")
                    clean_lines = [l for l in lines if not l.startswith("**") and l.strip()]
                    if clean_lines:
                        disk_free = clean_lines[0]
                        if "---LOG---" in clean_lines:
                            log_idx = clean_lines.index("---LOG---")
                            log_slice = clean_lines[log_idx+1:]
                            logs = "\n".join(log_slice)

                            for line in log_slice:
                                if "[EXCLUSIVE LAUNCH] Starting module:" in line:
                                    active_module = line.split(":")[-1].strip()
                                elif "[MODULE FINISHED]" in line:
                                    active_module = "Transitioning..."
            except Exception as e:
                logs = f"Telemetry read error: {e}"

            # 2. Extract recent benchmark summaries from local Project results
            recent_results = []
            total_sweeps = 0
            try:
                csv_files = sorted(
                    glob.glob("/Users/tahamajs/Documents/uni/Project/results/**/*summary.csv", recursive=True),
                    key=os.path.getmtime,
                    reverse=True
                )
                total_sweeps = len(csv_files)
                for cf in csv_files:
                    try:
                        # Extract architecture from path
                        path_parts = cf.replace('\\', '/').split('/')
                        detected_arch = "unknown"
                        for p in path_parts:
                            if p in ["distilbert", "roberta", "gpt2", "qwen3_5"]:
                                detected_arch = p
                                break
                        
                        df = pd.read_csv(cf)
                        df.columns = [c.strip().lower() for c in df.columns]
                        
                        # Find base accuracy if available
                        base_acc = 0.0
                        if 'accuracy_pct' in df.columns or 'accuracy' in df.columns:
                            for _, r in df.iterrows():
                                if str(r.get('module_name', r.get('module', ''))).strip().lower() == 'full':
                                    base_acc = float(r.get('accuracy_pct', r.get('accuracy', 0.0)))
                                    break
                        
                        for _, row in df.iterrows():
                            mod = str(row.get('module_name', row.get('module', row.get('attention_type', 'attn')))).replace('unified_', '').replace('_attention', '')
                            if mod == 'full':
                                continue
                            
                            # Parse accuracy using exact fields
                            acc = 0.0
                            for fld in ['accuracy_pct', 'accuracy', 'eval_accuracy', 'acc', 'mean_accuracy']:
                                if fld in row and pd.notna(row[fld]):
                                    try:
                                        acc = float(row[fld])
                                        break
                                    except:
                                        pass
                            
                            f1 = str(row.get('f1_score', row.get('f1', 'N/A')))
                            if f1 != 'N/A' and len(f1) > 6:
                                try:
                                    f1 = f"{float(f1):.4f}"
                                except:
                                    pass
                            
                            delta = "-"
                            if base_acc > 0 and acc > 0:
                                diff = acc - base_acc
                                delta = f"{diff:+.2f}%"
                            elif 'delta_acc' in row and pd.notna(row['delta_acc']):
                                delta = str(row['delta_acc'])
                            
                            status = str(row.get('status', 'PASS' if acc >= 56.0 else 'FAIL'))
                            arch_str = detected_arch if detected_arch != "unknown" else str(row.get('model', row.get('model_name', 'bert'))).replace('models\\', '').replace('_sst2', '')
                            
                            recent_results.append({
                                'module': mod,
                                'arch': arch_str,
                                'acc': acc,
                                'f1': f1,
                                'delta': delta,
                                'status': status
                            })
                            if len(recent_results) >= 15:
                                break
                        if len(recent_results) >= 15:
                            break
                    except:
                        continue
            except Exception as e:
                print(f"Error parsing summaries: {e}")

            resp_payload = {
                "gpu_mem": gpu_mem,
                "gpu_util": gpu_util,
                "gpu_temp": gpu_temp,
                "disk_free": disk_free,
                "active_module": active_module,
                "total_sweeps": total_sweeps,
                "logs": logs,
                "recent_results": recent_results
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp_payload).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), MissionControlHandler) as httpd:
        print(f"Mission Control webserver running at http://localhost:{PORT}")
        httpd.serve_forever()
