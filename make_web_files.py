import os
from pathlib import Path

WEB_DIR = Path(r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\web")
WEB_DIR.mkdir(parents=True, exist_ok=True)

html_content = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVD Attention Compression | GPU Orchestrator Live</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="/styles.css">
</head>
<body class="p-4 md:p-6 max-w-[1600px] mx-auto min-h-screen flex flex-col justify-between">
    <div>
        <!-- Top Bar -->
        <header class="flex flex-col md:flex-row items-start md:items-center justify-between pb-6 mb-6 border-b border-gray-800 gap-4">
            <div>
                <div class="flex items-center gap-3">
                    <span class="relative flex h-3 w-3">
                        <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                        <span class="relative inline-flex rounded-full h-3 w-3 bg-cyan-500"></span>
                    </span>
                    <h1 class="text-2xl font-black tracking-tight text-white flex items-center gap-2">
                        SVD ATTENTION PIPELINE <span class="text-xs px-2 py-0.5 rounded bg-cyan-950 border border-cyan-500 text-cyan-400 font-mono">PARALLEL v3.0</span>
                    </h1>
                </div>
                <p class="text-xs text-gray-400 mt-1 font-mono">Zero-Bloat Telemetry | 3 Concurrent Workers | 120-Run Matrix | GitHub Live Sync</p>
            </div>
            <div class="flex flex-wrap items-center gap-2">
                <span class="text-xs px-3 py-1.5 rounded-lg glass border border-emerald-500/40 text-emerald-400 font-mono flex items-center gap-2">
                    <span class="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></span> Git Auto-Push: Active
                </span>
                <span id="conn-badge" class="text-xs px-3 py-1.5 rounded-lg glass border border-cyan-500/40 text-cyan-400 font-mono">
                    STATUS: <span id="conn-status" class="font-bold">CONNECTING...</span>
                </span>
            </div>
        </header>

        <!-- Hardware & Global Suite Telemetry Grid -->
        <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
            <!-- GPU Load -->
            <div class="glass rounded-xl p-4 glow-cyan border-l-4 border-cyan-500">
                <p class="text-xs text-gray-400 font-medium">GPU Compute Load</p>
                <div class="flex items-baseline justify-between mt-1">
                    <p id="gpu-util" class="text-2xl font-black text-white font-mono">--%</p>
                    <span id="gpu-load-badge" class="text-[10px] px-1.5 py-0.5 rounded bg-gray-800 text-gray-400 font-mono">IDLE</span>
                </div>
                <div class="w-full bg-gray-800 rounded-full h-1.5 mt-3 overflow-hidden">
                    <div id="gpu-bar" class="bg-cyan-400 h-1.5 rounded-full transition-all duration-500" style="width: 0%"></div>
                </div>
            </div>

            <!-- VRAM Usage -->
            <div class="glass rounded-xl p-4 border-l-4 border-indigo-500">
                <p class="text-xs text-gray-400 font-medium">VRAM Allocation</p>
                <p class="text-2xl font-black text-white font-mono mt-1"><span id="vram-used">--</span> <span class="text-xs font-normal text-gray-400">/ <span id="vram-total">8.0</span> GB</span></p>
                <div class="w-full bg-gray-800 rounded-full h-1.5 mt-3 overflow-hidden">
                    <div id="vram-bar" class="bg-indigo-500 h-1.5 rounded-full transition-all duration-500" style="width: 0%"></div>
                </div>
            </div>

            <!-- System RAM -->
            <div class="glass rounded-xl p-4 border-l-4 border-purple-500">
                <p class="text-xs text-gray-400 font-medium">System RAM</p>
                <p class="text-2xl font-black text-white font-mono mt-1"><span id="ram-used">--</span> <span class="text-xs font-normal text-gray-400">/ <span id="ram-total">--</span> GB</span></p>
                <div class="w-full bg-gray-800 rounded-full h-1.5 mt-3 overflow-hidden">
                    <div id="ram-bar" class="bg-purple-500 h-1.5 rounded-full transition-all duration-500" style="width: 0%"></div>
                </div>
            </div>

            <!-- GPU Temp & Power -->
            <div class="glass rounded-xl p-4 border-l-4 border-amber-500">
                <p class="text-xs text-gray-400 font-medium">Power & Temp</p>
                <div class="flex items-baseline justify-between mt-1">
                    <p id="gpu-power" class="text-2xl font-black text-white font-mono">-- W</p>
                    <span id="gpu-temp" class="text-sm font-bold text-amber-400 font-mono">-- °C</span>
                </div>
                <p class="text-[10px] text-gray-500 font-mono mt-3">Target TDP: 140W Max</p>
            </div>

            <!-- Suite Progress -->
            <div class="glass rounded-xl p-4 border-l-4 border-emerald-500">
                <p class="text-xs text-gray-400 font-medium">Total Suite Progress</p>
                <div class="flex items-baseline justify-between mt-1">
                    <p class="text-2xl font-black text-white font-mono"><span id="completed-count">0</span><span class="text-xs font-normal text-gray-400"> / 120</span></p>
                    <span id="progress-pct" class="text-xs font-bold text-emerald-400 font-mono">0%</span>
                </div>
                <div class="w-full bg-gray-800 rounded-full h-1.5 mt-3 overflow-hidden">
                    <div id="suite-bar" class="bg-emerald-500 h-1.5 rounded-full transition-all duration-500" style="width: 0%"></div>
                </div>
            </div>

            <!-- Zero-Bloat Disk -->
            <div class="glass rounded-xl p-4 border-l-4 border-rose-500">
                <p class="text-xs text-gray-400 font-medium">Disk Footprint</p>
                <p class="text-2xl font-black text-white font-mono mt-1"><span id="storage-used">--</span> <span class="text-xs font-normal text-gray-400">MB</span></p>
                <p class="text-[10px] text-emerald-400 font-mono mt-3">Zero Checkpoints Saved</p>
            </div>
        </div>

        <!-- 3 Active Concurrency Workers Grid -->
        <div class="mb-6">
            <h2 class="text-sm font-bold tracking-wider text-gray-400 uppercase mb-3 flex items-center gap-2">
                <span>⚡ Active Concurrency Workers (3 GPU Streams)</span>
            </h2>
            <div id="workers-grid" class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <!-- Worker 1 -->
                <div id="wcard-1" class="glass rounded-xl p-4 border border-gray-800 transition-all">
                    <div class="flex justify-between items-center pb-2 border-b border-gray-800">
                        <span class="text-xs font-bold font-mono text-cyan-400 flex items-center gap-1.5">
                            <span class="h-2 w-2 rounded-full bg-cyan-400 animate-ping"></span> WORKER 1
                        </span>
                        <span id="wstatus-1" class="text-[10px] px-2 py-0.5 rounded bg-gray-800 text-gray-400 uppercase font-mono">Idle</span>
                    </div>
                    <div class="mt-3 space-y-1 font-mono text-xs">
                        <div class="flex justify-between"><span class="text-gray-500">Model:</span> <span id="wmodel-1" class="text-white font-bold">--</span></div>
                        <div class="flex justify-between"><span class="text-gray-500">Task:</span> <span id="wtask-1" class="text-cyan-300 font-bold uppercase">--</span></div>
                        <div class="flex justify-between"><span class="text-gray-500">Run:</span> <span id="widx-1" class="text-gray-300">--</span></div>
                    </div>
                </div>

                <!-- Worker 2 -->
                <div id="wcard-2" class="glass rounded-xl p-4 border border-gray-800 transition-all">
                    <div class="flex justify-between items-center pb-2 border-b border-gray-800">
                        <span class="text-xs font-bold font-mono text-indigo-400 flex items-center gap-1.5">
                            <span class="h-2 w-2 rounded-full bg-indigo-400 animate-ping"></span> WORKER 2
                        </span>
                        <span id="wstatus-2" class="text-[10px] px-2 py-0.5 rounded bg-gray-800 text-gray-400 uppercase font-mono">Idle</span>
                    </div>
                    <div class="mt-3 space-y-1 font-mono text-xs">
                        <div class="flex justify-between"><span class="text-gray-500">Model:</span> <span id="wmodel-2" class="text-white font-bold">--</span></div>
                        <div class="flex justify-between"><span class="text-gray-500">Task:</span> <span id="wtask-2" class="text-indigo-300 font-bold uppercase">--</span></div>
                        <div class="flex justify-between"><span class="text-gray-500">Run:</span> <span id="widx-2" class="text-gray-300">--</span></div>
                    </div>
                </div>

                <!-- Worker 3 -->
                <div id="wcard-3" class="glass rounded-xl p-4 border border-gray-800 transition-all">
                    <div class="flex justify-between items-center pb-2 border-b border-gray-800">
                        <span class="text-xs font-bold font-mono text-purple-400 flex items-center gap-1.5">
                            <span class="h-2 w-2 rounded-full bg-purple-400 animate-ping"></span> WORKER 3
                        </span>
                        <span id="wstatus-3" class="text-[10px] px-2 py-0.5 rounded bg-gray-800 text-gray-400 uppercase font-mono">Idle</span>
                    </div>
                    <div class="mt-3 space-y-1 font-mono text-xs">
                        <div class="flex justify-between"><span class="text-gray-500">Model:</span> <span id="wmodel-3" class="text-white font-bold">--</span></div>
                        <div class="flex justify-between"><span class="text-gray-500">Task:</span> <span id="wtask-3" class="text-purple-300 font-bold uppercase">--</span></div>
                        <div class="flex justify-between"><span class="text-gray-500">Run:</span> <span id="widx-3" class="text-gray-300">--</span></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Middle Layout: Live Curve Chart & Console Stream -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
            <!-- Chart Section -->
            <div class="lg:col-span-2 glass rounded-xl p-5 flex flex-col justify-between">
                <div class="flex items-center justify-between mb-4">
                    <div>
                        <h2 class="text-base font-bold text-white flex items-center gap-2">
                            <span>Live Step Loss Trajectory</span>
                        </h2>
                        <p class="text-xs text-gray-400 font-mono mt-0.5">Active Stream: <span id="active-model" class="text-cyan-400 font-bold">--</span> (<span id="active-task" class="text-white uppercase">--</span>)</p>
                    </div>
                    <span class="text-[10px] bg-cyan-950/80 text-cyan-400 border border-cyan-800 px-2 py-1 rounded font-mono">AUTOSCALE</span>
                </div>
                <div class="h-64 relative w-full">
                    <canvas id="lossChart"></canvas>
                </div>
            </div>

            <!-- Console Terminal Section -->
            <div class="glass rounded-xl p-5 flex flex-col h-[340px]">
                <div class="flex items-center justify-between pb-2 mb-2 border-b border-gray-800">
                    <span class="text-xs font-mono text-gray-400 font-bold flex items-center gap-2">
                        <span class="h-2 w-2 rounded-full bg-emerald-400"></span> ORCHESTRATOR CONSOLE
                    </span>
                    <span class="text-[10px] text-gray-500 font-mono">master.log</span>
                </div>
                <div id="log-terminal" class="flex-1 overflow-y-auto font-mono text-[11px] leading-relaxed text-gray-300 space-y-1 bg-black/40 p-3 rounded-lg border border-gray-900 select-all">
                    <p class="text-gray-600">> Connecting to orchestrator stream...</p>
                </div>
            </div>
        </div>

        <!-- Matrix Section: 120-Run Matrix -->
        <div class="glass rounded-xl p-5 mb-6">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between pb-4 mb-4 border-b border-gray-800 gap-2">
                <div>
                    <h2 class="text-base font-bold text-white flex items-center gap-2">
                        <span>120-Run GLUE Benchmark Matrix</span>
                    </h2>
                    <p class="text-xs text-gray-400 font-mono mt-0.5">15 SVD & Attention Architectures across all 8 GLUE Tasks</p>
                </div>
                <div class="flex items-center gap-4 text-xs font-mono">
                    <span class="flex items-center gap-1.5"><span class="h-2.5 w-2.5 rounded-full bg-emerald-500 inline-block"></span> Completed</span>
                    <span class="flex items-center gap-1.5"><span class="h-2.5 w-2.5 rounded-full bg-cyan-400 animate-pulse inline-block"></span> Running</span>
                    <span class="flex items-center gap-1.5"><span class="h-2.5 w-2.5 rounded-full bg-gray-600 inline-block"></span> Queued</span>
                </div>
            </div>
            
            <div id="matrix-container" class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3">
                <!-- Injected via JS -->
            </div>
        </div>
    </div>

    <footer class="text-center text-xs text-gray-600 font-mono pt-4 border-t border-gray-900">
        SVD Linear Attention Framework | Autonomous Research Pipeline | Zero-Bloat Telemetry
    </footer>

    <script src="/app.js"></script>
</body>
</html>
"""

js_content = """let lossChart;

function initChart() {
    const ctx = document.getElementById('lossChart').getContext('2d');
    lossChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Step Loss',
                data: [],
                borderColor: '#06b6d4',
                backgroundColor: 'rgba(6, 182, 212, 0.12)',
                borderWidth: 2,
                fill: true,
                tension: 0.3,
                pointRadius: 2,
                pointHoverRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: { grid: { color: 'rgba(75, 85, 99, 0.15)' }, ticks: { color: '#9ca3af', font: { size: 10, family: 'monospace' } } },
                y: { grid: { color: 'rgba(75, 85, 99, 0.15)' }, ticks: { color: '#9ca3af', font: { size: 10, family: 'monospace' } } }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

async function updateDashboard() {
    try {
        const res = await fetch('/api/stats');
        if (!res.ok) throw new Error("HTTP " + res.status);
        const data = await res.json();
        
        document.getElementById('conn-status').innerText = 'CONNECTED';
        document.getElementById('conn-status').className = 'font-bold text-emerald-400';

        // 1. GPU & Hardware
        if (data.gpu) {
            const util = data.gpu.gpu_util || 0;
            document.getElementById('gpu-util').innerText = util + "%";
            document.getElementById('gpu-bar').style.width = Math.min(util, 100) + "%";
            
            const badge = document.getElementById('gpu-load-badge');
            if (util > 70) {
                badge.innerText = 'MAX UTIL';
                badge.className = 'text-[10px] px-1.5 py-0.5 rounded bg-emerald-950 border border-emerald-500 text-emerald-400 font-mono font-bold animate-pulse';
            } else if (util > 30) {
                badge.innerText = 'ACTIVE';
                badge.className = 'text-[10px] px-1.5 py-0.5 rounded bg-cyan-950 border border-cyan-500 text-cyan-400 font-mono';
            } else {
                badge.innerText = 'IDLE';
                badge.className = 'text-[10px] px-1.5 py-0.5 rounded bg-gray-800 text-gray-400 font-mono';
            }

            const vUsed = (data.gpu.mem_used_mb / 1024).toFixed(1);
            const vTotal = (data.gpu.mem_total_mb / 1024).toFixed(1);
            document.getElementById('vram-used').innerText = vUsed;
            document.getElementById('vram-total').innerText = vTotal;
            document.getElementById('vram-bar').style.width = Math.min((vUsed / vTotal) * 100, 100) + "%";

            if (data.gpu.sys_ram_used_gb !== undefined) {
                document.getElementById('ram-used').innerText = data.gpu.sys_ram_used_gb;
                document.getElementById('ram-total').innerText = data.gpu.sys_ram_total_gb;
                document.getElementById('ram-bar').style.width = Math.min((data.gpu.sys_ram_used_gb / data.gpu.sys_ram_total_gb) * 100, 100) + "%";
            }

            document.getElementById('gpu-power').innerText = (data.gpu.power_w || 0) + " W";
            document.getElementById('gpu-temp').innerText = (data.gpu.temp_c || 0) + " °C";
        }

        // 2. Storage
        if (data.storage) {
            document.getElementById('storage-used').innerText = data.storage.used_mb;
        }

        // 3. Workers
        if (data.workers) {
            [1, 2, 3].forEach(id => {
                const w = data.workers[String(id)];
                if (w) {
                    const isRunning = w.status === 'running';
                    const card = document.getElementById('wcard-' + id);
                    const stBadge = document.getElementById('wstatus-' + id);
                    
                    stBadge.innerText = w.status.toUpperCase();
                    if (isRunning) {
                        card.className = 'glass rounded-xl p-4 border border-cyan-500/60 glow-cyan transition-all';
                        stBadge.className = 'text-[10px] px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 font-bold border border-cyan-500 font-mono';
                    } else {
                        card.className = 'glass rounded-xl p-4 border border-gray-800 transition-all';
                        stBadge.className = 'text-[10px] px-2 py-0.5 rounded bg-gray-800 text-gray-400 uppercase font-mono';
                    }
                    
                    document.getElementById('wmodel-' + id).innerText = w.attn || '--';
                    document.getElementById('wtask-' + id).innerText = w.task || '--';
                    document.getElementById('widx-' + id).innerText = w.idx || '--';
                }
            });
        }

        // 4. Matrix & Suite Progress
        if (data.matrix) {
            let totalRuns = 0;
            let completedRuns = 0;
            let html = '';
            
            data.matrix.forEach(t => {
                html += `<div class="glass rounded-lg p-2.5 border border-gray-800/80"><h3 class="text-xs font-black text-cyan-400 uppercase border-b border-gray-800 pb-1.5 mb-2 font-mono flex justify-between"><span>${t.task}</span></h3><div class="space-y-1.5">`;
                t.runs.forEach(r => {
                    totalRuns++;
                    let badgeStyle = 'bg-gray-900/60 text-gray-500 border-gray-800';
                    let icon = '○';
                    let extra = '';
                    if (r.status === 'completed') { 
                        badgeStyle = 'bg-emerald-950/60 text-emerald-400 border-emerald-800 font-bold'; 
                        icon = '✓'; 
                        completedRuns++; 
                        if (r.metrics) {
                            extra = `<span class="text-[9px] text-emerald-500 ml-1 font-mono">${r.metrics.acc}%</span>`;
                        }
                    } else if (r.status === 'running') { 
                        badgeStyle = 'bg-cyan-950/90 text-cyan-300 border-cyan-500 font-bold animate-pulse'; 
                        icon = '►'; 
                    }
                    
                    html += `<div class="flex items-center justify-between text-[11px] p-1 rounded border ${badgeStyle}">
                        <div class="truncate max-w-[85px]"><span class="mr-1 text-[9px]">${icon}</span>${r.attn_id}</div>
                        ${extra}
                    </div>`;
                });
                html += `</div></div>`;
            });
            document.getElementById('matrix-container').innerHTML = html;
            document.getElementById('completed-count').innerText = completedRuns;
            const pct = totalRuns > 0 ? Math.round((completedRuns / totalRuns) * 100) : 0;
            document.getElementById('progress-pct').innerText = pct + "%";
            document.getElementById('suite-bar').style.width = pct + "%";
        }

        // 5. Chart
        if (data.train) {
            document.getElementById('active-model').innerText = data.train.active_model;
            document.getElementById('active-task').innerText = data.train.active_task;
            
            if (data.train.loss_history && data.train.loss_history.length > 0) {
                const labels = data.train.loss_history.map(d => `Step ${d.step || Math.round(d.epoch*1000)}`);
                const losses = data.train.loss_history.map(d => d.loss);
                lossChart.data.labels = labels;
                lossChart.data.datasets[0].data = losses;
                lossChart.update();
            }
        }

        // 6. Logs
        if (data.logs && data.logs.length > 0) {
            const terminal = document.getElementById('log-terminal');
            terminal.innerHTML = data.logs.map(l => {
                let col = 'text-gray-300';
                if (l.includes('SUCCESS:')) col = 'text-emerald-400 font-bold';
                else if (l.includes('NOTICE:') || l.includes('Error:')) col = 'text-rose-400 font-bold';
                else if (l.includes('Launching:')) col = 'text-cyan-300';
                return `<p class="${col}">> ${l}</p>`;
            }).join('');
            terminal.scrollTop = terminal.scrollHeight;
        }

    } catch (err) {
        console.error("Dashboard update error: ", err);
        document.getElementById('conn-status').innerText = 'ERROR: ' + err.message;
        document.getElementById('conn-status').className = 'font-bold text-rose-500';
    }
}

window.onload = () => {
    initChart();
    updateDashboard();
    setInterval(updateDashboard, 2000);
};
"""

css_content = """body { 
    background-color: #030712; 
    color: #f3f4f6; 
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
}
.glass { 
    background: rgba(17, 24, 39, 0.85); 
    backdrop-filter: blur(16px); 
    border: 1px solid rgba(75, 85, 99, 0.3); 
}
.glow-cyan { 
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.25); 
}
.glow-green { 
    box-shadow: 0 0 20px rgba(34, 197, 94, 0.25); 
}
"""

with open(WEB_DIR / "index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

with open(WEB_DIR / "app.js", "w", encoding="utf-8") as f:
    f.write(js_content)

with open(WEB_DIR / "styles.css", "w", encoding="utf-8") as f:
    f.write(css_content)

print("Created modular web directory with index.html, app.js, and styles.css!")
