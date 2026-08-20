import re
import os

path = r"C:\Users\Hadi Schumacher\SVD_Project\Codes\framework\live_monitor.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

new_html = r'''HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SVD Compression Hub</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-gray-900 text-gray-100 font-sans p-6">
    <div class="max-w-7xl mx-auto space-y-6">
        <!-- Header -->
        <header class="flex justify-between items-end border-b border-gray-700 pb-4">
            <div>
                <h1 class="text-3xl font-bold text-cyan-400">GPU Orchestrator Live</h1>
                <p class="text-gray-400">Zero-Bloat Attention Compression Pipeline</p>
            </div>
            <div class="text-right">
                <p class="text-sm font-mono text-gray-500">Target: 192.168.1.100</p>
                <div class="flex items-center space-x-2 text-green-400 font-bold">
                    <span class="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span>
                    <span id="conn-status">CONNECTED</span>
                </div>
            </div>
        </header>

        <!-- System Stats -->
        <div class="grid grid-cols-4 gap-4">
            <div class="bg-gray-800 p-4 rounded-lg shadow-lg border border-gray-700">
                <p class="text-xs text-gray-400 uppercase">GPU Load</p>
                <p class="text-2xl font-bold text-cyan-300" id="gpu-util">--%</p>
            </div>
            <div class="bg-gray-800 p-4 rounded-lg shadow-lg border border-gray-700">
                <p class="text-xs text-gray-400 uppercase">VRAM Usage</p>
                <p class="text-2xl font-bold text-purple-400"><span id="vram-used">--</span> / <span id="vram-total">--</span> GB</p>
            </div>
            <div class="bg-gray-800 p-4 rounded-lg shadow-lg border border-gray-700">
                <p class="text-xs text-gray-400 uppercase">GPU Power</p>
                <p class="text-2xl font-bold text-orange-400" id="gpu-power">-- W</p>
            </div>
            <div class="bg-gray-800 p-4 rounded-lg shadow-lg border border-gray-700">
                <p class="text-xs text-gray-400 uppercase">GPU Temp</p>
                <p class="text-2xl font-bold text-red-400" id="gpu-temp">-- °C</p>
            </div>
        </div>

        <div class="grid grid-cols-3 gap-6">
            <!-- Active Run -->
            <div class="col-span-2 bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-xl font-bold text-white">Live Training Curve</h2>
                    <div class="text-sm font-mono bg-gray-700 px-3 py-1 rounded text-cyan-300">
                        <span id="active-model">Loading...</span> | Task: <span id="active-task">--</span> | Run: <span id="active-index">--/45</span>
                    </div>
                </div>
                <div class="h-64 w-full">
                    <canvas id="lossChart"></canvas>
                </div>
            </div>

            <!-- Trajectory Matrix -->
            <div class="col-span-1 bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700 overflow-y-auto" style="max-height: 400px;">
                <h2 class="text-xl font-bold text-white mb-4">Pipeline Matrix (<span id="completed-count">0</span>/45)</h2>
                <div id="matrix-container" class="space-y-4">
                    <!-- Matrix injected here -->
                </div>
            </div>
        </div>
        
        <!-- Live Terminal -->
        <div class="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
            <h2 class="text-xl font-bold text-white mb-4">Console Stream</h2>
            <div class="bg-black p-4 rounded font-mono text-sm h-48 overflow-y-auto text-green-400" id="terminal-out">
                > Connecting to orchestrator...
            </div>
        </div>
    </div>

    <script>
        let lossChart;
        
        function initChart() {
            const ctx = document.getElementById('lossChart').getContext('2d');
            lossChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Training Loss',
                        data: [],
                        borderColor: '#22d3ee',
                        backgroundColor: 'rgba(34, 211, 238, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    scales: {
                        y: {
                            beginAtZero: false,
                            grid: { color: '#374151' },
                            ticks: { color: '#9ca3af' }
                        },
                        x: {
                            grid: { display: false },
                            ticks: { color: '#9ca3af', maxTicksLimit: 10 }
                        }
                    },
                    plugins: { legend: { display: false } }
                }
            });
        }

        async function updateDashboard() {
            try {
                // Ensure we handle absolute and relative paths correctly
                const res = await fetch('/api/stats');
                if (!res.ok) throw new Error("HTTP " + res.status);
                const data = await res.json();
                
                document.getElementById('conn-status').innerText = 'CONNECTED';
                document.getElementById('conn-status').className = 'text-green-400';

                // GPU
                if (data.gpu) {
                    document.getElementById('gpu-util').innerText = data.gpu.gpu_util + "%";
                    document.getElementById('vram-used').innerText = (data.gpu.mem_used_mb / 1024).toFixed(1);
                    document.getElementById('vram-total').innerText = (data.gpu.mem_total_mb / 1024).toFixed(1);
                    document.getElementById('gpu-power').innerText = data.gpu.power_w + " W";
                    document.getElementById('gpu-temp').innerText = data.gpu.temp_c + " °C";
                }

                // Train active
                if (data.train) {
                    document.getElementById('active-model').innerText = data.train.active_model.toUpperCase();
                    document.getElementById('active-task').innerText = data.train.active_task.toUpperCase();
                    document.getElementById('active-index').innerText = data.train.active_index;
                    
                    if (data.train.loss_history && data.train.loss_history.length > 0) {
                        const labels = data.train.loss_history.map(d => `Step ${d.step || Math.round(d.epoch*1000)}`);
                        const losses = data.train.loss_history.map(d => d.loss);
                        lossChart.data.labels = labels;
                        lossChart.data.datasets[0].data = losses;
                        lossChart.update();
                    }
                }

                // Matrix
                if (data.matrix) {
                    let totalRuns = 0;
                    let completedRuns = 0;
                    let html = '';
                    
                    data.matrix.forEach(t => {
                        html += `<div class="mb-2"><h3 class="text-sm font-bold text-gray-400 uppercase border-b border-gray-700 pb-1 mb-2">${t.task}</h3><div class="space-y-1">`;
                        t.runs.forEach(r => {
                            totalRuns++;
                            let color = 'text-gray-500';
                            let icon = '○';
                            if (r.status === 'completed') { color = 'text-green-400'; icon = '●'; completedRuns++; }
                            else if (r.status === 'running') { color = 'text-blue-400 animate-pulse'; icon = '►'; }
                            
                            html += `<div class="flex items-center text-sm ${color}">
                                <span class="mr-2">${icon}</span> ${r.attn_id}
                            </div>`;
                        });
                        html += `</div></div>`;
                    });
                    document.getElementById('matrix-container').innerHTML = html;
                    document.getElementById('completed-count').innerText = completedRuns;
                }

                // Terminal
                if (data.logs) {
                    const term = document.getElementById('terminal-out');
                    term.innerHTML = data.logs.join('<br>');
                    term.scrollTop = term.scrollHeight;
                }

            } catch (err) {
                console.error("Dashboard update error: ", err);
                document.getElementById('conn-status').innerText = 'ERROR';
                document.getElementById('conn-status').className = 'text-red-500';
            }
        }

        initChart();
        setInterval(updateDashboard, 2000);
        updateDashboard();
    </script>
</body>
</html>
"""'''

# Extract old HTML_PAGE string boundaries using regex
old_content = content
start = content.find('HTML_PAGE = """')
if start == -1:
    start = content.find("HTML_PAGE = '''")
    
end = content.find('"""\n', start + 15)
if end == -1:
    end = content.find("'''\n", start + 15)

if start != -1 and end != -1:
    content = content[:start] + new_html + content[end+4:]
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("UI Replaced")
else:
    print("Could not find HTML_PAGE block!")
