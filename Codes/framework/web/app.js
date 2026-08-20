let lossChart;

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
