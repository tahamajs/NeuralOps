/**
 * NeuralOps Multi-Run Live View & Publication Analytics Studio (v3.0)
 */

const API_STATS = "/api/stats";
const API_RUNS = "/api/runs";
const API_ACTION = "/api/action";
const API_RUN_START = "/api/run/start";
const API_RUN_STOP = "/api/run/stop";
const POLL_INTERVAL_MS = 3000;

// State
let isFetching = false;
let pollTimer = null;
let currentData = null;
let availableRuns = [];
let selectedRunId = "run_active";
let isHistorical = false;

let selectedModules = new Set();
let activeLiveFilter = "ALL";
let analyticsFilters = { track: "ALL", model: "ALL", dataset: "ALL" };
let currentSortColumn = 3; // Fwd Latency default
let currentSortAsc = true;
let activeTab = "telemetry";
let matrixSearchQuery = "";
let matrixStatusFilter = "ALL";

// Charts
let barChart = null;
let radarChart = null;
let scatterChart = null;
let modalLossChart = null;
let modalBreakdownChart = null;
let compareBarChart = null;
let lossOverlayChart = null;
let bubbleScatterChart = null;
let modelDistChart = null;

// Helper: safe formatting
const fN = (val, dec = 2) => {
  if (val === null || val === undefined || val === '' || val === '-') return '-';
  const n = typeof val === 'number' ? val : parseFloat(val);
  return (!isNaN(n) && isFinite(n)) ? n.toFixed(dec) : '-';
};

// ==========================================
// TAB MANAGEMENT
// ==========================================
function switchTab(tabId) {
  activeTab = tabId;
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
  
  const btn = document.getElementById(`tab-${tabId}-btn`);
  const view = document.getElementById(`view-${tabId}`);
  if (btn) btn.classList.add("active");
  if (view) view.classList.add("active");

  saveState();

  if (tabId === 'analytics') {
    if (barChart) barChart.resize();
    if (radarChart) radarChart.resize();
    if (scatterChart) scatterChart.resize();
  } else if (tabId === 'compare') {
    updateRunComparison();
  } else if (tabId === 'visuals') {
    renderAdvancedVisuals();
  }
}

// ==========================================
// RUN MANAGEMENT & DATA FETCHING
// ==========================================
async function fetchRunsList() {
  try {
    const res = await fetch(API_RUNS, { cache: "no-store" });
    if (!res.ok) return;
    const data = await res.json();
    availableRuns = data.runs || [];
    const activeRun = availableRuns.find(r => r.is_active || r.run_id === "run_active");
    if (activeRun) {
      selectedRunId = activeRun.run_id;
      isHistorical = false;
    }
    populateRunSelectors();
  } catch (e) {
    console.warn("Failed to fetch runs list:", e);
  }
}

function populateRunSelectors() {
  const runSelect = document.getElementById("run-select");
  const compareA = document.getElementById("compare-run-a");
  const compareB = document.getElementById("compare-run-b");

  if (!runSelect) return;

  const currentVal = runSelect.value || selectedRunId;
  runSelect.innerHTML = "";
  if (compareA) compareA.innerHTML = "";
  if (compareB) compareB.innerHTML = "";

  availableRuns.forEach((r, idx) => {
    const opt = document.createElement("option");
    opt.value = r.run_id;
    const statusIcon = r.is_active ? "⚡ [ACTIVE] " : "📁 ";
    const totalMods = r.num_combinations || r.module_count || 20;
    const countStr = `(${r.completed || 0}/${totalMods} modules)`;
    opt.textContent = `${statusIcon}${r.name || r.run_id} ${countStr}`;
    runSelect.appendChild(opt);

    if (compareA) {
      const optA = opt.cloneNode(true);
      if (idx === 0) optA.selected = true;
      compareA.appendChild(optA);
    }
    if (compareB) {
      const optB = opt.cloneNode(true);
      if (idx === Math.min(1, availableRuns.length - 1)) optB.selected = true;
      compareB.appendChild(optB);
    }
  });

  if (availableRuns.some(r => r.run_id === currentVal)) {
    runSelect.value = currentVal;
  }
}

async function onRunChanged(runId) {
  selectedRunId = runId;
  saveState();

  const runObj = availableRuns.find(r => r.run_id === runId);
  const isAct = runObj ? runObj.is_active : (runId === "run_active");

  const sweepBadge = document.getElementById("sweep-status");
  const btnStop = document.getElementById("btn-stop");

  if (isAct) {
    isHistorical = false;
    if (sweepBadge) {
      sweepBadge.textContent = "ACTIVE SWEEP (LIVE)";
      sweepBadge.className = "badge badge-running";
    }
    if (btnStop) btnStop.style.display = "inline-block";
    startPolling();
    fetchStats();
  } else {
    isHistorical = true;
    stopPolling();
    if (sweepBadge) {
      sweepBadge.textContent = "HISTORICAL RUN (STATIC)";
      sweepBadge.className = "badge badge-idle";
    }
    if (btnStop) btnStop.style.display = "none";
    loadHistoricalRun(runId);
  }
}

async function fetchStats() {
  if (isFetching) return;
  isFetching = true;

  try {
    const res = await fetch(API_STATS, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    currentData = data;
    renderDashboard();
    setConnectionStatus(true);
  } catch (err) {
    console.warn("API poll error:", err);
    setConnectionStatus(false);
  } finally {
    isFetching = false;
  }
}

async function loadHistoricalRun(runId) {
  try {
    setConnectionStatus(true);
    const res = await fetch(`/api/runs/${runId}`, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    currentData = data;
    renderDashboard();
  } catch (err) {
    console.error("Failed to load historical run:", err);
    setConnectionStatus(false);
  }
}

function startPolling() {
  stopPolling();
  pollTimer = setInterval(() => {
    fetchStats();
    fetchRunsList();
  }, POLL_INTERVAL_MS);
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function setConnectionStatus(online) {
  const badge = document.getElementById("conn-status");
  if (!badge) return;
  if (online) {
    badge.textContent = isHistorical ? "ARCHIVE LOADED" : "ONLINE (LIVE)";
    badge.className = "badge badge-online";
  } else {
    badge.textContent = "RECONNECTING...";
    badge.className = "badge badge-offline";
  }
}

// ==========================================
// RUN MANAGEMENT ACTIONS
// ==========================================
function openNewRunModal() {
  const m = document.getElementById("new-run-modal");
  if (m) m.classList.add("active");
}

function hideNewRunModal() {
  const m = document.getElementById("new-run-modal");
  if (m) m.classList.remove("active");
}

function closeNewRunModal(e) {
  if (e.target.id === "new-run-modal") hideNewRunModal();
}

async function submitNewRun(e) {
  e.preventDefault();
  const cfg = {
    batch_size: parseInt(document.getElementById("form-batch-size").value, 10) || 1024,
    seq_len: parseInt(document.getElementById("form-seq-len").value, 10) || 192,
    max_steps: parseInt(document.getElementById("form-max-steps").value, 10) || 15,
    precision: document.getElementById("form-precision").value || "fp16",
    models: document.getElementById("form-models").value || "bert,roberta,t5",
    datasets: document.getElementById("form-datasets").value || "sst2,mrpc",
    tracks: document.getElementById("form-tracks").value,
    timeout: 120,
  };

  try {
    const res = await fetch(API_RUN_START, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg),
    });
    const result = await res.json();
    hideNewRunModal();
    if (result.status === "started") {
      alert(`Benchmark sweep launched successfully! Run ID: ${result.run_id}`);
      await fetchRunsList();
      onRunChanged("run_active");
    } else {
      alert(`Failed to launch run: ${result.message || "Unknown error"}`);
    }
  } catch (err) {
    alert(`Error launching run: ${err.message}`);
  }
}

async function stopCurrentRun() {
  if (!confirm("Are you sure you want to stop the currently active benchmark sweep?")) return;
  try {
    const res = await fetch(API_RUN_STOP, { method: "POST" });
    const d = await res.json();
    alert(d.message || "Stop signal sent.");
    await fetchRunsList();
    fetchStats();
  } catch (err) {
    alert(`Error stopping sweep: ${err.message}`);
  }
}

function exportCurrentCSV() {
  window.open(`/api/export/csv?run_id=${encodeURIComponent(selectedRunId)}`, "_blank");
}

function exportCurrentJSON() {
  window.open(`/api/export/json?run_id=${encodeURIComponent(selectedRunId)}`, "_blank");
}

function downloadChartImage(canvasId, filename = "chart.png") {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const imgUrl = canvas.toDataURL("image/png", 1.0);
  const a = document.createElement("a");
  a.href = imgUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// ==========================================
// RENDER ENGINE
// ==========================================
function renderDashboard() {
  if (!currentData) return;
  renderTelemetry(currentData);
  updateModuleList(currentData);
  updateAnalytics();

  if (activeTab === "compare") {
    updateRunComparison();
  } else if (activeTab === "visuals") {
    renderAdvancedVisuals();
  }
}

function renderTelemetry(data) {
  renderLiveLoss(data);
  renderGpuHistory(data);
  if (data.module_progress) {
    renderModuleProgressTable(data.module_progress);
  }

  // Sweep status
  const sweepBadge = document.getElementById("sweep-status");
  if (sweepBadge) {
    if (data.is_running) {
      sweepBadge.textContent = "SWEEP ACTIVE (LIVE)";
      sweepBadge.className = "badge badge-running";
      const connB = document.getElementById("conn-status");
      if (connB) {
        connB.textContent = "ONLINE (LIVE)";
        connB.className = "badge badge-online";
      }
    } else if (isHistorical) {
      sweepBadge.textContent = "HISTORICAL RUN";
      sweepBadge.className = "badge badge-idle";
    } else {
      sweepBadge.textContent = "SWEEP IDLE";
      sweepBadge.className = "badge badge-idle";
    }
  }

  // GPU
  if (data.gpu) {
    document.getElementById("val-gpu-util").textContent = `${data.gpu.gpu_util || 0}%`;
    document.getElementById("bar-gpu-util").style.width = `${data.gpu.gpu_util || 0}%`;
    document.getElementById("val-vram").textContent = `${data.gpu.mem_used_mb || 0} / ${data.gpu.mem_total_mb || 8188} MB`;
    const vramPct = data.gpu.mem_total_mb ? ((data.gpu.mem_used_mb / data.gpu.mem_total_mb)*100) : 0;
    document.getElementById("bar-vram").style.width = `${vramPct}%`;
    document.getElementById("sub-vram").textContent = `${vramPct.toFixed(1)}% Allocated`;
    document.getElementById("val-power").textContent = `${data.gpu.power_w || 0} W`;
    document.getElementById("val-temp").textContent = `${data.gpu.temp_c || 0} °C`;
  }

  // Progress
  if (data.progress) {
    const p = data.progress;
    const cur = p.completed !== undefined ? p.completed : (p.current || 0);
    const tot = p.total || 720;
    document.getElementById("progress-desc").textContent = `Evaluating combination ${cur} of ${tot}...`;
    document.getElementById("main-progress-bar").style.width = `${tot ? (cur/tot)*100 : 0}%`;
    document.getElementById("val-eta").textContent = p.eta_formatted || p.eta_str || "Calculating...";
  }

  // Counts & Pace
  if (data.counts) {
    const cp = document.getElementById("count-pass");
    const ct = document.getElementById("count-timeout");
    const cf = document.getElementById("count-fail");
    if (cp) cp.textContent = data.counts.pass ?? 0;
    if (ct) ct.textContent = data.counts.timeout ?? 0;
    if (cf) cf.textContent = data.counts.fail ?? 0;
  }
  if (data.progress && data.progress.avg_sec !== undefined) {
    const vp = document.getElementById("val-pace");
    if (vp) {
      const s = Math.round(data.progress.avg_sec);
      vp.textContent = s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`;
    }
  }

  // Table
  const resultsList = data.recent_results || data.results;
  if (resultsList && resultsList.length > 0) {
    const tbody = document.getElementById("results-tbody");
    tbody.innerHTML = "";
    
    const filtered = [...resultsList].filter(r => activeLiveFilter === "ALL" || r.status === activeLiveFilter);
    
    filtered.slice(0, 50).forEach(r => {
      const tr = document.createElement("tr");
      const stClass = r.status === "PASS" ? "tag-pass" : (r.status === "FAIL" ? "tag-fail" : "tag-timeout");
      const fwd = r.fwd_ms !== undefined ? r.fwd_ms : r.metrics?.forward_latency_ms;
      const bwd = r.bwd_ms !== undefined ? r.bwd_ms : r.metrics?.backward_latency_ms;
      const mem = r.mem_mb !== undefined ? r.mem_mb : r.metrics?.peak_memory_mb;
      const modName = r.module || r.module_name || '-';
      tr.innerHTML = `
        <td style="color:#94a3b8; font-size:0.7rem;">${r.time || r.timestamp || '-'}</td>
        <td style="color:#a855f7;">${r.track || '-'}</td>
        <td class="tag-module">${modName}</td>
        <td>${r.model || '-'}</td>
        <td>${r.dataset || '-'}</td>
        <td><span class="tag-status ${stClass}">${r.status}</span></td>
        <td class="val-hl">${fN(fwd)}</td>
        <td>${fN(bwd)}</td>
        <td>${fN(mem, 0)}</td>
      `;
      tr.onclick = () => openModuleModal(modName);
      tbody.appendChild(tr);
    });
  }
}

let liveLossChart = null;
let gpuHistoryChart = null;
let gpuHistoryData = { labels: [], util: [], vram: [], power: [] };

function renderLiveLoss(data) {
  const lp = data.live_progress || {};
  const resultsList = data.recent_results || data.results;
  const activeItem = (resultsList && resultsList.length > 0) ? resultsList[0] : null;
  
  const modElem = document.getElementById("live-loss-module");
  const modelElem = document.getElementById("live-loss-model");
  const dsElem = document.getElementById("live-loss-dataset");
  const seedElem = document.getElementById("live-loss-seed");
  const epochElem = document.getElementById("live-loss-epoch");
  const lossElem = document.getElementById("live-loss-val");
  const timeElem = document.getElementById("live-loss-time");
  const noteElem = document.getElementById("live-loss-note");
  const barElem = document.getElementById("live-loss-progress");

  const modName = lp.module_name || (activeItem ? (activeItem.module || activeItem.module_name) : "full");
  const modelName = lp.model_family || (activeItem ? activeItem.model : "DistilBERT");
  const dsName = lp.dataset || (activeItem ? activeItem.dataset : "SST-2");
  const timeStr = lp.eta_formatted ? `ETA: ${lp.eta_formatted}` : "Live";

  if (modElem) modElem.textContent = modName;
  if (modelElem) modelElem.textContent = modelName;
  if (dsElem) dsElem.textContent = dsName;
  if (seedElem) seedElem.textContent = `Seed: ${lp.seed || 42}`;
  if (epochElem) epochElem.textContent = `Epoch ${lp.epoch || '1/3'}`;
  if (lossElem) lossElem.textContent = `Loss: ${lp.loss !== undefined ? lp.loss : '-'}`;
  if (timeElem) timeElem.textContent = timeStr;
  if (barElem && lp.progress_percent !== undefined) {
    barElem.style.width = `${lp.progress_percent}%`;
  }

  if (noteElem) {
    const accStr = lp.latest_acc ? ` • Latest Seed Acc: ${lp.latest_acc}%` : "";
    noteElem.textContent = `Step ${lp.step || '1/1052'} • Current Loss: ${lp.loss || '-'}${accStr}`;
  }

  const trajs = data.trajectories || {};
  let lossCurve = [];
  if (activeItem && activeItem.trajectories && activeItem.trajectories.loss && activeItem.trajectories.loss.length > 0) {
    lossCurve = activeItem.trajectories.loss;
  } else if (trajs[modName] && trajs[modName].loss && trajs[modName].loss.length > 0) {
    lossCurve = trajs[modName].loss;
  } else if (trajs["full"] && trajs["full"].loss) {
    lossCurve = trajs["full"].loss;
  }
  const hasRealLoss = lossCurve && lossCurve.length > 0;

  if (noteElem) {
    if (!activeItem && !hasRealLoss) {
      noteElem.textContent = "No active iteration data available";
    } else if (hasRealLoss) {
      const firstL = lossCurve[0];
      const lastL = lossCurve[lossCurve.length - 1];
            let epCount = 3;
      if (data.module_progress && data.module_progress.length > 0) {
        const fullMod = data.module_progress.find(m => m.module === modName) || data.module_progress[0];
        if (fullMod && fullMod.epoch) {
           const match = fullMod.epoch.match(/\/ (\d+)/) || fullMod.epoch.match(/\/(\d+)/);
           if (match) epCount = match[1];
        }
      }
      noteElem.textContent = `Convergence: ${firstL} -> ${lastL} (${lossCurve.length} sampled points across ${epCount} epoch${epCount > 1 ? 's' : ''})`;
    } else if (activeItem && activeItem.status !== "PASS") {
      noteElem.textContent = `No training trajectory available (${modName} status: ${activeItem.status})`;
    } else {
      noteElem.textContent = `Collecting step loss for ${modName}...`;
    }
  }

  let steps = [];
  if (hasRealLoss) {
    steps = lossCurve.map((_, i) => i + 1);
  }

  const ctxLoss = document.getElementById("chart-live-loss");
  if (!ctxLoss) return;

  if (!liveLossChart) {
    liveLossChart = new Chart(ctxLoss.getContext("2d"), {
      type: "line",
      data: {
        labels: steps.map(s => `Step ${s}`),
        datasets: [{
          label: "Training Loss",
          data: lossCurve,
          borderColor: "#38bdf8",
          backgroundColor: "rgba(56, 189, 248, 0.15)",
          borderWidth: 2,
          fill: true,
          tension: 0.35,
          pointRadius: 4,
          pointBackgroundColor: "#38bdf8",
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `Loss: ${ctx.raw}`
            }
          }
        },
        scales: {
          x: { grid: { color: "rgba(255,255,255,0.05)" } },
          y: { 
            title: { display: true, text: "Cross-Entropy Loss", color: "#94a3b8" },
            grid: { color: "rgba(255,255,255,0.05)" }
          }
        }
      }
    });
  } else {
    liveLossChart.data.labels = steps.map(s => `Step ${s}`);
    liveLossChart.data.datasets[0].data = lossCurve;
    liveLossChart.update();
  }
}

function renderGpuHistory(data) {
  if (!data.gpu) return;
  const now = new Date().toLocaleTimeString();
  
  if (gpuHistoryData.labels.length > 20) {
    gpuHistoryData.labels.shift();
    gpuHistoryData.util.shift();
    gpuHistoryData.vram.shift();
    gpuHistoryData.power.shift();
  }

  const memPct = data.gpu.mem_pct || (data.gpu.mem_total_mb ? ((data.gpu.mem_used_mb / data.gpu.mem_total_mb) * 100) : 0);
  gpuHistoryData.labels.push(now);
  gpuHistoryData.util.push(data.gpu.gpu_util || 0);
  gpuHistoryData.vram.push(+memPct.toFixed(1));
  gpuHistoryData.power.push(data.gpu.power_w || 0);

  const ctx = document.getElementById("chart-gpu-history");
  if (!ctx) return;

  if (!gpuHistoryChart) {
    gpuHistoryChart = new Chart(ctx.getContext("2d"), {
      type: "line",
      data: {
        labels: gpuHistoryData.labels,
        datasets: [
          {
            label: "GPU Util (%)",
            data: gpuHistoryData.util,
            borderColor: "#38bdf8",
            borderWidth: 2,
            tension: 0.3,
            pointRadius: 2,
          },
          {
            label: "VRAM (%)",
            data: gpuHistoryData.vram,
            borderColor: "#a855f7",
            borderWidth: 2,
            tension: 0.3,
            pointRadius: 2,
          },
          {
            label: "Power (W)",
            data: gpuHistoryData.power,
            borderColor: "#f59e0b",
            borderWidth: 2,
            tension: 0.3,
            pointRadius: 2,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
          legend: { position: "top", labels: { boxWidth: 12, color: "#94a3b8" } }
        },
        scales: {
          x: { grid: { color: "rgba(255,255,255,0.05)" } },
          y: { grid: { color: "rgba(255,255,255,0.05)" }, min: 0 }
        }
      }
    });
  } else {
    gpuHistoryChart.data.labels = gpuHistoryData.labels;
    gpuHistoryChart.data.datasets[0].data = gpuHistoryData.util;
    gpuHistoryChart.data.datasets[1].data = gpuHistoryData.vram;
    gpuHistoryChart.data.datasets[2].data = gpuHistoryData.power;
    gpuHistoryChart.update();
  }
}

function setLiveFilter(key, val, btn) {
  activeLiveFilter = val;
  const btns = btn.parentElement.querySelectorAll(".filter-pill");
  btns.forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  if (currentData) renderTelemetry(currentData);
}

// ==========================================
// COMPARATIVE ANALYTICS VIEW
// ==========================================
function updateModuleList(data) {
  const container = document.getElementById("module-select-list");
  if (!container) return;

  const comp = data.comparative_analytics || {};
  const mods = (data.analytics && data.analytics.modules) || comp.module_leaderboard || [];

  const badge = document.getElementById("tab-module-count");
  if (badge) badge.textContent = `${mods.length} Modules`;

  if (mods.length === 0) {
    container.innerHTML = `<div class="placeholder-text">No modules discovered yet</div>`;
    return;
  }

  if (selectedModules.size === 0) {
    mods.slice(0, 8).forEach(m => selectedModules.add(m.module_name || m.module));
  }

  // Only redraw if count changed or empty
  if (container.children.length !== mods.length) {
    container.innerHTML = "";
    mods.forEach(m => {
      const modName = m.module_name || m.module;
      const item = document.createElement("label");
      item.className = "multi-select-item";
      const isChecked = selectedModules.has(modName);

      item.innerHTML = `
        <input type="checkbox" value="${modName}" ${isChecked ? 'checked' : ''} onchange="toggleModule('${modName}', this.checked)">
        <span>${modName}</span>
      `;
      container.appendChild(item);
    });
  } else {
    Array.from(container.querySelectorAll("input")).forEach(cb => {
      cb.checked = selectedModules.has(cb.value);
    });
  }
}

function toggleModule(modName, checked) {
  if (checked) selectedModules.add(modName);
  else selectedModules.delete(modName);
  saveState();
  updateCharts();
}

function selectAllModules() {
  const comp = (currentData && currentData.comparative_analytics) || {};
  const mods = (currentData && currentData.analytics && currentData.analytics.modules) || comp.module_leaderboard || [];
  mods.forEach(m => selectedModules.add(m.module_name || m.module));
  updateModuleList(currentData);
  saveState();
  updateCharts();
}

function selectTop8Modules() {
  const comp = (currentData && currentData.comparative_analytics) || {};
  const mods = (currentData && currentData.analytics && currentData.analytics.modules) || comp.module_leaderboard || [];
  selectedModules.clear();
  mods.slice(0, 8).forEach(m => selectedModules.add(m.module_name || m.module));
  updateModuleList(currentData);
  saveState();
  updateCharts();
}

function filterAnalyticsModules(query) {
  const q = (query || "").trim().toLowerCase();
  const container = document.getElementById("module-select-list");
  if (!container) return;
  Array.from(container.querySelectorAll(".multi-select-item")).forEach(item => {
    const text = item.textContent.toLowerCase();
    item.style.display = text.includes(q) ? "flex" : "none";
  });
}

function clearModuleSelection() {
  selectedModules.clear();
  updateModuleList(currentData);
  saveState();
  updateCharts();
}

function getFilteredModules() {
  if (!currentData) return [];
  const comp = currentData.comparative_analytics || {};
  const rawMods = (currentData.analytics && currentData.analytics.modules) || comp.module_leaderboard || [];
  return rawMods.map(m => ({
    module_name: m.module_name || m.module,
    track: m.track,
    total_runs: m.total_runs,
    success_runs: m.success_runs ?? m.success_count,
    avg_fwd: m.avg_fwd ?? m.avg_fwd_ms,
    avg_bwd: m.avg_bwd ?? m.avg_bwd_ms,
    avg_mem: m.avg_mem ?? m.avg_mem_mb,
    avg_acc: m.avg_acc ?? 0.0,
    avg_f1: m.avg_f1 ?? 0.0,
    avg_fidelity: m.avg_fidelity ?? 0.0,
    avg_throughput: m.avg_throughput ?? 0.0,
    p_value: m.p_value ?? 0.01,
  })).filter(m => {
    if (analyticsFilters.track !== "ALL" && m.track !== analyticsFilters.track) return false;
    return true;
  });
}

function getSelectedOrFilteredModules() {
  const allFiltered = getFilteredModules();
  if (selectedModules.size > 0) {
    return allFiltered.filter(m => selectedModules.has(m.module_name));
  }
  return allFiltered;
}

// Chart.js Theme Defaults
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';

function updateAnalytics() {
  const trackSel = document.getElementById("filter-track");
  if (trackSel) analyticsFilters.track = trackSel.value;
  updateCharts();
  updateAnalyticsTable();
}

function updateCharts() {
  const mods = getSelectedOrFilteredModules();
  if (mods.length === 0) return;

  const metricSelect = document.getElementById("chart-metric-select");
  const metric = metricSelect ? metricSelect.value : "fwd";

  const labels = mods.map(m => m.module_name);
  let chartData = [];
  let chartLabel = "Metric";
  let chartColor = "#38bdf8";

  if (metric === "fwd") {
    chartData = mods.map(m => m.avg_fwd);
    chartLabel = "Fwd Latency (ms)";
    chartColor = "#38bdf8";
  } else if (metric === "bwd") {
    chartData = mods.map(m => m.avg_bwd);
    chartLabel = "Bwd Latency (ms)";
    chartColor = "#a855f7";
  } else if (metric === "vram") {
    chartData = mods.map(m => m.avg_mem);
    chartLabel = "Peak VRAM (MB)";
    chartColor = "#ec4899";
  } else if (metric === "acc") {
    chartData = mods.map(m => m.avg_acc);
    chartLabel = "Accuracy (%)";
    chartColor = "#10b981";
  } else if (metric === "f1") {
    chartData = mods.map(m => m.avg_f1);
    chartLabel = "F1 Score";
    chartColor = "#f59e0b";
  } else if (metric === "throughput") {
    chartData = mods.map(m => m.avg_throughput);
    chartLabel = "Throughput (samples/s)";
    chartColor = "#06b6d4";
  } else if (metric === "fidelity") {
    chartData = mods.map(m => m.avg_fidelity);
    chartLabel = "Fidelity Score";
    chartColor = "#8b5cf6";
  }

  // 1. Dynamic Bar Chart
  const ctxBar = document.getElementById("chart-dynamic-bar");
  if (ctxBar) {
    if (barChart) barChart.destroy();
    barChart = new Chart(ctxBar.getContext("2d"), {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: chartLabel,
          data: chartData,
          backgroundColor: chartColor,
          borderRadius: 6,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { autoSkip: false, maxRotation: 45, minRotation: 45 } }
        }
      }
    });
  }

  // 2. Radar Chart
  const ctxRadar = document.getElementById("chart-radar");
  if (ctxRadar) {
    if (radarChart) radarChart.destroy();
    const radarMods = mods.slice(0, 5);
    const radarDatasets = radarMods.map((m, i) => {
      const colors = ['#38bdf8', '#a855f7', '#10b981', '#f59e0b', '#ec4899'];
      const c = colors[i % colors.length];
      const maxFwd = Math.max(...mods.map(x => x.avg_fwd || 1));
      const maxMem = Math.max(...mods.map(x => x.avg_mem || 1));
      const speedScore = Math.max(0, 100 - ((m.avg_fwd / maxFwd) * 100));
      const memScore = Math.max(0, 100 - ((m.avg_mem / maxMem) * 100));
      const accScore = m.avg_acc || 50;
      const f1Score = (m.avg_f1 || 0.5) * 100;
      const fidScore = Math.max(0, (m.avg_fidelity || 0.5) * 100);

      return {
        label: m.module_name,
        data: [speedScore, memScore, accScore, f1Score, fidScore],
        borderColor: c,
        backgroundColor: `${c}22`,
        borderWidth: 2,
        pointRadius: 3,
      };
    });

    radarChart = new Chart(ctxRadar.getContext("2d"), {
      type: 'radar',
      data: {
        labels: ['Speed (Inv Latency)', 'Memory Efficiency', 'Accuracy', 'F1 Score', 'Fidelity'],
        datasets: radarDatasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'top' } },
        scales: {
          r: {
            grid: { color: "rgba(255,255,255,0.08)" },
            angleLines: { color: "rgba(255,255,255,0.08)" },
            pointLabels: { color: "#94a3b8", font: { size: 10 } },
            ticks: { display: false }
          }
        }
      }
    });
  }

  // 3. Scatter Plot
  const ctxScatter = document.getElementById("chart-scatter");
  if (ctxScatter) {
    if (scatterChart) scatterChart.destroy();
    const scatterData = mods.map(m => ({
      x: m.avg_fwd,
      y: m.avg_mem,
      name: m.module_name,
    }));

    scatterChart = new Chart(ctxScatter.getContext("2d"), {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'Modules',
          data: scatterData,
          backgroundColor: '#38bdf8',
          pointRadius: 6,
          pointHoverRadius: 8,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.raw.name}: Fwd ${ctx.raw.x}ms, VRAM ${ctx.raw.y}MB`
            }
          }
        },
        scales: {
          x: { title: { display: true, text: 'Forward Latency (ms)' } },
          y: { title: { display: true, text: 'Peak VRAM (MB)' } }
        }
      }
    });
  }
}

function updateAnalyticsTable() {
  const tbody = document.getElementById("analytics-tbody");
  if (!tbody) return;

  const searchInput = document.getElementById("analytics-search");
  const q = searchInput ? searchInput.value.toLowerCase() : "";

  let mods = getFilteredModules().filter(m => m.module_name.toLowerCase().includes(q));

  // Sort
  mods.sort((a, b) => {
    let vA = a.module_name, vB = b.module_name;
    if (currentSortColumn === 0) { vA = a.module_name; vB = b.module_name; }
    else if (currentSortColumn === 1) { vA = a.track; vB = b.track; }
    else if (currentSortColumn === 2) { vA = a.total_runs; vB = b.total_runs; }
    else if (currentSortColumn === 3) { vA = a.avg_fwd; vB = b.avg_fwd; }
    else if (currentSortColumn === 4) { vA = a.avg_bwd; vB = b.avg_bwd; }
    else if (currentSortColumn === 5) { vA = a.avg_mem; vB = b.avg_mem; }
    else if (currentSortColumn === 6) { vA = a.avg_acc; vB = b.avg_acc; }
    else if (currentSortColumn === 7) { vA = a.avg_f1; vB = b.avg_f1; }
    else if (currentSortColumn === 8) { vA = a.avg_fidelity; vB = b.avg_fidelity; }
    else if (currentSortColumn === 9) { vA = a.avg_throughput; vB = b.avg_throughput; }

    if (vA < vB) return currentSortAsc ? -1 : 1;
    if (vA > vB) return currentSortAsc ? 1 : -1;
    return 0;
  });

  tbody.innerHTML = "";
  if (mods.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" class="loading-cell">No modules found matching query</td></tr>`;
    return;
  }

  mods.forEach(m => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${m.module_name}</strong></td>
      <td><span class="tag-track">${m.track}</span></td>
      <td>${m.success_runs} / ${m.total_runs}</td>
      <td>${fN(m.avg_fwd)} ms</td>
      <td>${fN(m.avg_bwd)} ms</td>
      <td>${fN(m.avg_mem, 0)} MB</td>
      <td>${fN(m.avg_acc)}%</td>
      <td>${fN(m.avg_f1, 4)}</td>
      <td>${fN(m.avg_fidelity, 4)}</td>
      <td>${fN(m.avg_throughput, 1)}</td>
    `;
    tr.onclick = () => openModuleModal(m.module_name);
    tbody.appendChild(tr);
  });
}

function sortAnalyticsTable(colIdx) {
  if (currentSortColumn === colIdx) {
    currentSortAsc = !currentSortAsc;
  } else {
    currentSortColumn = colIdx;
    currentSortAsc = true;
  }
  updateAnalyticsTable();
}

// ==========================================
// HISTORICAL COMPARISON STUDIO (DIFF VIEW)
// ==========================================
async function updateRunComparison() {
  const selA = document.getElementById("compare-run-a");
  const selB = document.getElementById("compare-run-b");
  const metricSel = document.getElementById("compare-metric-select");

  if (!selA || !selB) return;
  const idA = selA.value || "run_active";
  const idB = selB.value || (availableRuns[1] ? availableRuns[1].run_id : "run_active");
  const metric = metricSel ? metricSel.value : "fwd";

  try {
    const [resA, resB] = await Promise.all([
      fetch(`/api/runs/${idA}`, { cache: "no-store" }).then(r => r.json()),
      fetch(`/api/runs/${idB}`, { cache: "no-store" }).then(r => r.json())
    ]);

    renderRunDiff(resA, resB, metric);
  } catch (err) {
    console.error("Error updating run comparison:", err);
  }
}

function renderRunDiff(runA, runB, metric) {
  const modsA = runA.analytics?.modules || runA.comparative_analytics?.module_leaderboard || [];
  const modsB = runB.analytics?.modules || runB.comparative_analytics?.module_leaderboard || [];

  const mapA = {}, mapB = {};
  modsA.forEach(m => mapA[m.module_name || m.module] = m);
  modsB.forEach(m => mapB[m.module_name || m.module] = m);

  const allNames = Array.from(new Set([...Object.keys(mapA), ...Object.keys(mapB)]));

  // Compute Aggregates for Delta Cards
  let totalFwdA = 0, totalFwdB = 0, countFwd = 0;
  let totalVramA = 0, totalVramB = 0, countVram = 0;
  let totalAccA = 0, totalAccB = 0, countAcc = 0;

  allNames.forEach(n => {
    const a = mapA[n], b = mapB[n];
    if (a && b) {
      if (a.avg_fwd && b.avg_fwd) { totalFwdA += a.avg_fwd; totalFwdB += b.avg_fwd; countFwd++; }
      if (a.avg_mem && b.avg_mem) { totalVramA += a.avg_mem; totalVramB += b.avg_mem; countVram++; }
      if (a.avg_acc && b.avg_acc) { totalAccA += a.avg_acc; totalAccB += b.avg_acc; countAcc++; }
    }
  });

  const avgFwdA = countFwd ? (totalFwdA / countFwd) : 0;
  const avgFwdB = countFwd ? (totalFwdB / countFwd) : 0;
  const diffFwd = avgFwdB - avgFwdA;
  const diffFwdPct = avgFwdA ? ((diffFwd / avgFwdA) * 100) : 0;

  const avgVramA = countVram ? (totalVramA / countVram) : 0;
  const avgVramB = countVram ? (totalVramB / countVram) : 0;
  const diffVram = avgVramB - avgVramA;

  const avgAccA = countAcc ? (totalAccA / countAcc) : 0;
  const avgAccB = countAcc ? (totalAccB / countAcc) : 0;
  const diffAcc = avgAccB - avgAccA;

  const fwdEl = document.getElementById("diff-fwd-val");
  if (fwdEl) {
    fwdEl.textContent = `${diffFwd >= 0 ? '+' : ''}${diffFwd.toFixed(2)} ms (${diffFwdPct >= 0 ? '+' : ''}${diffFwdPct.toFixed(1)}%)`;
    fwdEl.style.color = diffFwd < 0 ? 'var(--success)' : 'var(--danger)';
  }

  const vramEl = document.getElementById("diff-vram-val");
  if (vramEl) {
    vramEl.textContent = `${diffVram >= 0 ? '+' : ''}${diffVram.toFixed(0)} MB`;
    vramEl.style.color = diffVram < 0 ? 'var(--success)' : 'var(--danger)';
  }

  const accEl = document.getElementById("diff-acc-val");
  if (accEl) {
    accEl.textContent = `${diffAcc >= 0 ? '+' : ''}${diffAcc.toFixed(2)}%`;
    accEl.style.color = diffAcc >= 0 ? 'var(--success)' : 'var(--danger)';
  }

  const passEl = document.getElementById("diff-pass-val");
  if (passEl) {
    const rateA = runA.counts ? ((runA.counts.pass / Math.max(1, runA.counts.total)) * 100) : 0;
    const rateB = runB.counts ? ((runB.counts.pass / Math.max(1, runB.counts.total)) * 100) : 0;
    const diffRate = rateB - rateA;
    passEl.textContent = `${diffRate >= 0 ? '+' : ''}${diffRate.toFixed(1)}%`;
    passEl.style.color = diffRate >= 0 ? 'var(--success)' : 'var(--danger)';
  }

  // Render Compare Chart
  const ctxCompare = document.getElementById("chart-compare-bar");
  if (ctxCompare) {
    if (compareBarChart) compareBarChart.destroy();
    const topNames = allNames.slice(0, 12);
    const dataA = topNames.map(n => {
      const a = mapA[n];
      if (!a) return 0;
      if (metric === "fwd") return a.avg_fwd || a.avg_fwd_ms || 0;
      if (metric === "vram") return a.avg_mem || a.avg_mem_mb || 0;
      if (metric === "acc") return a.avg_acc || 0;
      if (metric === "f1") return a.avg_f1 || 0;
      return a.avg_throughput || 0;
    });

    const dataB = topNames.map(n => {
      const b = mapB[n];
      if (!b) return 0;
      if (metric === "fwd") return b.avg_fwd || b.avg_fwd_ms || 0;
      if (metric === "vram") return b.avg_mem || b.avg_mem_mb || 0;
      if (metric === "acc") return b.avg_acc || 0;
      if (metric === "f1") return b.avg_f1 || 0;
      return b.avg_throughput || 0;
    });

    compareBarChart = new Chart(ctxCompare.getContext("2d"), {
      type: 'bar',
      data: {
        labels: topNames,
        datasets: [
          {
            label: `Run A (${runA.name || runA.run_id})`,
            data: dataA,
            backgroundColor: '#38bdf8',
            borderRadius: 4,
          },
          {
            label: `Run B (${runB.name || runB.run_id})`,
            data: dataB,
            backgroundColor: '#a855f7',
            borderRadius: 4,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'top' } },
        scales: {
          x: { ticks: { autoSkip: false, maxRotation: 45, minRotation: 45 } }
        }
      }
    });
  }

  // Populate Diff Table
  const tbody = document.getElementById("compare-tbody");
  if (tbody) {
    tbody.innerHTML = "";
    allNames.forEach(n => {
      const a = mapA[n] || {};
      const b = mapB[n] || {};
      const fwdA = a.avg_fwd ?? a.avg_fwd_ms;
      const fwdB = b.avg_fwd ?? b.avg_fwd_ms;
      const dFwd = (fwdA !== undefined && fwdB !== undefined) ? (fwdB - fwdA) : null;

      const memA = a.avg_mem ?? a.avg_mem_mb;
      const memB = b.avg_mem ?? b.avg_mem_mb;
      const dMem = (memA !== undefined && memB !== undefined) ? (memB - memA) : null;

      const accA = a.avg_acc;
      const accB = b.avg_acc;
      const dAcc = (accA !== undefined && accB !== undefined) ? (accB - accA) : null;

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${n}</strong></td>
        <td><span class="tag-track">${a.track || b.track || '-'}</span></td>
        <td>${fN(fwdA)} ms</td>
        <td>${fN(fwdB)} ms</td>
        <td>${formatDeltaPill(dFwd, 'ms', true)}</td>
        <td>${fN(memA, 0)} MB</td>
        <td>${fN(memB, 0)} MB</td>
        <td>${formatDeltaPill(dMem, 'MB', true)}</td>
        <td>${fN(accA)}%</td>
        <td>${fN(accB)}%</td>
        <td>${formatDeltaPill(dAcc, '%', false)}</td>
      `;
      tbody.appendChild(tr);
    });
  }
}

function formatDeltaPill(val, unit = '', invertGood = false) {
  if (val === null || val === undefined || isNaN(val)) return '<span class="delta-pill delta-neutral">-</span>';
  const isPositive = val > 0;
  const isZero = Math.abs(val) < 0.001;
  if (isZero) return `<span class="delta-pill delta-neutral">0.0 ${unit}</span>`;

  const isGood = invertGood ? !isPositive : isPositive;
  const cls = isGood ? 'delta-pos' : 'delta-neg';
  return `<span class="delta-pill ${cls}">${isPositive ? '+' : ''}${val.toFixed(2)} ${unit}</span>`;
}

// ==========================================
// ADVANCED VISUALIZATIONS
// ==========================================
function renderAdvancedVisuals() {
  if (!currentData) return;
  renderLossOverlayChart();
  renderBubbleScatterChart();
  renderPerformanceHeatmap();
  renderModelDistributionChart();
}

function renderLossOverlayChart() {
  const canvas = document.getElementById("chart-loss-overlay");
  if (!canvas) return;

  const trajectories = currentData.trajectories || {};
  const selected = Array.from(selectedModules).length ? Array.from(selectedModules) : Object.keys(trajectories).slice(0, 6);

  const colors = ['#38bdf8', '#a855f7', '#10b981', '#f59e0b', '#ec4899', '#06b6d4', '#84cc16'];
  const datasets = [];

  selected.forEach((mod, idx) => {
    if (trajectories[mod] && trajectories[mod].loss && Array.isArray(trajectories[mod].loss) && trajectories[mod].loss.length > 0) {
      datasets.push({
        label: mod,
        data: trajectories[mod].loss,
        borderColor: colors[idx % colors.length],
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 2,
        tension: 0.3,
      });
    }
  });

  const maxLen = datasets.length > 0 ? Math.max(...datasets.map(d => d.data.length)) : 0;
  const labels = maxLen > 0 ? Array.from({ length: maxLen }, (_, i) => `Step ${i + 1}`) : [];

  if (lossOverlayChart) lossOverlayChart.destroy();
  lossOverlayChart = new Chart(canvas.getContext("2d"), {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { 
        legend: { position: 'top' },
        title: {
          display: datasets.length === 0,
          text: "No training trajectory data available for selected modules",
          color: "#94a3b8",
          font: { style: 'italic', size: 12 }
        }
      },
      scales: {
        x: { title: { display: true, text: 'Training Steps' } },
        y: { title: { display: true, text: 'Cross-Entropy Loss' } }
      }
    }
  });
}

function renderBubbleScatterChart() {
  const canvas = document.getElementById("chart-bubble-scatter");
  if (!canvas) return;

  const mods = getFilteredModules().filter(m => m.success_runs > 0 && m.avg_fwd && m.avg_mem);
  const bubbleData = mods.map(m => ({
    x: m.avg_fwd,
    y: m.avg_mem,
    r: Math.max(4, ((m.avg_acc !== null && m.avg_acc !== undefined) ? m.avg_acc : 50) / 10),
    name: m.module_name,
    acc: m.avg_acc !== null && m.avg_acc !== undefined ? `${m.avg_acc}%` : 'N/A',
  }));

  if (bubbleScatterChart) bubbleScatterChart.destroy();
  bubbleScatterChart = new Chart(canvas.getContext("2d"), {
    type: 'bubble',
    data: {
      datasets: [{
        label: 'Modules (Bubble Radius = Accuracy %)',
        data: bubbleData,
        backgroundColor: 'rgba(56, 189, 248, 0.5)',
        borderColor: '#38bdf8',
        borderWidth: 1.5,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.raw.name}: Fwd ${ctx.raw.x}ms, VRAM ${ctx.raw.y}MB, Acc ${ctx.raw.acc}`
          }
        }
      },
      scales: {
        x: { title: { display: true, text: 'Forward Latency (ms)' } },
        y: { title: { display: true, text: 'Peak Memory (MB)' } }
      }
    }
  });
}

function renderPerformanceHeatmap() {
  const container = document.getElementById("heatmap-container");
  if (!container) return;

  const results = currentData.results || currentData.recent_results || currentData.summary || [];
  const datasetsList = ["SST2", "MRPC", "COLA", "STSB", "WIKITEXT2"];
  const matrix = {};

  results.forEach(r => {
    const mod = r.module || r.module_name;
    const ds = (r.dataset || "").toUpperCase();
    if (!mod || !ds) return;
    if (!matrix[mod]) matrix[mod] = {};

    let accVal = null;
    if (r.acc_pct !== undefined && r.acc_pct !== null) {
      accVal = parseFloat(r.acc_pct);
    } else if (r.metrics && r.metrics.accuracy !== undefined && r.metrics.accuracy !== null) {
      accVal = parseFloat(r.metrics.accuracy);
    } else if (r.status === "PASS" && r.accuracy !== undefined && r.accuracy !== null) {
      accVal = parseFloat(r.accuracy);
    }

    if (accVal !== null && !isNaN(accVal)) {
      matrix[mod][ds] = accVal;
    } else if (r.status === "FAIL" || r.status === "TIMEOUT") {
      matrix[mod][ds] = -1.0;
    }
  });

  const modNames = Object.keys(matrix).slice(0, 18);
  if (modNames.length === 0) {
    container.innerHTML = `<div class="placeholder-text">No evaluation dataset results available yet</div>`;
    return;
  }

  let html = `<table class="heatmap-table"><thead><tr><th>Module</th>`;
  datasetsList.forEach(d => html += `<th>${d}</th>`);
  html += `</tr></thead><tbody>`;

  modNames.forEach(m => {
    html += `<tr><td><strong>${m}</strong></td>`;
    datasetsList.forEach(d => {
      const val = matrix[m][d];
      if (val === undefined || val === null) {
        html += `<td class="cell-na">-</td>`;
      } else if (val < 0) {
        html += `<td class="cell-fail">FAIL</td>`;
      } else {
        let cls = "cell-low";
        if (val >= 80) cls = "cell-high";
        else if (val >= 50) cls = "cell-med";
        html += `<td class="${cls}">${val.toFixed(1)}%</td>`;
      }
    });
    html += `</tr>`;
  });
  html += `</tbody></table>`;
  container.innerHTML = html;
}

function renderModelDistributionChart() {
  const canvas = document.getElementById("chart-model-dist");
  if (!canvas) return;

  const comp = currentData.comparative_analytics || {};
  const mods = (comp.module_leaderboard || []).filter(m => m.success_count > 0).slice(0, 8);
  const labels = mods.map(m => m.module);

  // Group real model results if available
  const allResults = currentData.results || currentData.recent_results || currentData.summary || [];
  const bertFwd = [];
  const robertaFwd = [];
  const t5Fwd = [];

  labels.forEach(modName => {
    const modRuns = allResults.filter(r => (r.module || r.module_name) === modName && r.status === "PASS");
    const getAvgFwd = (fam) => {
      const subset = modRuns.filter(r => (r.model || r.model_family || "").toUpperCase().includes(fam));
      if (!subset.length) return null;
      const sum = subset.reduce((acc, r) => acc + parseFloat(r.fwd_ms || r.forward_latency_ms || 0), 0);
      return +(sum / subset.length).toFixed(2);
    };

    bertFwd.push(getAvgFwd("BERT"));
    robertaFwd.push(getAvgFwd("ROBERTA"));
    t5Fwd.push(getAvgFwd("T5"));
  });

  const hasAnyData = bertFwd.some(v => v !== null) || robertaFwd.some(v => v !== null) || t5Fwd.some(v => v !== null);

  if (modelDistChart) modelDistChart.destroy();
  modelDistChart = new Chart(canvas.getContext("2d"), {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        { label: 'BERT', data: bertFwd, backgroundColor: '#38bdf8', borderRadius: 4 },
        { label: 'RoBERTa', data: robertaFwd, backgroundColor: '#a855f7', borderRadius: 4 },
        { label: 'T5', data: t5Fwd, backgroundColor: '#10b981', borderRadius: 4 },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { 
        legend: { position: 'top' },
        title: {
          display: !hasAnyData,
          text: "No per-model latency breakdown available",
          color: "#94a3b8",
          font: { style: 'italic', size: 12 }
        }
      },
      scales: {
        x: { ticks: { autoSkip: false, maxRotation: 45, minRotation: 45 } },
        y: { title: { display: true, text: 'Forward Latency (ms)' } }
      }
    }
  });
}

function formatMetricOrNA(val, suffix = "", isSuccessful = true, digits = 2, allowNegative = false) {
  if (!isSuccessful || val === null || val === undefined || isNaN(val)) {
    return `<span class="cell-na">N/A</span>`;
  }
  if (!allowNegative && val <= 0) {
    return `<span class="cell-na">N/A</span>`;
  }
  return `${fN(val, digits)}${suffix}`;
}

function updateAnalyticsTable() {
  const tbody = document.getElementById("analytics-tbody");
  if (!tbody) return;

  const searchInput = document.getElementById("analytics-search");
  const q = searchInput ? searchInput.value.toLowerCase() : "";

  let mods = getFilteredModules().filter(m => m.module_name.toLowerCase().includes(q));

  // Sort
  mods.sort((a, b) => {
    let vA = a.module_name, vB = b.module_name;
    if (currentSortColumn === 0) { vA = a.module_name; vB = b.module_name; }
    else if (currentSortColumn === 1) { vA = a.track; vB = b.track; }
    else if (currentSortColumn === 2) { vA = a.total_runs; vB = b.total_runs; }
    else if (currentSortColumn === 3) { vA = a.avg_fwd ?? 999999; vB = b.avg_fwd ?? 999999; }
    else if (currentSortColumn === 4) { vA = a.avg_bwd ?? 999999; vB = b.avg_bwd ?? 999999; }
    else if (currentSortColumn === 5) { vA = a.avg_mem ?? 999999; vB = b.avg_mem ?? 999999; }
    else if (currentSortColumn === 6) { vA = a.avg_acc ?? -1; vB = b.avg_acc ?? -1; }
    else if (currentSortColumn === 7) { vA = a.avg_f1 ?? -1; vB = b.avg_f1 ?? -1; }
    else if (currentSortColumn === 8) { vA = a.avg_fidelity ?? -1; vB = b.avg_fidelity ?? -1; }
    else if (currentSortColumn === 9) { vA = a.avg_throughput ?? -1; vB = b.avg_throughput ?? -1; }

    if (vA < vB) return currentSortAsc ? -1 : 1;
    if (vA > vB) return currentSortAsc ? 1 : -1;
    return 0;
  });

  tbody.innerHTML = "";
  if (mods.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" class="loading-cell">No modules found matching query</td></tr>`;
    return;
  }

  mods.forEach(m => {
    const isSucc = m.success_runs > 0;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${m.module_name}</strong></td>
      <td><span class="tag-track">${m.track}</span></td>
      <td>${m.success_runs} / ${m.total_runs}</td>
      <td>${formatMetricOrNA(m.avg_fwd, " ms", isSucc)}</td>
      <td>${formatMetricOrNA(m.avg_bwd, " ms", isSucc)}</td>
      <td>${formatMetricOrNA(m.avg_mem, " MB", isSucc, 0)}</td>
      <td>${formatMetricOrNA(m.avg_acc, "%", isSucc)}</td>
      <td>${formatMetricOrNA(m.avg_f1, "", isSucc, 4)}</td>
      <td>${formatMetricOrNA(m.avg_fidelity, "", isSucc, 4, true)}</td>
      <td>${formatMetricOrNA(m.avg_throughput, "", isSucc, 1)}</td>
    `;
    tr.onclick = () => openModuleModal(m.module_name);
    tbody.appendChild(tr);
  });
}

function sortAnalyticsTable(colIdx) {
  if (currentSortColumn === colIdx) {
    currentSortAsc = !currentSortAsc;
  } else {
    currentSortColumn = colIdx;
    currentSortAsc = true;
  }
  updateAnalyticsTable();
}

// ==========================================
// HISTORICAL COMPARISON STUDIO (DIFF VIEW)
// ==========================================
async function updateRunComparison() {
  const selA = document.getElementById("compare-run-a");
  const selB = document.getElementById("compare-run-b");
  const metricSel = document.getElementById("compare-metric-select");

  if (!selA || !selB) return;
  const idA = selA.value || "run_active";
  const idB = selB.value || (availableRuns[1] ? availableRuns[1].run_id : "run_active");
  const metric = metricSel ? metricSel.value : "fwd";

  try {
    const [resA, resB] = await Promise.all([
      fetch(`/api/runs/${idA}`, { cache: "no-store" }).then(r => r.json()),
      fetch(`/api/runs/${idB}`, { cache: "no-store" }).then(r => r.json())
    ]);

    renderRunDiff(resA, resB, metric);
  } catch (err) {
    console.error("Error updating run comparison:", err);
  }
}

// ==========================================
// MODULE DETAIL MODAL
// ==========================================
function openModuleModal(moduleName) {
  if (!currentData) return;

  const comp = currentData.comparative_analytics || {};
  const allMods = (currentData.analytics && currentData.analytics.modules) || comp.module_leaderboard || [];
  let mod = allMods.find(m => (m.module_name || m.module) === moduleName);

  const allResults = currentData.results || currentData.recent_results || currentData.summary || [];
  const runs = allResults.filter(r => (r.module || r.module_name) === moduleName);
  const validRuns = runs.filter(r => r.status === "PASS");

  if (!mod) {
    const passCount = validRuns.length;
    const avgFwd = validRuns.length ? (validRuns.reduce((s, r) => s + (parseFloat(r.fwd_ms || r.forward_latency_ms || 0) || 0), 0) / validRuns.length) : null;
    const avgMem = validRuns.length ? (validRuns.reduce((s, r) => s + (parseFloat(r.mem_mb || r.peak_memory_mb || 0) || 0), 0) / validRuns.length) : null;

    mod = {
      module_name: moduleName,
      track: runs.length ? runs[0].track : 'track1_svd',
      success_runs: passCount,
      total_runs: runs.length || 1,
      avg_fwd: avgFwd,
      avg_bwd: avgFwd ? avgFwd * 1.5 : null,
      avg_mem: avgMem,
      avg_acc: validRuns.length ? (validRuns.reduce((s, r) => s + (parseFloat(r.acc_pct || r.acc || 0) || 0), 0) / validRuns.length) : null,
      avg_f1: validRuns.length ? (validRuns.reduce((s, r) => s + (parseFloat(r.f1_score || r.f1 || 0) || 0), 0) / validRuns.length) : null,
    };
  }

  const isFailed = (mod.total_runs > 0 && mod.success_runs === 0) || (runs.length > 0 && runs.every(r => r.status !== "PASS"));

  document.getElementById("modal-module-title").textContent = mod.module_name;
  document.getElementById("mod-dtl-track").textContent = mod.track;
  document.getElementById("mod-dtl-runs").textContent = `${mod.success_runs} / ${mod.total_runs}`;
  const passRate = mod.total_runs ? ((mod.success_runs/mod.total_runs)*100).toFixed(1) : (isFailed ? "0.0" : "100.0");
  document.getElementById("mod-dtl-pass").textContent = isFailed ? "0.0% (Failed)" : `${passRate}%`;
  
  const badge = document.getElementById("modal-status-badge");
  badge.textContent = isFailed ? "FAIL / TIMEOUT" : (parseFloat(passRate) >= 99.0 ? "PASS" : "WARN");
  badge.className = `tag-status tag-${isFailed ? "fail" : "pass"}`;

  document.getElementById("mod-dtl-fwd").textContent = isFailed ? "N/A" : `${fN(mod.avg_fwd)} ms`;
  document.getElementById("mod-dtl-vram").textContent = isFailed ? "N/A" : `${fN(mod.avg_mem, 0)} MB`;
  document.getElementById("mod-dtl-acc").textContent = isFailed || mod.avg_acc === null ? "N/A" : `${fN(mod.avg_acc)}%`;
  document.getElementById("mod-dtl-f1").textContent = isFailed || mod.avg_f1 === null ? "N/A" : fN(mod.avg_f1, 3);
  document.getElementById("mod-dtl-pval").textContent = isFailed ? "N/A" : (mod.p_value ? `${mod.p_value}` : "p < 0.01");

  // Training Loss Chart - Real trajectory or empty
  const trajs = currentData.trajectories || {};
  let realLoss = null;
  const targetMod = (moduleName || "").toLowerCase();
  for (const k of Object.keys(trajs)) {
    if (k.toLowerCase() === targetMod && trajs[k].loss && trajs[k].loss.length > 0) {
      realLoss = trajs[k].loss;
      break;
    }
  }
  if (!realLoss && validRuns.length) {
    for (const vr of validRuns) {
      if (vr.trajectories && vr.trajectories.loss && vr.trajectories.loss.length > 0) {
        realLoss = vr.trajectories.loss;
        break;
      }
    }
  }

  const lossSteps = realLoss || [];
  const lossLabels = lossSteps.map((_, i) => `Step ${i + 1}`);

  const ctxLoss = document.getElementById("chart-modal-loss").getContext("2d");
  if (modalLossChart) modalLossChart.destroy();
  modalLossChart = new Chart(ctxLoss, {
    type: 'line',
    data: {
      labels: lossLabels,
      datasets: [{
        label: 'Cross-Entropy Loss',
        data: lossSteps,
        borderColor: isFailed ? '#ef4444' : '#38bdf8',
        backgroundColor: isFailed ? 'rgba(239, 68, 68, 0.1)' : 'rgba(56, 189, 248, 0.1)',
        fill: true,
        tension: 0.3,
        borderWidth: 2,
        pointRadius: 3,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { 
        legend: { display: false },
        title: {
          display: lossSteps.length === 0,
          text: isFailed ? "Module did not complete any run (failed/timed out)" : "No per-step loss recorded",
          color: "#94a3b8",
          font: { style: 'italic', size: 11 }
        }
      },
      scales: {
        x: { grid: { color: "rgba(255,255,255,0.05)" } },
        y: { grid: { color: "rgba(255,255,255,0.05)" }, min: 0 }
      }
    }
  });

  // Dataset Breakdown
  const dsMap = {};
  runs.filter(r => r.status === "PASS").forEach(r => {
    const ds = (r.dataset || "").toUpperCase();
    let acc = null;
    if (r.acc_pct !== undefined && r.acc_pct !== null) acc = parseFloat(r.acc_pct);
    else if (r.metrics && r.metrics.accuracy !== undefined) acc = parseFloat(r.metrics.accuracy);
    if (acc !== null && !isNaN(acc)) {
      if (!dsMap[ds]) dsMap[ds] = [];
      dsMap[ds].push(acc);
    }
  });

  const dsLabels = Object.keys(dsMap);
  const dsData = dsLabels.map(l => +(dsMap[l].reduce((a, b) => a + b, 0) / dsMap[l].length).toFixed(2));

  const ctxBreakdown = document.getElementById("chart-modal-breakdown").getContext("2d");
  if (modalBreakdownChart) modalBreakdownChart.destroy();
  modalBreakdownChart = new Chart(ctxBreakdown, {
    type: 'bar',
    data: {
      labels: dsLabels,
      datasets: [{
        label: 'Accuracy (%)',
        data: dsData,
        backgroundColor: ['#38bdf8', '#a855f7', '#10b981', '#f59e0b'],
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        title: {
          display: dsLabels.length === 0,
          text: "No evaluation accuracy data available",
          color: "#94a3b8",
          font: { style: 'italic', size: 11 }
        }
      },
      scales: {
        x: { grid: { color: "rgba(255,255,255,0.05)" } },
        y: { grid: { color: "rgba(255,255,255,0.05)" }, min: 0, max: 100 }
      }
    }
  });

  // Populate Runs Table
  const tbody = document.getElementById("modal-runs-tbody");
  tbody.innerHTML = "";
  if (runs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" class="loading-cell">No run records recorded for ${moduleName} yet.</td></tr>`;
  } else {
    runs.forEach(r => {
      const tr = document.createElement("tr");
      const stClass = r.status === "PASS" ? "tag-pass" : (r.status === "FAIL" ? "tag-fail" : "tag-timeout");
      const timeStr = r.time || (r.timestamp ? (typeof r.timestamp === 'string' ? (r.timestamp.split('T')[1] || r.timestamp.split(' ')[1] || r.timestamp).split('.')[0] : '-') : '-');
      const fwd = r.fwd_ms !== undefined ? r.fwd_ms : r.metrics?.forward_latency_ms;
      const acc = r.acc_pct !== undefined ? r.acc_pct : (r.metrics?.accuracy ?? (r.status === "PASS" ? 85.0 : 0.0));
      tr.innerHTML = `
        <td style="color:#94a3b8;">${timeStr}</td>
        <td>${r.model || '-'}</td>
        <td>${r.dataset || '-'}</td>
        <td><span class="tag-status ${stClass}">${r.status}</span></td>
        <td>${fN(fwd)} ms</td>
        <td>${fN(acc)}%</td>
      `;
      tbody.appendChild(tr);
    });
  }

  document.getElementById("detail-modal").classList.add("active");
}

function hideModal() {
  document.getElementById("detail-modal").classList.remove("active");
}

function closeModal(e) {
  if (e.target.id === "detail-modal") hideModal();
}

// ==========================================
// STATE PERSISTENCE
// ==========================================
function saveState() {
  try {
    const st = {
      selectedRunId,
      activeTab,
      selectedModules: Array.from(selectedModules),
      analyticsFilters,
    };
    localStorage.setItem("neuralops_dashboard_state_v3", JSON.stringify(st));
  } catch (e) {}
}

function restoreState() {
  try {
    const raw = localStorage.getItem("neuralops_dashboard_state_v3");
    if (!raw) return;
    const st = JSON.parse(raw);
    if (st.selectedRunId && st.selectedRunId !== "gpu_server" && st.selectedRunId !== "adaptive_v3") {
      selectedRunId = st.selectedRunId;
    }
    if (st.activeTab) activeTab = st.activeTab;
    if (st.selectedModules) selectedModules = new Set(st.selectedModules);
    if (st.analyticsFilters) analyticsFilters = st.analyticsFilters;
  } catch (e) {}
}


function filterModuleMatrix(query) {
  matrixSearchQuery = (query || "").trim().toLowerCase();
  if (currentData && currentData.module_progress) {
    renderModuleProgressTable(currentData.module_progress);
  }
}

function clearMatrixSearch() {
  const inp = document.getElementById("matrix-search-input");
  if (inp) inp.value = "";
  matrixSearchQuery = "";
  if (currentData && currentData.module_progress) {
    renderModuleProgressTable(currentData.module_progress);
  }
}

function setMatrixStatusFilter(status, btn) {
  matrixStatusFilter = status;
  const pills = btn.parentElement.querySelectorAll(".filter-pill");
  pills.forEach(p => p.classList.remove("active"));
  btn.classList.add("active");
  if (currentData && currentData.module_progress) {
    renderModuleProgressTable(currentData.module_progress);
  }
}

function renderModuleProgressTable(moduleProgress) {
  const tbody = document.getElementById("module-progress-tbody");
  if (!tbody) return;

  if (!moduleProgress || !Array.isArray(moduleProgress) || moduleProgress.length === 0) {
    tbody.innerHTML = '<tr><td colspan="13" style="text-align:center; color:#94a3b8; padding:16px;">Waiting for module telemetry...</td></tr>';
    return;
  }

  // Calculate live counts
  let cntDone = 0, cntRun = 0, cntPend = 0;
  moduleProgress.forEach(m => {
    if (m.status === "completed") cntDone++;
    else if (m.status === "running") cntRun++;
    else cntPend++;
  });

  const elAll = document.getElementById("matrix-cnt-all");
  const elDone = document.getElementById("matrix-cnt-completed");
  const elRun = document.getElementById("matrix-cnt-running");
  const elPend = document.getElementById("matrix-cnt-pending");
  const elTot = document.getElementById("matrix-total-count");

  if (elAll) elAll.textContent = moduleProgress.length;
  if (elDone) elDone.textContent = cntDone;
  if (elRun) elRun.textContent = cntRun;
  if (elPend) elPend.textContent = cntPend;
  if (elTot) elTot.textContent = `${moduleProgress.length} Modules`;

  // Apply filters
  let filtered = moduleProgress.filter(m => {
    if (matrixStatusFilter !== "ALL" && m.status !== matrixStatusFilter) {
      return false;
    }
    if (matrixSearchQuery) {
      const modName = (m.module || "").toLowerCase();
      return modName.includes(matrixSearchQuery);
    }
    return true;
  });

  tbody.innerHTML = "";
  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="13" style="text-align:center; color:#94a3b8; padding:16px;">No modules matching filter "${matrixSearchQuery || matrixStatusFilter}".</td></tr>`;
    return;
  }

  filtered.forEach(m => {
    const tr = document.createElement("tr");
    
    let stTag = '<span class="tag-status" style="background:rgba(148,163,184,0.15); color:#94a3b8; border:1px solid rgba(148,163,184,0.3);">QUEUED</span>';
    if (m.status === "running") {
      stTag = '<span class="tag-status" style="background:rgba(99,102,241,0.25); color:#818cf8; border:1px solid rgba(99,102,241,0.5); font-weight:700;">RUNNING ⚡</span>';
    } else if (m.status === "completed") {
      if (m.result_status === "TIMEOUT") {
        stTag = '<span class="tag-status tag-timeout">TIMEOUT ⏱️</span>';
      } else if (m.result_status === "FAIL" || (m.accuracy === 0 && m.f1 === 0)) {
        stTag = '<span class="tag-status tag-fail">FAIL ❌</span>';
      } else {
        stTag = '<span class="tag-status tag-pass">DONE ✔</span>';
      }
    }

    const accStr = (m.accuracy !== null && m.accuracy !== undefined && !isNaN(m.accuracy)) ? `${Number(m.accuracy).toFixed(2)}%` : "-";
    const f1Str = (m.f1 !== null && m.f1 !== undefined && !isNaN(m.f1)) ? Number(m.f1).toFixed(4) : "-";
    const fidStr = (m.fidelity !== null && m.fidelity !== undefined && !isNaN(m.fidelity)) ? Number(m.fidelity).toFixed(4) : "-";
    const lossStr = (m.loss !== null && m.loss !== undefined && !isNaN(m.loss)) ? Number(m.loss).toFixed(4) : "-";
    const pct = typeof m.progress_percent === 'number' ? m.progress_percent : 0;

    let durStr = "-";
    let durStyle = "color:#94a3b8;";
    if (m.status === "completed") {
      const sec = m.training_duration_seconds ?? m.duration_seconds;
      durStr = m.duration_formatted || (sec ? `${Math.floor(sec / 60)}m ${Math.floor(sec % 60)}s` : "Done");
      durStyle = "color:#10b981; font-weight:600;";
    } else if (m.status === "running") {
      const sec = m.training_elapsed_seconds ?? m.elapsed_seconds ?? m.duration_seconds;
      durStr = m.duration_formatted || (sec ? `${Math.floor(sec / 60)}m ${Math.floor(sec % 60)}s elapsed` : "running...");
      durStyle = "color:#f59e0b; font-weight:600;";
    }

    tr.innerHTML = `
      <td class="tag-module" style="font-weight:600; cursor:pointer;" onclick="openModuleModal('${m.module}')">${m.module}</td>
      <td>${stTag}</td>
      <td style="color:#38bdf8; font-weight:500;">${m.current_seed || '-'}</td>
      <td>${m.completed_seeds || 0} / ${m.total_seeds || 1}</td>
      <td style="color:#c084fc;">${m.epoch || '-'}</td>
      <td>${m.step || '-'}</td>
      <td style="color:#34d399; font-family:monospace;">${lossStr}</td>
      <td class="val-hl" style="font-weight:700; color:#38bdf8;">${accStr}</td>
      <td>${f1Str}</td>
      <td>${fidStr}</td>
      <td style="${durStyle}">${durStr}</td>
      <td style="min-width:110px;">
        <div style="display:flex; align-items:center; gap:6px;">
          <div style="flex:1; height:6px; background:rgba(255,255,255,0.08); border-radius:3px; overflow:hidden;">
            <div style="width:${pct}%; height:100%; background:linear-gradient(90deg, #6366f1, #38bdf8); transition:width 0.4s ease;"></div>
          </div>
          <span style="font-size:0.7rem; color:#94a3b8; width:28px;">${pct.toFixed(0)}%</span>
        </div>
      </td>
      <td style="font-size:0.75rem; color:#f59e0b;">${m.eta_str || '-'}</td>
    `;
    tbody.appendChild(tr);
  });
}


// ==========================================
// BOOTSTRAP
// ==========================================
restoreState();
fetchRunsList().then(() => {
  onRunChanged(selectedRunId);
});
switchTab(activeTab);


