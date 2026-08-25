/**
 * Real-Time Dashboard & Comparative Analytics Controller for NeuralOps GPU Suite
 * Features:
 * - Dynamic Chart.js visualizations (Side-by-side comparison, Track latency distribution, Model family breakdown)
 * - Sortable and searchable Module Leaderboard
 * - Live Telemetry & Process Control
 * - Real-time stream filters and modal diagnostics
 */

const API_STATS = "/api/stats";
const API_ACTION = "/api/action";
const POLL_INTERVAL_MS = 3000;

let isFetching = false;
let currentResultsCache = [];
let currentLeaderboard = [];
let availableModules = [];

// Filters
let activeStatusFilter = "ALL";

// Chart.js Instances
let comparisonChart = null;
let tracksChart = null;
let modelsChart = null;

// Tab Management
function switchTab(tabId) {
  document.querySelectorAll(".tab-btn").forEach((btn) => btn.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach((view) => view.classList.remove("active"));

  if (tabId === "telemetry") {
    document.getElementById("tab-telemetry-btn").classList.add("active");
    document.getElementById("view-telemetry").classList.add("active");
  } else if (tabId === "analytics") {
    document.getElementById("tab-analytics-btn").classList.add("active");
    document.getElementById("view-analytics").classList.add("active");
    // Trigger chart resize/render
    if (comparisonChart) comparisonChart.resize();
    if (tracksChart) tracksChart.resize();
    if (modelsChart) modelsChart.resize();
  }
}

// -----------------------------------------------------------------------------
// Data Fetching & Core Controller
// -----------------------------------------------------------------------------
async function fetchStats() {
  if (isFetching) return;
  isFetching = true;

  try {
    const res = await fetch(API_STATS, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();
    renderDashboard(data);
    setConnectionStatus(true);
  } catch (err) {
    console.warn("API poll error:", err);
    setConnectionStatus(false);
  } finally {
    isFetching = false;
  }
}

function setConnectionStatus(online) {
  const badge = document.getElementById("conn-status");
  if (!badge) return;

  if (online) {
    badge.textContent = "ONLINE";
    badge.className = "badge badge-online";
  } else {
    badge.textContent = "RECONNECTING...";
    badge.className = "badge badge-offline";
  }
}

function renderDashboard(data) {
  // 1. Process / Sweep State
  const sweepBadge = document.getElementById("sweep-status");
  if (sweepBadge) {
    if (data.is_running) {
      sweepBadge.textContent = "SWEEP RUNNING";
      sweepBadge.className = "badge badge-running";
    } else {
      sweepBadge.textContent = "SWEEP IDLE";
      sweepBadge.className = "badge badge-idle";
    }
  }

  // 2. GPU Telemetry
  if (data.gpu) {
    const gpu = data.gpu;
    const util = gpu.gpu_util || 0;
    const memUsed = gpu.mem_used_mb || 0;
    const memTotal = gpu.mem_total_mb || 8188;
    const memPct = gpu.mem_pct || ((memUsed / memTotal) * 100).toFixed(1);
    const power = gpu.power_w || 0;
    const temp = gpu.temp_c || 0;

    document.getElementById("val-gpu-util").textContent = `${util}%`;
    document.getElementById("bar-gpu-util").style.width = `${Math.min(100, util)}%`;

    document.getElementById("val-vram").textContent = `${Math.round(memUsed)} / ${Math.round(memTotal)} MB`;
    document.getElementById("bar-vram").style.width = `${Math.min(100, memPct)}%`;
    document.getElementById("sub-vram").textContent = `${memPct}% Allocated`;

    document.getElementById("val-power").textContent = `${power.toFixed(1)} W`;
    document.getElementById("bar-power").style.width = `${Math.min(100, (power / 140) * 100)}%`;

    document.getElementById("val-temp").textContent = `${temp} °C`;
    document.getElementById("bar-temp").style.width = `${Math.min(100, (temp / 90) * 100)}%`;
  }

  // 3. Progress & ETA
  if (data.progress) {
    const p = data.progress;
    const completed = p.completed || 0;
    const total = p.total || 720;
    const pct = p.percent || ((completed / total) * 100).toFixed(1);

    document.getElementById("progress-desc").textContent = `Evaluating combination ${completed} of ${total} (${pct}% complete)`;
    document.getElementById("main-progress-bar").style.width = `${pct}%`;
    document.getElementById("val-eta").textContent = p.eta_formatted || "Calculating...";
    document.getElementById("val-pace").textContent = p.avg_sec ? p.avg_sec.toFixed(2) : "-";
  }

  // 4. Counts
  if (data.counts) {
    document.getElementById("count-pass").textContent = data.counts.pass || 0;
    document.getElementById("count-timeout").textContent = data.counts.timeout || 0;
    document.getElementById("count-fail").textContent = data.counts.fail || 0;
  }

  // 5. Recent Results Table
  if (data.recent_results) {
    currentResultsCache = data.recent_results;
    renderRecentStreamTable();
  }

  // 6. Comparative Analytics
  if (data.comparative_analytics) {
    renderComparativeAnalytics(data.comparative_analytics);
  }
}

// -----------------------------------------------------------------------------
// Recent Execution Stream & Filter
// -----------------------------------------------------------------------------
function setFilter(type, val, btn) {
  if (type === "status") {
    activeStatusFilter = val;
    document.querySelectorAll(".filter-pill").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    renderRecentStreamTable();
  }
}

function renderRecentStreamTable() {
  const tbody = document.getElementById("results-tbody");
  if (!tbody || !currentResultsCache) return;

  const filtered = currentResultsCache.filter((r) => {
    if (activeStatusFilter === "ALL") return true;
    return r.status === activeStatusFilter;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" class="loading-cell">No runs matching status filter "${activeStatusFilter}"</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered
    .map((r, idx) => {
      const statusClass =
        r.status === "PASS"
          ? "tag-pass"
          : r.status === "TIMEOUT"
          ? "tag-timeout"
          : "tag-fail";

      return `
      <tr class="clickable-row" onclick="openDetailModal(${idx})" title="Click to inspect full diagnostics & stack trace">
        <td>${r.time || "-"}</td>
        <td><code>${r.track}</code></td>
        <td><strong>${r.module}</strong></td>
        <td>${r.model}</td>
        <td>${r.dataset}</td>
        <td><span class="tag-status ${statusClass}">${r.status}</span></td>
        <td>${r.fwd_ms ? r.fwd_ms + " ms" : "-"}</td>
        <td>${r.bwd_ms ? r.bwd_ms + " ms" : "-"}</td>
        <td>${r.mem_mb ? r.mem_mb + " MB" : "-"}</td>
      </tr>
    `;
    })
    .join("");
}

// -----------------------------------------------------------------------------
// Comparative Analytics Lab & Chart.js Engine
// -----------------------------------------------------------------------------
function renderComparativeAnalytics(analytics) {
  if (!analytics) return;

  currentLeaderboard = analytics.module_leaderboard || [];
  availableModules = analytics.available_modules || [];

  // Update tab badge
  const countBadge = document.getElementById("tab-module-count");
  if (countBadge) countBadge.textContent = `${availableModules.length} Modules`;

  // 1. Populate Dropdowns if not populated or changed
  populateModuleDropdowns();

  // 2. Render Leaderboard Table
  renderLeaderboardTable();

  // 3. Update Head-to-Head Comparison Chart
  updateComparison();

  // 4. Update Overview Charts
  updateTrackChart(analytics.track_breakdown || {});
  updateModelChart(analytics.model_breakdown || {});
}

function populateModuleDropdowns() {
  const sel1 = document.getElementById("select-mod1");
  const sel2 = document.getElementById("select-mod2");
  if (!sel1 || !sel2 || availableModules.length === 0) return;

  const currentVal1 = sel1.value;
  const currentVal2 = sel2.value;

  if (sel1.options.length !== availableModules.length) {
    sel1.innerHTML = availableModules.map((m) => `<option value="${m}">${m}</option>`).join("");
    sel2.innerHTML = availableModules.map((m) => `<option value="${m}">${m}</option>`).join("");

    // Set defaults: first and second module
    if (availableModules.length > 0) sel1.value = currentVal1 && availableModules.includes(currentVal1) ? currentVal1 : availableModules[0];
    if (availableModules.length > 1) sel2.value = currentVal2 && availableModules.includes(currentVal2) ? currentVal2 : availableModules[1];
  }
}

function updateComparison() {
  const mod1Name = document.getElementById("select-mod1")?.value;
  const mod2Name = document.getElementById("select-mod2")?.value;

  if (!mod1Name || !mod2Name) return;

  const mod1Data = currentLeaderboard.find((m) => m.module === mod1Name) || {
    module: mod1Name, avg_fwd_ms: 0, avg_bwd_ms: 0, avg_total_ms: 0, avg_mem_mb: 0, success_rate_pct: 0
  };
  const mod2Data = currentLeaderboard.find((m) => m.module === mod2Name) || {
    module: mod2Name, avg_fwd_ms: 0, avg_bwd_ms: 0, avg_total_ms: 0, avg_mem_mb: 0, success_rate_pct: 0
  };

  // Update Summary Panes
  document.getElementById("comp-mod1-title").textContent = mod1Name;
  document.getElementById("comp-mod1-fwd").textContent = `${mod1Data.avg_fwd_ms} ms`;
  document.getElementById("comp-mod1-bwd").textContent = `${mod1Data.avg_bwd_ms} ms`;
  document.getElementById("comp-mod1-total").textContent = `${mod1Data.avg_total_ms} ms`;
  document.getElementById("comp-mod1-mem").textContent = `${mod1Data.avg_mem_mb} MB`;
  document.getElementById("comp-mod1-pass").textContent = `${mod1Data.success_rate_pct}%`;

  document.getElementById("comp-mod2-title").textContent = mod2Name;
  document.getElementById("comp-mod2-fwd").textContent = `${mod2Data.avg_fwd_ms} ms`;
  document.getElementById("comp-mod2-bwd").textContent = `${mod2Data.avg_bwd_ms} ms`;
  document.getElementById("comp-mod2-total").textContent = `${mod2Data.avg_total_ms} ms`;
  document.getElementById("comp-mod2-mem").textContent = `${mod2Data.avg_mem_mb} MB`;
  document.getElementById("comp-mod2-pass").textContent = `${mod2Data.success_rate_pct}%`;

  // Compute Delta Speedup
  const deltaBox = document.getElementById("val-comp-delta");
  const deltaSub = document.getElementById("sub-comp-delta");
  if (mod1Data.avg_total_ms > 0 && mod2Data.avg_total_ms > 0) {
    const ratio = (mod1Data.avg_total_ms / mod2Data.avg_total_ms).toFixed(2);
    if (ratio >= 1) {
      deltaBox.textContent = `${ratio}x Speedup`;
      deltaBox.style.color = "#34d399";
      deltaSub.textContent = `${mod2Name} is faster than ${mod1Name}`;
    } else {
      const invRatio = (mod2Data.avg_total_ms / mod1Data.avg_total_ms).toFixed(2);
      deltaBox.textContent = `${invRatio}x Slower`;
      deltaBox.style.color = "#f87171";
      deltaSub.textContent = `${mod2Name} is slower than ${mod1Name}`;
    }
  } else {
    deltaBox.textContent = "-";
    deltaSub.textContent = "Awaiting benchmark metrics";
  }

  // Render/Update Chart.js Comparison Bar Chart
  const ctx = document.getElementById("chart-comparison")?.getContext("2d");
  if (!ctx) return;

  const chartData = {
    labels: ["Fwd Latency (ms)", "Bwd Latency (ms)", "Total Step (ms)", "VRAM (MB / 10)"],
    datasets: [
      {
        label: mod1Name,
        data: [mod1Data.avg_fwd_ms, mod1Data.avg_bwd_ms, mod1Data.avg_total_ms, mod1Data.avg_mem_mb / 10],
        backgroundColor: "rgba(99, 102, 241, 0.7)",
        borderColor: "#6366f1",
        borderWidth: 1.5,
        borderRadius: 4,
      },
      {
        label: mod2Name,
        data: [mod2Data.avg_fwd_ms, mod2Data.avg_bwd_ms, mod2Data.avg_total_ms, mod2Data.avg_mem_mb / 10],
        backgroundColor: "rgba(16, 185, 129, 0.7)",
        borderColor: "#10b981",
        borderWidth: 1.5,
        borderRadius: 4,
      },
    ],
  };

  if (!comparisonChart) {
    comparisonChart = new Chart(ctx, {
      type: "bar",
      data: chartData,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#f8fafc", font: { family: "Plus Jakarta Sans", weight: 600 } } },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                if (ctx.dataIndex === 3) {
                  return `${ctx.dataset.label}: ${(ctx.raw * 10).toFixed(1)} MB`;
                }
                return `${ctx.dataset.label}: ${ctx.raw.toFixed(2)} ms`;
              },
            },
          },
        },
        scales: {
          x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255, 255, 255, 0.05)" } },
          y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255, 255, 255, 0.05)" } },
        },
      },
    });
  } else {
    comparisonChart.data = chartData;
    comparisonChart.update();
  }
}

function selectForCompare(modName) {
  const sel2 = document.getElementById("select-mod2");
  if (sel2) {
    sel2.value = modName;
    switchTab("analytics");
    updateComparison();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
}

function updateTrackChart(trackBreakdown) {
  const ctx = document.getElementById("chart-tracks")?.getContext("2d");
  if (!ctx) return;

  const tracks = Object.keys(trackBreakdown);
  const fwdData = tracks.map((t) => trackBreakdown[t].avg_fwd_ms);
  const bwdData = tracks.map((t) => trackBreakdown[t].avg_bwd_ms);

  const data = {
    labels: tracks.map((t) => t.replace("track", "Trk ")),
    datasets: [
      {
        label: "Avg Fwd Latency (ms)",
        data: fwdData,
        backgroundColor: "rgba(56, 189, 248, 0.7)",
        borderColor: "#38bdf8",
        borderWidth: 1,
        borderRadius: 4,
      },
      {
        label: "Avg Bwd Latency (ms)",
        data: bwdData,
        backgroundColor: "rgba(244, 63, 94, 0.7)",
        borderColor: "#f43f5e",
        borderWidth: 1,
        borderRadius: 4,
      },
    ],
  };

  if (!tracksChart) {
    tracksChart = new Chart(ctx, {
      type: "bar",
      data: data,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#f8fafc" } },
        },
        scales: {
          x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255, 255, 255, 0.05)" } },
          y: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255, 255, 255, 0.05)" } },
        },
      },
    });
  } else {
    tracksChart.data = data;
    tracksChart.update();
  }
}

function updateModelChart(modelBreakdown) {
  const ctx = document.getElementById("chart-models")?.getContext("2d");
  if (!ctx) return;

  const models = Object.keys(modelBreakdown);
  const latData = models.map((m) => modelBreakdown[m].avg_fwd_ms + modelBreakdown[m].avg_bwd_ms);
  const memData = models.map((m) => modelBreakdown[m].avg_mem_mb);

  const data = {
    labels: models,
    datasets: [
      {
        label: "Avg Total Latency (ms)",
        data: latData,
        backgroundColor: "rgba(168, 85, 247, 0.7)",
        borderColor: "#a855f7",
        borderWidth: 1,
        borderRadius: 4,
        yAxisID: "y",
      },
      {
        label: "Peak VRAM (MB)",
        data: memData,
        type: "line",
        borderColor: "#10b981",
        backgroundColor: "#10b981",
        borderWidth: 2,
        yAxisID: "y1",
      },
    ],
  };

  if (!modelsChart) {
    modelsChart = new Chart(ctx, {
      type: "bar",
      data: data,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#f8fafc" } },
        },
        scales: {
          x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(255, 255, 255, 0.05)" } },
          y: {
            type: "linear",
            display: true,
            position: "left",
            ticks: { color: "#94a3b8" },
            grid: { color: "rgba(255, 255, 255, 0.05)" },
          },
          y1: {
            type: "linear",
            display: true,
            position: "right",
            ticks: { color: "#10b981" },
            grid: { drawOnChartArea: false },
          },
        },
      },
    });
  } else {
    modelsChart.data = data;
    modelsChart.update();
  }
}

// -----------------------------------------------------------------------------
// Leaderboard Table
// -----------------------------------------------------------------------------
function filterLeaderboard() {
  renderLeaderboardTable();
}

function renderLeaderboardTable() {
  const tbody = document.getElementById("leaderboard-tbody");
  if (!tbody || !currentLeaderboard) return;

  const search = (document.getElementById("leaderboard-search")?.value || "").toLowerCase();

  const filtered = currentLeaderboard.filter((m) => {
    if (!search) return true;
    return m.module.toLowerCase().includes(search) || m.track.toLowerCase().includes(search);
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" class="loading-cell">No modules found matching "${search}"</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered
    .map((m, idx) => {
      const rankBadge =
        idx === 0 ? "🥇 1" : idx === 1 ? "🥈 2" : idx === 2 ? "🥉 3" : `#${idx + 1}`;

      const rateClass =
        m.success_rate_pct >= 90
          ? "tag-pass"
          : m.success_rate_pct >= 50
          ? "tag-timeout"
          : "tag-fail";

      return `
      <tr>
        <td><strong>${rankBadge}</strong></td>
        <td><strong>${m.module}</strong></td>
        <td><code>${m.track}</code></td>
        <td>${m.avg_fwd_ms > 0 ? m.avg_fwd_ms + " ms" : "-"}</td>
        <td>${m.avg_bwd_ms > 0 ? m.avg_bwd_ms + " ms" : "-"}</td>
        <td><strong>${m.avg_total_ms > 0 ? m.avg_total_ms + " ms" : "-"}</strong></td>
        <td>${m.avg_mem_mb > 0 ? m.avg_mem_mb + " MB" : "-"}</td>
        <td><span class="tag-status ${rateClass}">${m.success_rate_pct}%</span></td>
        <td>${m.success_count}/${m.total_runs}</td>
        <td>
          <button class="btn btn-secondary btn-sm" onclick="selectForCompare('${m.module}')">Compare</button>
        </td>
      </tr>
    `;
    })
    .join("");
}

// -----------------------------------------------------------------------------
// Diagnostics Modal
// -----------------------------------------------------------------------------
function openDetailModal(index) {
  const item = currentResultsCache[index];
  if (!item) return;

  document.getElementById("modal-module-title").textContent = item.module;
  document.getElementById("modal-track").textContent = item.track;
  document.getElementById("modal-model").textContent = item.model;
  document.getElementById("modal-dataset").textContent = item.dataset;
  document.getElementById("modal-time").textContent = item.time || "-";
  document.getElementById("modal-fwd").textContent = item.fwd_ms ? `${item.fwd_ms} ms` : "-";
  document.getElementById("modal-bwd").textContent = item.bwd_ms ? `${item.bwd_ms} ms` : "-";
  document.getElementById("modal-mem").textContent = item.mem_mb ? `${item.mem_mb} MB` : "-";

  const badge = document.getElementById("modal-status-badge");
  badge.textContent = item.status;
  badge.className = `tag-status ${
    item.status === "PASS"
      ? "tag-pass"
      : item.status === "TIMEOUT"
      ? "tag-timeout"
      : "tag-fail"
  }`;

  const errorBox = document.getElementById("modal-error-box");
  const detailText = item.error_detail || "None / Success";
  errorBox.textContent = detailText;

  if (item.status === "FAIL") {
    errorBox.className = "error-detail-box error-box-fail";
  } else if (item.status === "TIMEOUT") {
    errorBox.className = "error-detail-box error-box-timeout";
  } else {
    errorBox.className = "error-detail-box success-box";
  }

  const modal = document.getElementById("detail-modal");
  modal.style.display = "flex";
}

function hideModal() {
  const modal = document.getElementById("detail-modal");
  if (modal) modal.style.display = "none";
}

function closeModal(e) {
  if (e.target.id === "detail-modal") {
    hideModal();
  }
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") hideModal();
});

async function triggerAction(action) {
  const btn = action === "stop" ? document.getElementById("btn-stop") : document.getElementById("btn-start");
  if (btn) btn.disabled = true;

  try {
    const res = await fetch(API_ACTION, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
    const result = await res.json();
    console.log("Action response:", result);
    fetchStats();
  } catch (err) {
    alert(`Failed to trigger ${action}: ${err.message}`);
  } finally {
    if (btn) btn.disabled = false;
  }
}

// Initial fetch & setup recurring loop
document.addEventListener("DOMContentLoaded", () => {
  fetchStats();
  setInterval(fetchStats, POLL_INTERVAL_MS);
});
