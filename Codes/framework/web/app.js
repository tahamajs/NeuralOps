/**
 * Real-Time Dashboard Controller for NeuralOps GPU Sweep
 */

const API_STATS = "/api/stats";
const API_ACTION = "/api/action";
const POLL_INTERVAL_MS = 3000;

let isFetching = false;

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

    // GPU Util
    document.getElementById("val-gpu-util").textContent = `${util}%`;
    document.getElementById("bar-gpu-util").style.width = `${Math.min(100, util)}%`;

    // VRAM
    document.getElementById("val-vram").textContent = `${Math.round(memUsed)} / ${Math.round(memTotal)} MB`;
    document.getElementById("bar-vram").style.width = `${Math.min(100, memPct)}%`;
    document.getElementById("sub-vram").textContent = `${memPct}% Allocated`;

    // Power
    document.getElementById("val-power").textContent = `${power.toFixed(1)} W`;
    document.getElementById("bar-power").style.width = `${Math.min(100, (power / 140) * 100)}%`;

    // Temp
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
  const tbody = document.getElementById("results-tbody");
  if (tbody && data.recent_results && data.recent_results.length > 0) {
    tbody.innerHTML = data.recent_results
      .map((r) => {
        const statusClass =
          r.status === "PASS"
            ? "tag-pass"
            : r.status === "TIMEOUT"
            ? "tag-timeout"
            : "tag-fail";

        return `
        <tr>
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
}

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
