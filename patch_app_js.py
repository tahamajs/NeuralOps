from pathlib import Path
import re

p = Path("Codes/framework/web/app.js")
content = p.read_text(encoding="utf-8")

# 1. Update restoreState so it never loads stale historical run when active run exists
old_restore = '''function restoreState() {
  try {
    const raw = localStorage.getItem("neuralops_dashboard_state_v3");
    if (!raw) return;
    const st = JSON.parse(raw);
    if (st.selectedRunId) selectedRunId = st.selectedRunId;'''

new_restore = '''function restoreState() {
  try {
    const raw = localStorage.getItem("neuralops_dashboard_state_v3");
    if (!raw) return;
    const st = JSON.parse(raw);
    if (st.selectedRunId && st.selectedRunId !== "gpu_server" && st.selectedRunId !== "adaptive_v3") {
      selectedRunId = st.selectedRunId;
    }'''

content = content.replace(old_restore, new_restore, 1)

# 2. Update fetchRunsList to prioritize active run
old_fetch_runs = '''    availableRuns = data.runs || [];
    const activeRun = availableRuns.find(r => r.is_active || r.run_id === "run_active");
    if (activeRun) {
      selectedRunId = "run_active";
    }
    populateRunSelectors();'''

new_fetch_runs = '''    availableRuns = data.runs || [];
    const activeRun = availableRuns.find(r => r.is_active || r.run_id === "run_active");
    if (activeRun) {
      selectedRunId = activeRun.run_id;
      isHistorical = false;
    }
    populateRunSelectors();'''

content = content.replace(old_fetch_runs, new_fetch_runs, 1)

# 3. Add renderModuleProgressTable function
module_progress_fn = '''
function renderModuleProgressTable(moduleProgress) {
  const tbody = document.getElementById("module-progress-tbody");
  if (!tbody) return;

  if (!moduleProgress || moduleProgress.length === 0) {
    tbody.innerHTML = '<tr><td colspan="12" style="text-align:center; color:#94a3b8; padding:16px;">Waiting for module telemetry...</td></tr>';
    return;
  }

  tbody.innerHTML = "";
  moduleProgress.forEach(m => {
    const tr = document.createElement("tr");
    
    let stTag = '<span class="tag-status" style="background:rgba(148,163,184,0.15); color:#94a3b8; border:1px solid rgba(148,163,184,0.3);">QUEUED</span>';
    if (m.status === "running") {
      stTag = '<span class="tag-status" style="background:rgba(99,102,241,0.25); color:#818cf8; border:1px solid rgba(99,102,241,0.5); font-weight:700; animation:pulse 1.5s infinite;">RUNNING ⚡</span>';
    } else if (m.status === "completed") {
      stTag = '<span class="tag-status tag-pass">DONE ✔</span>';
    }

    const accStr = m.accuracy !== null && m.accuracy !== undefined ? `${m.accuracy.toFixed(2)}%` : "-";
    const f1Str = m.f1 !== null && m.f1 !== undefined ? m.f1.toFixed(4) : "-";
    const fidStr = m.fidelity !== null && m.fidelity !== undefined ? m.fidelity.toFixed(4) : "-";
    const lossStr = m.loss !== null && m.loss !== undefined ? m.loss.toFixed(4) : "-";
    const pct = m.progress_percent || 0;

    tr.innerHTML = `
      <td class="tag-module" style="font-weight:600; cursor:pointer;" onclick="openModuleModal('${m.module}')">${m.module}</td>
      <td>${stTag}</td>
      <td style="color:#38bdf8; font-weight:500;">${m.current_seed || '-'}</td>
      <td>${m.completed_seeds} / ${m.total_seeds}</td>
      <td style="color:#c084fc;">${m.epoch || '-'}</td>
      <td>${m.step || '-'}</td>
      <td style="color:#34d399; font-family:monospace;">${lossStr}</td>
      <td class="val-hl" style="font-weight:700; color:#38bdf8;">${accStr}</td>
      <td>${f1Str}</td>
      <td>${fidStr}</td>
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
'''

# 4. Update renderTelemetry to invoke renderModuleProgressTable and update badges
old_render_telemetry = '''function renderTelemetry(data) {
  renderLiveLoss(data);
  renderGpuHistory(data);'''

new_render_telemetry = '''function renderTelemetry(data) {
  renderLiveLoss(data);
  renderGpuHistory(data);
  if (data.module_progress) {
    renderModuleProgressTable(data.module_progress);
  }'''

content = content.replace(old_render_telemetry, new_render_telemetry, 1)

# 5. Update renderLiveLoss to handle live_progress
old_render_live_loss = '''function renderLiveLoss(data) {
  const resultsList = data.recent_results || data.results;
  const activeItem = (resultsList && resultsList.length > 0) ? resultsList[0] : null;
  
  const modElem = document.getElementById("live-loss-module");
  const modelElem = document.getElementById("live-loss-model");
  const dsElem = document.getElementById("live-loss-dataset");
  const timeElem = document.getElementById("live-loss-time");
  const noteElem = document.getElementById("live-loss-note");

  const modName = activeItem ? (activeItem.module || activeItem.module_name || "svd_linear_attention") : "svd_linear_attention";
  const modelName = activeItem ? (activeItem.model || "BERT") : "BERT";
  const dsName = activeItem ? (activeItem.dataset || "SST-2") : "SST-2";
  const timeStr = activeItem ? (activeItem.time || activeItem.timestamp || "Live") : "Live";

  if (modElem) modElem.textContent = modName;
  if (modelElem) modelElem.textContent = modelName;
  if (dsElem) dsElem.textContent = dsName;
  if (timeElem) timeElem.textContent = timeStr;
  const hasRealLoss = activeItem && activeItem.status === "PASS" && activeItem.trajectories && activeItem.trajectories.loss && Array.isArray(activeItem.trajectories.loss) && activeItem.trajectories.loss.length > 0;'''

new_render_live_loss = '''function renderLiveLoss(data) {
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

  const hasRealLoss = activeItem && activeItem.status === "PASS" && activeItem.trajectories && activeItem.trajectories.loss && Array.isArray(activeItem.trajectories.loss) && activeItem.trajectories.loss.length > 0;'''

content = content.replace(old_render_live_loss, new_render_live_loss, 1)

# Append module_progress_fn before window.addEventListener
content = content.replace("window.addEventListener('DOMContentLoaded', init);", module_progress_fn + "\nwindow.addEventListener('DOMContentLoaded', init);")

p.write_text(content, encoding="utf-8")
print("Codes/framework/web/app.js successfully patched!")
