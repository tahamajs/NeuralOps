from pathlib import Path

p = Path("Codes/framework/web/app.js")
content = p.read_text(encoding="utf-8")

fn_code = '''
function renderModuleProgressTable(moduleProgress) {
  const tbody = document.getElementById("module-progress-tbody");
  if (!tbody) return;

  if (!moduleProgress || !Array.isArray(moduleProgress) || moduleProgress.length === 0) {
    tbody.innerHTML = '<tr><td colspan="12" style="text-align:center; color:#94a3b8; padding:16px;">Waiting for module telemetry...</td></tr>';
    return;
  }

  tbody.innerHTML = "";
  moduleProgress.forEach(m => {
    const tr = document.createElement("tr");
    
    let stTag = '<span class="tag-status" style="background:rgba(148,163,184,0.15); color:#94a3b8; border:1px solid rgba(148,163,184,0.3);">QUEUED</span>';
    if (m.status === "running") {
      stTag = '<span class="tag-status" style="background:rgba(99,102,241,0.25); color:#818cf8; border:1px solid rgba(99,102,241,0.5); font-weight:700;">RUNNING ⚡</span>';
    } else if (m.status === "completed") {
      stTag = '<span class="tag-status tag-pass">DONE ✔</span>';
    }

    const accStr = (m.accuracy !== null && m.accuracy !== undefined && !isNaN(m.accuracy)) ? `${Number(m.accuracy).toFixed(2)}%` : "-";
    const f1Str = (m.f1 !== null && m.f1 !== undefined && !isNaN(m.f1)) ? Number(m.f1).toFixed(4) : "-";
    const fidStr = (m.fidelity !== null && m.fidelity !== undefined && !isNaN(m.fidelity)) ? Number(m.fidelity).toFixed(4) : "-";
    const lossStr = (m.loss !== null && m.loss !== undefined && !isNaN(m.loss)) ? Number(m.loss).toFixed(4) : "-";
    const pct = typeof m.progress_percent === 'number' ? m.progress_percent : 0;

    tr.innerHTML = `
      <td class="tag-module" style="font-weight:600; cursor:pointer;" onclick="openModuleModal('${m.module}')">${m.module}</td>
      <td>${stTag}</td>
      <td style="color:#38bdf8; font-weight:500;">${m.current_seed || '-'}</td>
      <td>${m.completed_seeds || 0} / ${m.total_seeds || 3}</td>
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

target_anchor = "// ==========================================\n// BOOTSTRAP\n// =========================================="
content = content.replace(target_anchor, fn_code + "\n\n" + target_anchor)

p.write_text(content, encoding="utf-8")
print("renderModuleProgressTable successfully added to Codes/framework/web/app.js!")
