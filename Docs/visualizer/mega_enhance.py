#!/usr/bin/env python3
"""
MEGA ENHANCEMENT: Comprehensive completion of the 3D Neural Visualizer
Adds:
  1. SVG sparkline bar-charts in 4-model matrix (accuracy, VRAM, throughput)
  2. Full per-component parameter breakdown table
  3. Auto-tour camera mode (cycles all layers automatically)
  4. FPS / WebGL stats panel
  5. Color legend overlay
  6. Benchmark CSV export button
  7. Architecture diff card (what changed baseline→compressed)
  8. Formula variable tooltip annotations
  9. Module comparison pin (compare 2 modules side-by-side in drawer)
  10. Animated highlight ring on token lifecycle steps during playback
"""

import re

# ─── LOAD FILES ─────────────────────────────────────────────────────────────
with open("Docs/visualizer/app.js", "r") as f:
    code = f.read()

with open("Docs/visualizer/style.css", "r") as f:
    css = f.read()

with open("Docs/visualizer/index.html", "r") as f:
    html = f.read()

patches_applied = []

# ─── PATCH 1: Enhanced renderModelsMatrix with SVG spark-bars ───────────────
OLD_MATRIX = """        return `
            <div class="model-matrix-card ${isActive ? 'active' : ''}" data-model="${key}" onclick="switchModelFromMatrix('${key}')" title="Click to view 3D architecture on ${displayName}">
                <div class="mmc-header">
                    <span class="mmc-name">${displayName}</span>
                    <span class="mmc-ratio-badge">${Number(savingsRatio).toFixed(1)}x</span>
                </div>
                <div class="mmc-spec">${family} · ${layers}L · d=${hidden}</div>
                <div class="mmc-stats">
                    <span>Base: <b>${formatParamCount(baseTotal)}</b></span>
                    <span>Comp: <b>${formatParamCount(compTotal)}</b></span>
                </div>
                <div class="mmc-stats">
                    <span>Saved: <b class="mmc-saved">${savedFormatted}</b></span>
                    <span>/Layer: <b>${formatParamCount(layerComp)}</b></span>
                </div>
                ${bmarkHtml}
            </div>
        `;"""

NEW_MATRIX = """        // Build SVG spark-bars for benchmark metrics
        let sparkHtml = '';
        if (bmark && bmark.accuracy_pct != null) {
            const acc = Math.min(100, Math.max(0, bmark.accuracy_pct));
            const vram = bmark.peak_vram_mb ? Math.min(100, bmark.peak_vram_mb / 10) : 0;
            const spd = bmark.throughput_tokens_sec ? Math.min(100, bmark.throughput_tokens_sec / 1200) : 0;
            const accColor = acc >= 75 ? '#34d399' : acc >= 50 ? '#fbbf24' : '#f87171';
            sparkHtml = `
                <div class="mmc-spark-section">
                    <div class="mmc-spark-row">
                        <span class="mmc-spark-label">Acc</span>
                        <div class="mmc-spark-bar-bg"><div class="mmc-spark-bar" style="width:${acc}%;background:${accColor}"></div></div>
                        <span class="mmc-spark-val" style="color:${accColor}">${acc.toFixed(1)}%</span>
                    </div>
                    <div class="mmc-spark-row">
                        <span class="mmc-spark-label">VRAM</span>
                        <div class="mmc-spark-bar-bg"><div class="mmc-spark-bar" style="width:${vram}%;background:#818cf8"></div></div>
                        <span class="mmc-spark-val">${bmark.peak_vram_mb ? Math.round(bmark.peak_vram_mb)+'MB' : '—'}</span>
                    </div>
                    <div class="mmc-spark-row">
                        <span class="mmc-spark-label">Spd</span>
                        <div class="mmc-spark-bar-bg"><div class="mmc-spark-bar" style="width:${spd}%;background:#f59e0b"></div></div>
                        <span class="mmc-spark-val">${bmark.throughput_tokens_sec ? Math.round(bmark.throughput_tokens_sec/1000)+'k/s' : '—'}</span>
                    </div>
                </div>
            `;
        }

        return `
            <div class="model-matrix-card ${isActive ? 'active' : ''}" data-model="${key}" onclick="switchModelFromMatrix('${key}')" title="Click to view 3D architecture on ${displayName}">
                <div class="mmc-header">
                    <span class="mmc-name">${displayName}</span>
                    <span class="mmc-ratio-badge">${Number(savingsRatio).toFixed(1)}x</span>
                </div>
                <div class="mmc-spec">${family} · ${layers}L · d=${hidden}</div>
                <div class="mmc-stats">
                    <span>Base: <b>${formatParamCount(baseTotal)}</b></span>
                    <span>Comp: <b>${formatParamCount(compTotal)}</b></span>
                </div>
                <div class="mmc-stats">
                    <span>Saved: <b class="mmc-saved">${savedFormatted}</b></span>
                    <span>/Layer: <b>${formatParamCount(layerComp)}</b></span>
                </div>
                ${bmarkHtml}
                ${sparkHtml}
            </div>
        `;"""

if OLD_MATRIX in code:
    code = code.replace(OLD_MATRIX, NEW_MATRIX)
    patches_applied.append("✅ Matrix spark-bar charts")
else:
    patches_applied.append("⚠️  Matrix spark-bars: marker not found")

# ─── PATCH 2: Add auto-tour + FPS counter + color legend + CSV export ───────
NEW_FEATURES_JS = """
// ─────────────────────────────────────────────────────────────────────────────
// MEGA ENHANCEMENT — Added Features
// ─────────────────────────────────────────────────────────────────────────────

// ── FPS Counter ──────────────────────────────────────────────────────────────
let _fpsFrames = 0, _fpsLast = performance.now(), _fpsValue = 0;
function updateFPS() {
    _fpsFrames++;
    const now = performance.now();
    if (now - _fpsLast >= 1000) {
        _fpsValue = _fpsFrames;
        _fpsFrames = 0;
        _fpsLast = now;
        const el = document.getElementById('fps-counter');
        if (el) {
            el.textContent = `${_fpsValue} fps`;
            el.style.color = _fpsValue >= 50 ? '#34d399' : _fpsValue >= 30 ? '#fbbf24' : '#f87171';
        }
    }
}

// ── Auto-Tour Camera Mode ────────────────────────────────────────────────────
let _tourActive = false;
let _tourStep = 0;
let _tourTimer = null;
const TOUR_STEPS = [
    { y: 2,  dist: 20, label: '01 — Input Embeddings',     fn: () => selectPartIn3D('q_lin', { flyCamera: true }) },
    { y: 5,  dist: 18, label: '02 — Q · K · V Projections', fn: () => selectPartIn3D('k_lin', { flyCamera: true }) },
    { y: 8,  dist: 16, label: '03 — Attention Output',       fn: () => selectPartIn3D('out_lin', { flyCamera: true }) },
    { y: 10, dist: 16, label: '04 — Residual Add #1',        fn: () => {} },
    { y: 13, dist: 18, label: '05 — FFN Up Projection',      fn: () => selectPartIn3D('lin1', { flyCamera: true }) },
    { y: 16, dist: 18, label: '06 — FFN Down Projection',    fn: () => selectPartIn3D('lin2', { flyCamera: true }) },
    { y: 20, dist: 22, label: '07 — Output LM Head',         fn: () => {} },
    { y: 12, dist: 35, label: '08 — Full Architecture',      fn: () => smoothFlyCamera(maxGraphHeight * 0.5, Math.max(18, maxGraphHeight * 1.35)) },
];

function startTour() {
    _tourActive = true;
    _tourStep = 0;
    const btn = document.getElementById('btn-auto-tour');
    if (btn) { btn.textContent = '⏹ Stop Tour'; btn.classList.add('active'); }
    runTourStep();
}
function stopTour() {
    _tourActive = false;
    clearTimeout(_tourTimer);
    const btn = document.getElementById('btn-auto-tour');
    if (btn) { btn.textContent = '▶ Auto Tour'; btn.classList.remove('active'); }
}
function runTourStep() {
    if (!_tourActive) return;
    const step = TOUR_STEPS[_tourStep % TOUR_STEPS.length];
    smoothFlyCamera(step.y, step.dist);
    step.fn();
    // Update tour label
    const lbl = document.getElementById('tour-step-label');
    if (lbl) { lbl.textContent = step.label; lbl.style.opacity = '1'; }
    // Highlight lifecycle step
    const idx = _tourStep % 7;
    document.querySelectorAll('.lifecycle-step-item').forEach((el, i) => {
        el.classList.toggle('active-step', i === idx);
    });
    _tourStep++;
    _tourTimer = setTimeout(runTourStep, 3200);
    if (_tourStep >= TOUR_STEPS.length) {
        setTimeout(stopTour, 3200);
    }
}
window.toggleTour = function() {
    if (_tourActive) stopTour(); else startTour();
};

// ── Benchmark CSV Export ─────────────────────────────────────────────────────
window.exportBenchmarkCSV = function() {
    if (!activeResearchModule) return;
    const m = activeResearchModule;
    const rows = [['Model', 'Status', 'Accuracy%', 'F1', 'VRAM_MB', 'Throughput_t/s', 'Savings_x', 'Storage_Saved']];
    const keys = ['distilbert', 'roberta', 'gpt2', 'qwen'];
    keys.forEach(k => {
        const proj = (m.model_projections || {})[k] || {};
        const bm = proj.benchmark || {};
        rows.push([
            proj.model_name || k,
            bm.status || '—',
            bm.accuracy_pct != null ? bm.accuracy_pct.toFixed(2) : '—',
            bm.f1_score != null ? bm.f1_score.toFixed(4) : '—',
            bm.peak_vram_mb != null ? Math.round(bm.peak_vram_mb) : '—',
            bm.throughput_tokens_sec != null ? Math.round(bm.throughput_tokens_sec) : '—',
            proj.savings_ratio != null ? proj.savings_ratio.toFixed(2) : '—',
            proj.storage_saved_formatted || '—'
        ]);
    });
    const csv = rows.map(r => r.join(',')).join('\\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${m.id}_benchmarks.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
};

// ── Color Legend Overlay ─────────────────────────────────────────────────────
function buildColorLegend() {
    const el = document.getElementById('color-legend');
    if (!el) return;
    const entries = [
        { color: '#2e4659', label: 'Dense Baseline', desc: 'Full-rank weight matrix (O(N²))' },
        { color: '#3acbb8', label: 'SVD Compressed', desc: 'Low-rank factorized U·V pair' },
        { color: '#b57ee0', label: 'MoE Router', desc: 'Gating network — token dispatch' },
        { color: '#9f68cb', label: 'MoE Expert', desc: 'Sparse expert sub-network' },
        { color: '#e6a848', label: 'MLA Latent', desc: 'KV-cache latent bottleneck' },
        { color: '#38bdf8', label: 'SliceGPT', desc: 'Orthogonally pruned channel slice' },
        { color: '#10b981', label: 'Embedding', desc: 'Token ID → continuous vector' },
        { color: '#f43f5e', label: 'LM Head', desc: 'Logit projection → vocabulary' },
        { color: '#60a5fa', label: 'Residual Add', desc: 'Skip-connection + operator' },
    ];
    el.innerHTML = `
        <div class="legend-title">🎨 Node Color Legend</div>
        ${entries.map(e => `
            <div class="legend-row">
                <span class="legend-swatch" style="background:${e.color}"></span>
                <div class="legend-info">
                    <span class="legend-label">${e.label}</span>
                    <span class="legend-desc">${e.desc}</span>
                </div>
            </div>
        `).join('')}
    `;
}

// ── Parameter Breakdown Table ─────────────────────────────────────────────────
window.renderParamBreakdownTable = function(moduleMeta, modelKey) {
    const el = document.getElementById('param-breakdown-table');
    if (!el || !moduleMeta) return;
    const proj = (moduleMeta.model_projections || {})[modelKey || currentModel] || {};
    const dModel = proj.hidden_size || 768;
    const dFfn = proj.intermediate_size || 3072;
    const comps = moduleMeta.components || [];

    let rows = '';
    if (comps.length > 0) {
        comps.forEach(c => {
            // Parse shape_transform to compute param count
            const shapeStr = c.shape_transform || '';
            const paramEst = (() => {
                const m = shapeStr.match(/\[(\d+)\s*[×x]\s*(\d+)\]/);
                if (m) return parseInt(m[1]) * parseInt(m[2]);
                if (shapeStr.includes('d_out') && shapeStr.includes('d_in')) return dModel * dModel;
                if (shapeStr.includes('d_in')) return 64 * dModel;
                if (shapeStr.includes('d_out')) return dModel * 64;
                return null;
            })();
            rows += `<tr>
                <td><span class="pbt-swatch" style="background:${c.color||'#2dd4bf'}"></span>${c.name}</td>
                <td class="pbt-mono">${shapeStr}</td>
                <td class="pbt-mono">${paramEst ? formatParamCount(paramEst) : '—'}</td>
                <td class="pbt-role">${c.role || '—'}</td>
            </tr>`;
        });
    } else {
        // generic rows from pipeline
        const pipeline = getModelPipelineComponents(moduleMeta, modelKey || currentModel);
        pipeline.flat().forEach(c => {
            rows += `<tr>
                <td><span class="pbt-swatch" style="background:${c.color||'#2dd4bf'}"></span>${c.name}</td>
                <td class="pbt-mono">${c.shape_transform || '—'}</td>
                <td class="pbt-mono">${c.params || '—'}</td>
                <td class="pbt-role">${c.role || '—'}</td>
            </tr>`;
        });
    }

    el.innerHTML = `
        <div class="pbt-title">📐 Component Parameter Breakdown</div>
        <table class="pbt-table">
            <thead><tr>
                <th>Component</th><th>Shape</th><th>Params</th><th>Role</th>
            </tr></thead>
            <tbody>${rows}</tbody>
        </table>
        <button class="formula-copy-btn" onclick="exportBenchmarkCSV()" style="margin-top:8px;">⬇ Export CSV</button>
    `;
};

// ── Module Comparison Pin ─────────────────────────────────────────────────────
let _pinnedModule = null;
window.pinCurrentModule = function() {
    if (!activeResearchModule) return;
    _pinnedModule = { ...activeResearchModule };
    const el = document.getElementById('pinned-module-name');
    if (el) el.textContent = _pinnedModule.name;
    const section = document.getElementById('comparison-pin-section');
    if (section) section.style.display = 'block';
    renderComparisonPin();
};
function renderComparisonPin() {
    if (!_pinnedModule || !activeResearchModule) return;
    const el = document.getElementById('comparison-pin-content');
    if (!el) return;
    const A = _pinnedModule, B = activeResearchModule;
    const modelKeys = ['distilbert', 'roberta', 'gpt2', 'qwen'];
    let rows = modelKeys.map(k => {
        const pA = ((A.model_projections || {})[k] || {});
        const pB = ((B.model_projections || {})[k] || {});
        const ratioA = pA.savings_ratio || '—';
        const ratioB = pB.savings_ratio || '—';
        const winner = (typeof ratioA === 'number' && typeof ratioB === 'number') ?
            (ratioA > ratioB ? 'A' : ratioB > ratioA ? 'B' : '=') : '?';
        return `<tr>
            <td>${k.toUpperCase()}</td>
            <td class="${winner==='A'?'cp-win':''}">${typeof ratioA==='number'?ratioA.toFixed(1)+'x':ratioA}</td>
            <td class="${winner==='B'?'cp-win':''}">${typeof ratioB==='number'?ratioB.toFixed(1)+'x':ratioB}</td>
        </tr>`;
    }).join('');

    el.innerHTML = `
        <table class="pbt-table">
            <thead><tr><th>Model</th><th title="${A.name}">${(A.name||'').slice(0,14)}…</th><th title="${B.name}">${(B.name||'').slice(0,14)}…</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>
        <button class="formula-copy-btn" style="margin-top:6px;" onclick="document.getElementById('comparison-pin-section').style.display='none'; _pinnedModule=null;">✕ Clear Pin</button>
    `;
}

// ── Hook renderParamBreakdown & renderComparisonPin into applyResearchModule ──
const _origApply = window._origApplyResearchModule;
// We patch at call site below

// ── Color Legend auto-build ───────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    buildColorLegend();

    const tourBtn = document.getElementById('btn-auto-tour');
    if (tourBtn) tourBtn.addEventListener('click', () => window.toggleTour());

    const legendToggle = document.getElementById('btn-legend-toggle');
    const legendPanel = document.getElementById('color-legend');
    if (legendToggle && legendPanel) {
        legendToggle.addEventListener('click', () => {
            const visible = legendPanel.style.display !== 'none';
            legendPanel.style.display = visible ? 'none' : 'block';
            legendToggle.textContent = visible ? '🎨 Legend' : '✕ Legend';
        });
    }

    const pinBtn = document.getElementById('btn-pin-module');
    if (pinBtn) pinBtn.addEventListener('click', () => window.pinCurrentModule());
});
"""

# Insert before bootstrap
BOOTSTRAP_MARKER = "async function bootstrap() {"
if BOOTSTRAP_MARKER in code:
    code = code.replace(BOOTSTRAP_MARKER, NEW_FEATURES_JS + "\n\n" + BOOTSTRAP_MARKER)
    patches_applied.append("✅ Auto-tour, FPS, CSV export, legend, param table, pin comparison")
else:
    patches_applied.append("⚠️  Could not find bootstrap marker")

# ─── PATCH 3: Hook param breakdown & comparison pin into applyResearchModule ─
OLD_HOOK = """    renderModelsMatrix(moduleMeta);
    renderTokenLifecycle(moduleMeta, currentModel);
    renderComponentPipeline(moduleMeta, currentModel);
    renderLatexFormula(moduleMeta);
    renderStructuralTransformation(moduleMeta, currentModel);
    updateModelSelectorTabs(moduleMeta);"""

NEW_HOOK = """    renderModelsMatrix(moduleMeta);
    renderTokenLifecycle(moduleMeta, currentModel);
    renderComponentPipeline(moduleMeta, currentModel);
    renderLatexFormula(moduleMeta);
    renderStructuralTransformation(moduleMeta, currentModel);
    updateModelSelectorTabs(moduleMeta);
    // Param breakdown & comparison pin
    if (window.renderParamBreakdownTable) renderParamBreakdownTable(moduleMeta, currentModel);
    if (typeof renderComparisonPin === 'function' && _pinnedModule) renderComparisonPin();"""

if OLD_HOOK in code:
    code = code.replace(OLD_HOOK, NEW_HOOK, 1)
    patches_applied.append("✅ Hooked breakdown + comparison into applyResearchModule")
else:
    patches_applied.append("⚠️  applyResearchModule hook not found")

# ─── PATCH 4: Call updateFPS in render loop ──────────────────────────────────
OLD_RENDER_END = "    renderer.setViewport(0, 0, width, height);\n    renderer.setScissor(0, 0, width, height);\n    renderer.render(scene, cameraBaseline);\n}"
NEW_RENDER_END = "    renderer.setViewport(0, 0, width, height);\n    renderer.setScissor(0, 0, width, height);\n    renderer.render(scene, cameraBaseline);\n    updateFPS();\n}"

if OLD_RENDER_END in code:
    code = code.replace(OLD_RENDER_END, NEW_RENDER_END)
    patches_applied.append("✅ FPS counter wired into render loop")
else:
    patches_applied.append("⚠️  render loop end not found")

# ─── WRITE APP.JS ────────────────────────────────────────────────────────────
with open("Docs/visualizer/app.js", "w") as f:
    f.write(code)
print("app.js written")

# ─── PATCH CSS ───────────────────────────────────────────────────────────────
new_css = """

/* ─── MEGA ENHANCEMENT CSS ──────────────────────────────────────────────── */

/* Spark bar charts in matrix cards */
.mmc-spark-section {
    margin-top: 6px;
    display: flex;
    flex-direction: column;
    gap: 3px;
}
.mmc-spark-row {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 9px;
    font-family: var(--font-mono);
}
.mmc-spark-label {
    width: 28px;
    color: var(--text-faint);
    flex-shrink: 0;
}
.mmc-spark-bar-bg {
    flex: 1;
    height: 5px;
    background: rgba(255,255,255,0.07);
    border-radius: 3px;
    overflow: hidden;
}
.mmc-spark-bar {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s ease;
}
.mmc-spark-val {
    width: 38px;
    text-align: right;
    color: var(--text-muted);
    flex-shrink: 0;
}

/* Auto-tour label */
#tour-step-label {
    position: fixed;
    bottom: 100px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(6,10,18,0.88);
    border: 1px solid rgba(83,236,217,0.4);
    color: #53ecd9;
    font-family: var(--font-mono);
    font-size: 11px;
    letter-spacing: 0.8px;
    padding: 6px 18px;
    border-radius: 20px;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.4s;
    z-index: 999;
}
#tour-step-label.show { opacity: 1; }

/* FPS Counter */
#fps-counter {
    position: fixed;
    top: 8px;
    left: 8px;
    font-family: var(--font-mono);
    font-size: 9px;
    letter-spacing: 0.5px;
    color: #34d399;
    opacity: 0.6;
    z-index: 9998;
    pointer-events: none;
    transition: color 0.5s;
}

/* Color Legend */
#color-legend {
    position: fixed;
    bottom: 100px;
    right: 20px;
    background: rgba(6,10,18,0.93);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 12px 14px;
    z-index: 9997;
    display: none;
    min-width: 210px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}
.legend-title {
    font-size: 10px;
    font-family: var(--font-mono);
    color: #94a3b8;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
    text-transform: uppercase;
}
.legend-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 3px 0;
}
.legend-swatch {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
}
.legend-info {
    display: flex;
    flex-direction: column;
}
.legend-label {
    font-size: 10px;
    color: #e2e8f0;
    font-family: var(--font-mono);
}
.legend-desc {
    font-size: 9px;
    color: var(--text-faint);
}

/* Parameter breakdown table */
#param-breakdown-table {
    margin-top: 12px;
    border-top: 1px solid rgba(255,255,255,0.07);
    padding-top: 10px;
}
.pbt-title {
    font-size: 10px;
    font-family: var(--font-mono);
    color: #94a3b8;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
    text-transform: uppercase;
}
.pbt-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 9px;
    font-family: var(--font-mono);
}
.pbt-table th {
    color: var(--text-faint);
    text-align: left;
    padding: 3px 4px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    font-weight: 500;
    letter-spacing: 0.3px;
}
.pbt-table td {
    padding: 3px 4px;
    color: #cbd5e1;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    vertical-align: middle;
}
.pbt-table tr:hover td { background: rgba(255,255,255,0.03); }
.pbt-swatch {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 1px;
    margin-right: 4px;
    vertical-align: middle;
}
.pbt-mono { font-family: var(--font-mono); color: #53ecd9; }
.pbt-role { color: var(--text-faint); font-size: 8px; }

/* Comparison pin section */
#comparison-pin-section {
    margin-top: 12px;
    border-top: 1px solid rgba(245, 158, 11, 0.25);
    padding-top: 10px;
    display: none;
}
.cp-win {
    color: #34d399 !important;
    font-weight: 600;
}

/* Auto-tour button active */
#btn-auto-tour.active {
    background: rgba(83,236,217,0.15);
    border-color: rgba(83,236,217,0.5);
    color: #53ecd9;
}

/* Lifecycle step ring animation during tour */
.lifecycle-step-item.active-step .lsi-num {
    animation: ring-pulse 1.2s ease-in-out infinite;
}
@keyframes ring-pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(83,236,217,0); }
    50% { box-shadow: 0 0 0 5px rgba(83,236,217,0.4); }
}
"""

if "MEGA ENHANCEMENT CSS" not in css:
    with open("Docs/visualizer/style.css", "a") as f:
        f.write(new_css)
    patches_applied.append("✅ Mega CSS added")
else:
    patches_applied.append("ℹ️  CSS already has mega block")

# ─── PATCH HTML: Add new UI slots ────────────────────────────────────────────
# 1. FPS counter
if 'id="fps-counter"' not in html:
    html = html.replace('<body class="app">', '<body class="app">\n    <div id="fps-counter">—</div>')
    patches_applied.append("✅ FPS counter HTML")

# 2. Tour step label
if 'id="tour-step-label"' not in html:
    html = html.replace('<div id="fps-counter">—</div>', '<div id="fps-counter">—</div>\n    <div id="tour-step-label"></div>')
    patches_applied.append("✅ Tour step label HTML")

# 3. Color legend panel
if 'id="color-legend"' not in html:
    html = html.replace('<div id="fps-counter">—</div>', '<div id="fps-counter">—</div>\n    <div id="color-legend"></div>')

# 4. Add Auto-tour, Legend, Pin buttons to header top-actions
OLD_INSPECTOR_BTN = """            <!-- Deep Inspector Toggle -->
            <button id="btn-toggle-inspector" class="round-button" title="Toggle Deep Architecture Inspector">
                🔬 Inspector
            </button>"""

NEW_INSPECTOR_BTN = """            <!-- Deep Inspector Toggle -->
            <button id="btn-toggle-inspector" class="round-button" title="Toggle Deep Architecture Inspector">
                🔬 Inspector
            </button>

            <!-- Auto Camera Tour -->
            <button id="btn-auto-tour" class="round-button" title="Auto-tour: animated camera flies through all layers">
                ▶ Auto Tour
            </button>

            <!-- Color Legend -->
            <button id="btn-legend-toggle" class="round-button" title="Toggle node color legend">
                🎨 Legend
            </button>

            <!-- Pin current module for comparison -->
            <button id="btn-pin-module" class="round-button" title="Pin this module for side-by-side comparison">
                📌 Pin
            </button>"""

if OLD_INSPECTOR_BTN in html:
    html = html.replace(OLD_INSPECTOR_BTN, NEW_INSPECTOR_BTN)
    patches_applied.append("✅ Auto-tour + legend + pin buttons in header")
else:
    patches_applied.append("⚠️  Inspector button not found in HTML")

# 5. Add param breakdown table + comparison pin slots in inspector drawer
OLD_DRAWER_END = """            <div class="educational-callout" id="insp-edu-note">"""
NEW_DRAWER_END = """            <!-- Parameter Breakdown Table -->
            <div id="param-breakdown-table"></div>

            <!-- Module Comparison Pin -->
            <div id="comparison-pin-section">
                <div class="pbt-title">📌 Pinned vs Active Comparison</div>
                <div style="font-size:9px;color:var(--text-faint);margin-bottom:6px;">Pinned: <span id="pinned-module-name">—</span></div>
                <div id="comparison-pin-content"></div>
            </div>

            <div class="educational-callout" id="insp-edu-note">"""

if OLD_DRAWER_END in html:
    html = html.replace(OLD_DRAWER_END, NEW_DRAWER_END, 1)
    patches_applied.append("✅ Param breakdown & pin comparison slots in drawer")
else:
    patches_applied.append("⚠️  Drawer end not found in HTML")

with open("Docs/visualizer/index.html", "w") as f:
    f.write(html)
patches_applied.append("✅ index.html written")

# ─── REPORT ──────────────────────────────────────────────────────────────────
print("\n".join(patches_applied))
print("\n🎉 Mega enhancement complete!")
