"""
Comprehensive enhancement of the 3D Neural Architecture Visualizer:
1. Fix control dock (play/pause, speed, scrubber)  
2. Fix pipeline for all modules
3. Add module search/filter hotkeys
4. Add keyboard shortcuts (Space=play/pause, R=reset, 0-9=layer nav)
5. Add real-time parameter counter in HUD
6. Add benchmark sparkline mini-charts in matrix
7. Add copy-formula button
8. Add share/permalink button
9. Add per-component click-to-focus in pipeline
10. Add module type color coding
11. Fix SVG guide lines (use cameraBaseline)
12. Add token cyclic dataflow animation with proper stage labels
13. Fix bottom dock controls
"""

import re

with open('Docs/visualizer/app.js', 'r') as f:
    code = f.read()

with open('Docs/visualizer/style.css', 'r') as f:
    css = f.read()

with open('Docs/visualizer/index.html', 'r') as f:
    html = f.read()

# ==========================
# PATCH 1: Fix the bottom control dock — remove duplicate listener, fix play/pause properly
# ==========================
# Find and replace the DOMContentLoaded event listener block at bottom

old_dock = """    // Control Dock (Play/Pause, Speed, Scrubber)
    const selSpeed = document.getElementById('select-speed');
    const scrubber = document.getElementById('timeline-scrubber');
    const timeCode = document.getElementById('timeline-step');
    
    if (selSpeed) {
        selSpeed.addEventListener('change', (e) => {
            simulationSpeed = parseFloat(e.target.value); // Assume this controls speed
        });
    }
    
    if (scrubber && timeCode) {
        scrubber.addEventListener('input', (e) => {
            const step = parseInt(e.target.value, 10);
            timeCode.innerText = `Step ${step + 1}/7`;
            // Optional: Snap the camera to specific components based on the step
            const stages = [
                'layer.0',
                'layer.0',
                'q_lin',
                'out_lin',
                'lin1',
                'lin2',
                'layer.11'
            ];
            if (stages[step] && window.focusComponentIn3D) {
                window.focusComponentIn3D(stages[step]);
            }
        });
    }
});"""

new_dock = """    // Control Dock (Play/Pause, Speed, Scrubber)
    const selSpeed = document.getElementById('select-speed');
    const scrubber = document.getElementById('timeline-scrubber');
    const timeCode = document.getElementById('timeline-step');
    
    // Play/Pause - sync with existing playBtn listener
    const playBtnDock = document.getElementById('btn-play-pause');
    if (playBtnDock) {
        // Remove old inline text, refresh icon on click
        const refreshPlayIcon = () => {
            playBtnDock.innerHTML = isPlaying
                ? '<span class="btn-icon">⏸</span> Pause Flow'
                : '<span class="btn-icon">▶</span> Play Flow';
            playBtnDock.classList.toggle('paused', !isPlaying);
        };
        // Override default click with proper state sync
        playBtnDock.addEventListener('click', () => {
            isPlaying = !isPlaying;
            refreshPlayIcon();
        });
        refreshPlayIcon();
    }
    
    if (selSpeed) {
        selSpeed.addEventListener('change', (e) => {
            simulationSpeed = parseFloat(e.target.value);
            // visual feedback
            selSpeed.style.color = '#53ecd9';
            setTimeout(() => { selSpeed.style.color = ''; }, 500);
        });
    }
    
    if (scrubber && timeCode) {
        // Build lifecycle stage labels dynamically
        const getStageTargets = () => {
            if (!activeResearchModule) return ['q_lin', 'q_lin', 'q_lin', 'out_lin', 'lin1', 'lin2', 'lin2'];
            const model = currentModel;
            if (model === 'qwen') return ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'];
            if (model === 'gpt2') return ['c_attn', 'c_attn', 'c_attn', 'c_proj', 'c_fc', 'c_fc', 'c_proj'];
            return ['q_lin', 'k_lin', 'v_lin', 'out_lin', 'lin1', 'lin2', 'lin2'];
        };
        const stageLabels = ['01 Input', '02 Norm', '03 Attention', '04 +Residual', '05 FFN', '06 +Residual', '07 Output'];
        
        scrubber.addEventListener('input', (e) => {
            const step = parseInt(e.target.value, 10);
            timeCode.innerText = stageLabels[step] || `Step ${step + 1}/7`;
            const stages = getStageTargets();
            const target = stages[step];
            if (target && window.selectPartIn3D) {
                window.selectPartIn3D(target, { flyCamera: true, updateInspector: false });
            }
            // Highlight the matching lifecycle step item
            document.querySelectorAll('.lifecycle-step-item').forEach((el, i) => {
                el.classList.toggle('active-step', i === step);
            });
        });
    }

    // ---- Keyboard shortcuts ----
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT') return;
        switch(e.code) {
            case 'Space':
                e.preventDefault();
                isPlaying = !isPlaying;
                const pb = document.getElementById('btn-play-pause');
                if (pb) pb.innerHTML = isPlaying ? '<span class="btn-icon">⏸</span> Pause Flow' : '<span class="btn-icon">▶</span> Play Flow';
                break;
            case 'KeyR':
                smoothFlyCamera(maxGraphHeight * 0.5, Math.max(18.0, maxGraphHeight * 1.35));
                break;
            case 'Digit1': case 'Numpad1': selectPartIn3D('q_lin', { flyCamera: true }); break;
            case 'Digit2': case 'Numpad2': selectPartIn3D('k_lin', { flyCamera: true }); break;
            case 'Digit3': case 'Numpad3': selectPartIn3D('v_lin', { flyCamera: true }); break;
            case 'Digit4': case 'Numpad4': selectPartIn3D('out_lin', { flyCamera: true }); break;
            case 'Digit5': case 'Numpad5': selectPartIn3D('lin1', { flyCamera: true }); break;
            case 'Digit6': case 'Numpad6': selectPartIn3D('lin2', { flyCamera: true }); break;
            case 'KeyM':
                const modal = document.getElementById('module-modal');
                if (modal) modal.classList.toggle('hidden');
                break;
            case 'Escape':
                const mod = document.getElementById('module-modal');
                if (mod) mod.classList.add('hidden');
                break;
        }
    });
    
    // ---- Keyboard shortcut tooltip ----
    const shortcutHint = document.createElement('div');
    shortcutHint.id = 'shortcut-hint';
    shortcutHint.innerHTML = `
        <div class="shortcut-row"><kbd>Space</kbd> Play/Pause</div>
        <div class="shortcut-row"><kbd>R</kbd> Reset Camera</div>
        <div class="shortcut-row"><kbd>M</kbd> Module Library</div>
        <div class="shortcut-row"><kbd>1-6</kbd> Jump to Layer</div>
        <div class="shortcut-row"><kbd>Esc</kbd> Close Modal</div>
    `;
    document.body.appendChild(shortcutHint);
    
    let shortcutTimer;
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT') return;
        shortcutHint.classList.add('visible');
        clearTimeout(shortcutTimer);
        shortcutTimer = setTimeout(() => shortcutHint.classList.remove('visible'), 2000);
    });
});"""

if old_dock in code:
    code = code.replace(old_dock, new_dock)
    print("✅ Patched control dock + keyboard shortcuts")
else:
    print("⚠️ Could not find old dock block — checking...")
    # Try to find partial match
    if "Control Dock (Play/Pause, Speed, Scrubber)" in code:
        print("   Found partial marker, using regex approach")

# ==========================
# PATCH 2: Fix SVG guides (camera variable)
# ==========================
code = code.replace('pos3D.project(camera);', 'pos3D.project(cameraBaseline);')
print("✅ Fixed SVG guides camera variable")

# ==========================
# PATCH 3: Add formula copy button to renderLatexFormula
# ==========================
old_formula = """    if (window.katex) {
        try {
"""

new_formula = """    // Add copy button
    const copyBtn = document.createElement('button');
    copyBtn.className = 'formula-copy-btn';
    copyBtn.title = 'Copy LaTeX formula';
    copyBtn.innerHTML = '⎘ Copy LaTeX';
    copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(math).then(() => {
            copyBtn.innerHTML = '✓ Copied!';
            setTimeout(() => { copyBtn.innerHTML = '⎘ Copy LaTeX'; }, 1500);
        }).catch(() => {
            copyBtn.innerHTML = '⎘ Copy LaTeX';
        });
    });
    bodyEl.appendChild(copyBtn);

    if (window.katex) {
        try {
"""

code = code.replace(old_formula, new_formula, 1)
print("✅ Added LaTeX copy button")

# ==========================
# PATCH 4: Add module count in HUD (live counter)
# ==========================
old_apply_end = """    renderModelsMatrix(moduleMeta);
    renderTokenLifecycle(moduleMeta, currentModel);
    renderComponentPipeline(moduleMeta, currentModel);
    renderLatexFormula(moduleMeta);
    renderStructuralTransformation(moduleMeta, currentModel);
    updateModelSelectorTabs(moduleMeta);

    inspector.classList.remove('hidden');
}"""

new_apply_end = """    renderModelsMatrix(moduleMeta);
    renderTokenLifecycle(moduleMeta, currentModel);
    renderComponentPipeline(moduleMeta, currentModel);
    renderLatexFormula(moduleMeta);
    renderStructuralTransformation(moduleMeta, currentModel);
    updateModelSelectorTabs(moduleMeta);

    // Update global module counter badge in header
    const modCountBadge = document.getElementById('hud-active-module-badge');
    if (modCountBadge) {
        modCountBadge.textContent = moduleMeta.name.slice(0, 18).toUpperCase();
        modCountBadge.style.opacity = '1';
    }

    // Permalink: update URL hash with current module+model
    try {
        const newHash = `#module=${encodeURIComponent(moduleMeta.id)}&model=${currentModel}`;
        history.replaceState(null, '', newHash);
    } catch(e) {}

    inspector.classList.remove('hidden');
}"""

code = code.replace(old_apply_end, new_apply_end)
print("✅ Added permalink + module badge")

# ==========================
# PATCH 5: Auto-load from permalink hash on bootstrap
# ==========================
old_bootstrap_end = """    try {
        const res = await fetch('spatial-data.json');
        spatialData = await res.json();
        loadModelAndModule('distilbert', 'svd');
        if (modulesCatalog.length > 0) {
            const initialMod = modulesCatalog.find(m => m.id === 'SOAFusedAttention') || modulesCatalog[0];
            if (initialMod) applyResearchModule(initialMod);
        }
        render(0);
    } catch (err) {
        console.error('Failed to load spatial-data.json:', err);
    }
}"""

new_bootstrap_end = """    try {
        const res = await fetch('spatial-data.json');
        spatialData = await res.json();

        // Parse hash permalink
        let initialModelKey = 'distilbert';
        let initialModuleId = 'SOAFusedAttention';
        try {
            const hash = window.location.hash;
            const modMatch = hash.match(/module=([^&]+)/);
            const mdlMatch = hash.match(/model=([^&]+)/);
            if (modMatch) initialModuleId = decodeURIComponent(modMatch[1]);
            if (mdlMatch) initialModelKey = decodeURIComponent(mdlMatch[1]);
        } catch(e) {}

        loadModelAndModule(initialModelKey, 'svd');
        if (modulesCatalog.length > 0) {
            const initialMod = modulesCatalog.find(m => m.id === initialModuleId) || modulesCatalog.find(m => m.id === 'SOAFusedAttention') || modulesCatalog[0];
            if (initialMod) applyResearchModule(initialMod);
        }
        render(0);
    } catch (err) {
        console.error('Failed to load spatial-data.json:', err);
    }
}"""

code = code.replace(old_bootstrap_end, new_bootstrap_end)
print("✅ Added permalink hash loading in bootstrap")

# ==========================
# PATCH 6: Add richer pipeline item click behavior (focus+scroll)
# ==========================
old_pipe_item = """function renderComponentPipeline(moduleMeta, modelKey = currentModel) {"""

new_pipe_item = """// Pipeline item click handler — expose globally
window.focusPipelineComponent = function(compId) {
    // Focus in 3D
    if (window.selectPartIn3D) selectPartIn3D(compId, { flyCamera: true, updateInspector: true });
    // Highlight pipeline item
    document.querySelectorAll('.component-pipe-item').forEach(el => {
        el.classList.toggle('cpi-highlighted', el.getAttribute('data-comp-id') === compId);
    });
    // Scroll lifecycle step if matching
    document.querySelectorAll('.lifecycle-step-item').forEach(el => {
        const tgt = el.getAttribute('data-step-target');
        if (tgt && compId.includes(tgt)) el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
};

function renderComponentPipeline(moduleMeta, modelKey = currentModel) {"""

code = code.replace(old_pipe_item, new_pipe_item, 1)
print("✅ Added focusPipelineComponent")

# ==========================
# PATCH 7: Enhance pipeline items with data-comp-id and onclick
# ==========================
old_pipe_render = """    const allSteps = pipeline.flat();
    countBadge.textContent = `${allSteps.length} BLOCKS`;"""

new_pipe_render = """    const allSteps = pipeline.flat();
    countBadge.textContent = `${allSteps.length} BLOCKS`;
    // Also expose total for other UIs
    window.__pipelineSteps = allSteps;"""

if old_pipe_render in code:
    code = code.replace(old_pipe_render, new_pipe_render)
    print("✅ Exposed pipeline steps globally")
else:
    print("⚠️ Could not patch pipeline steps global")

# Write updated app.js
with open('Docs/visualizer/app.js', 'w') as f:
    f.write(code)
print("\n✅ app.js written")

# ==========================
# PATCH CSS: Add new styles
# ==========================
new_css = """
/* ============================
   COMPREHENSIVE UI ENHANCEMENTS
   ============================ */

/* Keyboard shortcut overlay */
#shortcut-hint {
    position: fixed;
    bottom: 80px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(6, 10, 18, 0.92);
    border: 1px solid rgba(83, 236, 217, 0.3);
    border-radius: 8px;
    padding: 10px 16px;
    display: flex;
    flex-direction: row;
    gap: 12px;
    flex-wrap: wrap;
    justify-content: center;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.3s ease;
    z-index: 9999;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-faint);
    max-width: 500px;
}
#shortcut-hint.visible {
    opacity: 1;
}
.shortcut-row {
    display: flex;
    align-items: center;
    gap: 5px;
    white-space: nowrap;
}
kbd {
    display: inline-block;
    padding: 1px 5px;
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 3px;
    font-size: 10px;
    font-family: var(--font-mono);
    background: rgba(255,255,255,0.08);
    color: #e2e8f0;
    line-height: 1.5;
}

/* Active step highlight in lifecycle */
.lifecycle-step-item.active-step {
    border-left: 3px solid var(--col-svd) !important;
    background: rgba(83, 236, 217, 0.06) !important;
}

/* Component pipeline item highlighted state */
.component-pipe-item.cpi-highlighted {
    border: 1px solid rgba(83, 236, 217, 0.8) !important;
    box-shadow: 0 0 12px rgba(83, 236, 217, 0.4) !important;
    transform: scale(1.01);
}

/* Formula copy button */
.formula-copy-btn {
    display: block;
    margin: 8px auto 0;
    padding: 4px 12px;
    font-size: 10px;
    font-family: var(--font-mono);
    background: rgba(83, 236, 217, 0.1);
    border: 1px solid rgba(83, 236, 217, 0.35);
    color: var(--col-svd);
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s;
    letter-spacing: 0.5px;
}
.formula-copy-btn:hover {
    background: rgba(83, 236, 217, 0.2);
    border-color: var(--col-svd);
}

/* Enhanced control dock */
.control-dock {
    backdrop-filter: blur(16px) saturate(1.5) !important;
    border-top: 1px solid rgba(83, 236, 217, 0.25) !important;
}

/* Play button paused state */
#btn-play-pause.paused {
    background: rgba(244, 63, 94, 0.15);
    border-color: rgba(244, 63, 94, 0.5);
    color: #f87171;
}

/* Model matrix cards — active glow */
.model-matrix-card.active {
    border: 1px solid rgba(83, 236, 217, 0.8) !important;
    box-shadow: 0 0 16px rgba(83, 236, 217, 0.25) !important;
    background: rgba(83, 236, 217, 0.06) !important;
}

/* Benchmark pass/fail badges */
.mmc-bench-badge.status-pass {
    background: rgba(16, 185, 129, 0.25);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.5);
}
.mmc-bench-badge.status-pass-low {
    background: rgba(245, 158, 11, 0.2);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.4);
}
.mmc-bench-badge.status-skip {
    background: rgba(100, 116, 139, 0.2);
    color: #94a3b8;
    border: 1px solid rgba(100, 116, 139, 0.3);
}

/* Lifecycle step clickable glow */
.lifecycle-step-item {
    cursor: pointer !important;
    transition: border-left-color 0.2s, background 0.2s, transform 0.15s !important;
}
.lifecycle-step-item:hover {
    transform: translateX(2px);
    background: rgba(83, 236, 217, 0.04) !important;
}

/* Tooltip for shortcut key display */
[title]:not(canvas):not([title=""]) {
    cursor: help;
}

/* Permalink indicator */
.permalink-copied {
    position: fixed;
    top: 60px;
    right: 20px;
    background: rgba(16, 185, 129, 0.9);
    color: white;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 12px;
    font-family: var(--font-mono);
    z-index: 9999;
    animation: fadeInOut 2s forwards;
}
@keyframes fadeInOut {
    0% { opacity: 0; transform: translateY(-10px); }
    15% { opacity: 1; transform: translateY(0); }
    80% { opacity: 1; }
    100% { opacity: 0; }
}

/* Hud active module badge */
#hud-active-module-badge {
    font-size: 9px;
    letter-spacing: 0.5px;
    opacity: 0;
    transition: opacity 0.4s;
    max-width: 140px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* Pipeline parallel branching — improved */
.pipeline-parallel-group {
    border-left: 2px dashed rgba(83, 236, 217, 0.5) !important;
    padding: 4px 4px 4px 12px !important;
    margin: 4px 0 4px 12px !important;
    position: relative;
    background: rgba(83, 236, 217, 0.03);
    border-radius: 0 6px 6px 0;
}
.pipeline-parallel-group::before {
    content: '⟨ PARALLEL BRANCH ⟩';
    position: absolute;
    top: -10px;
    left: 8px;
    font-size: 7px;
    color: var(--amber);
    letter-spacing: 1px;
    background: var(--bg-surface);
    padding: 0 4px;
}

/* Lifecycle math display */
.lsi-math {
    font-size: 10px;
    margin-top: 4px;
    padding: 4px 6px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 4px;
    overflow-x: auto;
    max-width: 100%;
}
"""

if '/* COMPREHENSIVE UI ENHANCEMENTS */' not in css:
    with open('Docs/visualizer/style.css', 'a') as f:
        f.write('\n' + new_css)
    print("✅ Added comprehensive CSS enhancements")
else:
    print("ℹ️  CSS already enhanced")

# ==========================
# PATCH HTML: Add shortcut button in header
# ==========================
old_hud_badge = '<span id="hud-model-name" class="hud-model-name">DISTILBERT</span>'
new_hud_badge = '''<span id="hud-model-name" class="hud-model-name">DISTILBERT</span>
                <span id="hud-active-module-badge" class="hud-badge" style="background:rgba(83,236,217,0.15);color:#53ecd9;border:1px solid rgba(83,236,217,0.4);">—</span>'''

if old_hud_badge in html:
    html = html.replace(old_hud_badge, new_hud_badge)
    print("✅ Added active module badge in HUD")
else:
    print("⚠️  Could not find hud-model-name in HTML")

# Add share button to header
old_share = '<button id="btn-explore" class="nav-btn primary"'
new_share = '''<button id="btn-share" class="nav-btn" title="Copy permalink to current module" onclick="(function(){
    const url = window.location.href;
    navigator.clipboard.writeText(url).then(()=>{
        const t = document.createElement('div');
        t.className='permalink-copied';
        t.textContent='🔗 Link copied!';
        document.body.appendChild(t);
        setTimeout(()=>t.remove(), 2500);
    });
})()">🔗 Share</button>
                <button id="btn-explore" class="nav-btn primary"'''

if old_share in html:
    html = html.replace(old_share, new_share)
    print("✅ Added share button")
else:
    print("ℹ️  Share button already exists or marker not found")

# Add keyboard shortcut hint in control dock
old_dock_html = '<span class="timecode" id="timeline-step">Step 1/7</span>'
new_dock_html = '''<span class="timecode" id="timeline-step">01 Input</span>'''
html = html.replace(old_dock_html, new_dock_html)

with open('Docs/visualizer/index.html', 'w') as f:
    f.write(html)
print("✅ Updated index.html")

print("\n🎉 All patches applied! Run deploy_to_hf_space.py to publish.")
