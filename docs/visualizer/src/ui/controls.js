/**
 * ui/controls.js
 * ==============
 * Comprehensive Interactive Control Hub:
 * - Play/Pause, speed multiplier, timeline scrubber with 7-stage labels
 * - 2D/3D perspective toggles, Zoom in/out, Auto-Fit Camera
 * - Architectural Color Legend with real semantic swatches
 * - Pinned Module comparator state with toast notifications
 * - Model selector tabs, compression tabs, LOD buttons, mechanism buttons
 * - Keyboard Shortcuts modal & global hotkey navigation
 */

import { store } from '../state/store.js';

const TIMELINE_STAGES = [
    { step: '01 Stage', label: 'Input Embeddings' },
    { step: '02 Stage', label: 'Pre-Attention Norm' },
    { step: '03 Stage', label: 'Multi-Head Attention' },
    { step: '04 Stage', label: 'Residual Add Junction (+)' },
    { step: '05 Stage', label: 'Pre-FFN Norm' },
    { step: '06 Stage', label: 'Feed-Forward Network (FFN)' },
    { step: '07 Stage', label: 'Residual Add Junction (+)' }
];

const COLOR_PALETTE_ITEMS = [
    { name: 'Dense Baseline', swatch: '#3B82F6', desc: 'Standard full-rank projections (O(N²))' },
    { name: 'SVD Low-Rank', swatch: '#2DD4BF', desc: 'Factor V & U bottleneck (83.3% cut)' },
    { name: 'Mixture-of-Experts', swatch: '#A855F7', desc: 'Top-2 dynamic gating router & experts' },
    { name: 'Latent MLA', swatch: '#F59E0B', desc: 'DeepSeek 512-dim compressed KV cache' },
    { name: 'Residual Highway', swatch: '#10B981', desc: 'Additive skip connections (+)' },
    { name: 'Fallback Nodes', swatch: '#F97316', desc: 'Uncompressed layers (reconstruction err > τ)' },
    { name: 'Anchors / Norm', swatch: '#64748B', desc: 'RMSNorm / LayerNorm (0.2% params)' }
];

/**
 * Initializes all visualizer control surfaces and event listeners.
 * @param {object} handlers - Action handlers
 */
export function setupControls(handlers = {}) {
    // 1. Play / Pause Button
    const playBtn = document.getElementById('btn-play-pause');
    if (playBtn) {
        playBtn.addEventListener('click', () => {
            const isPlaying = handlers.onPlayPause ? handlers.onPlayPause() : false;
            playBtn.innerText = isPlaying ? '⏸ Pause' : '▶ Play';
            playBtn.classList.toggle('active', isPlaying);
        });
    }

    // 2. Simulation Speed Multiplier (both select and button controls)
    const speedSelect = document.getElementById('select-speed');
    if (speedSelect) {
        speedSelect.addEventListener('change', (e) => {
            const val = parseFloat(e.target.value) || 1.0;
            if (handlers.onSpeedChange) handlers.onSpeedChange(val);
        });
    }

    document.querySelectorAll('.btn-speed').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.btn-speed').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const speed = parseFloat(btn.dataset.speed || btn.textContent.replace('x', '')) || 1.0;
            if (speedSelect) speedSelect.value = speed.toString();
            if (handlers.onSpeedChange) handlers.onSpeedChange(speed);
        });
    });

    // 3. Timeline Scrubber & Stage Labels
    const scrubber = document.getElementById('timeline-scrubber');
    const stepLabel = document.getElementById('timeline-step');
    const titleLabel = document.getElementById('timeline-label');
    if (scrubber) {
        scrubber.addEventListener('input', (e) => {
            const step = Math.max(0, Math.min(6, parseInt(e.target.value, 10) || 0));
            const stage = TIMELINE_STAGES[step] || TIMELINE_STAGES[0];
            if (stepLabel) stepLabel.innerText = stage.step;
            if (titleLabel) titleLabel.innerText = stage.label;
            if (handlers.onScrub) handlers.onScrub(step);
        });
    }

    // 4. 2D / 3D Perspective Toggle
    const btn3d = document.getElementById('btn-view-3d');
    const btn2d = document.getElementById('btn-view-2d');
    if (btn3d && btn2d) {
        btn3d.addEventListener('click', () => {
            btn3d.classList.add('active');
            btn2d.classList.remove('active');
            if (handlers.onSwitchView) handlers.onSwitchView('3d');
        });
        btn2d.addEventListener('click', () => {
            btn2d.classList.add('active');
            btn3d.classList.remove('active');
            if (handlers.onSwitchView) handlers.onSwitchView('2d');
        });
    }

    // 5. Deep Inspector Drawer Toggle
    const inspBtn = document.getElementById('btn-toggle-inspector');
    const closeInspBtn = document.getElementById('btn-close-insp');
    const inspector = document.getElementById('inspector');
    const detailInspectBtn = document.getElementById('btn-detail-inspect');

    if (inspBtn && inspector) {
        inspBtn.addEventListener('click', () => {
            const isHidden = inspector.classList.contains('hidden');
            inspector.classList.toggle('hidden');
            inspBtn.setAttribute('aria-expanded', isHidden ? 'true' : 'false');
            if (document.body) document.body.classList.toggle('inspector-open', isHidden);
        });
    }
    if (detailInspectBtn && inspector) {
        detailInspectBtn.addEventListener('click', () => {
            inspector.classList.remove('hidden');
            if (inspBtn) inspBtn.setAttribute('aria-expanded', 'true');
            if (document.body) document.body.classList.add('inspector-open');
        });
    }
    if (closeInspBtn && inspector) {
        closeInspBtn.addEventListener('click', () => {
            inspector.classList.add('hidden');
            if (inspBtn) inspBtn.setAttribute('aria-expanded', 'false');
            if (document.body) document.body.classList.remove('inspector-open');
        });
    }

    // 6. Camera Auto-Fit Reset Button
    const resetBtn = document.getElementById('btn-reset-cam');
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            if (handlers.onResetCamera) handlers.onResetCamera();
        });
    }

    // 7. Architectural Color Legend Toggle & DOM Generation
    const legendBtn = document.getElementById('btn-legend-toggle');
    const colorLegend = document.getElementById('color-legend');
    if (colorLegend && !colorLegend.hasChildNodes()) {
        let legendHtml = `
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span class="legend-title">ARCHITECTURE COLOR PALETTE</span>
                <button id="btn-close-legend" class="btn-icon-only" style="width:16px; height:16px; font-size:10px; line-height:1;">✕</button>
            </div>
            <div style="display:flex; flex-direction:column; gap:4px;">
        `;
        COLOR_PALETTE_ITEMS.forEach(item => {
            legendHtml += `
                <div class="legend-row">
                    <span class="legend-swatch" style="background:${item.swatch};"></span>
                    <div class="legend-info">
                        <span class="legend-label">${item.name}</span>
                        <span class="legend-desc">${item.desc}</span>
                    </div>
                </div>
            `;
        });
        legendHtml += `</div>`;
        colorLegend.innerHTML = legendHtml;

        const closeLegend = document.getElementById('btn-close-legend');
        if (closeLegend) closeLegend.onclick = () => { colorLegend.style.display = 'none'; };
    }

    if (legendBtn && colorLegend) {
        legendBtn.addEventListener('click', () => {
            const isVisible = colorLegend.style.display === 'block';
            colorLegend.style.display = isVisible ? 'none' : 'block';
            legendBtn.classList.toggle('active', !isVisible);
        });
    }

    // 8. Pinned Module Comparator State
    const pinBtn = document.getElementById('btn-pin-module');
    if (pinBtn) {
        pinBtn.addEventListener('click', () => {
            const curMod = store.get('activeResearchModule') || { name: 'SVDLinearAttention', paradigm: 'svd', rank: 64 };
            const existingPin = store.get('pinnedModule');

            if (existingPin && (existingPin.id === curMod.id || existingPin.name === curMod.name)) {
                // Unpin
                store.set({ pinnedModule: null });
                pinBtn.classList.remove('active');
                pinBtn.style.background = '';
                showToast('📌 Unpinned module');
            } else {
                // Pin active module
                store.set({ pinnedModule: curMod });
                pinBtn.classList.add('active');
                pinBtn.style.background = 'rgba(45,212,191,0.3)';
                pinBtn.style.color = '#2dd4bf';
                showToast(`📌 Pinned: ${curMod.name || curMod.id}`);

                const activeNodeLabel = document.getElementById('cmp-selected-node');
                if (activeNodeLabel) {
                    activeNodeLabel.textContent = `[PINNED] ${curMod.name || curMod.id}`;
                    activeNodeLabel.style.color = '#2dd4bf';
                }
            }
        });
    }

    // 9. Browse All Modules Button (opens catalog modal)
    const browseAllBtn = document.getElementById('btn-browse-all-modules');
    const catalogBtn = document.getElementById('btn-open-catalog');
    if (browseAllBtn && catalogBtn) {
        browseAllBtn.addEventListener('click', () => catalogBtn.click());
    }

    // 10. Model Tabs Binding
    document.querySelectorAll('.model-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.model-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            const modelKey = tab.dataset.model || 'distilbert';
            if (handlers.onSwitchModel) handlers.onSwitchModel(modelKey);
        });
    });

    // 11. Compression Tabs Binding
    document.querySelectorAll('.comp-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.id === 'btn-browse-all-modules') return;
            document.querySelectorAll('.comp-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            const compKey = tab.dataset.module || tab.dataset.comp || 'svd';
            if (handlers.onSwitchCompression) handlers.onSwitchCompression(compKey);
        });
    });

    // 12. Level of Detail (LOD) Buttons
    document.querySelectorAll('.btn-lod').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.btn-lod').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const lod = btn.dataset.lod || 'hybrid';
            if (handlers.onSwitchLOD) handlers.onSwitchLOD(lod);
        });
    });

    // 13. Mechanism Nav Buttons (Active tab styling synchronization)
    document.querySelectorAll('.mechanism-button').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.mechanism-button').forEach(b => {
                b.classList.remove('active');
                b.setAttribute('aria-selected', 'false');
            });
            btn.classList.add('active');
            btn.setAttribute('aria-selected', 'true');
        });
    });

    // 14. Keyboard Shortcuts Modal & Trigger Button
    const keysBtn = document.getElementById('btn-keyboard-shortcuts');
    if (keysBtn) {
        keysBtn.addEventListener('click', () => openKeyboardShortcutsModal());
    }

    // 15. Global Hotkeys Handler
    window.addEventListener('keydown', (e) => {
        if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA')) return;

        if (e.code === 'Space') {
            e.preventDefault();
            if (playBtn) playBtn.click();
        } else if (e.key === 'r' || e.key === 'R') {
            if (handlers.onResetCamera) handlers.onResetCamera();
        } else if (e.key === 'm' || e.key === 'M') {
            if (catalogBtn) catalogBtn.click();
        } else if (e.key === 'u' || e.key === 'U') {
            const btn = document.getElementById('btn-open-unfolding');
            if (btn) btn.click();
        } else if (e.key === 'h' || e.key === 'H') {
            const btn = document.getElementById('btn-open-heatmap');
            if (btn) btn.click();
        } else if (e.key === 't' || e.key === 'T') {
            const btn = document.getElementById('btn-open-prediction');
            if (btn) btn.click();
        } else if (e.key === 'c' || e.key === 'C') {
            const btn = document.getElementById('btn-open-compression');
            if (btn) btn.click();
        } else if (e.key === 'g' || e.key === 'G') {
            const btn = document.getElementById('btn-open-lessons');
            if (btn) btn.click();
        } else if (e.key === 'l' || e.key === 'L') {
            const btn = document.getElementById('btn-open-learn');
            if (btn) btn.click();
        } else if (e.key === 'v' || e.key === 'V') {
            const btn = document.getElementById('btn-record-video');
            if (btn) btn.click();
        } else if (e.key === 'p' || e.key === 'P') {
            const btn = document.getElementById('btn-present-mode');
            if (btn) btn.click();
        } else if (e.key === '?' || (e.shiftKey && e.key === '/')) {
            openKeyboardShortcutsModal();
        } else if (e.key === 'Escape') {
            document.querySelectorAll('.modal-overlay').forEach(m => m.classList.add('hidden'));
            if (inspector) {
                inspector.classList.add('hidden');
                if (document.body) document.body.classList.remove('inspector-open');
                if (inspBtn) inspBtn.setAttribute('aria-expanded', 'false');
            }
            if (colorLegend) colorLegend.style.display = 'none';
        }
    });
}

/**
 * Displays an elegant visual toast message.
 * @param {string} msg - Notification message
 */
function showToast(msg) {
    let toast = document.querySelector('.permalink-copied');
    if (!toast) {
        toast = document.createElement('div');
        toast.className = 'permalink-copied';
        document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.style.display = 'block';
    setTimeout(() => {
        if (toast) toast.remove();
    }, 2500);
}

/**
 * Displays the Keyboard Shortcuts Cheatsheet Modal.
 */
export function openKeyboardShortcutsModal() {
    let modal = document.getElementById('keyboard-shortcuts-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'keyboard-shortcuts-modal';
        modal.className = 'modal-overlay hidden';
        modal.setAttribute('role', 'dialog');
        modal.setAttribute('aria-modal', 'true');
        modal.innerHTML = `
            <div class="modal-container" style="max-width: 640px; display:flex; flex-direction:column;">
                <div class="modal-header">
                    <div>
                        <span class="hud-badge" style="background:#38bdf8; color:#000; font-weight:800;">COMMAND CHEATSHEET</span>
                        <h2>Visualizer Keyboard Shortcuts</h2>
                        <p class="modal-subtitle">Fast navigation and interactive controls for demonstrations and research exploration.</p>
                    </div>
                    <button id="btn-close-keys-modal" class="btn-icon-only modal-close">✕</button>
                </div>

                <div style="padding:20px 24px; overflow-y:auto; display:flex; flex-direction:column; gap:12px; font-size:12px;">
                    <div style="display:grid; grid-template-columns: 120px 1fr; gap:10px; align-items:center; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:8px;">
                        <kbd style="background:#080f1c; border:1px solid rgba(255,255,255,0.2); padding:3px 8px; border-radius:4px; font-family:monospace; color:#38bdf8; text-align:center;">Space</kbd>
                        <span>Play / Pause continuous 3D particle simulation</span>
                    </div>
                    <div style="display:grid; grid-template-columns: 120px 1fr; gap:10px; align-items:center; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:8px;">
                        <kbd style="background:#080f1c; border:1px solid rgba(255,255,255,0.2); padding:3px 8px; border-radius:4px; font-family:monospace; color:#38bdf8; text-align:center;">P</kbd>
                        <span>Toggle Minimal Presentation Mode (5-min presentation)</span>
                    </div>
                    <div style="display:grid; grid-template-columns: 120px 1fr; gap:10px; align-items:center; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:8px;">
                        <kbd style="background:#080f1c; border:1px solid rgba(255,255,255,0.2); padding:3px 8px; border-radius:4px; font-family:monospace; color:#38bdf8; text-align:center;">L</kbd>
                        <span>Open Educational Learn Mode Primer (English Masterclass)</span>
                    </div>
                    <div style="display:grid; grid-template-columns: 120px 1fr; gap:10px; align-items:center; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:8px;">
                        <kbd style="background:#080f1c; border:1px solid rgba(255,255,255,0.2); padding:3px 8px; border-radius:4px; font-family:monospace; color:#38bdf8; text-align:center;">G</kbd>
                        <span>Open Guided Lessons Mode (5 Interactive Masterclasses)</span>
                    </div>
                    <div style="display:grid; grid-template-columns: 120px 1fr; gap:10px; align-items:center; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:8px;">
                        <kbd style="background:#080f1c; border:1px solid rgba(255,255,255,0.2); padding:3px 8px; border-radius:4px; font-family:monospace; color:#38bdf8; text-align:center;">H</kbd>
                        <span>Open Interactive Attention Heatmap Explorer</span>
                    </div>
                    <div style="display:grid; grid-template-columns: 120px 1fr; gap:10px; align-items:center; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:8px;">
                        <kbd style="background:#080f1c; border:1px solid rgba(255,255,255,0.2); padding:3px 8px; border-radius:4px; font-family:monospace; color:#c084fc; text-align:center;">U</kbd>
                        <span>Open Attention Square Deconstruction &amp; Matrix Unfolding Engine</span>
                    </div>
                    <div style="display:grid; grid-template-columns: 120px 1fr; gap:10px; align-items:center; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:8px;">
                        <kbd style="background:#080f1c; border:1px solid rgba(255,255,255,0.2); padding:3px 8px; border-radius:4px; font-family:monospace; color:#38bdf8; text-align:center;">T</kbd>
                        <span>Open Live Autoregressive Token Prediction Playground</span>
                    </div>
                    <div style="display:grid; grid-template-columns: 120px 1fr; gap:10px; align-items:center; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:8px;">
                        <kbd style="background:#080f1c; border:1px solid rgba(255,255,255,0.2); padding:3px 8px; border-radius:4px; font-family:monospace; color:#38bdf8; text-align:center;">C</kbd>
                        <span>Open SVD Low-Rank Bottleneck Compression Simulator</span>
                    </div>
                    <div style="display:grid; grid-template-columns: 120px 1fr; gap:10px; align-items:center; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:8px;">
                        <kbd style="background:#080f1c; border:1px solid rgba(255,255,255,0.2); padding:3px 8px; border-radius:4px; font-family:monospace; color:#38bdf8; text-align:center;">V</kbd>
                        <span>Open 360° Orbit Video Exporter</span>
                    </div>
                    <div style="display:grid; grid-template-columns: 120px 1fr; gap:10px; align-items:center; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:8px;">
                        <kbd style="background:#080f1c; border:1px solid rgba(255,255,255,0.2); padding:3px 8px; border-radius:4px; font-family:monospace; color:#38bdf8; text-align:center;">M</kbd>
                        <span>Open Research Module Library (241 Modules)</span>
                    </div>
                    <div style="display:grid; grid-template-columns: 120px 1fr; gap:10px; align-items:center; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:8px;">
                        <kbd style="background:#080f1c; border:1px solid rgba(255,255,255,0.2); padding:3px 8px; border-radius:4px; font-family:monospace; color:#38bdf8; text-align:center;">+ / -</kbd>
                        <span>Smooth Zoom In / Zoom Out camera distance</span>
                    </div>
                    <div style="display:grid; grid-template-columns: 120px 1fr; gap:10px; align-items:center; border-bottom:1px solid rgba(255,255,255,0.06); padding-bottom:8px;">
                        <kbd style="background:#080f1c; border:1px solid rgba(255,255,255,0.2); padding:3px 8px; border-radius:4px; font-family:monospace; color:#38bdf8; text-align:center;">R</kbd>
                        <span>Auto-Fit Camera to active model geometry</span>
                    </div>
                    <div style="display:grid; grid-template-columns: 120px 1fr; gap:10px; align-items:center;">
                        <kbd style="background:#080f1c; border:1px solid rgba(255,255,255,0.2); padding:3px 8px; border-radius:4px; font-family:monospace; color:#38bdf8; text-align:center;">Esc</kbd>
                        <span>Close any active modal or inspector drawer</span>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        const closeBtn = document.getElementById('btn-close-keys-modal');
        if (closeBtn) closeBtn.onclick = () => modal.classList.add('hidden');
        modal.onclick = (e) => { if (e.target === modal) modal.classList.add('hidden'); };
    }

    modal.classList.remove('hidden');
}
