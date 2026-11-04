/**
 * ui/hud.js
 * =========
 * Real-time HUD and head-to-head parameter comparison display synchronizer.
 */

import { MODEL_SPECS } from '../constants.js';

/**
 * Updates floating head-to-head comparison cards and topbar status.
 * @param {string} modelKey - Selected model
 * @param {string} moduleKey - Selected compression module
 * @param {object} modelData - Model spatial dataset
 */
export function updateComparisonHUD(modelKey, moduleKey, modelData) {
    if (!modelData) return;

    const mod = (modelData.modules && modelData.modules[moduleKey]) || modelData;

    // Floating Compare Card
    const cmpModelEl = document.getElementById('cmp-model-name');
    const cmpBaseEl = document.getElementById('cmp-baseline-params');
    const cmpCompEl = document.getElementById('cmp-compressed-params');
    const cmpRatioEl = document.getElementById('cmp-ratio');
    const cmpStorageEl = document.getElementById('cmp-storage');

    const spec = MODEL_SPECS[modelKey];
    const modelDisplayName = spec ? spec.name.replace('-Base', '').replace('-Small', '') : modelKey.toUpperCase();

    if (cmpModelEl) cmpModelEl.innerText = modelDisplayName;
    if (cmpBaseEl) cmpBaseEl.innerText = modelData.total_baseline_params_formatted || '—';
    if (cmpCompEl) cmpCompEl.innerText = mod.total_compressed_params_formatted || '—';
    if (cmpRatioEl) {
        const ratio = mod.overall_compression_ratio || 1.0;
        const pct = Math.round((1 - 1 / Math.max(0.001, ratio)) * 100);
        cmpRatioEl.innerText = `${ratio.toFixed(2)}x (-${pct}%)`;
    }
    if (cmpStorageEl) cmpStorageEl.innerText = mod.storage_saved_formatted || '—';

    // Top Navigation active mechanism buttons
    document.querySelectorAll('.mechanism-button').forEach(btn => {
        const isActive = (btn.dataset.model === modelKey && btn.dataset.module === moduleKey);
        btn.classList.toggle('active', isActive);
        btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });
}
