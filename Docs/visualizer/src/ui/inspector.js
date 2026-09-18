/**
 * ui/inspector.js
 * ===============
 * Deep Architecture Inspector Drawer controller and telemetry updater.
 */

import { getEducationalExplanation } from '../educational/explanations.js';
import { renderStructuralTransformation } from '../educational/transformation.js';
import { renderTokenLifecycle } from '../educational/token_lifecycle.js';
import { renderComponentPipeline } from '../educational/component_pipeline.js';
import { renderLatexFormula } from '../educational/formula_renderer.js';
import { formatParamCount } from '../utils/format.js';

/**
 * Updates the slide-out inspector drawer with full telemetry for a selected node.
 * @param {object} node - Selected node metadata
 * @param {boolean} isLeft - True if baseline, False if compressed
 * @param {object} activeModule - Current active research module
 * @param {string} currentModel - Active model key
 */
export function updateInspectorWithNode(node, isLeft, activeModule, currentModel) {
    if (!node) return;

    const drawer = document.getElementById('inspector');
    if (drawer && drawer.classList.contains('hidden')) {
        drawer.classList.remove('hidden');
    }

    const meta = node.metadata || {};
    const dOut = (meta.shape && meta.shape[0]) || 768;
    const dIn = (meta.shape && meta.shape[1]) || 768;

    // Badges & Labels
    const badgeEl = document.getElementById('insp-type-badge');
    if (badgeEl) badgeEl.innerText = (node.type || 'LAYER').toUpperCase();

    const titleEl = document.getElementById('insp-title');
    if (titleEl) titleEl.innerText = node.label || node.id || '—';

    const shapeEl = document.getElementById('insp-shape');
    if (shapeEl) {
        shapeEl.innerText = (meta.shape && Array.isArray(meta.shape))
            ? `[${meta.shape.join(' × ')}]`
            : `[${dOut} × ${dIn}]`;
    }

    const paramsEl = document.getElementById('insp-params');
    if (paramsEl) {
        paramsEl.innerText = meta.params || (meta.params_raw ? meta.params_raw.toLocaleString() : '—');
    }

    const storageEl = document.getElementById('insp-storage');
    if (storageEl) storageEl.innerText = meta.storage || '—';

    const effEl = document.getElementById('insp-efficiency');
    if (effEl) effEl.innerText = meta.compression || (isLeft ? '1.0x Baseline' : 'Compressed');

    // Matrix Dimensions & Volume
    const dimsEl = document.getElementById('insp-tensor-dims');
    if (dimsEl) dimsEl.innerText = `[${dOut} × ${dIn}]`;

    const aspectEl = document.getElementById('insp-aspect-ratio');
    if (aspectEl) aspectEl.innerText = dIn ? `${(dOut / dIn).toFixed(2)}:1` : '1.0:1';

    const volEl = document.getElementById('insp-spatial-vol');
    if (volEl) {
        const vol = (node.width || 1) * (node.height || 1) * (node.depth || 1);
        volEl.innerText = `${vol.toFixed(2)} u³`;
    }

    const headsEl = document.getElementById('insp-heads-info');
    if (headsEl) {
        if (meta.rank) headsEl.innerText = `Rank r=${meta.rank}`;
        else if (meta.heads) headsEl.innerText = `${meta.heads} Heads`;
        else headsEl.innerText = `Dim d=${dIn}`;
    }

    // Comparative Card
    const compBaselineEl = document.getElementById('insp-comp-baseline');
    const compCompressedEl = document.getElementById('insp-comp-compressed');
    const savingsBadgeEl = document.getElementById('insp-savings-badge');

    if (compBaselineEl) {
        compBaselineEl.innerText = meta.dense_equivalent_params_formatted || formatParamCount(dOut * dIn);
    }
    if (compCompressedEl) {
        compCompressedEl.innerText = meta.params || formatParamCount(meta.params_raw);
    }
    if (savingsBadgeEl) {
        const ratio = meta.compression_ratio || 1.0;
        const pct = Math.round((1 - 1 / Math.max(0.001, ratio)) * 100);
        savingsBadgeEl.innerText = ratio > 1.0 ? `-${pct}%` : '0%';
    }

    // Explanations
    const { descHtml, noteHtml } = getEducationalExplanation(node, isLeft);
    const descEl = document.getElementById('insp-desc');
    const noteEl = document.getElementById('insp-edu-note');
    if (descEl) descEl.innerHTML = descHtml;
    if (noteEl) noteEl.innerHTML = noteHtml;

    // Synchronize Floating Detail Card
    const cardTitle = document.getElementById('detail-title');
    const cardKicker = document.getElementById('detail-kicker');
    const cardDesc = document.getElementById('detail-desc');
    const factShape = document.getElementById('fact-shape');
    const factParams = document.getElementById('fact-params');
    const factSubspace = document.getElementById('fact-subspace');
    const factDelta = document.getElementById('fact-delta');

    if (cardTitle) cardTitle.innerText = node.label || node.id || 'Component';
    if (cardKicker) cardKicker.innerText = `${isLeft ? 'BASELINE' : 'COMPRESSED'} · ${(node.type || 'LAYER').toUpperCase()}`;
    if (cardDesc) cardDesc.innerHTML = descHtml;
    if (factShape) factShape.innerText = `[${dOut} × ${dIn}]`;
    if (factParams) factParams.innerText = meta.params || (meta.params_raw ? formatParamCount(meta.params_raw) : formatParamCount(dOut * dIn));
    if (factSubspace) factSubspace.innerText = meta.rank ? `Rank r=${meta.rank}` : (meta.heads ? `${meta.heads} Heads` : `Dim d=${dIn}`);
    if (factDelta) factDelta.innerText = meta.storage || (meta.compression || (isLeft ? '1.0x Base' : 'Compressed'));

    // Sub-renderers
    if (activeModule) {
        renderStructuralTransformation(activeModule, currentModel);
        renderTokenLifecycle(activeModule, currentModel);
        renderComponentPipeline(activeModule, currentModel);
        renderLatexFormula(activeModule);
    }
}
