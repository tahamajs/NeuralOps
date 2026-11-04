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
import { renderAttentionInspector } from '../educational/attention_inspector.js';
import { renderLayerDepthProfile } from '../educational/layer_depth_profile.js';
import { formatParamCount } from '../utils/format.js';
import { store } from '../state/store.js';

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
        if (typeof document !== 'undefined' && document.body) {
            document.body.classList.add('inspector-open');
        }
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

    // Resolve active research module or synthesize default from node & model
    const curMod = activeModule || store.get('activeResearchModule') || {
        id: 'svd_linear',
        name: 'SVD Linear Factorization',
        file: 'Codes/framework/compression_engine/svd_linear.py',
        type: 'Compression Engine',
        category: 'Low-Rank SVD & Factorization',
        paradigm: 'svd',
        icon: '⚡',
        rank: (meta && meta.rank) || 64,
        math_latex: 'W \\approx U_r \\Sigma_r V_r^\\top = A B^\\top, \\quad r \\ll \\min(d_{\\text{in}}, d_{\\text{out}})',
        components: [
            { id: "v_factor", name: "Factor V (Low-Rank Subspace Projection)", role: "Projection to Rank r=64", shape_transform: "[64 × d_in]", color: "#3acbb8", math_latex: "z_t = x_t V_r^\\top, \\quad V_r \\in \\mathbb{R}^{64 \\times d_{\\text{in}}}" },
            { id: "u_factor", name: "Factor U (Full-Rank Expansion Projection)", role: "Expansion to Output Space", shape_transform: "[d_out × 64]", color: "#2dd4bf", math_latex: "y_t = z_t (\\Sigma_r U_r^\\top) + b, \\quad U_r \\in \\mathbb{R}^{d_{\\text{out}} \\times 64}" }
        ]
    };

    // Sub-renderers: Always executed with complete data
    renderAttentionInspector(curMod || node, currentModel);
    renderLayerDepthProfile(node, currentModel);
    renderStructuralTransformation(curMod, currentModel);
    renderTokenLifecycle(curMod, currentModel);
    renderComponentPipeline(curMod, currentModel);
    renderLatexFormula(curMod, null, node, isLeft, currentModel);

    // Render Python Implementation & Paper Citation Box
    const descSection = document.getElementById('insp-desc');
    if (descSection) {
        const fileRef = curMod.file ? `<code>${curMod.file}</code>` : '—';
        const classesRef = curMod.classes && curMod.classes.length ? curMod.classes.map(c => `<span class="badge" style="background:rgba(56,189,248,0.15);color:#7dd3fc;border:1px solid rgba(56,189,248,0.3);margin-right:4px;">${c}</span>`).join('') : '<span class="badge" style="background:rgba(56,189,248,0.15);color:#7dd3fc;border:1px solid rgba(56,189,248,0.3);">Linear / Attention Layer</span>';
        descSection.innerHTML = `
            <div style="margin-bottom:8px;font-size:11px;line-height:1.6;color:var(--text-main);">
                ${descHtml}
            </div>
            <div style="background:rgba(6,13,24,0.7);border:1px solid rgba(255,255,255,0.08);border-radius:6px;padding:8px 10px;margin-top:8px;font-size:11px;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <span style="color:var(--text-faint);font-size:10px;text-transform:uppercase;letter-spacing:0.5px;">📄 Code Implementation &amp; Paper Spec</span>
                    <button class="round-button btn-doc-read" onclick="window.openModuleDoc && window.openModuleDoc(window._activeResearchModule)" style="font-size:9px;padding:2px 8px;background:rgba(16,185,129,0.15);color:#34d399;border:1px solid rgba(16,185,129,0.4);cursor:pointer;border-radius:4px;">📖 Read Full Paper Doc</button>
                </div>
                <div style="margin-bottom:4px;word-break:break-all;"><strong style="color:var(--cyan);">Source:</strong> ${fileRef}</div>
                <div style="margin-top:4px;"><strong style="color:var(--cyan);">PyTorch Classes:</strong> ${classesRef}</div>
            </div>
        `;
        window._activeResearchModule = curMod;
    }
}
