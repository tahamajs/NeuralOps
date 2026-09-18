/**
 * educational/transformation.js
 * ==============================
 * Before-vs-After Structural Transformation Explainer.
 * Renders the structural delta, step-by-step tensor conversions, and comparison tables
 * for SVD, MoE, MLA, and SliceGPT paradigms.
 */

import { MODEL_SPECS } from '../constants.js';
import { formatParamCount, formatByteCount } from '../utils/format.js';

/**
 * Renders the structural transformation card in the deep inspector.
 * @param {object} moduleMeta - Active research module
 * @param {string} modelKey - Selected model ('distilbert' | 'roberta' | 'gpt2' | 'qwen')
 */
export function renderStructuralTransformation(moduleMeta, modelKey) {
    const cardEl = document.getElementById('transformation-card');
    if (!cardEl) return;

    const spec = MODEL_SPECS[modelKey] || MODEL_SPECS.distilbert;
    const proj = (moduleMeta.model_projections && moduleMeta.model_projections[modelKey]) || {};
    const paradigm = moduleMeta.paradigm || 'svd';

    const d = spec.hidden_size;
    const dFfn = spec.intermediate_size;
    const r = 64;

    const diagramEl = document.getElementById('tf-diagram-body');
    const tableEl = document.getElementById('tf-table-body');
    const explEl = document.getElementById('tf-explanation-text');
    const badgeEl = document.getElementById('tf-paradigm-badge');

    if (badgeEl) badgeEl.innerText = `${spec.name.toUpperCase()} · ${(moduleMeta.category || paradigm).toUpperCase()}`;

    const layerBaseP = proj.baseline_layer_params || spec.base_layer_params;
    const layerCompP = proj.compressed_layer_params || proj.compressed_layer_p || Math.round(layerBaseP / (proj.savings_ratio || 4.0));
    const ratio = proj.savings_ratio || (layerBaseP / Math.max(1, layerCompP));
    const savedBytes = proj.storage_saved_formatted || formatByteCount(proj.storage_saved_bytes || Math.max(0, (spec.base_total - (proj.total_compressed_params || spec.base_total / ratio)) * 2));

    if (paradigm === 'svd') {
        let denseShape = `[${d} × ${d}]`;
        let denseP = d * d;
        let vShape = `[${r} × ${d}]`;
        let vP = r * d;
        let uShape = `[${d} × ${r}]`;
        let uP = d * r;

        if (modelKey === 'gpt2') {
            denseShape = `[${d} × ${d * 3}] (Fused QKV)`;
            denseP = d * (d * 3);
            vShape = `[${r} × ${d * 3}]`;
            vP = r * (d * 3);
            uShape = `[${d} × ${r}]`;
            uP = d * r;
        } else if (modelKey === 'qwen') {
            denseShape = `[${d} × ${dFfn}] (SwiGLU)`;
            denseP = d * dFfn;
            vShape = `[${r} × ${dFfn}]`;
            vP = r * dFfn;
            uShape = `[${d} × ${r}]`;
            uP = d * r;
        }

        if (diagramEl) {
            diagramEl.innerHTML = `
                <div class="tf-step dense" data-step-key="dense" onclick="window.selectPartIn3D && window.selectPartIn3D('dense')" title="Click to illuminate Dense Layer in 3D">
                    <span class="tf-step-name">Dense Layer</span>
                    <span class="tf-step-shape">${denseShape}</span>
                    <span class="tf-step-param">${formatParamCount(denseP)}</span>
                </div>
                <span class="tf-arrow">➔ SVD ➔</span>
                <div class="tf-step compressed" data-step-key="v_factor" onclick="window.selectPartIn3D && window.selectPartIn3D('v_factor')" title="Click to illuminate Factor V in 3D">
                    <span class="tf-step-name">Factor V (Proj)</span>
                    <span class="tf-step-shape">${vShape}</span>
                    <span class="tf-step-param">${formatParamCount(vP)}</span>
                </div>
                <span class="tf-arrow">➔ Squeeze ➔</span>
                <div class="tf-step compressed" data-step-key="u_factor" onclick="window.selectPartIn3D && window.selectPartIn3D('u_factor')" title="Click to illuminate Factor U in 3D">
                    <span class="tf-step-name">Factor U (Exp)</span>
                    <span class="tf-step-shape">${uShape}</span>
                    <span class="tf-step-param">${formatParamCount(uP)}</span>
                </div>
            `;
        }
        if (tableEl) {
            tableEl.innerHTML = `
                <tr>
                    <td>Architecture</td>
                    <td>Dense Monolithic Layer</td>
                    <td>Low-Rank Bottleneck (V + U)</td>
                    <td class="delta-positive">${Number(ratio).toFixed(1)}x Parameter Compression</td>
                </tr>
                <tr>
                    <td>Layer Parameters</td>
                    <td>${formatParamCount(layerBaseP)}</td>
                    <td>${formatParamCount(layerCompP)}</td>
                    <td class="delta-positive">-${Math.round((1 - 1/ratio)*100)}% Memory Saved</td>
                </tr>
                <tr>
                    <td>Weight Storage</td>
                    <td>${formatByteCount(layerBaseP * 4)}</td>
                    <td>${formatByteCount(layerCompP * 4)}</td>
                    <td class="delta-positive">${savedBytes} Total Savings</td>
                </tr>
            `;
        }
        if (explEl) {
            explEl.innerHTML = `
                💡 <strong>Mathematical Mechanics:</strong> Decomposes dense weight matrix $W \\in \\mathbb{R}^{d_{\\text{out}} \\times d_{\\text{in}}}$ into two low-rank matrices $V \\in \\mathbb{R}^{r \\times d_{\\text{in}}}$ and $U \\in \\mathbb{R}^{d_{\\text{out}} \\times r}$ ($r=${r}$). Token features compress through rank bottleneck $r$ before re-expanding to model dimension.
            `;
        }
    } else {
        // Generic fallback for MoE, MLA, SliceGPT, Quant
        if (tableEl) {
            tableEl.innerHTML = `
                <tr>
                    <td>Architecture</td>
                    <td>Dense Baseline</td>
                    <td>${moduleMeta.name}</td>
                    <td class="delta-positive">${Number(ratio).toFixed(1)}x Compression</td>
                </tr>
                <tr>
                    <td>Layer Parameters</td>
                    <td>${formatParamCount(layerBaseP)}</td>
                    <td>${formatParamCount(layerCompP)}</td>
                    <td class="delta-positive">-${Math.round((1 - 1/ratio)*100)}% Memory Saved</td>
                </tr>
            `;
        }
        if (explEl) {
            explEl.innerHTML = `💡 <strong>Mathematical Mechanics:</strong> ${moduleMeta.description || 'Optimized architectural transformation.'}`;
        }
    }
}
