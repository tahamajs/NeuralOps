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
            let modelSpecificNote = '';
            if (modelKey === 'gpt2') {
                modelSpecificNote = ` In <strong>GPT-2</strong>, Q, K, and V projections are unified into a single fused Conv1D tensor $W_{\\text{c\\_attn}} \\in \\mathbb{R}^{${d} \\times ${d * 3}}$, factorized directly into rank $r=${r}$.`;
            } else if (modelKey === 'qwen') {
                modelSpecificNote = ` In <strong>Qwen2.5</strong>, Grouped Query Attention splits Query into 14 heads while Key/Value share only 2 heads ($d_{\\text{kv}}=128$), and the FFN utilizes SwiGLU 3-factor gating ($W_{\\text{gate}}, W_{\\text{up}}, W_{\\text{down}}$).`;
            } else if (modelKey === 'roberta') {
                modelSpecificNote = ` In <strong>RoBERTa-Base</strong> (12 layers), token mixing is full bidirectional self-attention with independent $[${d} \\times ${d}]$ matrices per projection operator.`;
            } else {
                modelSpecificNote = ` In <strong>DistilBERT</strong> (6 layers), half the transformer layers of BERT-base are retained while maintaining $>97\\%$ of linguistic capacity via knowledge distillation.`;
            }

            explEl.innerHTML = `
                💡 <strong>Mathematical Mechanics:</strong> Decomposes dense weight matrix $W \\in \\mathbb{R}^{d_{\\text{out}} \\times d_{\\text{in}}}$ into two low-rank matrices $V \\in \\mathbb{R}^{r \\times d_{\\text{in}}}$ and $U \\in \\mathbb{R}^{d_{\\text{out}} \\times r}$ ($r=${r}$). Token features compress through rank bottleneck $r$ before re-expanding to model dimension.${modelSpecificNote}
            `;
        }
    } else if (paradigm === 'moe') {
        if (diagramEl) {
            diagramEl.innerHTML = `
                <div class="tf-step dense" onclick="window.selectPartIn3D && window.selectPartIn3D('dense')" title="Dense FFN">
                    <span class="tf-step-name">Dense FFN (4×d)</span>
                    <span class="tf-step-shape">[${d} × ${dFfn}]</span>
                    <span class="tf-step-param">${formatParamCount(d * dFfn)}</span>
                </div>
                <span class="tf-arrow">➔ Gate ➔</span>
                <div class="tf-step compressed" onclick="window.selectPartIn3D && window.selectPartIn3D('router')" title="Router Gate">
                    <span class="tf-step-name">Router (Top-1)</span>
                    <span class="tf-step-shape">[${d} × 4]</span>
                    <span class="tf-step-param">${formatParamCount(d * 4)}</span>
                </div>
                <span class="tf-arrow">➔ Dispatch ➔</span>
                <div class="tf-step compressed" onclick="window.selectPartIn3D && window.selectPartIn3D('expert')" title="Expert Subnetwork">
                    <span class="tf-step-name">4x Sparse Experts</span>
                    <span class="tf-step-shape">4 × [${d} × ${Math.round(dFfn/2)}]</span>
                    <span class="tf-step-param">Active: 1/4 FLOPs</span>
                </div>
            `;
        }
        if (tableEl) {
            tableEl.innerHTML = `
                <tr><td>Mechanism</td><td>Dense Standard FFN</td><td>Top-1 Mixture-of-Experts</td><td class="delta-positive">Sparse Conditional Computation</td></tr>
                <tr><td>Active FLOPs</td><td>100% Quadratic</td><td>~28% per Token</td><td class="delta-positive">3.5x Throughput Acceleration</td></tr>
                <tr><td>Param Capacity</td><td>${formatParamCount(layerBaseP)}</td><td>${formatParamCount(layerCompP)}</td><td class="delta-positive">Multiplied Expert Knowledge</td></tr>
            `;
        }
        if (explEl) {
            explEl.innerHTML = `💡 <strong>Paper Mechanism (Shazeer et al. / ToMoE):</strong> Tokens are dynamically routed via gating network $G(x) = \\text{Softmax}(\\text{TopK}(x \\cdot W_g))$. Only the most specialized expert is evaluated per token, breaking the trade-off between parameter size and computational cost.`;
        }
    } else if (paradigm === 'mla') {
        if (diagramEl) {
            diagramEl.innerHTML = `
                <div class="tf-step dense" onclick="window.selectPartIn3D && window.selectPartIn3D('dense')">
                    <span class="tf-step-name">Standard MHA KV</span>
                    <span class="tf-step-shape">[2 × ${d} × ${d}]</span>
                    <span class="tf-step-param">${formatParamCount(2 * d * d)}</span>
                </div>
                <span class="tf-arrow">➔ Compress ➔</span>
                <div class="tf-step compressed" onclick="window.selectPartIn3D && window.selectPartIn3D('mla')">
                    <span class="tf-step-name">Latent KV Vector</span>
                    <span class="tf-step-shape">[${d} × 512]</span>
                    <span class="tf-step-param">512d Latent Compression</span>
                </div>
            `;
        }
        if (tableEl) {
            tableEl.innerHTML = `
                <tr><td>Mechanism</td><td>Full KV Cache Multi-Head</td><td>Multi-Head Latent Attention (MLA)</td><td class="delta-positive">93% KV Cache Reduction</td></tr>
                <tr><td>KV Footprint</td><td>${formatByteCount(d * 128 * 2)} per token</td><td>${formatByteCount(512 * 2)} per token</td><td class="delta-positive">Large Batch Context Serving</td></tr>
            `;
        }
        if (explEl) {
            explEl.innerHTML = `💡 <strong>Paper Mechanism (DeepSeek MLA):</strong> Compresses Key and Value projection matrices into a low-rank latent vector $c^{KV} \\in \\mathbb{R}^{d_c}$. Only the latent vector is cached in GPU VRAM during autoregressive generation!`;
        }
    } else {
        if (diagramEl) {
            diagramEl.innerHTML = `
                <div class="tf-step dense" onclick="window.selectPartIn3D && window.selectPartIn3D('dense')">
                    <span class="tf-step-name">Dense Layer</span>
                    <span class="tf-step-shape">[${d} × ${d}]</span>
                    <span class="tf-step-param">${formatParamCount(layerBaseP)}</span>
                </div>
                <span class="tf-arrow">➔ Compress ➔</span>
                <div class="tf-step compressed" onclick="window.selectPartIn3D && window.selectPartIn3D('compressed')">
                    <span class="tf-step-name">${moduleMeta.name || 'Compressed'}</span>
                    <span class="tf-step-shape">Optimized</span>
                    <span class="tf-step-param">${formatParamCount(layerCompP)}</span>
                </div>
            `;
        }
        if (tableEl) {
            tableEl.innerHTML = `
                <tr><td>Architecture</td><td>Dense Baseline</td><td>${moduleMeta.name}</td><td class="delta-positive">${Number(ratio).toFixed(1)}x Compression</td></tr>
                <tr><td>Layer Parameters</td><td>${formatParamCount(layerBaseP)}</td><td>${formatParamCount(layerCompP)}</td><td class="delta-positive">-${Math.round((1 - 1/ratio)*100)}% Memory Saved</td></tr>
            `;
        }
        if (explEl) {
            explEl.innerHTML = `💡 <strong>Paper Mechanism & Math:</strong> ${moduleMeta.description || 'Optimized architectural transformation derived directly from research implementation.'}`;
        }
    }

    if (window.katex && explEl) {
        const raw = explEl.innerHTML;
        explEl.innerHTML = raw.replace(/\$([^\$]+)\$/g, (match, formula) => {
            try {
                return window.katex.renderToString(formula, { throwOnError: false, displayMode: false });
            } catch (e) {
                return match;
            }
        });
    }
}
