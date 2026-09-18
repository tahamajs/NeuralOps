/**
 * educational/token_lifecycle.js
 * ===============================
 * 7-Stage Token Lifecycle & Step-by-Step State Evolution Tracker.
 * Tracks what enters, what gets transformed, what gets added (+), and what exits
 * at each transformer stage.
 */

import { MODEL_SPECS } from '../constants.js';

/**
 * Renders the 7-stage lifecycle progress list in the inspector drawer.
 * @param {object} moduleMeta - Active research module
 * @param {string} modelKey - Selected model
 */
export function renderTokenLifecycle(moduleMeta, modelKey = 'distilbert') {
    const listEl = document.getElementById('lifecycle-steps-list');
    if (!listEl) return;

    const spec = MODEL_SPECS[modelKey] || MODEL_SPECS.distilbert;
    const d = spec.hidden_size;
    const dFfn = spec.intermediate_size;
    const paradigm = moduleMeta ? (moduleMeta.paradigm || 'svd') : 'svd';

    const steps = [
        {
            num: '01',
            title: 'Input Token Ingestion',
            sub: `Discrete token IDs $[t_1, \\dots, t_S] \\to$ continuous manifold $X^{(0)} \\in \\mathbb{R}^{S \\times ${d}}$ via embedding lookup table.`,
            nodeTarget: 'input_embeddings'
        },
        {
            num: '02',
            title: 'Pre-Attention Normalization',
            sub: `RMSNorm / LayerNorm stabilizes token distribution: $\\bar{X} = \\text{Norm}(X^{(0)})$. Zero-mean unit-variance.`,
            nodeTarget: 'pre_attn_norm'
        },
        {
            num: '03',
            title: 'Self-Attention / Subspace Projection',
            sub: paradigm === 'svd'
                ? `Tokens project into low-rank eigenspace $r=64$ via Factor $V$, then expand via Factor $U$: $\\Delta_{\\text{attn}} = (\\bar{X} V^\\top) U^\\top$.`
                : (modelKey === 'qwen'
                    ? `Grouped Query Attention: 14 Query heads query 2 shared Key/Value head pairs, reducing KV cache by 7x.`
                    : `Multi-head contextual mixing: $A = \\text{Softmax}(Q K^\\top / \\sqrt{d}) V$.`),
            nodeTarget: 'attention_core'
        },
        {
            num: '04',
            title: 'Highway Residual Addition #1 (+)',
            sub: `Original token identity $X_{\\text{in}}$ carries forward via skip conduit: $X_{\\text{mid}} = X_{\\text{in}} + \\Delta_{\\text{attn}}$.`,
            nodeTarget: 'attn_add_node'
        },
        {
            num: '05',
            title: 'Post-Attention Normalization',
            sub: `Second normalization pass: $\\bar{X}_{\\text{mid}} = \\text{Norm}(X_{\\text{mid}})$. Prepares features for non-linear memory retrieval.`,
            nodeTarget: 'post_attn_norm'
        },
        {
            num: '06',
            title: 'Feed-Forward / Factual Memory Retrieval',
            sub: paradigm === 'moe'
                ? `Router dispatches tokens to specialized Expert subnetworks: $\\Delta_{\\text{ffn}} = \\sum_i G(x)_i E_i(x)$.`
                : (modelKey === 'qwen'
                    ? `SwiGLU Non-linear activation: $\\text{SwiGLU}(x) = (x W_{\\text{gate}}) \\cdot \\text{SiLU}(x W_{\\text{up}}) W_{\\text{down}}$.`
                    : `MLP 4x expansion into intermediate dimension $d_{\\text{ffn}} = ${dFfn}$, followed by down-projection.`),
            nodeTarget: 'ffn_core'
        },
        {
            num: '07',
            title: 'Highway Residual Addition #2 (+)',
            sub: `Layer output accumulation: $X_{\\text{out}} = X_{\\text{mid}} + \\Delta_{\\text{ffn}}$. Token exits layer and advances to next block.`,
            nodeTarget: 'ffn_add_node'
        }
    ];

    listEl.innerHTML = steps.map(s => `
        <div class="lifecycle-step" onclick="window.selectPartIn3D && window.selectPartIn3D('${s.nodeTarget}')" title="Click to illuminate stage in 3D">
            <span class="step-num">${s.num}</span>
            <div class="step-content">
                <strong>${s.title}</strong>
                <p class="lifecycle-step-sub">${s.sub}</p>
            </div>
        </div>
    `).join('');

    if (window.katex) {
        listEl.querySelectorAll('.lifecycle-step-sub').forEach(el => {
            const html = el.innerHTML;
            const replaced = html.replace(/\$([^\$]+)\$/g, (match, formula) => {
                try {
                    return window.katex.renderToString(formula, { throwOnError: false, displayMode: false });
                } catch (e) {
                    return match;
                }
            });
            el.innerHTML = replaced;
        });
    }
}
