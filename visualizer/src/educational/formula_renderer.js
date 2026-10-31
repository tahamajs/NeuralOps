/**
 * educational/formula_renderer.js
 * ================================
 * High-definition LaTeX formula rendering using KaTeX with multi-tier mathematical formulations:
 * 1. Operational Node Formulation: Forward pass equation for the active selected 3D component.
 * 2. Architectural Paradigm Theorem: Global low-rank, sparse, MoE, or attention formulation.
 * 3. Theoretical Scaling & Complexity: Asymptotic FLOPs and parameter bounds.
 * 4. Interactive LaTeX clipboard copy with status feedback.
 */

import { escapeHtml } from '../utils/dom.js';

/**
 * Returns the exact operational mathematical formulation for an inspected 3D node.
 * @param {object} node - Inspected 3D mesh node
 * @param {boolean} isLeft - True if dense baseline, False if compressed
 * @param {string} currentModel - Active model key
 * @returns {object|null} { title, formula, desc, complexity }
 */
export function getNodeMathematicalFormulation(node, isLeft = false, currentModel = 'distilbert') {
    if (!node) return null;
    const id = (node.id || '').toLowerCase();
    const label = (node.label || '').toLowerCase();
    const subType = (node.sub_type || '').toLowerCase();
    const type = (node.type || '').toLowerCase();

    // Query Linear / Projections
    if (id.includes('q_lin') || id.includes('q_proj') || label.includes('q_lin') || label.includes('query')) {
        if (subType === 'v_factor' || label.includes('v_factor')) {
            return {
                title: 'Query Low-Rank Factor V Projection',
                formula: 'z_Q = X V_{Q, r}^\\top, \\quad V_{Q, r} \\in \\mathbb{R}^{r \\times d_{\\text{in}}}, \\; r=64',
                desc: 'Compresses input representation from hidden dimension into rank-64 latent subspace.',
                complexity: '\\mathcal{O}(B \\times S \\times d_{\\text{in}} \\times r)'
            };
        }
        if (subType === 'u_factor' || label.includes('u_factor')) {
            return {
                title: 'Query Low-Rank Factor U Expansion',
                formula: 'Q = z_Q (\\Sigma_r U_{Q, r}^\\top) + b_Q, \\quad U_{Q, r} \\in \\mathbb{R}^{d_{\\text{out}} \\times r}',
                desc: 'Expands rank-64 latent subspace back to full query tensor dimension.',
                complexity: '\\mathcal{O}(B \\times S \\times r \\times d_{\\text{out}})'
            };
        }
        return {
            title: 'Dense Query Linear Projection',
            formula: 'Q = X W_Q + b_Q, \\quad W_Q \\in \\mathbb{R}^{d_{\\text{model}} \\times d_k}',
            desc: 'Linear transformation generating Query representations for scaled dot-product attention.',
            complexity: '\\mathcal{O}(B \\times S \\times d_{\\text{model}}^2)'
        };
    }

    // Key Linear / Projections
    if (id.includes('k_lin') || id.includes('k_proj') || label.includes('k_lin') || label.includes('key')) {
        if (subType === 'v_factor' || label.includes('v_factor')) {
            return {
                title: 'Key Low-Rank Factor V Projection',
                formula: 'z_K = X V_{K, r}^\\top, \\quad V_{K, r} \\in \\mathbb{R}^{r \\times d_{\\text{in}}}, \\; r=64',
                desc: 'Projects input into low-rank Key subspace before KV caching.',
                complexity: '\\mathcal{O}(B \\times S \\times d_{\\text{in}} \\times r)'
            };
        }
        if (subType === 'u_factor' || label.includes('u_factor')) {
            return {
                title: 'Key Low-Rank Factor U Expansion',
                formula: 'K = z_K (\\Sigma_r U_{K, r}^\\top) + b_K, \\quad U_{K, r} \\in \\mathbb{R}^{d_{\\text{out}} \\times r}',
                desc: 'Expands Key vector from bottleneck rank to full head projection space.',
                complexity: '\\mathcal{O}(B \\times S \\times r \\times d_{\\text{out}})'
            };
        }
        return {
            title: 'Dense Key Linear Projection',
            formula: 'K = X W_K + b_K, \\quad W_K \\in \\mathbb{R}^{d_{\\text{model}} \\times d_k}',
            desc: 'Projects sequence tokens into addressable Key representations.',
            complexity: '\\mathcal{O}(B \\times S \\times d_{\\text{model}}^2)'
        };
    }

    // Value Linear / Projections
    if (id.includes('v_lin') || id.includes('v_proj') || label.includes('v_lin') || label.includes('value')) {
        if (subType === 'v_factor' || label.includes('v_factor')) {
            return {
                title: 'Value Low-Rank Factor V Projection',
                formula: 'z_V = X V_{V, r}^\\top, \\quad V_{V, r} \\in \\mathbb{R}^{r \\times d_{\\text{in}}}, \\; r=64',
                desc: 'Projects input into compact rank-64 Value representation.',
                complexity: '\\mathcal{O}(B \\times S \\times d_{\\text{in}} \\times r)'
            };
        }
        if (subType === 'u_factor' || label.includes('u_factor')) {
            return {
                title: 'Value Low-Rank Factor U Expansion',
                formula: 'V = z_V (\\Sigma_r U_{V, r}^\\top) + b_V, \\quad U_{V, r} \\in \\mathbb{R}^{d_{\\text{out}} \\times r}',
                desc: 'Recovers contextual content vectors for attention aggregation.',
                complexity: '\\mathcal{O}(B \\times S \\times r \\times d_{\\text{out}})'
            };
        }
        return {
            title: 'Dense Value Linear Projection',
            formula: 'V = X W_V + b_V, \\quad W_V \\in \\mathbb{R}^{d_{\\text{model}} \\times d_v}',
            desc: 'Linear transformation generating content Value vectors.',
            complexity: '\\mathcal{O}(B \\times S \\times d_{\\text{model}}^2)'
        };
    }

    // Output Projection
    if (id.includes('out_lin') || id.includes('out_proj') || label.includes('out_lin') || label.includes('output projection')) {
        if (subType === 'v_factor') {
            return {
                title: 'Output Low-Rank Factor V Projection',
                formula: 'z_O = \\text{Concat}(h_1, \\dots, h_H) V_{O, r}^\\top, \\quad V_{O, r} \\in \\mathbb{R}^{r \\times d_{\\text{in}}}',
                desc: 'Projects concatenated multi-head context into rank-64 output bottleneck.',
                complexity: '\\mathcal{O}(B \\times S \\times d_{\\text{in}} \\times r)'
            };
        }
        if (subType === 'u_factor') {
            return {
                title: 'Output Low-Rank Factor U Expansion',
                formula: '\\text{Out} = z_O (\\Sigma_r U_{O, r}^\\top) + b_O, \\quad U_{O, r} \\in \\mathbb{R}^{d_{\\text{out}} \\times r}',
                desc: 'Expands output context into final hidden representation.',
                complexity: '\\mathcal{O}(B \\times S \\times r \\times d_{\\text{out}})'
            };
        }
        return {
            title: 'Multi-Head Attention Output Projection',
            formula: '\\text{AttnOut} = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_H) W_O + b_O',
            desc: 'Blends and projects concatenated attention head outputs back into the residual stream.',
            complexity: '\\mathcal{O}(B \\times S \\times d_{\\text{model}}^2)'
        };
    }

    // FFN Intermediate
    if (id.includes('intermediate') || id.includes('ffn.dense') || id.includes('mlp.fc1') || id.includes('gate_proj')) {
        return {
            title: 'Feed-Forward Network Expansion (FFN-1)',
            formula: 'h = \\text{GELU}(X W_1 + b_1), \\quad W_1 \\in \\mathbb{R}^{d_{\\text{model}} \\times d_{\\text{ffn}}}',
            desc: 'High-dimensional non-linear feature expansion (4x hidden dimension).',
            complexity: '\\mathcal{O}(4 \\cdot B \\times S \\times d_{\\text{model}}^2)'
        };
    }

    // FFN Output
    if (id.includes('ffn.output') || id.includes('mlp.fc2') || id.includes('down_proj') || (id.includes('output') && !id.includes('attention'))) {
        return {
            title: 'Feed-Forward Network Contraction (FFN-2)',
            formula: 'Y = h W_2 + b_2, \\quad W_2 \\in \\mathbb{R}^{d_{\\text{ffn}} \\times d_{\\text{model}}}',
            desc: 'Projects expanded non-linear representation back to residual highway dimension.',
            complexity: '\\mathcal{O}(4 \\cdot B \\times S \\times d_{\\text{model}}^2)'
        };
    }

    // Normalization
    if (id.includes('norm') || label.includes('norm') || type === 'norm') {
        return {
            title: 'Normalization Layer (LayerNorm / RMSNorm)',
            formula: '\\text{LN}(x) = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\odot \\gamma + \\beta, \\quad \\mu = \\frac{1}{d} \\sum_{i=1}^d x_i',
            desc: 'Stabilizes activation variance across hidden dimension coordinates.',
            complexity: '\\mathcal{O}(B \\times S \\times d)'
        };
    }

    // Embeddings
    if (id.includes('embed') || label.includes('embed')) {
        return {
            title: 'Token & Position Embedding Lookup',
            formula: 'X_0 = \\text{Embedding}(T) + E_{\\text{pos}}, \\quad X_0 \\in \\mathbb{R}^{S \\times d_{\\text{model}}}',
            desc: 'Discrete token index projection into continuous geometric hyperspace.',
            complexity: '\\mathcal{O}(B \\times S \\times d_{\\text{model}})'
        };
    }

    // MoE Router & Experts
    if (id.includes('router') || subType === 'router') {
        return {
            title: 'Top-K MoE Gating Router',
            formula: 'G(x) = \\text{Softmax}(\\text{TopK}(x W_g, K=2)), \\quad W_g \\in \\mathbb{R}^{d \\times E}',
            desc: 'Dynamic routing dispatching tokens to top-2 specialized neural sub-networks.',
            complexity: '\\mathcal{O}(B \\times S \\times d \\times E)'
        };
    }
    if (id.includes('expert') || subType.startsWith('expert')) {
        return {
            title: 'Specialized Feed-Forward Expert',
            formula: 'E_i(x) = \\text{GELU}(x W_{i, 1} + b_{i, 1}) W_{i, 2} + b_{i, 2}',
            desc: 'Domain-specialized capacity layer activated only for matching token representations.',
            complexity: '\\mathcal{O}(2 \\cdot B \\times S \\times d \\times d_{\\text{expert}})'
        };
    }

    // Residual Highway Skip
    if (id.includes('residual') || label.includes('residual')) {
        return {
            title: 'Residual Add Highway (+)',
            formula: 'X^{(l+1)} = X^{(l)} + \\mathcal{F}(X^{(l)}), \\quad \\nabla_X \\mathcal{L} = \\nabla_Y \\mathcal{L} \\cdot (I + \\nabla \\mathcal{F})',
            desc: 'Provides an uninterrupted gradient highway preventing vanishing gradients during backpropagation.',
            complexity: '\\mathcal{O}(B \\times S \\times d)'
        };
    }

    return null;
}

/**
 * Renders high-definition mathematical formulation into the Inspector Drawer.
 * @param {object|null} moduleMeta - Module metadata containing math or math_latex
 * @param {string|null} formulaOverride - Optional explicit formula string
 * @param {object|null} node - Active selected 3D node
 * @param {boolean} isLeft - True if baseline, False if compressed
 * @param {string} currentModel - Active model key
 */
export function renderLatexFormula(moduleMeta, formulaOverride = null, node = null, isLeft = false, currentModel = 'distilbert') {
    const bodyEl = document.getElementById('formula-latex-body');
    if (!bodyEl) return;
    bodyEl.innerHTML = '';

    const nodeMath = getNodeMathematicalFormulation(node, isLeft, currentModel);
    const moduleMath = formulaOverride || (moduleMeta && (moduleMeta.math_latex || moduleMeta.math)) || 'W \\approx U_r \\Sigma_r V_r^\\top = A B^\\top, \\quad r \\ll \\min(d_{\\text{in}}, d_{\\text{out}})';

    // Container for multiple mathematical aspects
    const container = document.createElement('div');
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.gap = '10px';

    // 1. Node-Specific Operational Formulation (if node is selected)
    if (nodeMath) {
        const nodeCard = document.createElement('div');
        nodeCard.className = 'latex-formula-subcard';
        nodeCard.style.background = 'rgba(6, 13, 24, 0.75)';
        nodeCard.style.border = '1px solid rgba(56, 189, 248, 0.3)';
        nodeCard.style.borderRadius = '6px';
        nodeCard.style.padding = '8px 10px';

        nodeCard.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                <span style="font-size:9px; font-weight:700; color:#38bdf8; text-transform:uppercase; letter-spacing:0.5px;">⚡ ${escapeHtml(nodeMath.title)}</span>
                <button class="formula-copy-btn" data-formula="${escapeHtml(nodeMath.formula)}" style="padding:1px 6px; font-size:9px;">⎘ Copy</button>
            </div>
            <div class="katex-math-render" style="text-align:center; padding:4px 0; color:#e0f2fe; font-size:12px;">$$ ${nodeMath.formula} $$</div>
            <div style="font-size:10px; color:var(--text-faint); margin-top:4px; line-height:1.4;">${escapeHtml(nodeMath.desc)}</div>
        `;
        container.appendChild(nodeCard);
    }

    // 2. Module Architectural Paradigm Formulation
    const modCard = document.createElement('div');
    modCard.className = 'latex-formula-subcard';
    modCard.style.background = 'rgba(6, 13, 24, 0.75)';
    modCard.style.border = '1px solid rgba(45, 212, 191, 0.3)';
    modCard.style.borderRadius = '6px';
    modCard.style.padding = '8px 10px';

    const modTitle = (moduleMeta && (moduleMeta.name || moduleMeta.id)) ? `📐 Paradigm Invariant: ${moduleMeta.name || moduleMeta.id}` : '📐 Architectural Invariant';

    modCard.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <span style="font-size:9px; font-weight:700; color:#2dd4bf; text-transform:uppercase; letter-spacing:0.5px;">${escapeHtml(modTitle)}</span>
            <button class="formula-copy-btn" data-formula="${escapeHtml(moduleMath)}" style="padding:1px 6px; font-size:9px;">⎘ Copy</button>
        </div>
        <div class="katex-math-render" style="text-align:center; padding:4px 0; color:#ccfbf1; font-size:12px;">$$ ${moduleMath} $$</div>
    `;
    container.appendChild(modCard);

    // 3. Complexity & Parameter Savings Invariant
    const paradigm = (moduleMeta && moduleMeta.paradigm) || 'svd';
    let scalingFormula = '\\Delta \\text{Params} = d_{\\text{in}} d_{\\text{out}} - (d_{\\text{in}} r + r d_{\\text{out}}), \\quad r=64';
    let scalingDesc = 'Linear parameter reduction: 83.3% parameter savings across projection matrices.';

    if (paradigm === 'moe') {
        scalingFormula = '\\text{Active FLOPs} = 2 \\times \\text{FLOPs}_{E} \\ll 8 \\times \\text{FLOPs}_{E} \\quad (\\text{Top-2 Routing})';
        scalingDesc = 'Executes only 2 active experts per token out of full expert capacity.';
    } else if (paradigm === 'mla') {
        scalingFormula = '\\text{KV Cache} = 2 \\times L \\times d_c \\times S \\times B \\times 2 \\text{ bytes} \\quad (d_c = 512 \\ll n_h d_h)';
        scalingDesc = 'DeepSeek multi-head latent compression saves 93.3% KV cache bandwidth.';
    } else if (paradigm === 'pruning') {
        scalingFormula = '\\text{Sparsity} = 1 - \\frac{\\text{nnz}(W)}{d_{\\text{in}} d_{\\text{out}}} = 45\\%, \\quad W_{\\text{pruned}} = (Q^\\top W Q)_{[:, \\mathcal{I}]}';
        scalingDesc = 'Direct channel slicing removes computational rows and columns with minimal perplexity shift.';
    }

    const scalingCard = document.createElement('div');
    scalingCard.className = 'latex-formula-subcard';
    scalingCard.style.background = 'rgba(6, 13, 24, 0.75)';
    scalingCard.style.border = '1px solid rgba(168, 85, 247, 0.25)';
    scalingCard.style.borderRadius = '6px';
    scalingCard.style.padding = '8px 10px';

    scalingCard.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
            <span style="font-size:9px; font-weight:700; color:#c084fc; text-transform:uppercase; letter-spacing:0.5px;">📊 Scaling &amp; Complexity Invariant</span>
            <button class="formula-copy-btn" data-formula="${escapeHtml(scalingFormula)}" style="padding:1px 6px; font-size:9px;">⎘ Copy</button>
        </div>
        <div class="katex-math-render" style="text-align:center; padding:4px 0; color:#f3e8ff; font-size:11px;">$$ ${scalingFormula} $$</div>
        <div style="font-size:10px; color:var(--text-faint); margin-top:4px; line-height:1.4;">${escapeHtml(scalingDesc)}</div>
    `;
    container.appendChild(scalingCard);

    bodyEl.appendChild(container);

    // Bind all copy buttons
    bodyEl.querySelectorAll('.formula-copy-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const textToCopy = btn.getAttribute('data-formula') || moduleMath;
            if (navigator.clipboard) {
                navigator.clipboard.writeText(textToCopy).then(() => {
                    const orig = btn.innerHTML;
                    btn.innerHTML = '✓ Copied!';
                    btn.style.color = '#34d399';
                    setTimeout(() => {
                        btn.innerHTML = orig;
                        btn.style.color = '';
                    }, 1500);
                }).catch(() => {});
            }
        });
    });

    // Render KaTeX for all math cards
    if (typeof window !== 'undefined' && window.katex) {
        bodyEl.querySelectorAll('.katex-math-render').forEach(el => {
            const raw = el.textContent.replace(/\$\$/g, '').trim();
            if (raw) {
                try {
                    window.katex.render(raw, el, { displayMode: true, throwOnError: false });
                } catch (err) {
                    console.warn('[formula_renderer] KaTeX error:', err);
                }
            }
        });
    }
}
