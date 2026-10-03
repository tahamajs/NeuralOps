/**
 * educational/matrix_unfolding.js
 * ===============================
 * High-Definition Interactive Attention Matrix Unfolding & Step-by-Step Deconstruction Engine.
 * Visually demystifies how a 1D sequence of tokens expands into a 2D S × S interaction square
 * ("opening the square"), and how advanced compression architectures (SVD, GQA, MLA, Linear Attention)
 * compress, restructure, or completely bypass the quadratic bottleneck.
 */

import { escapeHtml } from '../utils/dom.js';

export const MATRIX_UNFOLDING_STEPS = [
    {
        step: 1,
        id: 'input_tokens',
        title: 'Step 1: Input Token Ingress (1D Sequence)',
        phase: 'REPRESENTATION',
        badge: '1D SEQUENCE · S × d',
        color: '#38bdf8',
        formula: 'X = \\begin{bmatrix} x_1^\\top \\\\ x_2^\\top \\\\ \\vdots \\\\ x_S^\\top \\end{bmatrix} \\in \\mathbb{R}^{S \\times d_{\\text{model}}}',
        tensor_dims: {
            input: 'Tokens: S=4',
            shape: '[4 × 768]',
            nature: '1D Sequence of Independent Token Vectors'
        },
        narrative: 'Tokens enter the transformer stack from embedding lookup tables. Each token is encoded as a 1D latent activation vector of dimension d_model = 768. At this stage, tokens exist independently along the sequence dimension with zero cross-token interaction or contextual blending.',
        visual_type: '1d_tokens',
        implication: 'Memory footprint is strictly linear O(S · d). If no cross-attention took place, sequence length could scale to billions of tokens without quadratic bottlenecks.'
    },
    {
        step: 2,
        id: 'qkv_projections',
        title: 'Step 2: Linear Subspace Projections (Q, K, V)',
        phase: 'PROJECTION',
        badge: 'PARALLEL SUBSPACES',
        color: '#818cf8',
        formula: 'Q = X W_Q + b_Q, \\quad K = X W_K + b_K, \\quad V = X W_V + b_V, \\quad W_\\ast \\in \\mathbb{R}^{d_{\\text{model}} \\times d_k}',
        tensor_dims: {
            input: 'X: [4 × 768]',
            weights: 'W_Q, W_K, W_V: [768 × 64] (per head)',
            output: 'Q, K, V: [4 × 64]'
        },
        narrative: 'Input vectors are multiplied by three learned projection matrices to produce Query (Q), Key (K), and Value (V) tensors. Query vectors define "what context this position seeks"; Key vectors define "what attributes this position broadcasts"; Value vectors carry the actual semantic information to be routed.',
        visual_type: 'qkv_strips',
        implication: 'In standard MHA, each head projects to d_k = d_model / H. In our SVD low-rank compression research, these dense projection matrices are factored into bottleneck twins W ≈ U_r Σ_r V_r^T, reducing parameter storage by 83.3%.'
    },
    {
        step: 3,
        id: 'open_square',
        title: 'Step 3: Opening the S × S Interaction Square (Q · Kᵀ)',
        phase: 'QUADRATIC EXPANSION',
        badge: 'THE QUADRATIC WALL · S × S',
        color: '#ef4444',
        formula: '\\text{Logits}_{ij} = q_i \\cdot k_j^\\top = \\sum_{m=1}^{d_k} q_{i,m} k_{j,m} \\in \\mathbb{R}^{S \\times S}',
        tensor_dims: {
            left: 'Q: [S × d_k] = [4 × 64]',
            right: 'Kᵀ: [d_k × S] = [64 × 4]',
            result: 'Square Matrix: [S × S] = [4 × 4] = 16 cells'
        },
        narrative: 'THE CORE TRANSFORMATION: The 1D sequence unfolds into a 2D all-to-all interaction square! Every Query row i sweeps across every Key column j via inner dot-product. What was a 1D sequence of length S now explodes into an S × S grid of pairwise communication channels. At S=4,096 tokens, this square contains 16.7 million cells per head!',
        visual_type: 'matrix_square_raw',
        implication: 'Asymptotic complexity jumps from linear O(S) to quadratic O(S² · d). This quadratic explosion makes full attention prohibitively expensive for long context windows on standard GPU high-bandwidth memory (HBM).'
    },
    {
        step: 4,
        id: 'variance_scaling',
        title: 'Step 4: Variance Stabilization (Scaling by 1/√dₖ)',
        phase: 'STABILIZATION',
        badge: 'VARIANCE NORMALIZATION',
        color: '#2dd4bf',
        formula: 'S_{ij} = \\frac{\\text{Logits}_{ij}}{\\sqrt{d_k}} = \\frac{q_i k_j^\\top}{\\sqrt{64}} = 0.125 \\times (q_i k_j^\\top)',
        tensor_dims: {
            head_dim: 'd_k = 64',
            scale_factor: '1 / √64 = 0.125',
            variance_before: 'Var(Q·Kᵀ) = d_k = 64',
            variance_after: 'Var(Scaled) = 1.00'
        },
        narrative: 'If components of Q and K are independent random variables with mean 0 and unit variance, their dot product has variance d_k (standard deviation √d_k = 8). Without scaling, large dot-products (e.g. +16.0 or +24.0) push softmax into extreme saturation plateaus where gradients vanish (∂softmax/∂z ≈ 0). Dividing by √d_k restores Var = 1.0, preserving healthy training dynamics.',
        visual_type: 'matrix_square_scaled',
        implication: 'Without 1/√d_k scaling, multi-head attention cannot train deep networks (gradient vanishing catastrophe). With scaling, gradient flow remains robust across all sequence lengths.'
    },
    {
        step: 5,
        id: 'masking',
        title: 'Step 5: Attention Masking (Causal vs Structural)',
        phase: 'TEMPORAL ORDERING',
        badge: 'CAUSAL BLINDING',
        color: '#f59e0b',
        formula: 'M_{ij} = \\begin{cases} 0 & j \\le i \\\\ -\\infty & j > i \\end{cases}, \\quad \\tilde{S}_{ij} = S_{ij} + M_{ij}',
        tensor_dims: {
            mask_type: 'Autoregressive Lower-Triangular',
            masked_cells: 'Upper-triangle (j > i) set to -∞',
            visible_tokens: 'Token i can only attend to tokens 1..i'
        },
        narrative: 'In generative autoregressive models (GPT-2, Qwen, LLaMA), tokens are strictly forbidden from peeking into future tokens. An additive mask with -∞ is applied to the upper triangle. When exponentiated during softmax, exp(-∞) becomes exactly 0.0, strictly preserving temporal causality.',
        visual_type: 'matrix_square_masked',
        implication: 'In bidirectional models (DistilBERT, RoBERTa), the entire square is active. In causal models, 50% of the square compute is masked out during inference decoding.'
    },
    {
        step: 6,
        id: 'softmax',
        title: 'Step 6: Softmax Normalization (Stochastic Routing)',
        phase: 'PROBABILITY NORMALIZATION',
        badge: 'ROW-WISE PROBABILITIES · Σ = 1.0',
        color: '#34d399',
        formula: 'A_{ij} = \\frac{\\exp(\\tilde{S}_{ij})}{\\sum_{l=1}^S \\exp(\\tilde{S}_{il})}, \\quad \\sum_{j=1}^S A_{ij} = 1.0 \\; (100\\%)',
        tensor_dims: {
            matrix: 'A ∈ [0, 1]^(S × S)',
            row_sum: 'Exactly 1.00 per row',
            entropy: 'Quantifies attention sharpness vs dispersion'
        },
        narrative: 'Softmax exponentiates the masked scaled logits and normalizes across each row independently. This transforms raw affinity scores into true probability distributions where every horizontal row sums to exactly 1.0 (100%). The square is now an exact routing heatmap describing how much weight each token allocates to past positions.',
        visual_type: 'matrix_square_softmax',
        implication: 'Softmax produces an attention distribution that is non-negative and convex, guaranteeing stability during backward gradient propagation.'
    },
    {
        step: 7,
        id: 'value_aggregation',
        title: 'Step 7: Folding the Square Back to Vectors (A · V)',
        phase: 'CONTEXT AGGREGATION',
        badge: 'FOLDING THE SQUARE · S × d_k',
        color: '#a855f7',
        formula: 'O_i = \\sum_{j=1}^S A_{ij} V_j \\in \\mathbb{R}^{d_k}, \\quad \\text{Context} = A \\cdot V \\in \\mathbb{R}^{S \\times d_k}',
        tensor_dims: {
            probability_square: 'A: [S × S] = [4 × 4]',
            value_tensor: 'V: [S × d_k] = [4 × 64]',
            context_tensor: 'O: [S × d_k] = [4 × 64]'
        },
        narrative: 'The 2D attention square multiplies the Value matrix V, collapsing the quadratic S × S matrix back into a 1D sequence of contextualized representations! Each token representation is now a convex combination (weighted average) of all attended value vectors, enriched with global sequence context.',
        visual_type: 'value_mixing',
        implication: 'The 2D interaction square has completed its job: it routed semantic value payloads between tokens and folded back into sequence vectors ready for the next layer.'
    },
    {
        step: 8,
        id: 'output_projection',
        title: 'Step 8: Multi-Head Merge & Output Projection (W_O)',
        phase: 'MULTI-HEAD INTEGRATION',
        badge: 'FINAL MERGE · S × d_model',
        color: '#ec4899',
        formula: '\\text{Out} = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_H) W_O, \\quad W_O \\in \\mathbb{R}^{(H \\cdot d_k) \\times d_{\\text{model}}}',
        tensor_dims: {
            heads: 'H = 12 heads × 64d = 768d',
            concat_shape: '[4 × 768]',
            projection_weight: 'W_O: [768 × 768]',
            final_shape: '[4 × 768]'
        },
        narrative: 'All parallel head outputs are concatenated side-by-side (12 heads × 64 dim = 768 dim) and projected through the output weight matrix W_O. This merges the multiple diverse representation subspaces (syntactic agreement, coreference, factual retrieval) into a unified vector, which is then added to the residual bypass highway.',
        visual_type: 'multi_head_out',
        implication: 'Linear projection back to d_model allows seamless summation with the identity residual highway: X_next = RMSNorm(X + Out).'
    }
];

export const PARADIGM_SQUARE_EFFECTS = {
    dense_mha: {
        id: 'dense_mha',
        name: 'Standard Dense Multi-Head Attention',
        badge: 'O(S²) DENSE SQUARE',
        color: '#38bdf8',
        square_mode: 'Full S × S Square Materialized',
        flops: '\\mathcal{O}(2 S^2 d)',
        memory: '2 \\times B \\times H \\times S^2 \\text{ bytes}',
        description: 'Materializes the full dense S × S pairwise matrix in GPU memory. Requires quadratic compute and memory, creating the standard context-length ceiling.'
    },
    svd_lowrank: {
        id: 'svd_lowrank',
        name: 'SVD Low-Rank Attention (Our Research)',
        badge: 'FACTOR V/U PINCHED (r=64)',
        color: '#2dd4bf',
        square_mode: 'Factored Subspace Projections',
        flops: '\\mathcal{O}(2 S r d + 2 S^2 d_k)',
        memory: '2 \\times r \\times d \\ll d^2 \\; (83.3\\% \\text{ Weight Cut})',
        description: 'Replaces dense W_Q, W_K, W_V, W_O with low-rank factor twins: V (d → r) and U (r → d). Preserves full attention resolution while eliminating up to 83.3% parameter traffic.'
    },
    gqa: {
        id: 'gqa',
        name: 'Grouped-Query Attention (GQA)',
        badge: 'SHARED KV COLUMNS (7:1 RATIO)',
        color: '#f59e0b',
        square_mode: 'Compressed Key/Value Columns',
        flops: '\\mathcal{O}(2 S^2 d)',
        memory: '\\frac{1}{7} \\text{ to } \\frac{1}{4} \\text{ KV Cache Bandwidth}',
        description: 'Shares 1 Key and 1 Value head across multiple Query heads. Slashes autoregressive decoding memory traffic by 75-87.5% while maintaining full multi-head expressive query resolution.'
    },
    mla: {
        id: 'mla',
        name: 'DeepSeek Multi-Head Latent Attention (MLA)',
        badge: 'LATENT VECTOR c^{KV} (512d)',
        color: '#a855f7',
        square_mode: 'Decompression Absorbed into Q',
        flops: '\\mathcal{O}(2 S \\cdot 512 \\cdot d)',
        memory: '93.3\\% \\text{ KV Cache Reduction}',
        description: 'Compresses Key and Value into a compact 512-dim latent vector. Absorbs decompression matrices directly into the Query projection, eliminating full KV cache storage in high-bandwidth memory.'
    },
    linear_attention: {
        id: 'linear_attention',
        name: 'Linear Kernel Attention (Performer / Cosformer)',
        badge: 'O(S) ZERO S × S SQUARE',
        color: '#ec4899',
        square_mode: 'Bypasses S × S Square Entirely!',
        flops: '\\mathcal{O}(S \\cdot d^2) \\ll \\mathcal{O}(S^2 \\cdot d)',
        memory: '\\mathcal{O}(d^2) \\text{ Constant Cache Memory}',
        description: 'Never opens the S × S square! Uses associativity (Q Kᵀ) V = Q (Kᵀ V) with kernel feature maps φ to compute Kᵀ V ∈ ℝ^(d × d) first. Asymptotic complexity drops to strictly linear O(S).'
    },
    sliding_window: {
        id: 'sliding_window',
        name: 'Sliding Window Local Attention (Mistral / SWA)',
        badge: 'BANDED DIAGONAL (W=512)',
        color: '#06b6d4',
        square_mode: 'Banded S × W Diagonal Strip',
        flops: '\\mathcal{O}(S \\cdot W \\cdot d)',
        memory: '\\mathcal{O}(W \\cdot d) \\text{ Bounded Window Memory}',
        description: 'Only opens a diagonal band of width W tokens around each position. Tokens outside the local window are masked to zero, bounding memory to O(W) regardless of total prompt length.'
    }
};

/**
 * Computes exact numerical matrix states for the 4-token demonstrator.
 * @param {boolean} causal - If true, applies lower-triangular causal mask
 * @param {boolean} scaled - If true, scales by 1/sqrt(64) = 0.125
 * @returns {object} Numerical matrix values across all pipeline stages
 */
export function computeDemonstrationMatrices(causal = true, scaled = true) {
    const tokens = ['[CLS]', 'The', 'Neural', 'Model'];
    const S = tokens.length;
    const dk = 64;
    const scaleFactor = scaled ? (1.0 / Math.sqrt(dk)) : 1.0;

    // Representative unscaled dot products with realistic variance
    const rawLogits = [
        [14.4,  4.8, -6.4, -18.2],
        [ 5.2, 16.8,  6.0,  -4.8],
        [ 3.1,  7.4, 18.2,   5.6],
        [ 1.8,  4.2,  8.0,  19.4]
    ];

    // Scaled logits
    const scaledLogits = rawLogits.map(row => row.map(v => v * scaleFactor));

    // Masked logits
    const maskedLogits = scaledLogits.map((row, i) =>
        row.map((v, j) => (causal && j > i) ? -Infinity : v)
    );

    // Softmax probabilities
    const probabilities = maskedLogits.map(row => {
        const validVals = row.filter(v => v !== -Infinity);
        const maxVal = validVals.length > 0 ? Math.max(...validVals) : 0;
        const exps = row.map(v => v === -Infinity ? 0 : Math.exp(v - maxVal));
        const sumExp = exps.reduce((a, b) => a + b, 0);
        return exps.map(e => sumExp > 0 ? e / sumExp : 0);
    });

    return {
        tokens,
        S,
        dk,
        scaleFactor,
        rawLogits,
        scaledLogits,
        maskedLogits,
        probabilities
    };
}

/**
 * Renders the Matrix Unfolding Visualizer into a container element.
 * @param {HTMLElement} container - Target container
 * @param {object} options - Options { activeStep, activeParadigm, causal, scaled, onStepChange }
 */
export function renderMatrixUnfoldingVisualizer(container, options = {}) {
    if (!container) return;

    let currentStepIdx = options.activeStep || 0;
    let currentParadigm = options.activeParadigm || 'dense_mha';
    let causal = options.causal !== undefined ? options.causal : true;
    let scaled = options.scaled !== undefined ? options.scaled : true;
    let isPlaying = false;
    let playTimer = null;

    function render() {
        const step = MATRIX_UNFOLDING_STEPS[currentStepIdx];
        const paradigm = PARADIGM_SQUARE_EFFECTS[currentParadigm] || PARADIGM_SQUARE_EFFECTS.dense_mha;
        const demo = computeDemonstrationMatrices(causal, scaled);

        // Generate matrix cells for visual display
        let gridHtml = '';
        if (step.visual_type === '1d_tokens') {
            gridHtml = `
                <div style="display:flex; gap:10px; width:100%; justify-content:center; padding:16px 0;">
                    ${demo.tokens.map((tok, i) => `
                        <div style="flex:1; max-width:110px; background:#0e1726; border:1px solid #38bdf8; border-radius:6px; padding:12px 8px; text-align:center;">
                            <div style="font-size:10px; color:#94a3b8; font-family:monospace;">Token x_${i+1}</div>
                            <div style="font-size:13px; font-weight:800; color:#f8fafc; margin:4px 0;">${escapeHtml(tok)}</div>
                            <div style="font-size:9px; color:#38bdf8; font-family:monospace;">[1 × 768]</div>
                        </div>
                    `).join('')}
                </div>
            `;
        } else if (step.visual_type === 'qkv_strips') {
            gridHtml = `
                <div style="display:flex; flex-direction:column; gap:10px; width:100%; padding:10px 0;">
                    <div style="display:flex; gap:8px; align-items:center;">
                        <span style="width:70px; font-weight:800; color:#38bdf8; font-size:11px; font-family:monospace;">Query Q:</span>
                        <div style="flex:1; display:flex; gap:6px;">
                            ${demo.tokens.map((t, i) => `<span style="flex:1; background:rgba(56,189,248,0.15); border:1px solid #38bdf8; color:#7dd3fc; border-radius:4px; padding:6px; text-align:center; font-size:10px; font-family:monospace;">q_${i+1} [1×64]</span>`).join('')}
                        </div>
                    </div>
                    <div style="display:flex; gap:8px; align-items:center;">
                        <span style="width:70px; font-weight:800; color:#f59e0b; font-size:11px; font-family:monospace;">Key K:</span>
                        <div style="flex:1; display:flex; gap:6px;">
                            ${demo.tokens.map((t, i) => `<span style="flex:1; background:rgba(245,158,11,0.15); border:1px solid #f59e0b; color:#fde68a; border-radius:4px; padding:6px; text-align:center; font-size:10px; font-family:monospace;">k_${i+1} [1×64]</span>`).join('')}
                        </div>
                    </div>
                    <div style="display:flex; gap:8px; align-items:center;">
                        <span style="width:70px; font-weight:800; color:#34d399; font-size:11px; font-family:monospace;">Value V:</span>
                        <div style="flex:1; display:flex; gap:6px;">
                            ${demo.tokens.map((t, i) => `<span style="flex:1; background:rgba(52,211,153,0.15); border:1px solid #34d399; color:#a7f3d0; border-radius:4px; padding:6px; text-align:center; font-size:10px; font-family:monospace;">v_${i+1} [1×64]</span>`).join('')}
                        </div>
                    </div>
                </div>
            `;
        } else {
            // Render 4x4 Attention Square Matrix
            gridHtml = `
                <div style="display:flex; flex-direction:column; gap:4px; width:100%; max-width:440px; margin:0 auto; padding:8px;">
                    <div style="display:flex; gap:4px; margin-left:60px;">
                        ${demo.tokens.map(t => `<div style="flex:1; text-align:center; font-size:10px; font-weight:700; color:#94a3b8;">${escapeHtml(t)}</div>`).join('')}
                    </div>
                    ${demo.tokens.map((tokI, i) => `
                        <div style="display:flex; gap:6px; align-items:center;">
                            <span style="width:54px; font-size:10px; font-weight:700; color:#94a3b8; text-align:right;">${escapeHtml(tokI)}</span>
                            <div style="display:flex; gap:4px; flex:1;">
                                ${demo.tokens.map((tokJ, j) => {
                                    let cellVal = '';
                                    let cellBg = '#0e1726';
                                    let cellColor = '#94a3b8';
                                    let isMasked = causal && j > i;

                                    if (step.id === 'open_square') {
                                        const raw = demo.rawLogits[i][j];
                                        cellVal = raw > 0 ? `+${raw.toFixed(1)}` : raw.toFixed(1);
                                        cellBg = raw > 10 ? 'rgba(239,68,68,0.3)' : (raw > 0 ? 'rgba(56,189,248,0.2)' : 'rgba(15,23,42,0.6)');
                                        cellColor = raw > 10 ? '#fca5a5' : '#7dd3fc';
                                    } else if (step.id === 'variance_scaling') {
                                        const sc = demo.scaledLogits[i][j];
                                        cellVal = sc > 0 ? `+${sc.toFixed(2)}` : sc.toFixed(2);
                                        cellBg = 'rgba(45,212,191,0.2)';
                                        cellColor = '#5eead4';
                                    } else if (step.id === 'masking') {
                                        if (isMasked) {
                                            cellVal = '-∞';
                                            cellBg = 'rgba(239,68,68,0.15)';
                                            cellColor = '#f87171';
                                        } else {
                                            const sc = demo.scaledLogits[i][j];
                                            cellVal = sc > 0 ? `+${sc.toFixed(2)}` : sc.toFixed(2);
                                            cellBg = 'rgba(45,212,191,0.15)';
                                            cellColor = '#5eead4';
                                        }
                                    } else {
                                        // Softmax probabilities
                                        const prob = demo.probabilities[i][j];
                                        cellVal = isMasked ? '0.0%' : `${(prob * 100).toFixed(1)}%`;
                                        const alpha = Math.max(0.1, prob);
                                        cellBg = isMasked ? 'rgba(15,23,42,0.4)' : `rgba(56, 189, 248, ${alpha})`;
                                        cellColor = isMasked ? '#475569' : (prob > 0.4 ? '#ffffff' : '#93c5fd');
                                    }

                                    return `
                                        <div style="flex:1; height:42px; background:${cellBg}; border:1px solid rgba(255,255,255,0.08); border-radius:4px; display:flex; align-items:center; justify-content:center; font-family:monospace; font-size:11px; font-weight:700; color:${cellColor};" title="Cell (${i+1}, ${j+1}): ${tokI} → ${tokJ}">
                                            ${cellVal}
                                        </div>
                                    `;
                                }).join('')}
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        container.innerHTML = `
            <div class="matrix-unfolding-card" style="background:#070d18; border:1px solid rgba(56,189,248,0.3); border-radius:10px; padding:18px; display:flex; flex-direction:column; gap:16px;">
                <!-- Header Banner -->
                <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:10px; border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:12px;">
                    <div>
                        <div style="display:flex; align-items:center; gap:8px;">
                            <span class="hud-badge" style="background:#8b5cf6; color:#fff; font-weight:800; font-size:10px; padding:2px 8px;">INTERACTIVE MATRIX DECONSTRUCTION</span>
                            <span class="badge" style="background:${step.color}22; color:${step.color}; border:1px solid ${step.color}66; font-size:10px; font-weight:700;">${step.badge}</span>
                        </div>
                        <h3 style="font-size:18px; font-weight:800; color:#fff; margin-top:6px;">${step.title}</h3>
                    </div>

                    <!-- Step Stepper Controls -->
                    <div style="display:flex; align-items:center; gap:6px;">
                        <button id="btn-unfold-prev" class="round-button" style="padding:4px 10px; font-size:11px; background:rgba(255,255,255,0.06); color:#cbd5e1; border:1px solid rgba(255,255,255,0.15);" ${currentStepIdx === 0 ? 'disabled style="opacity:0.4;"' : ''}>◀ Prev</button>
                        <button id="btn-unfold-play" class="round-button" style="padding:4px 12px; font-size:11px; background:rgba(56,189,248,0.2); color:#38bdf8; border:1px solid #38bdf8;">${isPlaying ? '⏸ Pause' : '▶ Auto Play'}</button>
                        <button id="btn-unfold-next" class="round-button" style="padding:4px 10px; font-size:11px; background:rgba(255,255,255,0.06); color:#cbd5e1; border:1px solid rgba(255,255,255,0.15);" ${currentStepIdx === MATRIX_UNFOLDING_STEPS.length - 1 ? 'disabled style="opacity:0.4;"' : ''}>Next ▶</button>
                    </div>
                </div>

                <!-- Step Navigator Pills (1 to 8) -->
                <div style="display:flex; gap:6px; overflow-x:auto; padding-bottom:4px;">
                    ${MATRIX_UNFOLDING_STEPS.map((s, idx) => `
                        <button class="round-button btn-step-pill" data-idx="${idx}" style="flex:1; font-size:10px; padding:4px 6px; white-space:nowrap; ${idx === currentStepIdx ? `background:${s.color}; color:#040e1d; font-weight:800; border-color:${s.color};` : 'background:rgba(255,255,255,0.04); color:#94a3b8;'}">
                            ${idx + 1}. ${s.id.replace(/_/g, ' ').toUpperCase()}
                        </button>
                    `).join('')}
                </div>

                <!-- Architectural Paradigm Selector Bar -->
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; background:#0b1329; border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:8px 12px;">
                    <div style="font-size:11px; font-weight:700; color:#38bdf8;">COMPARE ARCHITECTURE PARADIGM:</div>
                    <div style="display:flex; gap:4px; flex-wrap:wrap;">
                        ${Object.keys(PARADIGM_SQUARE_EFFECTS).map(k => {
                            const p = PARADIGM_SQUARE_EFFECTS[k];
                            const isActive = k === currentParadigm;
                            return `
                                <button class="round-button btn-paradigm-pill" data-key="${k}" style="font-size:10px; padding:3px 8px; ${isActive ? `background:${p.color}33; color:${p.color}; border:1px solid ${p.color}; font-weight:700;` : 'background:rgba(255,255,255,0.03); color:#94a3b8;'}">
                                    ${escapeHtml(p.name.split(' ')[0])}
                                </button>
                            `;
                        }).join('')}
                    </div>
                </div>

                <!-- Visual Animation Canvas Container -->
                <div style="background:#050914; border:1px solid rgba(56,189,248,0.2); border-radius:8px; padding:16px; display:flex; flex-direction:column; align-items:center;">
                    <div style="display:flex; justify-content:space-between; width:100%; font-size:11px; color:#94a3b8; margin-bottom:10px;">
                        <span>VISUAL TENSOR REPRESENTATION</span>
                        <span style="font-family:monospace; color:#38bdf8;">${step.tensor_dims.shape || step.tensor_dims.result || ''}</span>
                    </div>
                    ${gridHtml}
                </div>

                <!-- Mathematical Equation Card -->
                <div style="background:#091224; border-left:3px solid ${step.color}; border-radius:6px; padding:10px 14px; text-align:center;">
                    <div class="katex-math-render" style="color:#7dd3fc; font-size:13px;">$$ ${step.formula} $$</div>
                </div>

                <!-- Pedagogical Narrative & Matrix Implications -->
                <div style="display:grid; grid-template-columns: 3fr 2fr; gap:12px; font-size:12px; line-height:1.5;">
                    <div style="background:#0b1329; border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:12px; color:#cbd5e1;">
                        <strong style="color:#fff; display:block; margin-bottom:4px;">📖 What is happening in this step:</strong>
                        ${step.narrative}
                    </div>
                    <div style="background:#0b1329; border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:12px; color:#cbd5e1;">
                        <strong style="color:${paradigm.color}; display:block; margin-bottom:4px;">📐 Paradigm Matrix Implication (${paradigm.name}):</strong>
                        <div style="font-size:11px; color:#f8fafc; font-weight:700; margin-bottom:4px;">${paradigm.square_mode}</div>
                        <div style="font-size:11px; color:#94a3b8;">${paradigm.description}</div>
                        <div style="margin-top:6px; font-family:monospace; color:#38bdf8; font-size:10px;">FLOPs: $$ ${paradigm.flops} $$</div>
                    </div>
                </div>

                <!-- Interactive Toggles (Causal Masking & Variance Scaling) -->
                <div style="display:flex; justify-content:space-between; align-items:center; background:#080f1c; border-top:1px solid rgba(255,255,255,0.06); padding-top:10px; font-size:11px; color:#94a3b8;">
                    <div style="display:flex; gap:12px; align-items:center;">
                        <label style="display:flex; align-items:center; gap:6px; cursor:pointer;">
                            <input id="chk-causal-mask" type="checkbox" ${causal ? 'checked' : ''} style="accent-color:#a855f7;">
                            <span>Causal Autoregressive Mask (-∞)</span>
                        </label>
                        <label style="display:flex; align-items:center; gap:6px; cursor:pointer;">
                            <input id="chk-variance-scaling" type="checkbox" ${scaled ? 'checked' : ''} style="accent-color:#2dd4bf;">
                            <span>Scale by 1/√dₖ (0.125)</span>
                        </label>
                    </div>
                    <span style="font-family:monospace; color:#34d399;">Step ${currentStepIdx + 1} of ${MATRIX_UNFOLDING_STEPS.length}</span>
                </div>
            </div>
        `;

        // Render KaTeX math
        if (typeof window !== 'undefined' && window.katex) {
            container.querySelectorAll('.katex-math-render').forEach(el => {
                const raw = el.textContent.replace(/\$\$/g, '').trim();
                if (raw) {
                    try {
                        window.katex.render(raw, el, { displayMode: true, throwOnError: false });
                    } catch (e) {}
                }
            });
        }

        // Attach Handlers
        const btnPrev = container.querySelector('#btn-unfold-prev');
        const btnNext = container.querySelector('#btn-unfold-next');
        const btnPlay = container.querySelector('#btn-unfold-play');

        if (btnPrev) btnPrev.onclick = () => {
            if (currentStepIdx > 0) {
                currentStepIdx--;
                render();
            }
        };

        if (btnNext) btnNext.onclick = () => {
            if (currentStepIdx < MATRIX_UNFOLDING_STEPS.length - 1) {
                currentStepIdx++;
                render();
            }
        };

        if (btnPlay) btnPlay.onclick = () => {
            isPlaying = !isPlaying;
            if (isPlaying) {
                playTimer = setInterval(() => {
                    currentStepIdx = (currentStepIdx + 1) % MATRIX_UNFOLDING_STEPS.length;
                    render();
                }, 3200);
            } else {
                clearInterval(playTimer);
            }
            render();
        };

        container.querySelectorAll('.btn-step-pill').forEach(btn => {
            btn.onclick = () => {
                currentStepIdx = parseInt(btn.getAttribute('data-idx'), 10);
                render();
            };
        });

        container.querySelectorAll('.btn-paradigm-pill').forEach(btn => {
            btn.onclick = () => {
                currentParadigm = btn.getAttribute('data-key');
                render();
            };
        });

        const chkCausal = container.querySelector('#chk-causal-mask');
        if (chkCausal) chkCausal.onchange = (e) => {
            causal = e.target.checked;
            render();
        };

        const chkScaled = container.querySelector('#chk-variance-scaling');
        if (chkScaled) chkScaled.onchange = (e) => {
            scaled = e.target.checked;
            render();
        };
    }

    render();
}

/**
 * Opens the Full-Screen Attention Square Unfolding & Matrix Deconstruction Modal.
 */
export function openMatrixUnfoldingModal() {
    let modal = document.getElementById('attention-unfolding-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'attention-unfolding-modal';
        modal.className = 'modal-overlay hidden';
        modal.setAttribute('role', 'dialog');
        modal.setAttribute('aria-modal', 'true');
        modal.innerHTML = `
            <div class="modal-container" style="max-width: 980px; max-height: 94vh; display:flex; flex-direction:column;">
                <div class="modal-header">
                    <div>
                        <span class="hud-badge" style="background:#8b5cf6; color:#fff;">FEATURE 1.8 · MATRIX DECONSTRUCTION</span>
                        <h2>Attention Square Deconstruction &amp; Matrix Unfolding Engine</h2>
                        <p class="modal-subtitle">Step-by-step interactive proof showing how a 1D sequence expands into a 2D S × S interaction square and how compression paradigms conquer the quadratic wall.</p>
                    </div>
                    <button id="btn-close-unfold-modal" class="btn-icon-only modal-close">✕</button>
                </div>
                <div id="unfolding-modal-body" style="flex:1; overflow-y:auto; padding:20px;"></div>
            </div>
        `;
        document.body.appendChild(modal);

        const closeBtn = modal.querySelector('#btn-close-unfold-modal');
        if (closeBtn) closeBtn.onclick = () => modal.classList.add('hidden');
        modal.onclick = (e) => {
            if (e.target === modal) modal.classList.add('hidden');
        };
    }

    const body = modal.querySelector('#unfolding-modal-body');
    if (body) {
        renderMatrixUnfoldingVisualizer(body, {
            activeStep: 0,
            activeParadigm: 'dense_mha',
            causal: true,
            scaled: true
        });
    }

    modal.classList.remove('hidden');
}
