const fs = require('fs');
let code = fs.readFileSync('Docs/visualizer/app.js', 'utf8');

const renderTokenLifecycle = `function renderTokenLifecycle(moduleMeta, modelKey = currentModel) {
    const listEl = document.getElementById('lifecycle-steps-list');
    if (!listEl) return;

    const proj = (moduleMeta && moduleMeta.model_projections && moduleMeta.model_projections[modelKey]) || {};
    const dModel = proj.hidden_size || (modelKey === 'qwen' ? 896 : 768);
    const dFfn = proj.intermediate_size || (modelKey === 'qwen' ? 4864 : 3072);
    const vocabSize = VOCAB_SIZES[modelKey] || 30522;
    
    const isAttn = (moduleMeta && (moduleMeta.category || "").toLowerCase().includes("attention")) || (moduleMeta && (moduleMeta.type || "").toLowerCase().includes("attention"));
    const hasCustom = (moduleMeta && moduleMeta.components && moduleMeta.components.length > 0);

    const steps = [
        {
            num: 1,
            title: 'Token Ingestion & Position Encoding',
            badge: 'INPUT',
            badgeColor: '#10b981',
            badgeBg: 'rgba(16, 185, 129, 0.2)',
            enters: \`Token IDs [t_1, ..., t_S] ∈ {1..\${vocabSize.toLocaleString()}}\`,
            added: modelKey === 'qwen' 
                ? 'RoPE Rotary Position Frequencies (Θ_i) injected at Q & K' 
                : \`+ W_pos [Positional Vector] ∈ ℝ^{Seq × \${dModel}}\`,
            exits: \`Continuous Hidden States X^{(0)} ∈ ℝ^{Seq × \${dModel}}\`,
            math: modelKey === 'qwen' 
                ? 'X^{(0)} = W_e[T], \\\\quad \\\\mathbf{q}_m = \\\\mathbf{R}_{\\\\Theta, m}^d (W_q x_m), \\\\; \\\\mathbf{k}_n = \\\\mathbf{R}_{\\\\Theta, n}^d (W_k x_n)' 
                : 'X^{(0)} = W_{\\\\text{embed}}[T] + W_{\\\\text{pos}}[P] \\\\in \\\\mathbb{R}^{S \\\\times ' + dModel + '}',
            target: 'input_embeddings'
        },
        {
            num: 2,
            title: modelKey === 'qwen' ? 'RMS Normalization (Pre-Attn)' : 'Layer Normalization (Pre-Attn)',
            badge: 'NORM',
            badgeColor: '#38bdf8',
            badgeBg: 'rgba(56, 189, 248, 0.2)',
            enters: \`Incoming Tokens X ∈ ℝ^{Seq × \${dModel}}\`,
            added: 'Variance Normalization & Scale γ',
            exits: \`Normalized Tokens X̃ ∈ ℝ^{Seq × \${dModel}}\`,
            math: modelKey === 'qwen'
                ? '\\\\tilde{X} = \\\\frac{X}{\\\\text{RMS}(X)} \\\\odot \\\\gamma, \\\\quad \\\\text{RMS}(X) = \\\\sqrt{\\\\frac{1}{d}\\\\sum x_i^2 + \\\\epsilon}'
                : '\\\\tilde{X} = \\\\frac{X - \\\\mu}{\\\\sqrt{\\\\sigma^2 + \\\\epsilon}} \\\\odot \\\\gamma + \\\\beta',
            target: modelKey === 'qwen' ? 'q_proj' : 'v_factor'
        },
        {
            num: 3,
            title: (isAttn && hasCustom) ? moduleMeta.name : (modelKey === 'qwen' ? 'Grouped-Query Attention Contextualization' : 'Self-Attention Contextualization'),
            badge: (isAttn && hasCustom) ? 'CUSTOM ATTN' : 'ATTENTION',
            badgeColor: '#2dd4bf',
            badgeBg: 'rgba(45, 212, 191, 0.2)',
            enters: \`Normalized Tokens X̃ ∈ ℝ^{Seq × \${dModel}}\`,
            added: (isAttn && hasCustom) ? moduleMeta.components.map(c => c.name).join(' ➔ ') : 'Context Interaction Matrix: Softmax(Q K^T / √d_k) · V',
            exits: \`Attention Context Delta O ∈ ℝ^{Seq × \${dModel}}\`,
            math: (isAttn && hasCustom && moduleMeta.math_latex) ? moduleMeta.math_latex : '\\\\text{Attn}(X) = \\\\text{Softmax}\\\\left(\\\\frac{Q K^T}{\\\\sqrt{d_k}} + M\\\\right) V \\\\cdot W_o',
            target: (isAttn && hasCustom) ? moduleMeta.components[0].id : (modelKey === 'qwen' ? 'o_proj' : (modelKey === 'gpt2' ? 'c_proj' : 'out_lin'))
        },
        {
            num: 4,
            title: '(+) Attention Residual Skip Highway',
            badge: '(+) ADD #1',
            badgeColor: '#0284c7',
            badgeBg: 'rgba(2, 132, 199, 0.25)',
            enters: \`Original Tokens X_in + Context Delta O\`,
            added: '+ X_in (Unmodified incoming tokens added via bypass highway)',
            exits: \`Accumulated Tokens X_mid = X_in + Dropout(O)\`,
            math: 'X_{\\\\text{mid}} = X_{\\\\text{in}} + \\\\text{Dropout}\\\\left(\\\\text{Attn}(X_{\\\\text{in}})\\\\right)',
            target: 'attn.residual_add'
        },
        {
            num: 5,
            title: (!isAttn && hasCustom) ? moduleMeta.name : (modelKey === 'qwen' ? 'SwiGLU Gated Feed-Forward Network' : 'Feed-Forward Knowledge Retrieval (GELU)'),
            badge: (!isAttn && hasCustom) ? 'CUSTOM FFN' : 'FFN',
            badgeColor: '#a855f7',
            badgeBg: 'rgba(168, 85, 247, 0.2)',
            enters: \`Contextualized Tokens X_mid ∈ ℝ^{Seq × \${dModel}}\`,
            added: (!isAttn && hasCustom) ? moduleMeta.components.map(c => c.name).join(' ➔ ') : (modelKey === 'qwen' 
                ? \`Bifurcated Non-Linear Gating: SiLU(X W_g) ⊙ (X W_u) across \${dFfn} intermediate dims\` 
                : \`4× Dimension Expansion GELU(X W_1 + b_1) across \${dFfn} hidden dims\`),
            exits: \`Factual Knowledge Delta Y_ffn ∈ ℝ^{Seq × \${dModel}}\`,
            math: (!isAttn && hasCustom && moduleMeta.math_latex) ? moduleMeta.math_latex : (modelKey === 'qwen'
                ? 'Y_{\\\\text{ffn}} = \\\\left( \\\\text{SiLU}(X_{\\\\text{mid}} W_g) \\\\odot (X_{\\\\text{mid}} W_u) \\\\right) W_d'
                : 'Y_{\\\\text{ffn}} = \\\\text{GELU}(X_{\\\\text{mid}} W_1 + b_1) W_2 + b_2'),
            target: (!isAttn && hasCustom) ? moduleMeta.components[0].id : (modelKey === 'qwen' ? 'gate_proj' : (modelKey === 'gpt2' ? 'c_fc' : 'intermediate'))
        },
        {
            num: 6,
            title: '(+) FFN Residual Skip Highway',
            badge: '(+) ADD #2',
            badgeColor: '#7c3aed',
            badgeBg: 'rgba(124, 58, 237, 0.25)',
            enters: \`Intermediate Tokens X_mid + FFN Knowledge Delta Y_ffn\`,
            added: '+ X_mid (Intermediate tokens added via bypass highway)',
            exits: \`Layer Output Tokens X_layer = X_mid + Dropout(Y_ffn)\`,
            math: 'X_{\\\\text{layer}} = X_{\\\\text{mid}} + \\\\text{Dropout}\\\\left(\\\\text{FFN}(X_{\\\\text{mid}})\\\\right)',
            target: 'ffn.residual_add'
        },
        {
            num: 7,
            title: 'Output LM Head & Next-Token Softmax',
            badge: 'OUTPUT',
            badgeColor: '#f43f5e',
            badgeBg: 'rgba(244, 63, 94, 0.2)',
            enters: \`Final Hidden Representation X^{(L)} ∈ ℝ^{Seq × \${dModel}}\`,
            added: \`Final Norm + Unembedding Projection W_vocab^T ∈ ℝ^{\${dModel} × \${vocabSize}}\`,
            exits: \`Probability Distribution P(t_{S+1}) ∈ [0, 1]^{\${vocabSize.toLocaleString()}}\`,
            math: '\\\\hat{y} = \\\\text{Softmax}\\\\left(\\\\text{Norm}(X^{(L)}) W_{\\\\text{vocab}}^T\\\\right) \\\\in \\\\mathbb{R}^{S \\\\times V}',
            target: 'output_head'
        }
    ];

    listEl.innerHTML = steps.map(s => \`
        <div class="lifecycle-step-item" data-step-target="\${s.target}" onclick="selectPartIn3D('\${s.target}')" title="Click to illuminate \${s.title} in 3D">
            <div class="lsi-header">
                <div class="lsi-title-group">
                    <span class="lsi-num">0\${s.num}</span>
                    <span class="lsi-title">\${s.title}</span>
                </div>
                <span class="lsi-badge" style="color: \${s.badgeColor}; background: \${s.badgeBg}; border: 1px solid \${s.badgeColor};">\${s.badge}</span>
            </div>
            <div class="lsi-flow">
                <span class="lsi-in">In: \${escapeHtml(s.enters)}</span>
                <span class="lsi-arrow">➔</span>
                <span class="lsi-added">Added: \${escapeHtml(s.added)}</span>
                <span class="lsi-arrow">➔</span>
                <span class="lsi-out">Out: \${escapeHtml(s.exits)}</span>
            </div>
            <div class="lsi-math" data-math="\${encodeURIComponent(s.math)}">$$ \${s.math} $$</div>
        </div>
    \`).join('');

    if (window.katex) {
        listEl.querySelectorAll('.lsi-math').forEach(el => {
            const raw = decodeURIComponent(el.getAttribute('data-math') || '');
            if (raw) {
                try { katex.render(raw, el, { displayMode: false, throwOnError: false }); } catch(e) {}
            }
        });
    }
}
`;

if (!code.includes('function renderTokenLifecycle(')) {
    const endIdx = code.indexOf('function renderComponentPipeline(');
    if (endIdx !== -1) {
        code = code.substring(0, endIdx) + renderTokenLifecycle + '\n\n' + code.substring(endIdx);
        fs.writeFileSync('Docs/visualizer/app.js', code);
        console.log("Restored renderTokenLifecycle");
    } else {
        console.log("Could not find renderComponentPipeline");
    }
} else {
    console.log("renderTokenLifecycle already exists");
}
