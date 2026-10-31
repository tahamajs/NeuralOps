/**
 * educational/learn_modal.js
 * ==========================
 * Interactive Educational "Learn Mode" Dialog.
 * Explains foundational Transformer and Compression concepts in clear, intuitive English:
 * - What is a Transformer?
 * - What is Attention & Why Multi-Head?
 * - What is the KV Cache Bottleneck?
 * - What is SVD & Low-Rank Factorization?
 * - What is Mixture-of-Experts (MoE)?
 * - What is Multi-Head Latent Attention (MLA)?
 * - What is Structured Pruning (SliceGPT)?
 * - What is Quantization (INT4 / FP4)?
 */

export const LEARN_TOPICS = [
    {
        id: 'transformer',
        title: 'What is a Transformer?',
        icon: '🧠',
        summary: 'The dominant deep learning architecture powering modern Large Language Models.',
        formula: 'X^{(l+1)} = \\text{FFN}(\\text{LN}(X^{(l)} + \\text{Attn}(\\text{LN}(X^{(l)}))))',
        body: `
            <p>Introduced in the landmark 2017 paper <em>"Attention Is All You Need"</em> by Vaswani et al., the Transformer replaces sequential recurrent networks (RNNs/LSTMs) with an entirely attention-based mechanism. This allows models to process all tokens in a sequence simultaneously in parallel on modern GPUs.</p>
            <p>Every transformer layer consists of two primary sub-blocks: <strong>Multi-Head Self-Attention</strong> (which mixes contextual information across different positions in the sequence) and a <strong>Feed-Forward Network (FFN)</strong> (which acts as an associative key-value memory retrieving factual information).</p>
            <p>Residual skip connections and normalization layers wrap both sub-blocks, ensuring gradients can flow smoothly across dozens of layers during training without vanishing.</p>
        `
    },
    {
        id: 'attention',
        title: 'What is Attention & Why Multi-Head?',
        icon: '⚡',
        summary: 'Dynamically routing context between tokens across multiple independent geometric subspaces.',
        formula: '\\text{MHA}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_H) W_O',
        body: `
            <p>Self-attention enables each word or token to dynamically determine how much weight to place on other words in the sentence. For example, in <em>"The bank of the river"</em>, the token <em>"bank"</em> strongly attends to <em>"river"</em> to clarify its semantic meaning.</p>
            <p><strong>Why Multi-Head?</strong> A single attention matrix can only focus on one type of relationship at a time. By splitting the hidden dimension $d_{\\text{model}}$ into $H$ heads (e.g. 12 heads of dimension 64), the model can attend to multiple distinct semantic features simultaneously: one head tracks syntactic dependencies, another tracks pronouns, and another tracks long-range factual topics.</p>
        `
    },
    {
        id: 'kv_cache',
        title: 'What is the KV Cache Bottleneck?',
        icon: '💾',
        summary: 'The primary memory bandwidth constraint during autoregressive LLM token generation.',
        formula: '\\text{Cache Size} = 2 \\times L \\times H \\times d_{\\text{head}} \\times \\text{SeqLen} \\times B \\times 2 \\text{ bytes}',
        body: `
            <p>During text generation (decoding), an LLM generates tokens one by one. In order to compute attention for the newest token without recomputing past tokens, the Key and Value vectors for all past tokens must be preserved in GPU VRAM.</p>
            <p>As sequence length grows to 32k, 128k, or 1M tokens, the KV cache quickly consumes tens of gigabytes of GPU memory, often exceeding the parameter size of the model itself! This memory bandwidth bottleneck severely limits the concurrent batch size that can be served on GPUs.</p>
        `
    },
    {
        id: 'svd',
        title: 'What is SVD & Low-Rank Compression?',
        icon: '📉',
        summary: 'Approximating heavy dense matrices via factorized rank-r bottleneck twins.',
        formula: 'W \\approx U_r \\Sigma_r V_r^\\top = A B^\\top, \\quad r \\ll \\min(d_{\\text{in}}, d_{\\text{out}})',
        body: `
            <p>Singular Value Decomposition (SVD) is a fundamental theorem of linear algebra proving that any matrix $W \\in \\mathbb{R}^{m \\times n}$ can be decomposed into orthogonal singular vectors and singular values. According to the Eckart-Young-Mirsky theorem, truncating SVD to rank $r$ yields the mathematically optimal rank-$r$ approximation in Frobenius norm.</p>
            <p>In this framework, dense projection matrices ($768 \\times 768$) are factorized into paired bottleneck linear layers ($768 \\times 64$ and $64 \\times 768$). This cuts parameters and memory storage by over 80% while retaining over 90% of model accuracy.</p>
        `
    },
    {
        id: 'moe',
        title: 'What is Mixture of Experts (MoE)?',
        icon: '🔀',
        summary: 'Activating only a sparse subset of parameters per token to maximize parameter capacity without increasing compute.',
        formula: 'y = \\sum_{i=1}^k G(x)_i E_i(x), \\quad G(x) = \\text{Softmax}(\\text{TopK}(x W_g))',
        body: `
            <p>In standard dense models, every single parameter is executed for every token. Mixture of Experts (MoE) replaces the dense Feed-Forward layer with multiple parallel "Expert" subnetworks (e.g. 4, 8, or 64 experts).</p>
            <p>A lightweight learned router network calculates gating scores for each token and dispatches it only to the Top-1 or Top-2 most relevant experts. This enables models to possess hundreds of billions of parameters while only consuming the FLOPs of a much smaller model during inference.</p>
        `
    },
    {
        id: 'mla',
        title: 'What is Multi-Head Latent Attention (MLA)?',
        icon: '💎',
        summary: 'Compressing Key and Value tensors into a compact latent vector prior to caching.',
        formula: 'c_t^{KV} = W^{DKV} x_t, \\quad K = W^{UK} c_t^{KV}, \\; V = W^{UV} c_t^{KV}',
        body: `
            <p>Pioneered in DeepSeek-V2 and DeepSeek-V3, Multi-Head Latent Attention (MLA) revolutionizes KV cache efficiency. Instead of storing separate full-rank Key and Value tensors for all 16+ heads, MLA uses a down-projection matrix to compress them into a single 512-dimensional latent vector $c_t^{KV}$.</p>
            <p>During autoregressive decoding, only this compact latent vector is stored in VRAM. The attention computation can even absorb the decompression matrix directly into the queries, completely eliminating the need to decompress keys during matrix multiplication!</p>
        `
    },
    {
        id: 'pruning',
        title: 'What is Structured Pruning & SliceGPT?',
        icon: '✂️',
        summary: 'Physically deleting rows and columns to produce smaller dense networks without sparse hardware overhead.',
        formula: 'W_{\\text{slice}} = W Q P_r, \\quad Q^\\top Q = I, \\; P_r = [I_r, 0]^\\top',
        body: `
            <p>Traditional unstructured pruning zeroes out individual weights, creating sparse matrices that require specialized software kernels to achieve speedups. Structured pruning, such as <strong>SliceGPT (ICLR 2024)</strong>, removes entire rows and columns.</p>
            <p>SliceGPT uses orthogonal transformation matrices $Q$ derived from activation covariance PCA to concentrate signal energy into the principal coordinates, allowing the model to slice away unneeded dimensions and run immediately faster on standard dense hardware.</p>
        `
    },
    {
        id: 'quantization',
        title: 'What is Quantization (INT4 / FP4)?',
        icon: '🎯',
        summary: 'Compressing 16-bit floating point representations down to 4-bit integers with per-channel calibration.',
        formula: '\\hat{w} = \\text{clamp}\\left(\\left\\lfloor \\frac{w}{s} \\right\\rceil, -8, 7\\right) \\times s',
        body: `
            <p>Standard neural network weights are stored in 16-bit half-precision (FP16 or BF16), requiring 2 bytes per parameter. Post-training quantization maps these continuous values onto a discrete grid of 8-bit (INT8) or 4-bit (INT4/FP4) integers.</p>
            <p>By computing per-channel or per-block scaling factors $s$, 4-bit quantization reduces memory requirements by 4× (e.g. from 14GB down to 3.5GB for a 7B model), allowing large LLMs to run locally on consumer GPUs and laptops.</p>
        `
    }
];

/**
 * Initializes and renders the interactive Learn Modal.
 */
export function initLearnModal() {
    let modal = document.getElementById('learn-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'learn-modal';
        modal.className = 'modal-overlay hidden';
        modal.setAttribute('role', 'dialog');
        modal.setAttribute('aria-modal', 'true');
        modal.innerHTML = `
            <div class="modal-container" style="max-width: 860px; max-height: 88vh; display: flex; flex-direction: column;">
                <div class="modal-header">
                    <div>
                        <span class="hud-badge" style="background:#10b981; color:#fff;">EDUCATIONAL ATLAS &amp; RESEARCH PRIMER</span>
                        <h2 id="learn-modal-title">Transformer &amp; Attention Masterclass</h2>
                        <p class="modal-subtitle">Core mathematical derivations, engineering trade-offs, and memory mechanics explained in plain English.</p>
                    </div>
                    <button id="btn-close-learn-modal" class="btn-icon-only modal-close" aria-label="Close Learn Modal">✕</button>
                </div>
                <div style="display: flex; gap: 16px; flex: 1; min-height: 0; padding: 14px 20px; overflow: hidden;">
                    <!-- Sidebar Topic Selector -->
                    <div id="learn-topic-nav" style="width: 260px; overflow-y: auto; display: flex; flex-direction: column; gap: 6px; border-right: 1px solid rgba(255,255,255,0.08); padding-right: 12px;">
                        ${LEARN_TOPICS.map((t, idx) => `
                            <button class="learn-topic-btn ${idx === 0 ? 'active' : ''}" data-topic-id="${t.id}" style="text-align: left; padding: 8px 10px; border-radius: 6px; background: ${idx === 0 ? 'rgba(34, 213, 247, 0.15)' : 'rgba(255,255,255,0.03)'}; border: 1px solid ${idx === 0 ? 'rgba(34, 213, 247, 0.4)' : 'rgba(255,255,255,0.06)'}; color: ${idx === 0 ? '#38bdf8' : 'var(--text-main)'}; font-size: 11px; cursor: pointer; transition: all 0.2s;">
                                <div style="font-weight: 700; margin-bottom: 2px;">${t.icon} ${t.title}</div>
                                <div style="font-size: 9px; color: var(--text-faint); line-height: 1.3; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${t.summary}</div>
                            </button>
                        `).join('')}
                    </div>
                    <!-- Topic Content Body -->
                    <div id="learn-topic-content" style="flex: 1; overflow-y: auto; padding-left: 6px;"></div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        const closeBtn = modal.querySelector('#btn-close-learn-modal');
        if (closeBtn) closeBtn.addEventListener('click', () => modal.classList.add('hidden'));
        modal.addEventListener('click', (e) => {
            if (e.target === modal) modal.classList.add('hidden');
        });
    }

    function renderTopic(topicId) {
        const topic = LEARN_TOPICS.find(t => t.id === topicId) || LEARN_TOPICS[0];
        const contentEl = document.getElementById('learn-topic-content');
        if (!contentEl) return;

        contentEl.innerHTML = `
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="font-size: 24px;">${topic.icon}</span>
                <h3 style="font-size: 18px; color: var(--text-primary); margin: 0;">${topic.title}</h3>
            </div>
            <p style="font-size: 12px; color: var(--cyan); font-weight: 500; margin-bottom: 12px;">${topic.summary}</p>
            
            <div class="latex-formula-card" style="margin-bottom: 14px; background: rgba(14,31,50,0.8); border: 1px solid rgba(133,198,235,0.2); border-radius: 8px; padding: 10px;">
                <div style="font-size: 10px; color: var(--text-faint); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Canonical Mathematical Invariant</div>
                <div class="learn-math" style="font-size: 13px; text-align: center; color: #38bdf8;">$$ ${topic.formula} $$</div>
            </div>

            <div class="markdown-body" style="font-size: 12px; line-height: 1.7; color: var(--text-main);">
                ${topic.body}
            </div>
        `;

        if (window.katex) {
            contentEl.querySelectorAll('.learn-math').forEach(el => {
                const raw = el.textContent.replace(/\$\$/g, '').trim();
                if (raw) {
                    try {
                        window.katex.render(raw, el, { displayMode: true, throwOnError: false });
                    } catch (e) {}
                }
            });
        }
    }

    // Attach topic click handlers
    document.querySelectorAll('.learn-topic-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.learn-topic-btn').forEach(b => {
                b.style.background = 'rgba(255,255,255,0.03)';
                b.style.borderColor = 'rgba(255,255,255,0.06)';
                b.style.color = 'var(--text-main)';
            });
            btn.style.background = 'rgba(34, 213, 247, 0.15)';
            btn.style.borderColor = 'rgba(34, 213, 247, 0.4)';
            btn.style.color = '#38bdf8';
            renderTopic(btn.dataset.topicId);
        });
    });

    renderTopic(LEARN_TOPICS[0].id);
}

/**
 * Opens the Learn Modal.
 */
export function openLearnModal() {
    initLearnModal();
    const modal = document.getElementById('learn-modal');
    if (modal) modal.classList.remove('hidden');
}
