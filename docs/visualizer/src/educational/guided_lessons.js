/**
 * educational/guided_lessons.js
 * ==============================
 * Interactive Guided Lessons Mode (5 Bite-Sized Transformer Masterclasses).
 * Features step-by-step curriculum, interactive in-lesson micro-widgets,
 * 3D scene camera & component highlighting, and completion badges.
 */

export const GUIDED_LESSONS = [
    {
        id: 'attention_foundations',
        number: 1,
        title: 'Foundations of Attention: Q, K, V & Softmax',
        badge: 'CORE RETRIEVAL ENGINE',
        color: '#38bdf8',
        componentTarget: 'attention_matrix',
        cameraTargetY: 2.2,
        cameraDist: 12.5,
        steps: [
            {
                title: 'Associative Information Retrieval: Q, K, and V',
                explanation: 'Attention reimagines communication across sequence positions as an associative memory retrieval. Query (Q = X W_Q) specifies what each token is looking for. Key (K = X W_K) specifies what attributes each token represents. Value (V = X W_V) contains the substantive information payload to be routed. Pairwise inner products quantify query-key relevance.',
                math: '\\text{Scores}_{ij} = \\frac{q_i \\cdot k_j^T}{\\sqrt{d_k}} = \\frac{\\sum_{m=1}^{d_k} q_{i,m} k_{j,m}}{\\sqrt{d_k}}',
                widget: {
                    type: 'softmax_scale_demo',
                    defaultDk: 64
                }
            },
            {
                title: 'Variance Stabilization: Why Scale by 1/√d_k?',
                explanation: 'If components of q and k are independent random variables with mean 0 and variance 1, their dot product has mean 0 and variance d_k (standard deviation √d_k). For large head dimension d_k = 64 or 128, unscaled logits grow huge, pushing softmax into extreme saturation regions where gradients vanish (∂softmax/∂z ≈ 0)! Dividing by √d_k enforces Var = 1, ensuring stable backpropagation gradients.',
                math: '\\mathbb{E}[q \\cdot k] = 0, \\quad \\mathrm{Var}(q \\cdot k) = d_k \\implies \\mathrm{Var}\\left(\\frac{q \\cdot k}{\\sqrt{d_k}}\\right) = 1',
                widget: {
                    type: 'softmax_scale_demo',
                    defaultDk: 64
                }
            },
            {
                title: 'Softmax Normalization & Convex Combination',
                explanation: 'Softmax exponentiates scaled logits and normalizes them into non-negative probabilities summing to 1 across all keys. The final output is an exact convex combination (weighted average) of value vectors V, routing information dynamically across the context window.',
                math: '\\alpha_{ij} = \\frac{\\exp\\left(\\frac{q_i k_j^T}{\\sqrt{d_k}}\\right)}{\\sum_{l=1}^N \\exp\\left(\\frac{q_i k_l^T}{\\sqrt{d_k}}\\right)}, \\quad \\text{Attn}(Q, K, V)_i = \\sum_{j=1}^N \\alpha_{ij} V_j',
                widget: {
                    type: 'none'
                }
            },
            {
                title: 'Multi-Head Subspace Specialization',
                explanation: 'Instead of performing a single attention pooling over the entire d_model dimension, Multi-Head Attention projects Q, K, V into h independent subspaces of dimension d_k = d_model / h. This allows the model to simultaneously track syntactic dependencies, semantic coreference, and positional ordering in parallel.',
                math: '\\text{MHA}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_h) W^O, \\quad \\text{head}_i = \\text{Attn}(Q W_i^Q, K W_i^K, V W_i^V)',
                widget: {
                    type: 'none'
                }
            }
        ]
    },
    {
        id: 'quadratic_attention',
        number: 2,
        title: 'Why Attention Is Quadratic',
        badge: 'O(N²) EXPLOSION',
        color: '#ef4444',
        componentTarget: 'attention_matrix',
        cameraTargetY: 2.5,
        cameraDist: 14.0,
        steps: [
            {
                title: 'The All-to-All Token Interaction',
                explanation: 'Standard Multi-Head Attention computes pairwise dot products between every Query token and every Key token: QKᵀ / √d. For a prompt of N tokens, this creates a dense N × N attention matrix.',
                math: 'A = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) \\in \\mathbb{R}^{N \\times N}',
                widget: {
                    type: 'quadratic_calc',
                    defaultSeq: 2048
                }
            },
            {
                title: 'The Sequence Length Scaling Trap',
                explanation: 'When sequence length N doubles from 1,024 to 2,048, the number of pairwise calculations does not double — it quadruples (4×). At 8,192 tokens, attention consumes 67.1 million elements per head!',
                math: '\\text{Memory} = 2 \\times N^2 \\times H \\times L \\text{ bytes (FP16)}',
                widget: {
                    type: 'quadratic_calc',
                    defaultSeq: 8192
                }
            },
            {
                title: 'The Solution: Sub-Quadratic Attention',
                explanation: 'To scale to 32k or 128k context windows, researchers developed Low-Rank SVD, Linear Kernel Attention O(N), and Sparse MoE, reducing asymptotic complexity and memory footprint dramatically.',
                math: '\\mathcal{O}(N^2 \\cdot d) \\longrightarrow \\mathcal{O}(N \\cdot r \\cdot d)',
                widget: {
                    type: 'none'
                }
            }
        ]
    },
    {
        id: 'svd_compression',
        number: 3,
        title: 'How SVD Compresses Attention',
        badge: 'LOW-RANK BOTTLENECK',
        color: '#38bdf8',
        componentTarget: 'v_factor',
        cameraTargetY: 2.2,
        cameraDist: 12.0,
        steps: [
            {
                title: 'Singular Value Decomposition (SVD)',
                explanation: 'Any linear projection weight matrix W ∈ ℝ^(d × d) can be factored into orthogonal matrices U, V and singular values Σ: W = U Σ Vᵀ.',
                math: 'W = U \\Sigma V^T = \\sum_{i=1}^{d} \\sigma_i u_i v_i^T',
                widget: {
                    type: 'svd_rank_demo',
                    defaultRank: 64
                }
            },
            {
                title: 'Rank-r Truncation & Low-Rank Bottleneck',
                explanation: 'By keeping only the top-r largest singular values, we create an optimal low-rank approximation W ≈ U_r (Σ_r V_rᵀ). Instead of d² parameters, we store only 2·r·d parameters.',
                math: '\\text{Parameters: } 2 \\times r \\times d \\ll d^2 \\quad (r=64, d=768 \\implies 83.3\\% \\text{ cut})',
                widget: {
                    type: 'svd_rank_demo',
                    defaultRank: 32
                }
            },
            {
                title: 'Inference Without Weight Reconstruction',
                explanation: 'During forward inference, tokens pass through Factor V first (ℝ^d → ℝ^r) and then Factor U (ℝ^r → ℝ^d), reducing FLOPs and memory traffic without ever needing to materialize the dense W matrix.',
                math: 'y = x \\cdot W \\approx (x \\cdot V_r^T) \\cdot U_r',
                widget: {
                    type: 'none'
                }
            }
        ]
    },
    {
        id: 'kv_cache_bottleneck',
        number: 4,
        title: 'The KV Cache Bottleneck',
        badge: 'MEMORY BANDWIDTH BOUND',
        color: '#f59e0b',
        componentTarget: 'k_proj',
        cameraTargetY: 2.0,
        cameraDist: 13.0,
        steps: [
            {
                title: 'Autoregressive Decoding & Memory Traffic',
                explanation: 'During text generation, LLMs generate one token per forward pass. To avoid recomputing past keys and values, past K and V states are saved in GPU high-bandwidth memory (HBM) as the "KV Cache".',
                math: '\\text{KV Cache Size} = 2 \\times 2 \\times n_{\\text{layers}} \\times n_{\\text{heads}} \\times d_{\\text{head}} \\times N_{\\text{ctx}}',
                widget: {
                    type: 'kv_cache_calc',
                    heads: 12
                }
            },
            {
                title: 'Multi-Query (MQA) & Grouped-Query (GQA)',
                explanation: 'In Standard MHA, each Query head has its own Key and Value head (12:12). Grouped-Query Attention (GQA) shares 1 Key/Value head across groups of Query heads (e.g. 4:1), cutting KV Cache bandwidth by 4×!',
                math: '\\text{GQA Ratio} = \\frac{n_{\\text{KV heads}}}{n_{\\text{Q heads}}} = \\frac{1}{4} \\text{ or } \\frac{1}{8}',
                widget: {
                    type: 'kv_cache_calc',
                    heads: 2
                }
            }
        ]
    },
    {
        id: 'sparse_vs_lowrank_vs_linear',
        number: 5,
        title: 'Sparse vs Low-Rank vs Linear Attention',
        badge: 'PARADIGM COMPARISON',
        color: '#10b981',
        componentTarget: 'router',
        cameraTargetY: 3.5,
        cameraDist: 16.0,
        steps: [
            {
                title: 'Three Paths to Efficiency',
                explanation: '1. Sparse Attention (MoE / Local Window): Only attends to subsets of tokens. 2. Low-Rank (SVD / LoRA): Projects to a low-dimensional subspace rank r. 3. Linear Attention: Uses kernel feature maps φ(Q)(φ(K)ᵀV) to compute KV first in O(N).',
                math: '\\text{Complexity: } \\text{Dense } \\mathcal{O}(N^2) \\quad \\text{Linear } \\mathcal{O}(N d^2) \\quad \\text{SVD } \\mathcal{O}(N r d)',
                widget: {
                    type: 'paradigm_picker'
                }
            },
            {
                title: 'Hardware Trade-offs',
                explanation: 'Sparse models require complex irregular memory accesses. Linear attention changes the softmax kernel. SVD maintains standard dense GEMM hardware accelerators while drastically reducing matrix footprint.',
                math: '\\text{Roofline: Low-Rank preserves compute density; Sparse optimizes parameter scale.}',
                widget: {
                    type: 'paradigm_picker'
                }
            }
        ]
    },
    {
        id: 'flash_attention_hardware',
        number: 6,
        title: 'Hardware-Aware Attention (FlashAttention)',
        badge: 'SRAM TILING & IO',
        color: '#ec4899',
        componentTarget: 'layernorm',
        cameraTargetY: 1.8,
        cameraDist: 11.0,
        steps: [
            {
                title: 'GPU Memory Hierarchy: HBM vs SRAM',
                explanation: 'Modern GPUs have fast, tiny SRAM (~200 KB per Streaming Multiprocessor, 19 TB/s) and slower, large HBM (80 GB, 2-3 TB/s). Standard attention is bottle-necked by writing the intermediate N×N attention matrix to HBM and reading it back.',
                math: '\\text{HBM Bandwidth: } 3.35 \\text{ TB/s (H100)} \\ll \\text{SRAM Bandwidth: } 19 \\text{ TB/s}',
                widget: {
                    type: 'io_tiling_calc'
                }
            },
            {
                title: 'Tiling & Online Softmax',
                explanation: 'FlashAttention loads blocks of Q, K, V into fast on-chip SRAM, computes attention incrementally using online softmax normalization, and writes only the final output back to HBM — achieving 2-4× wall-clock speedup without modifying math!',
                math: '\\text{IO Complexity: } \\mathcal{O}\\left(\\frac{N^2 d}{M}\\right) \\text{ where } M = \\text{SRAM size}',
                widget: {
                    type: 'io_tiling_calc'
                }
            }
        ]
    }
];

/**
 * Computes N^2 quadratic attention calculation statistics.
 * @param {number} seqLen - Sequence length N
 * @param {number} numHeads - Heads (default 12)
 * @param {number} numLayers - Layers (default 12)
 * @returns {object} Quadratic scaling metrics
 */
export function computeQuadraticStats(seqLen = 2048, numHeads = 12, numLayers = 12) {
    const N = Math.max(64, Math.min(65536, seqLen));
    const matrixEntriesPerHead = N * N;
    const totalEntriesPerLayer = matrixEntriesPerHead * numHeads;
    const fullModelEntries = totalEntriesPerLayer * numLayers;
    // FP16 = 2 bytes
    const memoryBytes = fullModelEntries * 2;
    const memoryMB = memoryBytes / (1024 * 1024);

    return {
        seqLen: N,
        matrixEntriesPerHead,
        totalEntriesPerLayer,
        fullModelEntries,
        memoryMB,
        ratioTo512: (N * N) / (512 * 512)
    };
}

/**
 * Computes softmax attention logits and gradient health under scaled vs unscaled regimes.
 * Demonstrates mathematically why dividing by sqrt(d_k) is mandatory.
 * @param {number} dk - Head dimension (e.g. 64)
 * @param {boolean} scaled - True if scaled by 1/sqrt(dk)
 * @returns {object} { dk, scaled, scaleFactor, probs, gradMagnitude, vanishingRisk }
 */
export function computeAttentionScalingDemo(dk = 64, scaled = true) {
    const scale = scaled ? 1.0 / Math.sqrt(dk) : 1.0;
    const rawDotProducts = [1.8 * Math.sqrt(dk), 0.5 * Math.sqrt(dk), -0.7 * Math.sqrt(dk), -2.2 * Math.sqrt(dk)];
    const logits = rawDotProducts.map(s => s * scale);
    const maxL = Math.max(...logits);
    const exps = logits.map(l => Math.exp(l - maxL));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map(e => e / sumExps);
    const maxProb = Math.max(...probs);
    // Gradient magnitude indicator: p * (1 - p)
    const gradMagnitude = maxProb * (1 - maxProb);
    return {
        dk,
        scaled,
        scaleFactor: scale,
        logits,
        probs,
        gradMagnitude,
        vanishingRisk: gradMagnitude < 0.05 ? 'CRITICAL VANISHING GRADIENTS (< 0.05)' : 'HEALTHY GRADIENTS (> 0.20)'
    };
}

/**
 * Opens the Interactive Guided Lessons Modal.
 * @param {object} context - Engine context { camera, controls, smoothFlyCamera, selectPartIn3D }
 */
export function openGuidedLessonsModal(context = {}) {
    let modal = document.getElementById('guided-lessons-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'guided-lessons-modal';
        modal.className = 'modal-overlay hidden';
        modal.setAttribute('role', 'dialog');
        modal.setAttribute('aria-modal', 'true');
        modal.innerHTML = `
            <div class="modal-container" style="max-width: 960px; max-height: 92vh; display:flex; flex-direction:column;">
                <div class="modal-header">
                    <div>
                        <span class="hud-badge" style="background:#10b981; color:#fff;">FEATURE 1.7 · GUIDED LESSONS</span>
                        <h2>Interactive Transformer Masterclass &amp; Guided Lessons</h2>
                        <p class="modal-subtitle">6 bite-sized architectural masterclasses with real-time 3D camera guidance, micro-calculators, and interactive proofs.</p>
                    </div>
                    <button id="btn-close-lessons-modal" class="btn-icon-only modal-close">✕</button>
                </div>

                <!-- Lesson Navigation Tabs -->
                <div id="lessons-tab-bar" style="padding:10px 24px; background:#080f1c; border-bottom:1px solid rgba(255,255,255,0.06); display:flex; gap:8px; overflow-x:auto;"></div>

                <!-- Active Lesson Content Body -->
                <div id="lesson-content-area" style="flex:1; overflow-y:auto; padding:24px 28px; display:flex; flex-direction:column; gap:20px;"></div>

                <!-- Footer Navigation Bar -->
                <div style="padding:14px 28px; background:#080f1c; border-top:1px solid rgba(255,255,255,0.06); display:flex; justify-content:space-between; align-items:center;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span id="lesson-step-indicator" style="font-size:12px; font-weight:700; color:#94a3b8;">Step 1 of 3</span>
                        <span id="lesson-completion-badge" class="hidden" style="background:#10b981; color:#fff; font-size:11px; font-weight:800; padding:2px 8px; border-radius:12px;">
                            🎓 Lesson Complete!
                        </span>
                    </div>

                    <div style="display:flex; gap:10px;">
                        <button id="btn-prev-lesson-step" class="round-button" style="font-size:12px; padding:6px 14px;">◀ Previous</button>
                        <button id="btn-next-lesson-step" class="round-button btn-catalog" style="font-size:12px; padding:6px 18px; background:#0284c7; color:#fff;">Next Step ▶</button>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        const closeBtn = document.getElementById('btn-close-lessons-modal');
        if (closeBtn) closeBtn.onclick = () => modal.classList.add('hidden');
        modal.onclick = (e) => { if (e.target === modal) modal.classList.add('hidden'); };
    }

    let activeLessonIdx = 0;
    let activeStepIdx = 0;
    const completedLessons = new Set();

    function renderTabs() {
        const tabBar = document.getElementById('lessons-tab-bar');
        if (!tabBar) return;
        tabBar.innerHTML = GUIDED_LESSONS.map((lesson, idx) => {
            const isActive = idx === activeLessonIdx;
            const isDone = completedLessons.has(lesson.id);
            const check = isDone ? '✓ ' : '';
            return `
                <button class="round-button lesson-tab-btn" data-idx="${idx}" style="font-size:11px; white-space:nowrap; padding:5px 12px; ${isActive ? 'background:rgba(56,189,248,0.25); border-color:#38bdf8; color:#38bdf8;' : ''}">
                    ${check}Lesson ${lesson.number}: ${lesson.title}
                </button>
            `;
        }).join('');

        tabBar.querySelectorAll('.lesson-tab-btn').forEach(btn => {
            btn.onclick = () => {
                activeLessonIdx = parseInt(btn.getAttribute('data-idx'), 10);
                activeStepIdx = 0;
                renderLesson();
            };
        });
    }

    function renderLesson() {
        renderTabs();
        const lesson = GUIDED_LESSONS[activeLessonIdx];
        const step = lesson.steps[activeStepIdx];
        const contentArea = document.getElementById('lesson-content-area');
        const stepIndicator = document.getElementById('lesson-step-indicator');
        const badge = document.getElementById('lesson-completion-badge');
        const btnPrev = document.getElementById('btn-prev-lesson-step');
        const btnNext = document.getElementById('btn-next-lesson-step');

        if (!contentArea || !step) return;

        if (stepIndicator) {
            stepIndicator.textContent = `Lesson ${lesson.number} · Step ${activeStepIdx + 1} of ${lesson.steps.length}`;
        }

        if (badge) {
            if (completedLessons.has(lesson.id)) {
                badge.classList.remove('hidden');
            } else {
                badge.classList.add('hidden');
            }
        }

        if (btnPrev) {
            btnPrev.disabled = activeStepIdx === 0 && activeLessonIdx === 0;
            btnPrev.style.opacity = btnPrev.disabled ? '0.4' : '1';
        }

        if (btnNext) {
            const isLast = activeStepIdx === lesson.steps.length - 1;
            btnNext.textContent = isLast ? (activeLessonIdx === GUIDED_LESSONS.length - 1 ? 'Finish Masterclass 🎓' : 'Complete & Next Lesson ➔') : 'Next Step ▶';
        }

        // Trigger 3D Camera Guidance
        if (context.smoothFlyCamera && context.camera && context.controls) {
            context.smoothFlyCamera(context.camera, context.controls, lesson.cameraTargetY, lesson.cameraDist, 0);
        }
        if (context.selectPartIn3D && lesson.componentTarget) {
            context.selectPartIn3D(lesson.componentTarget);
        }

        // Render Step HTML
        let widgetHtml = '';
        if (step.widget && step.widget.type === 'softmax_scale_demo') {
            const initialDk = step.widget.defaultDk || 64;
            const stats = computeAttentionScalingDemo(initialDk, true);
            widgetHtml = `
                <div style="background:#080f1c; border:1px solid rgba(56,189,248,0.2); border-radius:8px; padding:16px; margin-top:12px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:8px;">
                        <span style="font-weight:700; color:#38bdf8; font-size:12px;">INTERACTIVE SOFTMAX VARIANCE &amp; GRADIENT PROVING GROUND</span>
                        <div style="display:flex; gap:6px;">
                            <button id="btn-scale-toggle" class="round-button" style="font-size:10px; padding:3px 10px; background:rgba(56,189,248,0.2); color:#38bdf8; border:1px solid #38bdf8; cursor:pointer;">
                                ✔ Scaled (1/√dₖ)
                            </button>
                        </div>
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center; font-size:11px; color:#94a3b8; margin-bottom:4px;">
                        <span>Head Dimension (d_k)</span>
                        <span id="lesson-scale-badge" style="font-family:monospace; color:#38bdf8; font-weight:700;">d_k = ${initialDk} · Scaled by 1/√d_k</span>
                    </div>
                    <input id="lesson-dk-slider" type="range" min="16" max="128" step="16" value="${initialDk}" style="width:100%; accent-color:#38bdf8; cursor:pointer;">

                    <div style="margin-top:14px; background:#091224; border-radius:6px; padding:12px; border:1px solid rgba(255,255,255,0.06);">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                            <span style="font-size:11px; color:#94a3b8;">Attended Probability Distribution</span>
                            <span id="lesson-grad-health" style="font-size:11px; color:#34d399; font-weight:700;">${stats.vanishingRisk} (Grad: ${stats.gradMagnitude.toFixed(4)})</span>
                        </div>
                        <div id="lesson-prob-bars" style="display:flex; gap:12px; height:90px; align-items:flex-end;">
                            ${stats.probs.map((p, idx) => `
                                <div style="flex:1; display:flex; flex-direction:column; align-items:center; gap:4px; height:100%; justify-content:flex-end;">
                                    <div style="font-size:9px; color:#94a3b8;">Key ${idx + 1}</div>
                                    <div style="width:100%; height:60px; background:#0e1726; border-radius:3px; display:flex; align-items:flex-end; padding:2px;">
                                        <div style="width:100%; height:${Math.max(4, Math.round(p * 100))}%; background:#38bdf8; border-radius:2px; transition:all 0.2s ease;"></div>
                                    </div>
                                    <div style="font-size:10px; font-weight:700; color:#fff;">${(p * 100).toFixed(1)}%</div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            `;
        } else if (step.widget && step.widget.type === 'quadratic_calc') {
            const initialSeq = step.widget.defaultSeq || 2048;
            const stats = computeQuadraticStats(initialSeq);
            widgetHtml = `
                <div style="background:#080f1c; border:1px solid rgba(56,189,248,0.2); border-radius:8px; padding:16px; margin-top:12px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <span style="font-weight:700; color:#38bdf8; font-size:12px;">INTERACTIVE SEQUENCE LENGTH EXPLORER (N)</span>
                        <span id="lesson-seq-badge" style="font-family:monospace; background:rgba(56,189,248,0.2); padding:2px 8px; border-radius:4px; color:#38bdf8; font-weight:800;">${initialSeq} tokens</span>
                    </div>
                    <input id="lesson-seq-slider" type="range" min="512" max="16384" step="512" value="${initialSeq}" style="width:100%; accent-color:#38bdf8; cursor:pointer;">
                    <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; margin-top:14px; font-size:11px;">
                        <div style="background:#0e1726; padding:10px; border-radius:6px;">
                            <div style="color:#94a3b8; font-size:10px;">QKᵀ Matrix Elements</div>
                            <div id="lesson-qk-elems" style="font-size:15px; font-weight:800; color:#38bdf8; margin-top:2px;">${(stats.matrixEntriesPerHead / 1e6).toFixed(2)}M / head</div>
                        </div>
                        <div style="background:#0e1726; padding:10px; border-radius:6px;">
                            <div style="color:#94a3b8; font-size:10px;">Growth vs 512 Tokens</div>
                            <div id="lesson-growth-ratio" style="font-size:15px; font-weight:800; color:#ef4444; margin-top:2px;">${stats.ratioTo512.toFixed(1)}× Explosion</div>
                        </div>
                        <div style="background:#0e1726; padding:10px; border-radius:6px;">
                            <div style="color:#94a3b8; font-size:10px;">Attention Memory (FP16)</div>
                            <div id="lesson-mem-mb" style="font-size:15px; font-weight:800; color:#a855f7; margin-top:2px;">${stats.memoryMB.toFixed(1)} MB</div>
                        </div>
                    </div>
                </div>
            `;
        }

        contentArea.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                    <span class="hud-badge" style="background:${lesson.color}; color:#fff;">${lesson.badge}</span>
                    <h3 style="font-size:20px; font-weight:800; color:#fff; margin-top:8px;">${step.title}</h3>
                </div>
            </div>

            <div style="font-size:14px; line-height:1.65; color:#cbd5e1; background:#0b1329; padding:16px 20px; border-radius:8px; border:1px solid rgba(255,255,255,0.06);">
                ${step.explanation}
            </div>

            ${step.math ? `
                <div style="background:#050914; padding:12px 18px; border-radius:6px; font-family:monospace; color:#38bdf8; font-size:13px; border-left:3px solid ${lesson.color}; overflow-x:auto;">
                    <div class="katex-math-render">$$ ${step.math} $$</div>
                </div>
            ` : ''}

            ${widgetHtml}
        `;

        // Render KaTeX in lesson content
        if (typeof window !== 'undefined' && window.katex) {
            contentArea.querySelectorAll('.katex-math-render').forEach(el => {
                const raw = el.textContent.replace(/\$\$/g, '').trim();
                if (raw) {
                    try {
                        window.katex.render(raw, el, { displayMode: true, throwOnError: false });
                    } catch (e) {}
                }
            });
        }

        // Wire Softmax scale proving ground widget if present
        const dkSlider = document.getElementById('lesson-dk-slider');
        const scaleBtn = document.getElementById('btn-scale-toggle');
        if (dkSlider && scaleBtn) {
            let isScaled = true;
            const refreshSoftmaxWidget = () => {
                const curDk = parseInt(dkSlider.value, 10);
                const res = computeAttentionScalingDemo(curDk, isScaled);
                const badge = document.getElementById('lesson-scale-badge');
                if (badge) badge.textContent = `d_k = ${curDk} · ${isScaled ? 'Scaled by 1/√d_k' : 'UNSCALED'}`;
                const health = document.getElementById('lesson-grad-health');
                if (health) {
                    health.innerHTML = `<span style="color:${isScaled ? '#34d399' : '#ef4444'}; font-weight:700;">${res.vanishingRisk}</span> (Grad: ${res.gradMagnitude.toFixed(4)})`;
                }
                const bars = document.getElementById('lesson-prob-bars');
                if (bars) {
                    bars.innerHTML = res.probs.map((p, idx) => `
                        <div style="flex:1; display:flex; flex-direction:column; align-items:center; gap:4px; height:100%; justify-content:flex-end;">
                            <div style="font-size:9px; color:#94a3b8;">Key ${idx + 1}</div>
                            <div style="width:100%; height:60px; background:#0e1726; border-radius:3px; display:flex; align-items:flex-end; padding:2px;">
                                <div style="width:100%; height:${Math.max(4, Math.round(p * 100))}%; background:${isScaled ? '#38bdf8' : '#ef4444'}; border-radius:2px; transition:all 0.2s ease;"></div>
                            </div>
                            <div style="font-size:10px; font-weight:700; color:#fff;">${(p * 100).toFixed(1)}%</div>
                        </div>
                    `).join('');
                }
            };

            dkSlider.oninput = refreshSoftmaxWidget;
            scaleBtn.onclick = () => {
                isScaled = !isScaled;
                scaleBtn.textContent = isScaled ? '✔ Scaled (1/√dₖ)' : '✖ Unscaled';
                scaleBtn.style.background = isScaled ? 'rgba(56,189,248,0.2)' : 'rgba(239,68,68,0.2)';
                scaleBtn.style.color = isScaled ? '#38bdf8' : '#f87171';
                scaleBtn.style.borderColor = isScaled ? '#38bdf8' : '#ef4444';
                refreshSoftmaxWidget();
            };
        }

        // Wire in-lesson interactive slider if present
        const seqSlider = document.getElementById('lesson-seq-slider');
        if (seqSlider) {
            seqSlider.oninput = (e) => {
                const val = parseInt(e.target.value, 10);
                const s = computeQuadraticStats(val);
                const b = document.getElementById('lesson-seq-badge');
                if (b) b.textContent = `${val} tokens`;
                const qk = document.getElementById('lesson-qk-elems');
                if (qk) qk.textContent = `${(s.matrixEntriesPerHead / 1e6).toFixed(2)}M / head`;
                const gr = document.getElementById('lesson-growth-ratio');
                if (gr) gr.textContent = `${s.ratioTo512.toFixed(1)}× Explosion`;
                const mb = document.getElementById('lesson-mem-mb');
                if (mb) mb.textContent = `${s.memoryMB.toFixed(1)} MB`;
            };
        }
    }

    const btnPrev = document.getElementById('btn-prev-lesson-step');
    const btnNext = document.getElementById('btn-next-lesson-step');

    if (btnPrev) {
        btnPrev.onclick = () => {
            if (activeStepIdx > 0) {
                activeStepIdx--;
            } else if (activeLessonIdx > 0) {
                activeLessonIdx--;
                activeStepIdx = GUIDED_LESSONS[activeLessonIdx].steps.length - 1;
            }
            renderLesson();
        };
    }

    if (btnNext) {
        btnNext.onclick = () => {
            const lesson = GUIDED_LESSONS[activeLessonIdx];
            if (activeStepIdx < lesson.steps.length - 1) {
                activeStepIdx++;
            } else {
                completedLessons.add(lesson.id);
                if (activeLessonIdx < GUIDED_LESSONS.length - 1) {
                    activeLessonIdx++;
                    activeStepIdx = 0;
                }
            }
            renderLesson();
        };
    }

    renderLesson();
    modal.classList.remove('hidden');
}
