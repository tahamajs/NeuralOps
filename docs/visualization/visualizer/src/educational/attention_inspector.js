/**
 * educational/attention_inspector.js
 * ==================================
 * Specialized Attention Deep-Dive Breakdown & Masterclass Educational Tutor.
 * Provides complete architectural telemetry and conceptual explanations when attention is inspected:
 * 1. The Q · K · V Associative Retrieval Engine (Query, Key, Value intuition)
 * 2. Scaled Dot-Product & Softmax normalization mechanics (why scale by √d_k)
 * 3. The Quadratic O(N²) Sequence Length Bottleneck ($512^2 \to 4096^2 \to 32k^2$)
 * 4. Multi-Head Attention Subspace decomposition (orthogonal semantic subspaces)
 * 5. Modern Attention Compression Paradigms (SVD Bottleneck, GQA, MLA, Sliding Window)
 * 6. Granular projections telemetry, step-by-step computation pipeline, and analytical KV cache chart
 */

import { detectAttentionVariant, ATTENTION_VARIANTS } from './attention_types.js';
import { MODEL_SPECS } from '../constants.js';
import { escapeHtml } from '../utils/dom.js';
import { renderMatrixUnfoldingVisualizer, openMatrixUnfoldingModal } from './matrix_unfolding.js';

/**
 * Renders the comprehensive Attention Masterclass and telemetry card in the inspector drawer.
 * @param {object} moduleMeta - Active module metadata
 * @param {string} currentModel - Selected model key
 */
export function renderAttentionInspector(moduleMeta, currentModel = 'distilbert') {
    const container = document.getElementById('attention-deep-dive-card');
    if (!container) return;

    const variant = detectAttentionVariant(moduleMeta, currentModel);
    const spec = MODEL_SPECS[currentModel] || MODEL_SPECS.distilbert;

    const dModel = spec.hidden_size;
    const hQ = variant.q_heads || 12;
    const hKV = variant.kv_heads || 12;
    const dHead = variant.d_head || Math.round(dModel / hQ);

    // Build KV Cache Comparison Data
    const cacheTypes = [
        { name: 'Standard MHA', heads: hQ, perToken: 2 * hQ * dHead * 2, color: '#3B82F6' },
        { name: 'Grouped-Query (GQA)', heads: Math.max(1, Math.round(hQ / 6)), perToken: 2 * Math.max(1, Math.round(hQ / 6)) * dHead * 2, color: '#2DD4BF' },
        { name: 'Multi-Query (MQA)', heads: 1, perToken: 2 * 1 * dHead * 2, color: '#F59E0B' },
        { name: 'DeepSeek MLA', heads: 'Latent', perToken: 512 * 2, color: '#A855F7' }
    ];
    const maxCache = cacheTypes[0].perToken;

    // Computation graph steps HTML
    const compStepsHtml = (variant.computation_graph || []).map((step) => `
        <div class="attn-comp-step" style="display:flex; gap:8px; align-items:flex-start; padding:6px 8px; background:rgba(255,255,255,0.03); border-radius:5px; border:1px solid rgba(255,255,255,0.05);">
            <span class="attn-step-idx" style="font-size:9px; font-weight:800; color:var(--cyan); background:rgba(34,213,247,0.15); padding:2px 6px; border-radius:3px;">${step.step}</span>
            <div class="attn-step-body" style="flex:1;">
                <div class="attn-step-title" style="font-size:11px; font-weight:700; color:#f1f5f9; margin-bottom:2px;">
                    ${escapeHtml(step.title)}: <code class="attn-op-math" style="color:#38bdf8; font-family:var(--font-mono); font-size:10px;">$$ ${step.op} $$</code>
                </div>
                <div class="attn-step-desc" style="font-size:10px; color:var(--text-faint); line-height:1.4;">${escapeHtml(step.desc)}</div>
            </div>
        </div>
    `).join('');

    // Sample Attention Heatmap HTML
    const sampleMatrix = variant.sample_attention_matrix || [
        [0.70, 0.15, 0.10, 0.05],
        [0.20, 0.60, 0.15, 0.05],
        [0.10, 0.20, 0.60, 0.10],
        [0.05, 0.10, 0.25, 0.60]
    ];
    const tokenLabels = ['[CLS]', 'The', 'Neural', 'Model'];
    const heatmapRowsHtml = sampleMatrix.map((row, i) => `
        <div class="attn-hm-row" style="display:flex; align-items:center; gap:6px;">
            <span class="attn-hm-label" style="width:48px; font-size:10px; font-family:var(--font-mono); color:var(--text-faint); text-align:right;">${tokenLabels[i] || 'T' + i}</span>
            <div class="attn-hm-cells" style="display:flex; gap:4px; flex:1;">
                ${row.map(val => {
                    const alpha = Math.max(0.12, val);
                    const bg = `rgba(34, 213, 247, ${alpha})`;
                    const textCol = val > 0.4 ? '#050a12' : '#eef8ff';
                    return `<span class="attn-hm-cell" style="flex:1; text-align:center; padding:3px 0; border-radius:3px; font-size:10px; font-family:var(--font-mono); font-weight:600; background:${bg}; color:${textCol};" title="Score: ${(val * 100).toFixed(1)}%">${val.toFixed(2)}</span>`;
                }).join('')}
            </div>
        </div>
    `).join('');

    container.innerHTML = `
        <div class="transformation-header" style="border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:8px; margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span class="badge" style="background:${variant.color}22; color:${variant.color}; border:1px solid ${variant.color}66; font-size:11px; font-weight:700;">
                        ${variant.icon} ${escapeHtml(variant.name)}
                    </span>
                    <span style="font-size:10px; color:var(--text-faint);">${spec.name} · ${escapeHtml(variant.family)}</span>
                </div>
                <span class="hud-badge" style="background:rgba(56,189,248,0.15); color:#38bdf8; font-size:9px;">MASTERCLASS</span>
            </div>
        </div>

        <!-- 1. The Q · K · V Associative Retrieval Masterclass -->
        <div class="attn-qkv-explainer-card" style="background:rgba(6,13,24,0.75); border:1px solid rgba(56,189,248,0.25); border-radius:8px; padding:10px; margin-bottom:12px;">
            <div style="font-size:11px; font-weight:700; color:#38bdf8; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px;">
                🧠 The Attention Retrieval Engine (Q · K · V)
            </div>
            <div style="display:flex; flex-direction:column; gap:6px; font-size:10px; line-height:1.45;">
                <div style="display:flex; gap:8px; align-items:flex-start;">
                    <span style="background:rgba(56,189,248,0.2); color:#38bdf8; font-weight:800; padding:1px 6px; border-radius:3px; font-family:var(--font-mono);">Q</span>
                    <div><strong style="color:#e2e8f0;">Query ("What am I seeking?"):</strong> Current token's projection looking for contextual relationships in the sequence.</div>
                </div>
                <div style="display:flex; gap:8px; align-items:flex-start;">
                    <span style="background:rgba(245,158,11,0.2); color:#f59e0b; font-weight:800; padding:1px 6px; border-radius:3px; font-family:var(--font-mono);">K</span>
                    <div><strong style="color:#e2e8f0;">Key ("What do I represent?"):</strong> Other tokens' advertised attributes and syntactic roles in the sentence.</div>
                </div>
                <div style="display:flex; gap:8px; align-items:flex-start;">
                    <span style="background:rgba(16,185,129,0.2); color:#34d399; font-weight:800; padding:1px 6px; border-radius:3px; font-family:var(--font-mono);">V</span>
                    <div><strong style="color:#e2e8f0;">Value ("Content Payload"):</strong> The contextual semantic representation transmitted when Q and K match.</div>
                </div>
                <div style="display:flex; gap:8px; align-items:flex-start; border-top:1px solid rgba(255,255,255,0.06); padding-top:4px; margin-top:2px;">
                    <span style="background:rgba(168,85,247,0.2); color:#c084fc; font-weight:800; padding:1px 6px; border-radius:3px; font-family:var(--font-mono);">1/√d_k</span>
                    <div><strong style="color:#e2e8f0;">Scaling Factor:</strong> Normalizes variance to 1.0; prevents large dot-products from saturating the softmax into regions with near-zero gradients.</div>
                </div>
            </div>
        </div>

        <!-- 2. Head Dimension Architecture Indicator -->
        <div class="attn-head-indicator-card" style="background:rgba(14,31,50,0.7); border:1px solid rgba(133,198,235,0.2); border-radius:8px; padding:10px; margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; font-size:11px; margin-bottom:6px;">
                <span style="color:var(--text-faint); text-transform:uppercase; letter-spacing:0.5px;">HEAD DIMENSION TOPOLOGY</span>
                <strong style="color:var(--cyan);">${hQ} Query Heads × ${dHead}d = ${dModel}d</strong>
            </div>
            <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:6px; font-size:10px;">
                <div style="background:rgba(255,255,255,0.04); padding:4px 6px; border-radius:4px;">
                    <div style="color:var(--text-faint);">Query Heads (H_Q)</div>
                    <strong style="font-size:12px; color:#38bdf8;">${hQ}</strong>
                </div>
                <div style="background:rgba(255,255,255,0.04); padding:4px 6px; border-radius:4px;">
                    <div style="color:var(--text-faint);">KV Heads (H_KV)</div>
                    <strong style="font-size:12px; color:#f59e0b;">${hKV}</strong>
                </div>
                <div style="background:rgba(255,255,255,0.04); padding:4px 6px; border-radius:4px;">
                    <div style="color:var(--text-faint);">Scaling (1/√d_k)</div>
                    <strong style="font-size:11px; color:#34d399;">${(1 / Math.sqrt(dHead)).toFixed(3)}</strong>
                </div>
                <div style="background:rgba(255,255,255,0.04); padding:4px 6px; border-radius:4px;">
                    <div style="color:var(--text-faint);">Causal Mask</div>
                    <strong style="font-size:12px; color:${variant.causal ? '#a855f7' : '#94a3b8'};">${variant.causal ? 'Yes (Lower-Tri)' : 'No (Bidirectional)'}</strong>
                </div>
            </div>
        </div>

        <!-- 3. The O(N²) Quadratic Complexity Wall & Why Compression Matters -->
        <div style="background:rgba(6,13,24,0.75); border:1px solid rgba(239,68,68,0.25); border-radius:8px; padding:10px; margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-size:11px; font-weight:700; color:#f87171; text-transform:uppercase; letter-spacing:0.5px;">⚠️ The O(N²) Sequence Complexity Wall</span>
                <span style="font-size:9px; color:var(--text-faint);">Attention Matrix Entries</span>
            </div>
            <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:6px; font-size:10px; text-align:center;">
                <div style="background:rgba(255,255,255,0.04); padding:4px 6px; border-radius:4px;">
                    <div style="color:var(--text-faint);">Seq Len N=512</div>
                    <strong style="color:#e2e8f0; font-family:var(--font-mono);">262K</strong>
                    <div style="color:#34d399; font-size:9px;">Baseline (1×)</div>
                </div>
                <div style="background:rgba(255,255,255,0.04); padding:4px 6px; border-radius:4px;">
                    <div style="color:var(--text-faint);">Seq Len N=4,096</div>
                    <strong style="color:#f59e0b; font-family:var(--font-mono);">16.8M</strong>
                    <div style="color:#f59e0b; font-size:9px;">64× Explosion</div>
                </div>
                <div style="background:rgba(255,255,255,0.04); padding:4px 6px; border-radius:4px;">
                    <div style="color:var(--text-faint);">Seq Len N=32,768</div>
                    <strong style="color:#f87171; font-family:var(--font-mono);">1.07B</strong>
                    <div style="color:#f87171; font-size:9px;">4,096× (OOM Risk)</div>
                </div>
            </div>
            <div style="font-size:10px; color:#cbd5e1; margin-top:6px; line-height:1.4;">
                💡 <em>Engineering Invariant:</em> Full dense attention scales quadratically with sequence length. SVD factorization, GQA grouped sharing, and DeepSeek MLA latent compression reduce memory traffic by up to 93%, enabling ultra-long context inference on standard GPUs.
            </div>
        </div>

        <!-- 4. Weight Matrix Projections Breakdown -->
        <div style="margin-bottom:12px;">
            <div style="font-size:11px; font-weight:700; color:var(--text-primary); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px;">
                📐 Linear Projection Operators
            </div>
            <div style="display:grid; grid-template-columns: repeat(2, 1fr); gap:6px; font-size:11px; font-family:var(--font-mono);">
                <div style="background:rgba(6,13,24,0.6); border:1px solid rgba(255,255,255,0.06); padding:6px 8px; border-radius:4px;">
                    <span style="color:#38bdf8; font-weight:700;">W_Q:</span> ${variant.projections.W_Q || `[${dModel} × ${dModel}]`}
                </div>
                <div style="background:rgba(6,13,24,0.6); border:1px solid rgba(255,255,255,0.06); padding:6px 8px; border-radius:4px;">
                    <span style="color:#f59e0b; font-weight:700;">W_K:</span> ${variant.projections.W_K || `[${dModel} × ${dModel}]`}
                </div>
                <div style="background:rgba(6,13,24,0.6); border:1px solid rgba(255,255,255,0.06); padding:6px 8px; border-radius:4px;">
                    <span style="color:#f59e0b; font-weight:700;">W_V:</span> ${variant.projections.W_V || `[${dModel} × ${dModel}]`}
                </div>
                <div style="background:rgba(6,13,24,0.6); border:1px solid rgba(255,255,255,0.06); padding:6px 8px; border-radius:4px;">
                    <span style="color:#2dd4bf; font-weight:700;">W_O:</span> ${variant.projections.W_O || `[${dModel} × ${dModel}]`}
                </div>
            </div>
        </div>

        <!-- 5. Computation Pipeline -->
        <div style="margin-bottom:12px;">
            <div style="font-size:11px; font-weight:700; color:var(--text-primary); text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px;">
                ⚡ Step-by-Step Computation Graph
            </div>
            <div class="attn-comp-list" style="display:flex; flex-direction:column; gap:6px;">
                ${compStepsHtml}
            </div>
        </div>

        <!-- 6. KV Cache Size Benchmark Bar Chart -->
        <div style="background:rgba(14,31,50,0.7); border:1px solid rgba(133,198,235,0.2); border-radius:8px; padding:10px; margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-size:11px; font-weight:700; color:var(--text-primary); text-transform:uppercase; letter-spacing:0.5px;">💾 KV Cache Footprint (Bytes / Token)</span>
                <span style="font-size:10px; color:var(--text-faint);">Seq=4k Tokens</span>
            </div>
            <div style="display:flex; flex-direction:column; gap:6px;">
                ${cacheTypes.map(c => {
                    const pct = Math.max(10, Math.round((c.perToken / maxCache) * 100));
                    return `
                        <div>
                            <div style="display:flex; justify-content:space-between; font-size:10px; margin-bottom:2px;">
                                <span style="color:var(--text-main);">${escapeHtml(c.name)}</span>
                                <span style="color:${c.color}; font-family:var(--font-mono); font-weight:700;">${c.perToken} B/tok (${pct}%)</span>
                            </div>
                            <div style="height:6px; background:rgba(255,255,255,0.06); border-radius:3px; overflow:hidden;">
                                <div style="height:100%; width:${pct}%; background:${c.color}; border-radius:3px;"></div>
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        </div>

        <!-- 7. Interactive Matrix Deconstruction (Opening the S x S Square) -->
        <div style="background:rgba(15,23,42,0.85); border:1px solid rgba(168,85,247,0.35); border-radius:8px; padding:12px; margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; flex-wrap:wrap; gap:6px;">
                <span style="font-size:11px; font-weight:700; color:#c084fc; text-transform:uppercase; letter-spacing:0.5px;">
                    📐 Attention Matrix Square Deconstruction
                </span>
                <button id="btn-inspector-unfold-modal" class="round-button" style="font-size:10px; padding:3px 10px; background:rgba(168,85,247,0.2); color:#c084fc; border:1px solid #c084fc; cursor:pointer;" title="Open Fullscreen Matrix Deconstruction Engine">
                    ⛶ Full Engine
                </button>
            </div>
            <div style="font-size:10px; color:#cbd5e1; margin-bottom:10px; line-height:1.4;">
                Interactive step-by-step proof: Trace how a 1D sequence unfolds into an \(S \times S\) square (\(Q \cdot K^\top\)), scales by \(1/\sqrt{d_k}\), normalizes via Softmax, and folds back into value context vectors:
            </div>
            <div id="inspector-matrix-unfold-mount"></div>
        </div>

        <!-- 8. Interactive Sample Attention Heatmap -->
        <div style="background:rgba(6,13,24,0.7); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:10px; margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:11px; font-weight:700; color:var(--text-primary); text-transform:uppercase; letter-spacing:0.5px;">🔥 Attention Probability Heatmap</span>
                <span style="font-size:9px; color:var(--text-faint);">Softmax(Q K^T / √d)</span>
            </div>
            <div class="attn-heatmap-grid" style="display:flex; flex-direction:column; gap:4px;">
                ${heatmapRowsHtml}
            </div>
        </div>
    `;

    // Mount Interactive Matrix Unfolding Stepper
    const unfoldMount = container.querySelector('#inspector-matrix-unfold-mount');
    if (unfoldMount) {
        const pKey = variant.id === 'linear_attention' ? 'linear_attention' :
                     variant.id === 'gqa' ? 'gqa' :
                     variant.id === 'mla' ? 'mla' :
                     variant.id === 'sliding_window' ? 'sliding_window' :
                     (moduleMeta && moduleMeta.paradigm === 'svd' ? 'svd_lowrank' : 'dense_mha');

        renderMatrixUnfoldingVisualizer(unfoldMount, {
            activeStep: 2, // Step 3: Opening the S x S Square!
            activeParadigm: pKey,
            causal: variant.causal,
            scaled: true
        });
    }

    const modalBtn = container.querySelector('#btn-inspector-unfold-modal');
    if (modalBtn) modalBtn.onclick = () => openMatrixUnfoldingModal();

    // Render KaTeX for formulas
    if (typeof window !== 'undefined' && window.katex) {
        container.querySelectorAll('.attn-op-math').forEach(el => {
            const raw = el.textContent.replace(/\$\$/g, '').trim();
            if (raw) {
                try {
                    window.katex.render(raw, el, { displayMode: false, throwOnError: false });
                } catch (e) {}
            }
        });
    }
}
