/**
 * interactive/real_attention.js
 * ==============================
 * Real Transformer Attention Weights Client & Interactive Heatmap Engine.
 * Fetches true multi-head attention weights tensor [S × S] from the FastAPI/PyTorch backend,
 * with automated graceful fallback to offline synthetic simulation if backend is offline.
 */

import { escapeHtml } from '../utils/dom.js';

const API_BASE = (typeof window !== 'undefined' && window.A3_API_BASE) || 'http://localhost:8000';

/**
 * Fetches true attention matrix from the PyTorch inference backend.
 * @param {string} text - Prompt sentence
 * @param {string} modelId - Model ID (e.g. 'distilbert-base-uncased')
 * @param {number} layer - 0-indexed layer
 * @param {number} head - 0-indexed head
 * @returns {Promise<Object>} Attention data { tokens, attention, shape, entropy, isReal }
 */
export async function fetchRealAttention(text, modelId = 'distilbert-base-uncased', layer = 0, head = 0) {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 4000); // 4s timeout

        const res = await fetch(`${API_BASE}/attention`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text || 'The quick brown fox jumps over the lazy dog.',
                model_id: modelId,
                layer: layer,
                head: head
            }),
            signal: controller.signal
        });
        clearTimeout(timeoutId);

        if (!res.ok) throw new Error(`API returned HTTP ${res.status}`);
        const data = await res.json();
        return {
            ...data,
            isReal: true,
            source: 'PyTorch Inference Backend (Live)'
        };
    } catch (err) {
        // Graceful offline synthetic fallback
        return generateSyntheticAttentionFallback(text, layer, head, modelId);
    }
}

/**
 * High-fidelity synthetic fallback when backend server is offline or unreachable.
 */
export function generateSyntheticAttentionFallback(text = 'The quick brown fox jumps over the lazy dog', layer = 0, head = 0, modelId = 'distilbert') {
    const rawTokens = text.trim().split(/\s+/).slice(0, 16);
    const tokens = ['[CLS]', ...rawTokens, '[SEP]'];
    const S = tokens.length;
    const attention = [];

    // Synthesize structured attention with positional bias and layer-depth sharpening
    const isCausal = (modelId || '').includes('gpt') || (modelId || '').includes('qwen');
    const sharpness = 1.0 + (layer * 0.35) + (head * 0.1);

    for (let i = 0; i < S; i++) {
        const row = new Array(S).fill(0);
        let rowSum = 0;

        for (let j = 0; j < S; j++) {
            if (isCausal && j > i) {
                row[j] = 0;
                continue;
            }

            let logit = 1.0 / (1.0 + Math.abs(i - j) * 1.2);
            if (j === 0) logit += 0.8; // Attention sink ([CLS])
            if (i === j) logit += 1.5; // Self-attention affinity

            const score = Math.exp(logit * sharpness);
            row[j] = score;
            rowSum += score;
        }

        // Softmax normalization: row sums to 1.0
        for (let j = 0; j < S; j++) {
            row[j] = rowSum > 0 ? row[j] / rowSum : (1.0 / S);
        }
        attention.push(row);
    }

    // Shannon entropy
    let totalEntropy = 0;
    for (let i = 0; i < S; i++) {
        let rowH = 0;
        for (let j = 0; j < S; j++) {
            const p = attention[i][j];
            if (p > 1e-9) rowH -= p * Math.log2(p);
        }
        totalEntropy += rowH;
    }

    return {
        tokens,
        attention,
        shape: [S, S],
        model_id: modelId,
        layer,
        head,
        entropy: round(totalEntropy / S, 4),
        max_weight: round(Math.max(...attention.flat()), 4),
        isReal: false,
        source: 'Offline Synthetic Fallback'
    };
}

/**
 * Renders an interactive, research-grade Attention Heatmap into a DOM container.
 * @param {HTMLElement} container - Target DOM container
 * @param {Object} data - Attention response object
 * @param {Object} options - Interactivity and display callbacks
 */
export function renderRealAttentionHeatmap(container, data, options = {}) {
    if (!container) return;
    const { tokens, attention, isReal, entropy, layer, head, model_id } = data;
    const S = tokens.length;

    const statusBadge = isReal
        ? `<span class="hud-badge" style="background:#10b981; color:#000; font-weight:800;">● LIVE PYTORCH MODEL</span>`
        : `<span class="hud-badge" style="background:rgba(245,158,11,0.2); color:#fbbf24; border:1px solid rgba(245,158,11,0.4);">● OFFLINE SYNTHETIC MODE</span>`;

    container.innerHTML = `
        <div class="real-attn-explorer" style="display:flex; flex-direction:column; gap:12px; font-family:var(--font-mono); font-size:11px;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                <div style="display:flex; align-items:center; gap:8px;">
                    ${statusBadge}
                    <span style="color:var(--text-muted);">${escapeHtml(model_id)} · Layer ${layer} · Head ${head}</span>
                </div>
                <div style="display:flex; align-items:center; gap:12px;">
                    <span style="color:#38bdf8;">Entropy: <b>${entropy.toFixed(3)} bits</b></span>
                    <span style="color:var(--text-muted); font-size:10px;">Grid: ${S} × ${S} tokens</span>
                </div>
            </div>

            <!-- Heatmap Matrix Grid -->
            <div class="attn-matrix-wrapper" style="overflow-x:auto; background:#040911; padding:12px; border-radius:8px; border:1px solid var(--line);">
                <table style="border-collapse:collapse; margin:0 auto; user-select:none;">
                    <thead>
                        <tr>
                            <th style="padding:4px 6px; font-size:9px; color:var(--text-faint); text-align:right;">Q \\ K</th>
                            ${tokens.map(t => `<th style="padding:4px 6px; font-size:9px; color:#38bdf8; max-width:54px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${escapeHtml(t)}">${escapeHtml(t)}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
                        ${attention.map((row, i) => `
                            <tr>
                                <th style="padding:4px 6px; font-size:9px; color:#38bdf8; text-align:right; max-width:54px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${escapeHtml(tokens[i])}">
                                    ${escapeHtml(tokens[i])}
                                </th>
                                ${row.map((weight, j) => {
                                    const intensity = Math.min(1.0, Math.max(0, weight));
                                    // High-definition cyan-to-white color ramp
                                    const alpha = (0.06 + intensity * 0.94).toFixed(3);
                                    const textColor = intensity > 0.45 ? '#03121a' : '#94a3b8';
                                    const bgColor = `rgba(34, 213, 247, ${alpha})`;
                                    return `
                                        <td class="attn-cell"
                                            data-q="${i}"
                                            data-k="${j}"
                                            data-q-token="${escapeHtml(tokens[i])}"
                                            data-k-token="${escapeHtml(tokens[j])}"
                                            data-weight="${weight.toFixed(4)}"
                                            style="width:38px; height:28px; text-align:center; background:${bgColor}; color:${textColor}; font-size:9px; cursor:crosshair; border:1px solid rgba(255,255,255,0.04); transition:transform 0.15s ease;"
                                            title="Q: '${escapeHtml(tokens[i])}' → K: '${escapeHtml(tokens[j])}' | Attention: ${(weight * 100).toFixed(2)}%">
                                            ${weight > 0.08 ? (weight * 100).toFixed(0) + '%' : ''}
                                        </td>
                                    `;
                                }).join('')}
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>

            <!-- Interactive Cell Telemetry HUD -->
            <div id="attn-cell-hud" style="display:flex; justify-content:space-between; align-items:center; padding:8px 12px; background:rgba(8,18,30,0.8); border:1px solid var(--line); border-radius:6px; min-height:34px;">
                <span id="attn-hud-label" style="color:var(--text-muted);">Hover over any matrix cell to inspect directional attention affinity.</span>
                <span id="attn-hud-val" style="color:var(--cyan); font-weight:700;">—</span>
            </div>
        </div>
    `;

    // Hook up cell hover telemetry
    const hudLabel = container.querySelector('#attn-hud-label');
    const hudVal = container.querySelector('#attn-hud-val');
    const cells = container.querySelectorAll('.attn-cell');

    cells.forEach(cell => {
        cell.addEventListener('mouseenter', () => {
            cell.style.transform = 'scale(1.2)';
            cell.style.zIndex = '10';
            const q = cell.dataset.qToken;
            const k = cell.dataset.kToken;
            const w = parseFloat(cell.dataset.weight);
            if (hudLabel) hudLabel.innerHTML = `Query <b style="color:#fff;">"${q}"</b> attends to Key <b style="color:#fff;">"${k}"</b>:`;
            if (hudVal) hudVal.innerHTML = `${(w * 100).toFixed(2)}% <span style="font-size:9px; color:var(--text-muted); font-weight:normal;">(${w.toFixed(4)})</span>`;
        });

        cell.addEventListener('mouseleave', () => {
            cell.style.transform = 'scale(1.0)';
            cell.style.zIndex = '1';
        });

        if (options.onCellClick) {
            cell.addEventListener('click', () => {
                options.onCellClick({
                    queryIndex: parseInt(cell.dataset.q),
                    keyIndex: parseInt(cell.dataset.k),
                    queryToken: cell.dataset.qToken,
                    keyToken: cell.dataset.kToken,
                    weight: parseFloat(cell.dataset.weight)
                });
            });
        }
    });
}

function round(val, dec) {
    const factor = Math.pow(10, dec);
    return Math.round(val * factor) / factor;
}
