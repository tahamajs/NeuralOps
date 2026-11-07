/**
 * interactive/svd_decay.js
 * =========================
 * Singular Value Spectrum Decay Visualizer & Low-Rank Energy Preservation Engine.
 * Visually charts the singular values σ_i of weight matrices in log10 space,
 * dynamic rank cutoff thresholds, and cumulative Frobenius energy retention.
 */

import { escapeHtml } from '../utils/dom.js';

/**
 * Generates synthetic or real singular value spectrum with exponential decay.
 * @param {number} dim - Matrix dimension (e.g. 768)
 * @param {number} [decayRate=0.035] - Decay sharpness
 * @returns {number[]} Sorted singular values (descending)
 */
export function generateSingularValues(dim = 64, decayRate = 0.045) {
    const values = [];
    for (let i = 0; i < dim; i++) {
        // Power-law plus exponential decay mimicking empirical transformer weight spectra
        const v = Math.exp(-decayRate * i) * Math.pow(i + 1, -0.4);
        values.push(Math.max(1e-6, v));
    }
    return values;
}

/**
 * Computes energy preservation ratio (Frobenius norm retention) at rank r.
 */
export function computeSpectralEnergy(singularValues, r) {
    const totalEnergy = singularValues.reduce((acc, v) => acc + v * v, 0);
    const retained = singularValues.slice(0, r).reduce((acc, v) => acc + v * v, 0);
    return totalEnergy > 0 ? (retained / totalEnergy) : 0;
}

/**
 * Renders the Singular Value Decay SVG chart and interactive inspector into a DOM container.
 * @param {HTMLElement} container - DOM target container
 * @param {number[]} singularValues - Sorted array of singular values
 * @param {number} currentRank - Active low-rank cutoff r
 * @param {Object} [options] - Callbacks and customization
 */
export function renderSVDDecay(container, singularValues, currentRank = 32, options = {}) {
    if (!container) return;

    const values = singularValues || generateSingularValues(64);
    const N = values.length;
    const r = Math.min(N, Math.max(1, currentRank));

    const energy = computeSpectralEnergy(values, r);
    const energyPct = (energy * 100).toFixed(2);

    const W = 420;
    const H = 180;
    const padL = 40;
    const padR = 20;
    const padT = 20;
    const padB = 30;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const logVals = values.map(v => Math.log10(v));
    const yMin = Math.min(...logVals) * 1.05;
    const yMax = Math.max(...logVals) * 1.05;

    const scaleX = (idx) => padL + (idx / (N - 1)) * plotW;
    const scaleY = (logV) => padT + plotH - ((logV - yMin) / (yMax - yMin)) * plotH;

    const cutoffX = scaleX(r - 1);

    // Spectrum curve path
    const pathD = values.map((v, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(i).toFixed(1)} ${scaleY(logVals[i]).toFixed(1)}`).join(' ');

    // Area under retained spectrum
    const retainedAreaD = `${pathD.split(' ').slice(0, r * 2).join(' ')} L ${cutoffX.toFixed(1)} ${(padT + plotH).toFixed(1)} L ${padL} ${(padT + plotH).toFixed(1)} Z`;

    container.innerHTML = `
        <div class="svd-decay-widget" style="display:flex; flex-direction:column; gap:10px; font-family:var(--font-mono); font-size:11px; background:#040911; padding:14px; border:1px solid var(--line); border-radius:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="hud-badge" style="background:#a855f7; color:#fff; font-weight:800;">SINGULAR VALUE SPECTRUM</span>
                <span style="color:#22d5f7;">Energy Retained: <b>${energyPct}%</b></span>
            </div>

            <!-- SVG Decay Chart -->
            <svg id="svd-decay-svg" width="100%" height="180" viewBox="0 0 ${W} ${H}" style="background:#02060d; border:1px solid rgba(255,255,255,0.06); border-radius:6px; overflow:visible;">
                <!-- Axes -->
                <line x1="${padL}" y1="${padT + plotH}" x2="${W - padR}" y2="${padT + plotH}" stroke="rgba(255,255,255,0.15)" stroke-width="1" />
                <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}" stroke="rgba(255,255,255,0.15)" stroke-width="1" />

                <!-- Retained Area Gradient Fill -->
                <defs>
                    <linearGradient id="retainedGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stop-color="#a855f7" stop-opacity="0.4" />
                        <stop offset="100%" stop-color="#a855f7" stop-opacity="0.02" />
                    </linearGradient>
                </defs>
                <path d="${retainedAreaD}" fill="url(#retainedGrad)" />

                <!-- Full Spectrum Path -->
                <path d="${pathD}" fill="none" stroke="#64748b" stroke-width="1.5" stroke-dasharray="3,3" />

                <!-- Retained Spectrum Segment -->
                <path d="${values.slice(0, r).map((v, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(i).toFixed(1)} ${scaleY(logVals[i]).toFixed(1)}`).join(' ')}"
                    fill="none" stroke="#c084fc" stroke-width="2.5" stroke-linecap="round" />

                <!-- Cutoff Line at Rank r -->
                <line x1="${cutoffX}" y1="${padT}" x2="${cutoffX}" y2="${padT + plotH}" stroke="#22d5f7" stroke-width="2" stroke-dasharray="4,2">
                    <animate attributeName="opacity" values="0.7;1.0;0.7" dur="2s" repeatCount="indefinite" />
                </line>
                <text x="${cutoffX + 4}" y="${padT + 12}" fill="#22d5f7" font-size="9" font-weight="bold">r=${r}</text>

                <!-- Axis Labels -->
                <text x="${padL}" y="${H - 8}" fill="#64748b" font-size="8">i=1</text>
                <text x="${W - padR - 20}" y="${H - 8}" fill="#64748b" font-size="8">i=${N}</text>
                <text x="${padL - 32}" y="${padT + 10}" fill="#64748b" font-size="8">log σ</text>
            </svg>

            <!-- Telemetry Footer -->
            <div style="display:flex; justify-content:space-between; align-items:center; font-size:10px; color:var(--text-muted);">
                <span>First Value: <b style="color:#fff;">σ₁ = ${values[0].toFixed(3)}</b></span>
                <span>Cutoff: <b style="color:#22d5f7;">σ_${r} = ${values[r - 1].toFixed(4)}</b></span>
                <span>Tail: <b style="color:#64748b;">σ_${N} = ${values[N - 1].toFixed(5)}</b></span>
            </div>
        </div>
    `;
}
