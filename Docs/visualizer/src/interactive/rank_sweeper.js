/**
 * interactive/rank_sweeper.js
 * ============================
 * Research-Grade Live Rank Compression Sweeper & Pareto Frontier Explorer.
 * Allows interactive scrubbing through low-rank dimensions r ∈ [8..256] with
 * real-time parameter reduction, accuracy prediction, and SVG Pareto frontier mapping.
 */

import { escapeHtml } from '../utils/dom.js';

export const RANKS = [8, 16, 32, 64, 128, 192, 256];

/**
 * Computes exact compression metrics for a given rank and layer dimension.
 */
export function computeRankMetrics(rank, moduleData) {
    const din = (moduleData && moduleData.layerDims && moduleData.layerDims.din) || 768;
    const dout = (moduleData && moduleData.layerDims && moduleData.layerDims.dout) || 768;
    const baselineDense = din * dout; // 589,824 for 768x768
    const lowRankParams = rank * (din + dout); // r * 1536
    const savingsPercent = ((baselineDense - lowRankParams) / baselineDense) * 100;
    const compressionRatio = baselineDense / lowRankParams;

    const accCurve = (moduleData && moduleData.accuracyCurve) || {};
    const accuracy = accCurve[rank] ?? (accCurve[String(rank)] ?? 78.5);
    const baselineAcc = (moduleData && moduleData.baselineAccuracy) || 81.80;
    const accDelta = accuracy - baselineAcc;

    return {
        rank,
        din,
        dout,
        baselineDense,
        lowRankParams,
        savingsPercent: Math.max(0, savingsPercent),
        compressionRatio: Math.max(1.0, compressionRatio),
        accuracy,
        baselineAcc,
        accDelta
    };
}

/**
 * Renders the Rank Compression Sweeper HUD into a target container.
 * @param {HTMLElement} container - DOM target container
 * @param {Object} moduleData - Data from rank_curves.json
 * @param {Function} [onRankChange] - Optional callback triggered on slider input
 */
export function renderRankSweeper(container, moduleData, onRankChange = null) {
    if (!container) return;

    const mod = moduleData || {
        name: 'A3 (Adaptive SVD)',
        baselineParams: 66955779,
        baselineAccuracy: 81.80,
        layerDims: { din: 768, dout: 768 },
        accuracyCurve: { 8: 45.2, 16: 58.4, 32: 69.8, 64: 78.4, 128: 80.6, 192: 81.4, 256: 81.75 }
    };

    container.innerHTML = `
        <div class="rank-sweeper-widget" style="display:flex; flex-direction:column; gap:12px; font-family:var(--font-mono); font-size:11px; background:#040911; padding:14px; border:1px solid var(--line); border-radius:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span class="hud-badge" style="background:#22d5f7; color:#03121a; font-weight:800;">LIVE RANK SWEEPER</span>
                <span style="color:var(--text-muted); font-size:10px;">d_in=${mod.layerDims.din} · d_out=${mod.layerDims.dout}</span>
            </div>

            <!-- Slider Control -->
            <div style="display:flex; flex-direction:column; gap:6px;">
                <div style="display:flex; justify-content:space-between; font-size:10px; color:var(--text-muted);">
                    <span>Truncation Rank: <b id="rs-rank-display" style="color:var(--cyan); font-size:13px;">r=64</b></span>
                    <span id="rs-ratio-display" style="color:#10b981; font-weight:700;">6.00× compression</span>
                </div>
                <input type="range" id="rs-slider" min="0" max="${RANKS.length - 1}" value="3" step="1"
                    style="width:100%; cursor:pointer; accent-color:var(--cyan);">
                <div style="display:flex; justify-content:space-between; font-size:8px; color:var(--text-faint);">
                    ${RANKS.map(r => `<span>r=${r}</span>`).join('')}
                </div>
            </div>

            <!-- Readouts Grid -->
            <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:8px;">
                <div style="background:rgba(255,255,255,0.03); padding:8px; border-radius:6px; border:1px solid var(--line);">
                    <small style="display:block; color:var(--text-muted); font-size:8px; text-transform:uppercase;">Layer Parameters</small>
                    <strong id="rs-params-val" style="color:#fff; font-size:11px;">98,304</strong>
                    <span id="rs-savings-val" style="display:block; color:#10b981; font-size:9px;">-83.3% savings</span>
                </div>
                <div style="background:rgba(255,255,255,0.03); padding:8px; border-radius:6px; border:1px solid var(--line);">
                    <small style="display:block; color:var(--text-muted); font-size:8px; text-transform:uppercase;">Downstream Acc</small>
                    <strong id="rs-acc-val" style="color:#38bdf8; font-size:11px;">78.40%</strong>
                    <span id="rs-delta-val" style="display:block; color:#f59e0b; font-size:9px;">-3.40% vs dense</span>
                </div>
                <div style="background:rgba(255,255,255,0.03); padding:8px; border-radius:6px; border:1px solid var(--line);">
                    <small style="display:block; color:var(--text-muted); font-size:8px; text-transform:uppercase;">Dense Baseline</small>
                    <strong style="color:var(--text-muted); font-size:11px;">589,824</strong>
                    <span style="display:block; color:var(--text-muted); font-size:9px;">${mod.baselineAccuracy.toFixed(2)}% acc</span>
                </div>
            </div>

            <!-- SVG Pareto Frontier Chart -->
            <div style="display:flex; flex-direction:column; gap:4px;">
                <span style="font-size:9px; color:var(--text-faint); text-transform:uppercase;">Pareto Frontier (Compression vs Accuracy)</span>
                <svg id="rs-frontier-svg" width="100%" height="150" viewBox="0 0 380 150" style="background:#02060d; border:1px solid rgba(255,255,255,0.06); border-radius:6px; overflow:visible;"></svg>
            </div>
        </div>
    `;

    const slider = container.querySelector('#rs-slider');
    const rankDisplay = container.querySelector('#rs-rank-display');
    const ratioDisplay = container.querySelector('#rs-ratio-display');
    const paramsVal = container.querySelector('#rs-params-val');
    const savingsVal = container.querySelector('#rs-savings-val');
    const accVal = container.querySelector('#rs-acc-val');
    const deltaVal = container.querySelector('#rs-delta-val');
    const svg = container.querySelector('#rs-frontier-svg');

    const update = () => {
        const rankIdx = parseInt(slider.value, 10);
        const rank = RANKS[rankIdx];
        const metrics = computeRankMetrics(rank, mod);

        if (rankDisplay) rankDisplay.textContent = `r=${rank}`;
        if (ratioDisplay) ratioDisplay.textContent = `${metrics.compressionRatio.toFixed(2)}× compression`;
        if (paramsVal) paramsVal.textContent = metrics.lowRankParams.toLocaleString();
        if (savingsVal) savingsVal.textContent = `-${metrics.savingsPercent.toFixed(1)}% savings`;
        if (accVal) accVal.textContent = `${metrics.accuracy.toFixed(2)}%`;
        if (deltaVal) {
            const d = metrics.accDelta;
            deltaVal.textContent = `${d >= 0 ? '+' : ''}${d.toFixed(2)}% vs dense`;
            deltaVal.style.color = d >= -1.5 ? '#10b981' : (d >= -5.0 ? '#f59e0b' : '#ef4444');
        }

        drawFrontier(svg, rank, mod);

        if (onRankChange) {
            onRankChange(rank, metrics);
        }
    };

    slider.addEventListener('input', update);
    update();
}

/**
 * Draws the SVG Pareto Frontier curve with real coordinates.
 */
export function drawFrontier(svg, currentRank, mod) {
    if (!svg) return;

    const width = 380;
    const height = 150;
    const padL = 40;
    const padR = 20;
    const padT = 20;
    const padB = 30;

    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    // Data points: x = compression ratio [1..48], y = accuracy [40..85]
    const pts = RANKS.map(r => {
        const m = computeRankMetrics(r, mod);
        return { rank: r, ratio: m.compressionRatio, acc: m.accuracy };
    });

    const minRatio = 1.0;
    const maxRatio = Math.max(...pts.map(p => p.ratio)) * 1.05;
    const minAcc = 40.0;
    const maxAcc = 85.0;

    const scaleX = (r) => padL + ((r - minRatio) / (maxRatio - minRatio)) * plotW;
    const scaleY = (a) => padT + plotH - ((a - minAcc) / (maxAcc - minAcc)) * plotH;

    // Build SVG path
    const pathD = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(p.ratio).toFixed(1)} ${scaleY(p.acc).toFixed(1)}`).join(' ');

    let svgHtml = `
        <!-- Axes -->
        <line x1="${padL}" y1="${padT + plotH}" x2="${width - padR}" y2="${padT + plotH}" stroke="rgba(255,255,255,0.15)" stroke-width="1" />
        <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}" stroke="rgba(255,255,255,0.15)" stroke-width="1" />

        <!-- Axis Labels -->
        <text x="${padL}" y="${height - 8}" fill="#64748b" font-size="8" font-family="monospace">1×</text>
        <text x="${width - padR - 16}" y="${height - 8}" fill="#64748b" font-size="8" font-family="monospace">${maxRatio.toFixed(0)}×</text>
        <text x="${padL - 32}" y="${padT + plotH}" fill="#64748b" font-size="8" font-family="monospace">40%</text>
        <text x="${padL - 32}" y="${padT + 8}" fill="#64748b" font-size="8" font-family="monospace">85%</text>

        <!-- Dense Baseline Guideline -->
        <line x1="${padL}" y1="${scaleY(mod.baselineAccuracy).toFixed(1)}" x2="${width - padR}" y2="${scaleY(mod.baselineAccuracy).toFixed(1)}"
            stroke="rgba(239,68,68,0.4)" stroke-dasharray="3,3" stroke-width="1" />
        <text x="${width - padR - 55}" y="${(scaleY(mod.baselineAccuracy) - 4).toFixed(1)}" fill="#ef4444" font-size="8" font-family="monospace">Dense 81.8%</text>

        <!-- Pareto Frontier Path -->
        <path d="${pathD}" fill="none" stroke="#22d5f7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
    `;

    // Draw Points
    pts.forEach(p => {
        const cx = scaleX(p.ratio).toFixed(1);
        const cy = scaleY(p.acc).toFixed(1);
        const isCurrent = p.rank === currentRank;

        if (isCurrent) {
            svgHtml += `
                <!-- Operating Point Glow and Marker -->
                <circle cx="${cx}" cy="${cy}" r="8" fill="rgba(34, 213, 247, 0.25)" />
                <circle cx="${cx}" cy="${cy}" r="4.5" fill="#22d5f7" stroke="#fff" stroke-width="1.5" />
                <text x="${cx}" y="${(cy - 10)}" fill="#fff" font-size="9" font-weight="bold" font-family="monospace" text-anchor="middle">r=${p.rank}</text>
            `;
        } else {
            svgHtml += `
                <circle cx="${cx}" cy="${cy}" r="2.5" fill="#0369a1" stroke="#38bdf8" stroke-width="1" />
            `;
        }
    });

    svg.innerHTML = svgHtml;
}
