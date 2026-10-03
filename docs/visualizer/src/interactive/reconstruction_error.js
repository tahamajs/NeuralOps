/**
 * interactive/reconstruction_error.js
 * =====================================
 * Low-Rank Matrix Reconstruction Error & Residual Heatmap Engine.
 * Visualizes the dense weight matrix W, its low-rank SVD reconstruction W_r = U_r · V_r^T,
 * and the exact pointwise residual difference |W - W_r| with divergence color ramps.
 */

import { escapeHtml } from '../utils/dom.js';

/**
 * Synthesizes a structured weight matrix W and its rank-r reconstruction W_r.
 * @param {number} rows - Number of rows (e.g. 12)
 * @param {number} cols - Number of cols (e.g. 12)
 * @param {number} rank - Truncation rank r (e.g. 4)
 * @returns {Object} { W, Wr, diff, frobError }
 */
export function computeMatrixReconstruction(rows = 12, cols = 12, rank = 4) {
    const W = [];
    const r = Math.min(rank, Math.min(rows, cols));

    // Generate correlated latent factors for W
    const U = [];
    const V = [];
    const sigmas = [];

    for (let k = 0; k < Math.min(rows, cols); k++) {
        sigmas.push(Math.exp(-0.2 * k) * 2.0);
    }

    for (let i = 0; i < rows; i++) {
        U[i] = [];
        for (let k = 0; k < sigmas.length; k++) {
            U[i][k] = Math.sin((i + 1) * (k + 1) * 0.45);
        }
    }

    for (let j = 0; j < cols; j++) {
        V[j] = [];
        for (let k = 0; k < sigmas.length; k++) {
            V[j][k] = Math.cos((j + 1) * (k + 1) * 0.35);
        }
    }

    // Dense matrix W = sum_k sigma_k * u_k * v_k^T
    const Wr = [];
    const diff = [];
    let normW = 0;
    let normDiff = 0;

    for (let i = 0; i < rows; i++) {
        W[i] = [];
        Wr[i] = [];
        diff[i] = [];

        for (let j = 0; j < cols; j++) {
            let fullVal = 0;
            let lowRankVal = 0;

            for (let k = 0; k < sigmas.length; k++) {
                const prod = sigmas[k] * U[i][k] * V[j][k];
                fullVal += prod;
                if (k < r) lowRankVal += prod;
            }

            W[i][j] = fullVal;
            Wr[i][j] = lowRankVal;
            const err = Math.abs(fullVal - lowRankVal);
            diff[i][j] = err;

            normW += fullVal * fullVal;
            normDiff += err * err;
        }
    }

    const frobError = normW > 0 ? Math.sqrt(normDiff) / Math.sqrt(normW) : 0;

    return {
        rows,
        cols,
        rank: r,
        W,
        Wr,
        diff,
        frobError: Math.round(frobError * 10000) / 10000
    };
}

/**
 * Renders the Matrix Reconstruction Error Heatmap with diff toggle into DOM container.
 * @param {HTMLElement} container - DOM target container
 * @param {Object} reconData - Object returned by computeMatrixReconstruction
 * @param {Object} [options] - Options and mode defaults
 */
export function renderErrorHeatmap(container, reconData, options = {}) {
    if (!container) return;

    const data = reconData || computeMatrixReconstruction(12, 12, 4);
    let activeMode = options.defaultMode || 'diff'; // 'w', 'wr', 'diff'

    const renderGrid = () => {
        const matrix = activeMode === 'w' ? data.W : (activeMode === 'wr' ? data.Wr : data.diff);
        const maxVal = Math.max(1e-5, Math.max(...matrix.flat().map(v => Math.abs(v))));

        let modeDesc = 'Showing Absolute Pointwise Residual: |W - W_r|';
        if (activeMode === 'w') modeDesc = 'Showing Original Full-Rank Dense Matrix: W';
        if (activeMode === 'wr') modeDesc = `Showing Rank-${data.rank} SVD Low-Rank Approximation: W_r`;

        container.innerHTML = `
            <div class="recon-error-widget" style="display:flex; flex-direction:column; gap:12px; font-family:var(--font-mono); font-size:11px; background:#040911; padding:14px; border:1px solid var(--line); border-radius:10px;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <span class="hud-badge" style="background:#f43f5e; color:#fff; font-weight:800;">RECONSTRUCTION ERROR</span>
                        <span style="color:var(--text-muted); font-size:10px;">${data.rows} × ${data.cols} · Rank r=${data.rank}</span>
                    </div>
                    <span style="color:#38bdf8;">Frobenius Residual: <b>${(data.frobError * 100).toFixed(2)}%</b></span>
                </div>

                <!-- Mode Selector Radio Pills -->
                <div style="display:flex; gap:6px;">
                    <button class="diff-mode-btn ${activeMode === 'diff' ? 'active' : ''}" data-mode="diff"
                        style="padding:4px 10px; border-radius:6px; font-size:10px; cursor:pointer; font-family:var(--font-mono); border:1px solid ${activeMode === 'diff' ? '#f43f5e' : 'var(--line)'}; background:${activeMode === 'diff' ? 'rgba(244,63,94,0.2)' : 'transparent'}; color:${activeMode === 'diff' ? '#f43f5e' : 'var(--text-muted)'};">
                        |W - W_r| (Error Diff)
                    </button>
                    <button class="diff-mode-btn ${activeMode === 'w' ? 'active' : ''}" data-mode="w"
                        style="padding:4px 10px; border-radius:6px; font-size:10px; cursor:pointer; font-family:var(--font-mono); border:1px solid ${activeMode === 'w' ? '#38bdf8' : 'var(--line)'}; background:${activeMode === 'w' ? 'rgba(56,189,248,0.2)' : 'transparent'}; color:${activeMode === 'w' ? '#38bdf8' : 'var(--text-muted)'};">
                        W (Dense Baseline)
                    </button>
                    <button class="diff-mode-btn ${activeMode === 'wr' ? 'active' : ''}" data-mode="wr"
                        style="padding:4px 10px; border-radius:6px; font-size:10px; cursor:pointer; font-family:var(--font-mono); border:1px solid ${activeMode === 'wr' ? '#22d5f7' : 'var(--line)'}; background:${activeMode === 'wr' ? 'rgba(34,213,247,0.2)' : 'transparent'}; color:${activeMode === 'wr' ? '#22d5f7' : 'var(--text-muted)'};">
                        W_r (Rank-${data.rank})
                    </button>
                </div>

                <div style="font-size:9px; color:var(--text-faint);">${modeDesc}</div>

                <!-- Matrix Heatmap Table -->
                <div style="overflow-x:auto; background:#02060d; padding:10px; border-radius:6px; border:1px solid var(--line);">
                    <table style="border-collapse:collapse; margin:0 auto; user-select:none;">
                        <tbody>
                            ${matrix.map((row, i) => `
                                <tr>
                                    ${row.map((val, j) => {
                                        const normVal = Math.min(1.0, Math.abs(val) / maxVal);
                                        let bg = 'rgba(255,255,255,0.05)';
                                        if (activeMode === 'diff') {
                                            // Red heat ramp for error
                                            bg = `rgba(244, 63, 94, ${(0.05 + normVal * 0.95).toFixed(3)})`;
                                        } else {
                                            // Divergent blue/cyan ramp
                                            const c = val >= 0 ? '34, 213, 247' : '129, 140, 248';
                                            bg = `rgba(${c}, ${(0.05 + normVal * 0.95).toFixed(3)})`;
                                        }
                                        return `
                                            <td class="recon-cell"
                                                data-row="${i}" data-col="${j}"
                                                data-w="${data.W[i][j].toFixed(4)}"
                                                data-wr="${data.Wr[i][j].toFixed(4)}"
                                                data-diff="${data.diff[i][j].toFixed(4)}"
                                                style="width:24px; height:22px; text-align:center; background:${bg}; border:1px solid rgba(255,255,255,0.04); cursor:crosshair; transition:transform 0.1s;"
                                                title="Cell [${i},${j}]: W=${data.W[i][j].toFixed(3)} | Wr=${data.Wr[i][j].toFixed(3)} | Diff=${data.diff[i][j].toFixed(4)}">
                                            </td>
                                        `;
                                    }).join('')}
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>

                <!-- Hover Telemetry Readout -->
                <div id="recon-hud" style="display:flex; justify-content:space-between; align-items:center; padding:6px 10px; background:rgba(8,18,30,0.8); border:1px solid var(--line); border-radius:6px; font-size:10px;">
                    <span id="recon-hud-cell" style="color:var(--text-muted);">Hover cell to inspect pointwise fidelity.</span>
                    <span id="recon-hud-vals" style="color:#fff;">—</span>
                </div>
            </div>
        `;

        // Wire mode toggle buttons
        container.querySelectorAll('.diff-mode-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                activeMode = btn.dataset.mode;
                renderGrid();
            });
        });

        // Wire hover tooltips
        const hudCell = container.querySelector('#recon-hud-cell');
        const hudVals = container.querySelector('#recon-hud-vals');
        container.querySelectorAll('.recon-cell').forEach(cell => {
            cell.addEventListener('mouseenter', () => {
                cell.style.transform = 'scale(1.3)';
                cell.style.zIndex = '5';
                const r = cell.dataset.row;
                const c = cell.dataset.col;
                const w = cell.dataset.w;
                const wr = cell.dataset.wr;
                const df = cell.dataset.diff;
                if (hudCell) hudCell.innerHTML = `Cell [${r}, ${c}]:`;
                if (hudVals) hudVals.innerHTML = `W=<b style="color:#38bdf8;">${w}</b> · W_r=<b style="color:#22d5f7;">${wr}</b> · |Diff|=<b style="color:#f43f5e;">${df}</b>`;
            });
            cell.addEventListener('mouseleave', () => {
                cell.style.transform = 'scale(1.0)';
                cell.style.zIndex = '1';
            });
        });
    };

    renderGrid();
}
