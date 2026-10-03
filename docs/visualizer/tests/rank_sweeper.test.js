/**
 * tests/rank_sweeper.test.js
 * ==========================
 * Unit tests for Live Rank Compression Sweeper and Pareto Frontier Engine.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import { RANKS, computeRankMetrics, renderRankSweeper, drawFrontier } from '../src/interactive/rank_sweeper.js';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

describe('Live Rank Sweeper & Pareto Frontier Engine', () => {
    const rankCurvesPath = path.resolve(__dirname, '../src/data/rank_curves.json');
    const rankCurves = JSON.parse(fs.readFileSync(rankCurvesPath, 'utf-8'));

    it('verifies rank_curves.json contains verified compression modules', () => {
        assert.ok(rankCurves.a3, 'rank_curves must contain a3');
        assert.ok(rankCurves.asvd, 'rank_curves must contain asvd');
        assert.strictEqual(rankCurves.a3.paradigm, 'svd');
        assert.strictEqual(rankCurves.a3.layerDims.din, 768);
        assert.strictEqual(rankCurves.a3.layerDims.dout, 768);
    });

    it('computes exact rank metrics exhibiting monotonic parameter scaling with rank', () => {
        const a3 = rankCurves.a3;
        let prevParams = 0;
        let prevSavings = 100;

        RANKS.forEach(r => {
            const m = computeRankMetrics(r, a3);
            assert.strictEqual(m.lowRankParams, r * (768 + 768));
            assert.ok(m.lowRankParams > prevParams, 'Params must strictly increase with rank');
            assert.ok(m.savingsPercent < prevSavings, 'Savings must decrease as rank increases towards dense');
            prevParams = m.lowRankParams;
            prevSavings = m.savingsPercent;
        });

        // Rank 64 should have ~83.33% savings on 768x768 layer
        const r64 = computeRankMetrics(64, a3);
        assert.ok(Math.abs(r64.savingsPercent - 83.33) < 0.5);
        assert.ok(Math.abs(r64.compressionRatio - 6.0) < 0.1);
    });

    it('exhibits monotonic accuracy increase with rank along Pareto frontier', () => {
        const a3 = rankCurves.a3;
        let prevAcc = 0;

        RANKS.forEach(r => {
            const m = computeRankMetrics(r, a3);
            assert.ok(m.accuracy >= prevAcc, `Accuracy at r=${r} (${m.accuracy}%) must be >= r_prev (${prevAcc}%)`);
            prevAcc = m.accuracy;
        });
    });

    it('renders rank sweeper DOM and draws Pareto SVG without throwing', () => {
        const mockSvg = { innerHTML: '' };
        drawFrontier(mockSvg, 64, rankCurves.a3);
        assert.ok(mockSvg.innerHTML.includes('Pareto Frontier Path'));
        assert.ok(mockSvg.innerHTML.includes('r=64'));
        assert.ok(mockSvg.innerHTML.includes('Dense 81.8%'));

        const mockContainer = {
            innerHTML: '',
            querySelector: () => ({
                value: '3',
                addEventListener: () => {},
                textContent: '',
                style: {}
            }),
            querySelectorAll: () => []
        };
        renderRankSweeper(mockContainer, rankCurves.a3);
        assert.ok(mockContainer.innerHTML.includes('rank-sweeper-widget'));
        assert.ok(mockContainer.innerHTML.includes('LIVE RANK SWEEPER'));
    });
});
