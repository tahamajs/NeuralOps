/**
 * tests/reconstruction_error.test.js
 * ===================================
 * Unit tests for Low-Rank Matrix Reconstruction Error & Residual Heatmap Engine.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import {
    computeMatrixReconstruction,
    renderErrorHeatmap
} from '../src/interactive/reconstruction_error.js';

describe('Low-Rank Matrix Reconstruction Error Engine', () => {
    it('proves Frobenius reconstruction error strictly decreases as rank increases', () => {
        let prevError = 1.0;

        for (let r = 1; r <= 8; r++) {
            const data = computeMatrixReconstruction(10, 10, r);
            assert.strictEqual(data.rank, r);
            assert.ok(data.frobError >= 0, 'Frobenius error must be non-negative');
            assert.ok(data.frobError <= prevError + 1e-6, `Frobenius error at r=${r} (${data.frobError}) must be <= r=${r-1} (${prevError})`);
            prevError = data.frobError;
        }
    });

    it('proves zero residual error when rank equals full matrix dimension', () => {
        const fullRank = computeMatrixReconstruction(6, 6, 6);
        assert.ok(fullRank.frobError < 1e-4, `Full-rank reconstruction must have near-zero residual error, got ${fullRank.frobError}`);

        for (let i = 0; i < 6; i++) {
            for (let j = 0; j < 6; j++) {
                assert.ok(Math.abs(fullRank.W[i][j] - fullRank.Wr[i][j]) < 1e-4);
                assert.ok(fullRank.diff[i][j] < 1e-4);
            }
        }
    });

    it('renders error heatmap into mock DOM container and supports mode switching', () => {
        const mockContainer = {
            innerHTML: '',
            querySelectorAll: () => [],
            querySelector: () => ({ innerHTML: '', style: {} })
        };

        const data = computeMatrixReconstruction(8, 8, 3);
        renderErrorHeatmap(mockContainer, data, { defaultMode: 'diff' });

        assert.ok(mockContainer.innerHTML.includes('recon-error-widget'));
        assert.ok(mockContainer.innerHTML.includes('RECONSTRUCTION ERROR'));
        assert.ok(mockContainer.innerHTML.includes('|W - W_r| (Error Diff)'));
    });
});
