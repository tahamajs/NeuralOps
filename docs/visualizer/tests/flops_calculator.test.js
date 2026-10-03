/**
 * tests/flops_calculator.test.js
 * ==============================
 * Unit tests for Analytical FLOPs & Complexity Calculator.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import { computeFLOPs } from '../src/interactive/flops_calculator.js';

describe('Analytical FLOPs & Computational Complexity Calculator', () => {
    it('computes exact FLOP scaling with sequence length and model depth', () => {
        const arch = { numLayers: 12, hiddenDim: 768, numHeads: 12 };
        const flops512 = computeFLOPs(arch, 512, 64);
        const flops1024 = computeFLOPs(arch, 1024, 64);

        assert.strictEqual(flops512.numLayers, 12);
        assert.strictEqual(flops512.hiddenDim, 768);
        assert.ok(flops512.denseModelTotal > 0);

        // Sequence length doubled: pairwise attention FLOPs should scale 4x (quadratic S^2)
        const pairwise512 = flops512.pairwiseAttnFLOPs;
        const pairwise1024 = flops1024.pairwiseAttnFLOPs;
        assert.strictEqual(pairwise1024, pairwise512 * 4);
    });

    it('proves non-negative FLOP savings and speedup under low-rank SVD compression', () => {
        const arch = { numLayers: 12, hiddenDim: 768, numHeads: 12 };
        const res = computeFLOPs(arch, 512, 64);

        assert.ok(res.savedFLOPs > 0, 'FLOP savings must be positive');
        assert.ok(res.compressedModelTotal < res.denseModelTotal);
        assert.ok(res.speedup > 1.0, 'Speedup must be greater than 1.0x');
        assert.ok(res.compressedAttnProjections < res.denseAttnProjections);
    });
});
