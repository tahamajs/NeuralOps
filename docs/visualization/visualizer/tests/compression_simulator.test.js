/**
 * tests/compression_simulator.test.js
 * ===================================
 * Unit tests for Feature 1.3: SVD Compression Simulator.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import { calculateSVDMetrics, generateMatrixApproximation } from '../src/interactive/compression_simulator.js';

describe('Feature 1.3: SVD Compression Simulator', () => {
    it('computes exact parameter counts, compression ratio, and memory savings', () => {
        // Standard BERT/GPT-2: hiddenDim = 768, numLayers = 12, numProjections = 4
        const metrics = calculateSVDMetrics(64, 768, 12, 4);

        assert.strictEqual(metrics.singleBaselineParams, 768 * 768); // 589,824
        assert.strictEqual(metrics.singleCompressedParams, 2 * 64 * 768); // 98,304

        // Compression ratio: 2 * 64 / 768 = 128 / 768 = 1/6 ~ 0.16667
        assert.ok(Math.abs(metrics.compressionRatio - (128 / 768)) < 1e-5);
        assert.ok(metrics.reductionPercent > 83.0 && metrics.reductionPercent < 84.0);

        // Memory savings positive
        assert.ok(metrics.memorySavings.fp32MB > 0);
        assert.ok(metrics.memorySavings.fp16MB > 0);
        assert.ok(metrics.memorySavings.int8MB > 0);
        assert.strictEqual(metrics.memorySavings.fp32MB, metrics.memorySavings.fp16MB * 2);
    });

    it('exhibits monotonic quality score increase and speedup scaling with rank', () => {
        const lowRank = calculateSVDMetrics(16, 768, 12, 4);
        const midRank = calculateSVDMetrics(64, 768, 12, 4);
        const highRank = calculateSVDMetrics(256, 768, 12, 4);

        // Quality score should increase monotonically with rank
        assert.ok(lowRank.qualityScore < midRank.qualityScore);
        assert.ok(midRank.qualityScore < highRank.qualityScore);

        // Speedup should be higher for lower ranks (greater compression)
        assert.ok(lowRank.estimatedSpeedup > midRank.estimatedSpeedup);
        assert.ok(midRank.estimatedSpeedup > highRank.estimatedSpeedup);
        assert.ok(highRank.estimatedSpeedup >= 1.0);
    });

    it('generates low-rank matrix approximation with decreasing Frobenius residual error', () => {
        const approxR1 = generateMatrixApproximation(16, 1);
        const approxR4 = generateMatrixApproximation(16, 4);
        const approxR12 = generateMatrixApproximation(16, 12);
        const approxR16 = generateMatrixApproximation(16, 16);

        // Dimensions verification
        assert.strictEqual(approxR4.fullMatrix.length, 16);
        assert.strictEqual(approxR4.approxMatrix.length, 16);
        assert.strictEqual(approxR4.errorMatrix.length, 16);

        // Residual error must strictly decrease as rank increases
        assert.ok(approxR1.relativeFrobeniusError > approxR4.relativeFrobeniusError);
        assert.ok(approxR4.relativeFrobeniusError > approxR12.relativeFrobeniusError);
        assert.ok(approxR12.relativeFrobeniusError > approxR16.relativeFrobeniusError);
        assert.ok(approxR16.relativeFrobeniusError < 1e-4);
    });
});
