/**
 * tests/guided_lessons.test.js
 * =============================
 * Unit tests for Feature 1.7: Guided Lessons Masterclass.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import { GUIDED_LESSONS, computeQuadraticStats, computeAttentionScalingDemo } from '../src/educational/guided_lessons.js';

describe('Feature 1.7: Guided Lessons Mode', () => {
    it('verifies all 6 guided lessons are complete and structurally rigorous', () => {
        assert.strictEqual(GUIDED_LESSONS.length, 6);

        GUIDED_LESSONS.forEach((lesson, idx) => {
            assert.strictEqual(lesson.number, idx + 1);
            assert.ok(typeof lesson.id === 'string');
            assert.ok(typeof lesson.title === 'string');
            assert.ok(typeof lesson.badge === 'string');
            assert.ok(typeof lesson.componentTarget === 'string');
            assert.ok(lesson.cameraTargetY > 0);
            assert.ok(lesson.steps.length >= 2);

            lesson.steps.forEach(s => {
                assert.ok(typeof s.title === 'string');
                assert.ok(typeof s.explanation === 'string');
                assert.ok(s.explanation.length > 20);
            });
        });
    });

    it('proves mathematical necessity of 1/sqrt(dk) variance stabilization in attention', () => {
        const scaled = computeAttentionScalingDemo(64, true);
        const unscaled = computeAttentionScalingDemo(64, false);

        assert.strictEqual(scaled.scaled, true);
        assert.strictEqual(unscaled.scaled, false);

        // Scaled by 1/sqrt(64) = 1/8
        assert.strictEqual(scaled.scaleFactor, 0.125);
        assert.strictEqual(unscaled.scaleFactor, 1.0);

        // Sum of probabilities must sum to 1.0 in both cases
        const sumScaled = scaled.probs.reduce((a, b) => a + b, 0);
        const sumUnscaled = unscaled.probs.reduce((a, b) => a + b, 0);
        assert.ok(Math.abs(sumScaled - 1.0) < 1e-6);
        assert.ok(Math.abs(sumUnscaled - 1.0) < 1e-6);

        // Unscaled logits cause severe gradient saturation (max probability near 1.0, grad near 0)
        assert.ok(unscaled.gradMagnitude < 0.01, 'Unscaled attention should exhibit vanishing gradients');
        assert.ok(scaled.gradMagnitude > 0.10, 'Scaled attention should maintain healthy backprop gradient flow');
    });

    it('computes exact quadratic attention scaling metrics and explosion ratios', () => {
        const stats512 = computeQuadraticStats(512, 12, 12);
        assert.strictEqual(stats512.seqLen, 512);
        assert.strictEqual(stats512.matrixEntriesPerHead, 512 * 512);
        assert.strictEqual(stats512.ratioTo512, 1.0);

        const stats2048 = computeQuadraticStats(2048, 12, 12);
        assert.strictEqual(stats2048.matrixEntriesPerHead, 2048 * 2048);
        assert.strictEqual(stats2048.ratioTo512, 16.0); // (2048/512)^2 = 4^2 = 16

        const stats8192 = computeQuadraticStats(8192, 12, 12);
        assert.strictEqual(stats8192.matrixEntriesPerHead, 8192 * 8192);
        assert.strictEqual(stats8192.ratioTo512, 256.0); // (8192/512)^2 = 16^2 = 256
        assert.ok(stats8192.memoryMB > 18000.0); // > 18 GB across 12 layers and 12 heads
    });
});
