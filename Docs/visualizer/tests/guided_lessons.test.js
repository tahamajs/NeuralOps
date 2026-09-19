/**
 * tests/guided_lessons.test.js
 * =============================
 * Unit tests for Feature 1.7: Guided Lessons Masterclass.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import { GUIDED_LESSONS, computeQuadraticStats } from '../src/educational/guided_lessons.js';

describe('Feature 1.7: Guided Lessons Mode', () => {
    it('verifies all 5 guided lessons are complete and structurally rigorous', () => {
        assert.strictEqual(GUIDED_LESSONS.length, 5);

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
