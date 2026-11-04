/**
 * tests/matrix_unfolding.test.js
 * ==============================
 * Unit tests for Attention Square Deconstruction & Matrix Unfolding Engine.
 * Verifies all 8 expansion stages, mathematical proofs, and paradigm implications.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import {
    MATRIX_UNFOLDING_STEPS,
    PARADIGM_SQUARE_EFFECTS,
    computeDemonstrationMatrices,
    renderMatrixUnfoldingVisualizer
} from '../src/educational/matrix_unfolding.js';

describe('Attention Square Deconstruction & Matrix Unfolding Engine', () => {
    it('verifies all 8 unfolding steps are complete and mathematically formulated', () => {
        assert.strictEqual(MATRIX_UNFOLDING_STEPS.length, 8);

        MATRIX_UNFOLDING_STEPS.forEach((step, idx) => {
            assert.strictEqual(step.step, idx + 1);
            assert.ok(typeof step.id === 'string' && step.id.length > 0);
            assert.ok(typeof step.title === 'string' && step.title.length > 0);
            assert.ok(typeof step.formula === 'string' && step.formula.length > 0);
            assert.ok(typeof step.narrative === 'string' && step.narrative.length > 40);
            assert.ok(typeof step.tensor_dims === 'object');
            assert.ok(typeof step.implication === 'string');
        });
    });

    it('verifies all 6 architectural paradigms have rigorous square implications', () => {
        const expectedParadigms = ['dense_mha', 'svd_lowrank', 'gqa', 'mla', 'linear_attention', 'sliding_window'];

        expectedParadigms.forEach(key => {
            const p = PARADIGM_SQUARE_EFFECTS[key];
            assert.ok(p, `Missing paradigm config for: ${key}`);
            assert.ok(typeof p.name === 'string');
            assert.ok(typeof p.square_mode === 'string');
            assert.ok(typeof p.flops === 'string');
            assert.ok(typeof p.memory === 'string');
            assert.ok(typeof p.description === 'string');
        });
    });

    it('proves numerical matrix scaling, causal masking, and softmax row-sum invariant', () => {
        // Test with causal mask and scaling
        const causalScaled = computeDemonstrationMatrices(true, true);
        assert.strictEqual(causalScaled.S, 4);
        assert.strictEqual(causalScaled.scaleFactor, 0.125); // 1 / sqrt(64)

        // Verify causal masking on upper triangle (j > i)
        for (let i = 0; i < causalScaled.S; i++) {
            for (let j = 0; j < causalScaled.S; j++) {
                if (j > i) {
                    assert.strictEqual(causalScaled.maskedLogits[i][j], -Infinity);
                    assert.strictEqual(causalScaled.probabilities[i][j], 0.0);
                } else {
                    assert.ok(causalScaled.maskedLogits[i][j] !== -Infinity);
                    assert.ok(causalScaled.probabilities[i][j] > 0);
                }
            }
        }

        // Verify each row of probabilities sums to 1.0 (within epsilon)
        causalScaled.probabilities.forEach((row, i) => {
            const rowSum = row.reduce((a, b) => a + b, 0);
            assert.ok(Math.abs(rowSum - 1.0) < 1e-6, `Row ${i} sum must be 1.0, got ${rowSum}`);
        });

        // Test unscaled regime
        const unscaled = computeDemonstrationMatrices(false, false);
        assert.strictEqual(unscaled.scaleFactor, 1.0);
        // Unscaled logits should be 8x larger than scaled logits
        assert.strictEqual(unscaled.rawLogits[0][0], 14.4);
        assert.strictEqual(causalScaled.scaledLogits[0][0], 14.4 * 0.125); // 1.8
    });

    it('renders matrix unfolding visualizer into mock DOM container without throwing', () => {
        const dummyContainer = {
            innerHTML: '',
            querySelector: () => ({
                onclick: null,
                onchange: null
            }),
            querySelectorAll: () => []
        };

        renderMatrixUnfoldingVisualizer(dummyContainer, {
            activeStep: 2,
            activeParadigm: 'svd_lowrank',
            causal: true,
            scaled: true
        });

        assert.ok(dummyContainer.innerHTML.includes('INTERACTIVE MATRIX DECONSTRUCTION'));
        assert.ok(dummyContainer.innerHTML.includes('Step 3: Opening the S × S Interaction Square'));
        assert.ok(dummyContainer.innerHTML.includes('SVD Low-Rank Attention'));
    });
});
