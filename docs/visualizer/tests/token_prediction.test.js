/**
 * tests/token_prediction.test.js
 * ==============================
 * Unit tests for Feature 1.2: Live Token Prediction Playground.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import { predictNextToken, VOCAB_SUBSET } from '../src/interactive/token_prediction.js';

describe('Feature 1.2: Live Token Prediction Playground', () => {
    it('returns top candidate tokens with valid probabilities', () => {
        const predictions = predictNextToken('The deep learning', 1.0, 10);
        assert.ok(predictions.length <= 10);
        assert.ok(predictions.length > 0);

        // Check probability bounds
        predictions.forEach(p => {
            assert.ok(p.probability >= 0 && p.probability <= 1.0);
            assert.ok(typeof p.token === 'string');
            assert.ok(typeof p.logit === 'number');
        });

        // Top candidate has highest probability
        assert.ok(predictions[0].probability >= predictions[predictions.length - 1].probability);
    });

    it('sharpens distribution with low temperature and flattens with high temperature', () => {
        const cold = predictNextToken('The model', 0.1, 10);
        const hot = predictNextToken('The model', 2.0, 10);

        // Under cold temperature, top probability dominates significantly
        assert.ok(cold[0].probability > 0.65, `Cold top probability ${cold[0].probability} should be high`);
        // Under hot temperature, distribution is much more uniform
        assert.ok(hot[0].probability < 0.35, `Hot top probability ${hot[0].probability} should be flatter`);
    });

    it('respects top-K truncation', () => {
        const top3 = predictNextToken('Transformer attention', 1.0, 3);
        assert.strictEqual(top3.length, 3);
    });
});
