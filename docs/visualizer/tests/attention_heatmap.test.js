/**
 * tests/attention_heatmap.test.js
 * ===============================
 * Unit tests for Feature 1.1: Interactive Attention Heatmap Explorer.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import { tokenizeSentence, computeAttention } from '../src/interactive/attention_heatmap.js';

describe('Feature 1.1: Interactive Attention Heatmap Explorer', () => {
    it('tokenizes sentence with [CLS] and [SEP] boundary tokens', () => {
        const tokens = tokenizeSentence('The cat sat on the mat');
        assert.ok(tokens.length >= 7);
        assert.strictEqual(tokens[0], '[CLS]');
        assert.strictEqual(tokens[tokens.length - 1], '[SEP]');
    });

    it('computes exact 4D attention weights tensor [layers, heads, seq, seq]', () => {
        const res = computeAttention('The cat sat on the mat', 6, 12);
        assert.strictEqual(res.shape[0], 6); // 6 Layers
        assert.strictEqual(res.shape[1], 12); // 12 Heads
        const seqLen = res.tokens.length;
        assert.strictEqual(res.shape[2], seqLen);
        assert.strictEqual(res.shape[3], seqLen);

        // Verify softmax row stochasticity (sum of each row equals ~1.0)
        for (let l = 0; l < 6; l++) {
            for (let h = 0; h < 12; h++) {
                for (let i = 0; i < seqLen; i++) {
                    const rowSum = res.weights[l][h][i].reduce((acc, val) => acc + val, 0);
                    assert.ok(Math.abs(rowSum - 1.0) < 1e-4, `Softmax row sum ${rowSum} must be close to 1.0`);
                }
            }
        }
    });

    it('produces positive affinity scores for all token pairs', () => {
        const res = computeAttention('Quick brown fox', 2, 4);
        assert.ok(res.weights[0][0][0][0] > 0);
        assert.ok(res.weights[1][3][1][2] >= 0);
    });
});
