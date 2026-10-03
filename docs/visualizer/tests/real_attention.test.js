/**
 * tests/real_attention.test.js
 * ============================
 * Unit tests for Real Attention Weights Client, Synthetic Fallback, and Heatmap Renderer.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import {
    fetchRealAttention,
    generateSyntheticAttentionFallback,
    renderRealAttentionHeatmap
} from '../src/interactive/real_attention.js';

describe('Real Attention Weights & Offline Fallback Engine', () => {
    it('generates high-fidelity synthetic fallback with exact softmax row-sum invariant', () => {
        const fallback = generateSyntheticAttentionFallback('Attention is all you need', 0, 0, 'distilbert');
        assert.ok(fallback.tokens.length >= 6);
        assert.strictEqual(fallback.tokens[0], '[CLS]');
        assert.strictEqual(fallback.tokens[fallback.tokens.length - 1], '[SEP]');
        assert.strictEqual(fallback.isReal, false);

        const S = fallback.tokens.length;
        assert.strictEqual(fallback.shape[0], S);
        assert.strictEqual(fallback.shape[1], S);

        // Verify row-sum invariant: sum(row) == 1.0 (within 1e-4 tolerance)
        for (let i = 0; i < S; i++) {
            const sum = fallback.attention[i].reduce((acc, val) => acc + val, 0);
            assert.ok(Math.abs(sum - 1.0) < 1e-4, `Row ${i} sum must equal 1.0, got ${sum}`);
        }

        // Verify positive Shannon entropy
        assert.ok(fallback.entropy > 0, 'Entropy must be strictly positive');
    });

    it('enforces causal masking in autoregressive models (GPT-2, Qwen)', () => {
        const causalFallback = generateSyntheticAttentionFallback('Language modeling requires causality', 2, 0, 'gpt2');
        const S = causalFallback.tokens.length;

        // In causal attention, future tokens (j > i) must have probability 0
        for (let i = 0; i < S; i++) {
            for (let j = i + 1; j < S; j++) {
                assert.strictEqual(causalFallback.attention[i][j], 0, `Upper triangle cell [${i}, ${j}] must be 0 in causal models`);
            }
        }
    });

    it('gracefully falls back when backend API is offline', async () => {
        const res = await fetchRealAttention('Testing offline resilience');
        assert.ok(res.tokens.length > 0);
        assert.ok(res.attention.length > 0);
        assert.ok(res.entropy >= 0);
        assert.strictEqual(typeof res.isReal, 'boolean');
    });

    it('renders real attention heatmap into mock DOM container without throwing', () => {
        const mockContainer = {
            innerHTML: '',
            querySelector: () => ({ innerHTML: '', style: {} }),
            querySelectorAll: () => []
        };

        const data = generateSyntheticAttentionFallback('Quick test sentence', 1, 3, 'distilbert');
        renderRealAttentionHeatmap(mockContainer, data);

        assert.ok(mockContainer.innerHTML.includes('real-attn-explorer'));
        assert.ok(mockContainer.innerHTML.includes('attn-matrix-wrapper'));
        assert.ok(mockContainer.innerHTML.includes('SYNTHETIC MODE'));
    });
});
