/**
 * tests/constants.test.js
 * =======================
 * Validates configuration constants, model specs, and paradigm definitions.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import { VISUAL_CONFIG, MODEL_SPECS, VOCAB_SIZES, PARADIGMS, TOUR_STEPS } from '../src/constants.js';

describe('Constants and Specifications Verification', () => {
    it('verifies VISUAL_CONFIG token physics and layout settings', () => {
        assert.strictEqual(VISUAL_CONFIG.TOKENS.count, 360);
        assert.strictEqual(VISUAL_CONFIG.LAYOUT.baselineX, -4.5);
        assert.strictEqual(VISUAL_CONFIG.LAYOUT.compressedX, 4.5);
        assert.ok(VISUAL_CONFIG.ZOOM.macroThreshold > VISUAL_CONFIG.ZOOM.transitionStart);
        assert.ok(VISUAL_CONFIG.ZOOM.transitionStart > VISUAL_CONFIG.ZOOM.microThreshold);
    });

    it('verifies 4 foundational model specs', () => {
        const models = ['distilbert', 'roberta', 'gpt2', 'qwen'];
        for (const m of models) {
            assert.ok(MODEL_SPECS[m], `Missing model spec for ${m}`);
            assert.ok(MODEL_SPECS[m].num_layers > 0);
            assert.ok(MODEL_SPECS[m].hidden_size > 0);
            assert.ok(MODEL_SPECS[m].vocab_size > 0);
        }
        assert.strictEqual(MODEL_SPECS.distilbert.num_layers, 6);
        assert.strictEqual(MODEL_SPECS.roberta.num_layers, 12);
        assert.strictEqual(MODEL_SPECS.gpt2.num_layers, 12);
        assert.strictEqual(MODEL_SPECS.qwen.num_layers, 24);
    });

    it('verifies vocab sizes alignment', () => {
        assert.strictEqual(VOCAB_SIZES.distilbert, 30522);
        assert.strictEqual(VOCAB_SIZES.roberta, 50265);
        assert.strictEqual(VOCAB_SIZES.gpt2, 50257);
        assert.strictEqual(VOCAB_SIZES.qwen, 151936);
    });

    it('verifies 5 primary compression paradigms', () => {
        const expected = ['svd', 'moe', 'mla', 'slicegpt', 'quant'];
        for (const p of expected) {
            assert.ok(PARADIGMS[p], `Missing paradigm: ${p}`);
            assert.ok(PARADIGMS[p].math, `Missing math for paradigm: ${p}`);
        }
    });

    it('verifies tour steps count', () => {
        assert.strictEqual(TOUR_STEPS.length, 8);
    });
});
