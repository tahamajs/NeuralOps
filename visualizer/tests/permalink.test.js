/**
 * tests/permalink.test.js
 * =======================
 * Unit tests for Feature 1.5: Shareable Permalink per Module.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import { parseHashToState, serializeStateToHash } from '../src/interactions/permalink.js';

describe('Feature 1.5: Shareable Permalink per Module', () => {
    it('serializes state into URL hash string', () => {
        const state = {
            model: 'roberta',
            module: 'svd_linear_attention',
            view: '3d',
            layer: 3,
            rank: 64,
            inspector: true
        };

        const hash = serializeStateToHash(state);
        assert.ok(hash.startsWith('#'));
        assert.ok(hash.includes('model=roberta'));
        assert.ok(hash.includes('module=svd_linear_attention'));
        assert.ok(hash.includes('view=3d'));
        assert.ok(hash.includes('layer=3'));
        assert.ok(hash.includes('rank=64'));
        assert.ok(hash.includes('inspector=1'));
    });

    it('parses URL hash back into sanitized state object', () => {
        const hash = '#model=gpt2&module=dense_mha&view=2d&layer=5&rank=128&inspector=1';
        const parsed = parseHashToState(hash);

        assert.strictEqual(parsed.model, 'gpt2');
        assert.strictEqual(parsed.module, 'dense_mha');
        assert.strictEqual(parsed.view, '2d');
        assert.strictEqual(parsed.layer, 5);
        assert.strictEqual(parsed.rank, 128);
        assert.strictEqual(parsed.inspector, true);
    });

    it('filters out invalid model and view parameters', () => {
        const invalidHash = '#model=unsupported_net&view=4d_hyperspace&layer=999';
        const parsed = parseHashToState(invalidHash);

        assert.strictEqual(parsed.model, undefined);
        assert.strictEqual(parsed.view, undefined);
        assert.strictEqual(parsed.layer, undefined);
    });

    it('performs accurate round-trip state preservation', () => {
        const original = {
            model: 'qwen',
            module: 'moe_sparse',
            view: '3d',
            layer: 7
        };

        const hash = serializeStateToHash(original);
        const reconstructed = parseHashToState(hash);

        assert.strictEqual(reconstructed.model, original.model);
        assert.strictEqual(reconstructed.module, original.module);
        assert.strictEqual(reconstructed.view, original.view);
        assert.strictEqual(reconstructed.layer, original.layer);
    });
});
