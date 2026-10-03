/**
 * tests/head_specialization.test.js
 * =================================
 * Unit tests for Attention Head Functional Specialization and Entropy Map Engine.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import {
    SPECIALIZATION_TYPES,
    generateHeadSpecializationMatrix,
    renderHeadSpecializationMap
} from '../src/interactive/head_specialization.js';

describe('Attention Head Functional Specialization Engine', () => {
    it('verifies all 4 fundamental head specialization types are defined', () => {
        const types = ['positional', 'syntactic', 'semantic', 'sink'];
        types.forEach(t => {
            assert.ok(SPECIALIZATION_TYPES[t], `Missing specialization type ${t}`);
            assert.ok(SPECIALIZATION_TYPES[t].color);
            assert.ok(SPECIALIZATION_TYPES[t].desc);
        });
    });

    it('generates complete H × L matrix with valid entropy and functional assignments', () => {
        const matrix = generateHeadSpecializationMatrix(6, 12);
        assert.strictEqual(matrix.numLayers, 6);
        assert.strictEqual(matrix.numHeads, 12);
        assert.strictEqual(matrix.totalHeads, 72);

        // Verify type counts sum to total heads
        const sumTypes = Object.values(matrix.typeCounts).reduce((a, b) => a + b, 0);
        assert.strictEqual(sumTypes, 72);

        // Verify all 72 heads have valid layers, heads, entropy, and dominance scores
        matrix.layers.forEach((layer, l) => {
            assert.strictEqual(layer.length, 12);
            layer.forEach((head, h) => {
                assert.strictEqual(head.layer, l);
                assert.strictEqual(head.head, h);
                assert.ok(head.entropy > 0, 'Entropy must be positive');
                assert.ok(SPECIALIZATION_TYPES[head.type], `Invalid head type ${head.type}`);
            });
        });
    });

    it('renders head specialization grid into mock DOM container without throwing', () => {
        const mockContainer = {
            innerHTML: '',
            querySelectorAll: () => [],
            querySelector: () => ({ innerHTML: '', style: {} })
        };

        const data = generateHeadSpecializationMatrix(4, 8);
        renderHeadSpecializationMap(mockContainer, data);

        assert.ok(mockContainer.innerHTML.includes('head-spec-widget'));
        assert.ok(mockContainer.innerHTML.includes('HEAD SPECIALIZATION'));
        assert.ok(mockContainer.innerHTML.includes('32 Heads'));
    });
});
