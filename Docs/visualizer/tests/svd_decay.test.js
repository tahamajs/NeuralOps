/**
 * tests/svd_decay.test.js
 * =======================
 * Unit tests for Singular Value Decay Spectrum and Energy Preservation Engine.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import {
    generateSingularValues,
    computeSpectralEnergy,
    renderSVDDecay
} from '../src/interactive/svd_decay.js';

describe('Singular Value Decay Spectrum & Energy Engine', () => {
    it('generates strictly positive, monotonically descending singular values', () => {
        const sigmas = generateSingularValues(64, 0.05);
        assert.strictEqual(sigmas.length, 64);

        for (let i = 0; i < sigmas.length; i++) {
            assert.ok(sigmas[i] > 0, `Singular value ${i} must be strictly positive`);
            if (i > 0) {
                assert.ok(sigmas[i] <= sigmas[i - 1], `Spectrum must be monotonically descending: sigma[${i}] <= sigma[${i-1}]`);
            }
        }
    });

    it('proves monotonic increase of retained energy with rank', () => {
        const sigmas = generateSingularValues(64);
        let prevEnergy = 0;

        for (let r = 1; r <= 64; r += 8) {
            const energy = computeSpectralEnergy(sigmas, r);
            assert.ok(energy >= prevEnergy, `Energy at rank ${r} (${energy}) must exceed rank ${r-8} (${prevEnergy})`);
            assert.ok(energy <= 1.0, 'Energy fraction cannot exceed 1.0');
            prevEnergy = energy;
        }

        // Full rank must have 100% energy
        const fullEnergy = computeSpectralEnergy(sigmas, 64);
        assert.ok(Math.abs(fullEnergy - 1.0) < 1e-6);
    });

    it('renders SVD decay SVG chart into mock DOM container without throwing', () => {
        const mockContainer = { innerHTML: '' };
        const sigmas = generateSingularValues(32);
        renderSVDDecay(mockContainer, sigmas, 16);

        assert.ok(mockContainer.innerHTML.includes('svd-decay-widget'));
        assert.ok(mockContainer.innerHTML.includes('SINGULAR VALUE SPECTRUM'));
        assert.ok(mockContainer.innerHTML.includes('r=16'));
        assert.ok(mockContainer.innerHTML.includes('Energy Retained:'));
    });
});
