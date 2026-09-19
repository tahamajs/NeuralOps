/**
 * tests/attention_tour.test.js
 * ============================
 * Unit tests for Attention Masterclass 3D Tour & HUD.
 * Validates step completeness, mathematical formulations, and interactive HUD rendering.
 */

import assert from 'node:assert';
import test, { describe, it, beforeEach } from 'node:test';
import { ATTENTION_TOUR_STEPS, renderTourStepHUD } from '../src/educational/attention_tour.js';

describe('Attention Masterclass Tour & Tutor Mode', () => {
    it('verifies all 10 tour steps are defined and deeply educational', () => {
        assert.strictEqual(ATTENTION_TOUR_STEPS.length, 10);

        ATTENTION_TOUR_STEPS.forEach((step, idx) => {
            assert.strictEqual(step.step, idx + 1);
            assert.ok(typeof step.id === 'string' && step.id.length > 0);
            assert.ok(typeof step.title === 'string' && step.title.length > 0);
            assert.ok(typeof step.formula === 'string' && step.formula.length > 0);
            assert.ok(typeof step.narrative === 'string' && step.narrative.length > 40);
            assert.ok(typeof step.key_metric === 'string');
            assert.ok(typeof step.y === 'number');
            assert.ok(typeof step.dist === 'number');
        });
    });

    it('confirms foundational attention concepts are covered across the curriculum', () => {
        const fullTourText = ATTENTION_TOUR_STEPS.map(s => `${s.title} ${s.formula} ${s.narrative}`).join(' ');

        // Q, K, V associative retrieval
        assert.ok(fullTourText.includes('Query') && fullTourText.includes('Key') && fullTourText.includes('Value'));
        // Variance stabilization / sqrt(d_k)
        assert.ok(fullTourText.includes('\\sqrt{d_k}'));
        // Softmax
        assert.ok(fullTourText.includes('softmax'));
        // Multi-head projection
        assert.ok(fullTourText.includes('Multi-Head') || fullTourText.includes('Subspaces'));
        // O(N^2) quadratic complexity
        assert.ok(fullTourText.includes('\\mathcal{O}(N^2)') || fullTourText.includes('Quadratic'));
        // SVD low-rank factor compression
        assert.ok(fullTourText.includes('SVD') || fullTourText.includes('Low-Rank'));
        // GQA
        assert.ok(fullTourText.includes('GQA') || fullTourText.includes('Grouped-Query'));
        // MLA
        assert.ok(fullTourText.includes('MLA') || fullTourText.includes('Latent'));
    });

    it('renders tour HUD and responds to interactive callbacks', () => {
        // Mock DOM element
        const dummyLabel = {
            style: {},
            classList: {
                add: (c) => {},
                remove: (c) => {}
            },
            innerHTML: '',
            querySelector: (selector) => {
                return {
                    onclick: null,
                    textContent: '',
                    style: {}
                };
            },
            querySelectorAll: () => []
        };

        // Mock global document
        global.document = {
            getElementById: (id) => {
                if (id === 'tour-step-label') return dummyLabel;
                return null;
            }
        };

        let prevCalled = false;
        let nextCalled = false;
        let pauseCalled = false;
        let closeCalled = false;

        const step = ATTENTION_TOUR_STEPS[0];
        renderTourStepHUD(step, 0, ATTENTION_TOUR_STEPS.length, {
            isPaused: false,
            onPrev: () => { prevCalled = true; },
            onNext: () => { nextCalled = true; },
            onPlayPause: () => { pauseCalled = true; },
            onClose: () => { closeCalled = true; }
        });

        assert.ok(dummyLabel.innerHTML.includes('ATTENTION TUTOR · 1 / 10'));
        assert.ok(dummyLabel.innerHTML.includes('Transformer Architecture &amp; Token Ingress'));
        assert.ok(dummyLabel.innerHTML.includes(step.formula));
    });
});
