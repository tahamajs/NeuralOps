/**
 * tests/keyboard_navigation.test.js
 * =================================
 * Unit tests for Accessibility and Keyboard Navigation Engine.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import {
    announceLive,
    setupKeyboardNavigation,
    setActiveLayer,
    getActiveLayer
} from '../src/ui/keyboard.js';

describe('Accessibility & Keyboard Navigation Engine', () => {
    it('sets and clamps active layer index accurately', () => {
        setActiveLayer(5, 12);
        assert.strictEqual(getActiveLayer(), 5);
    });

    it('creates ARIA live status announcer in mock DOM', () => {
        let createdEl = null;
        global.document = {
            getElementById: () => createdEl,
            createElement: (tag) => {
                createdEl = {
                    tagName: tag,
                    id: '',
                    style: {},
                    textContent: '',
                    setAttribute: (k, v) => {}
                };
                return createdEl;
            },
            body: {
                appendChild: (el) => { createdEl = el; }
            }
        };

        announceLive('Layer 4 selected');
        assert.ok(createdEl !== null);
        assert.strictEqual(createdEl.id, 'a11y-announcer');
        assert.strictEqual(createdEl.textContent, 'Layer 4 selected');
    });

    it('wires arrow-key handlers for layer and step traversal', () => {
        let navigatedLayer = null;
        let steppedForward = false;

        let keyListener = null;
        global.window = {
            addEventListener: (evt, cb) => {
                if (evt === 'keydown') keyListener = cb;
            }
        };

        setupKeyboardNavigation({
            onNavigateLayer: (l) => { navigatedLayer = l; },
            onStepForward: () => { steppedForward = true; }
        });

        assert.ok(typeof keyListener === 'function');

        // Test ArrowUp
        keyListener({ key: 'ArrowUp', preventDefault: () => {} });
        assert.strictEqual(navigatedLayer, 6);

        // Test ArrowRight
        keyListener({ key: 'ArrowRight', preventDefault: () => {} });
        assert.strictEqual(steppedForward, true);
    });
});
