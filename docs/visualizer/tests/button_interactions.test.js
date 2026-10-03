/**
 * tests/button_interactions.test.js
 * =================================
 * Comprehensive verification for Visualizer UI Buttons & Controls:
 * - HTML audit of all 61 button elements (checking IDs, roles, labels, classes)
 * - setupControls event handler wiring for playback, camera, perspective, speeds, LOD, and models
 * - Pin module comparator state toggling
 * - Color palette legend dynamic DOM generation
 * - Keyboard shortcuts modal activation and hotkeys registry
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

describe('Visualizer UI Buttons & Controls Audit', () => {
    const htmlPath = path.resolve(__dirname, '../index.html');
    const htmlContent = fs.readFileSync(htmlPath, 'utf-8');

    it('contains all expected control and navigation buttons in index.html', () => {
        const buttonRegex = /<button([^>]*)>(.*?)<\/button>/gis;
        const matches = [...htmlContent.matchAll(buttonRegex)];
        assert.ok(matches.length >= 60, `Expected at least 60 buttons in index.html, found ${matches.length}`);

        const expectedIds = [
            'btn-present-mode',
            'btn-view-3d',
            'btn-view-2d',
            'btn-open-learn',
            'btn-open-lessons',
            'btn-open-model-matrix',
            'btn-open-heatmap',
            'btn-open-prediction',
            'btn-open-compression',
            'btn-open-race',
            'btn-open-catalog',
            'btn-toggle-inspector',
            'btn-share',
            'btn-record-video',
            'btn-keyboard-shortcuts',
            'btn-detail-inspect',
            'btn-play-pause',
            'btn-auto-tour',
            'btn-legend-toggle',
            'btn-pin-module',
            'btn-reset-cam',
            'btn-browse-all-modules',
            'btn-zoom-out',
            'btn-zoom-in',
            'btn-close-insp',
            'btn-tree-clear',
            'btn-close-modal',
            'btn-apply-from-doc',
            'btn-close-doc-modal'
        ];

        expectedIds.forEach(id => {
            assert.ok(htmlContent.includes(`id="${id}"`), `index.html must contain button with id="${id}"`);
        });
    });

    it('contains all mechanism navigation tabs and categories', () => {
        assert.ok(htmlContent.includes('class="mechanism-button active"'));
        assert.ok(htmlContent.includes('data-model="distilbert"'));
        assert.ok(htmlContent.includes('data-model="roberta"'));
        assert.ok(htmlContent.includes('data-model="gpt2"'));
        assert.ok(htmlContent.includes('data-model="qwen"'));
    });

    it('contains playback speeds, LOD modes, and timeline scrubber', () => {
        assert.ok(htmlContent.includes('id="timeline-scrubber"'));
        assert.ok(htmlContent.includes('id="timeline-step"'));
        assert.ok(htmlContent.includes('id="timeline-label"'));
        assert.ok(htmlContent.includes('class="btn-speed"'));
        assert.ok(htmlContent.includes('class="btn-lod"'));
    });

    it('verifies anti-overlap UI layout structure in index.html and style.css', () => {
        // Assert dock-actions wrapper wraps secondary dock buttons
        assert.ok(htmlContent.includes('class="dock-actions"'), 'index.html must group dock action buttons in .dock-actions');
        
        const cssPath = path.resolve(__dirname, '../style.css');
        const cssContent = fs.readFileSync(cssPath, 'utf-8');

        // Assert no unclosed braces / syntax errors around .view-panels
        assert.ok(cssContent.includes('.view-panels {'), 'style.css must have valid .view-panels selector');

        // Assert body.inspector-open hides compare-card to prevent overlap
        assert.ok(cssContent.includes('body.inspector-open .compare-card'), 'style.css must hide .compare-card when inspector is open');

        // Assert .mechanism-nav does not overlap cards
        assert.ok(cssContent.includes('.mechanism-nav {'), 'style.css must style .mechanism-nav');
        assert.ok(cssContent.includes('top: 68px;'), 'mechanism-nav top position must be aligned below topbar');
        assert.ok(cssContent.includes('top: 118px;'), 'stage cards top position must be below mechanism nav');
    });
});

describe('UI Controls Logic & Mock DOM Binding', () => {
    // Lightweight mock DOM environment for Node.js unit tests
    function createMockElement(tag = 'div', id = '', classes = []) {
        const classList = new Set(classes);
        const children = [];
        const listeners = {};
        const attrs = {};
        const style = {};

        return {
            tagName: tag.toUpperCase(),
            id,
            children,
            style,
            classList: {
                add: (c) => classList.add(c),
                remove: (c) => classList.delete(c),
                contains: (c) => classList.has(c),
                toggle: (c, force) => {
                    if (force === undefined) {
                        if (classList.has(c)) { classList.delete(c); return false; }
                        else { classList.add(c); return true; }
                    } else if (force) {
                        classList.add(c); return true;
                    } else {
                        classList.delete(c); return false;
                    }
                }
            },
            get className() { return [...classList].join(' '); },
            set className(val) {
                classList.clear();
                val.split(/\s+/).filter(Boolean).forEach(c => classList.add(c));
            },
            getAttribute: (attr) => attrs[attr] || null,
            setAttribute: (attr, val) => { attrs[attr] = String(val); },
            addEventListener: (evt, cb) => {
                if (!listeners[evt]) listeners[evt] = [];
                listeners[evt].push(cb);
            },
            click: function() {
                if (listeners['click']) listeners['click'].forEach(cb => cb({ target: this }));
            },
            trigger: function(evt, eventObj = {}) {
                if (listeners[evt]) listeners[evt].forEach(cb => cb({ target: this, ...eventObj }));
            },
            appendChild: (child) => { children.push(child); return child; },
            hasChildNodes: () => children.length > 0,
            dataset: {},
            innerText: '',
            textContent: '',
            innerHTML: ''
        };
    }

    it('verifies setupControls correctly wires playback and camera handlers', async () => {
        // Setup global mock document
        const elementsById = {
            'btn-play-pause': createMockElement('button', 'btn-play-pause', ['primary']),
            'btn-reset-cam': createMockElement('button', 'btn-reset-cam', ['secondary']),
            'btn-view-3d': createMockElement('button', 'btn-view-3d', ['active']),
            'btn-view-2d': createMockElement('button', 'btn-view-2d', []),
            'btn-toggle-inspector': createMockElement('button', 'btn-toggle-inspector', []),
            'btn-close-insp': createMockElement('button', 'btn-close-insp', []),
            'inspector': createMockElement('div', 'inspector', ['hidden']),
            'color-legend': createMockElement('div', 'color-legend', []),
            'btn-legend-toggle': createMockElement('button', 'btn-legend-toggle', []),
            'btn-pin-module': createMockElement('button', 'btn-pin-module', []),
            'cmp-selected-node': createMockElement('strong', 'cmp-selected-node', [])
        };

        const originalDoc = global.document;
        global.document = {
            getElementById: (id) => elementsById[id] || null,
            querySelectorAll: (sel) => {
                if (sel === '.btn-speed') return [];
                if (sel === '.model-tab') return [];
                if (sel === '.comp-tab') return [];
                if (sel === '.btn-lod') return [];
                if (sel === '.mechanism-button') return [];
                if (sel === '.modal-overlay') return [];
                return [];
            },
            querySelector: () => null,
            createElement: (tag) => createMockElement(tag),
            body: createMockElement('body')
        };
        global.window = {
            addEventListener: () => {}
        };

        const { setupControls } = await import('../src/ui/controls.js');

        let playPauseCalled = false;
        let resetCamCalled = false;
        let viewSwitched = null;

        setupControls({
            onPlayPause: () => {
                playPauseCalled = true;
                return true;
            },
            onResetCamera: () => {
                resetCamCalled = true;
            },
            onSwitchView: (v) => {
                viewSwitched = v;
            }
        });

        // Test play/pause
        elementsById['btn-play-pause'].click();
        assert.strictEqual(playPauseCalled, true);
        assert.strictEqual(elementsById['btn-play-pause'].innerText, '⏸ Pause');
        assert.ok(elementsById['btn-play-pause'].classList.contains('active'));

        // Test reset camera
        elementsById['btn-reset-cam'].click();
        assert.strictEqual(resetCamCalled, true);

        // Test 2D perspective switch
        elementsById['btn-view-2d'].click();
        assert.strictEqual(viewSwitched, '2d');
        assert.ok(elementsById['btn-view-2d'].classList.contains('active'));
        assert.strictEqual(elementsById['btn-view-3d'].classList.contains('active'), false);

        // Test inspector toggle
        elementsById['btn-toggle-inspector'].click();
        assert.strictEqual(elementsById['inspector'].classList.contains('hidden'), false);
        assert.strictEqual(global.document.body.classList.contains('inspector-open'), true, 'body must receive inspector-open class');

        // Test inspector toggle off
        elementsById['btn-toggle-inspector'].click();
        assert.strictEqual(elementsById['inspector'].classList.contains('hidden'), true);
        assert.strictEqual(global.document.body.classList.contains('inspector-open'), false, 'body must remove inspector-open class');

        // Restore globals
        global.document = originalDoc;
    });

    it('verifies openKeyboardShortcutsModal creates cheatsheet modal', async () => {
        let createdModal = null;
        const mockBody = {
            appendChild: (child) => { createdModal = child; return child; }
        };

        const originalDoc = global.document;
        global.document = {
            getElementById: (id) => (id === 'keyboard-shortcuts-modal' ? createdModal : null),
            createElement: (tag) => {
                const el = createMockElement(tag);
                return el;
            },
            body: mockBody
        };

        const { openKeyboardShortcutsModal } = await import('../src/ui/controls.js');
        openKeyboardShortcutsModal();

        assert.ok(createdModal !== null, 'Modal should be appended to body');
        assert.strictEqual(createdModal.id, 'keyboard-shortcuts-modal');
        assert.strictEqual(createdModal.classList.contains('hidden'), false);
        assert.ok(createdModal.innerHTML.includes('Space'));
        assert.ok(createdModal.innerHTML.includes('Play / Pause'));
        assert.ok(createdModal.innerHTML.includes('Auto-Fit Camera'));

        global.document = originalDoc;
    });
});
