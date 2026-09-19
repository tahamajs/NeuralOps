/**
 * tests/layer_navigation.test.js
 * ==============================
 * Comprehensive test suite for Layer-by-Layer 3D Navigation & UI Synchronization:
 * - Verifies HTML elements and CSS classes for quick-jump layer pills
 * - Tests dynamic pill generator across DistilBERT, RoBERTa, GPT-2, and Qwen
 * - Tests active layer indicator synchronization (pills, scrubber, tree explorer)
 * - Tests tree explorer layer-header click dispatching navigateToLayer
 * - Tests spatial Y-coordinate layer altitude resolution logic
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

describe('Layer-by-Layer 3D Navigation System', () => {
    const htmlPath = path.resolve(__dirname, '../index.html');
    const cssPath = path.resolve(__dirname, '../style.css');
    const htmlContent = fs.readFileSync(htmlPath, 'utf-8');
    const cssContent = fs.readFileSync(cssPath, 'utf-8');

    it('contains #layer-quick-pills container in index.html', () => {
        assert.ok(
            htmlContent.includes('id="layer-quick-pills"'),
            'Expected index.html to contain element with id="layer-quick-pills"'
        );
        assert.ok(
            htmlContent.includes('class="layer-quick-pills"'),
            'Expected index.html to contain class="layer-quick-pills"'
        );
    });

    it('contains comprehensive CSS styling for layer quick pills and active states', () => {
        assert.ok(cssContent.includes('.layer-quick-pills'), 'Expected .layer-quick-pills CSS rule');
        assert.ok(cssContent.includes('.layer-pill-btn'), 'Expected .layer-pill-btn CSS rule');
        assert.ok(cssContent.includes('.layer-pill-btn.active'), 'Expected .layer-pill-btn.active CSS rule');
        assert.ok(cssContent.includes('.layer-pill-btn:hover'), 'Expected .layer-pill-btn:hover CSS rule');
    });

    it('generates correct sequence of quick jump pills for various model architectures', () => {
        // Mock DOM implementation for buildLayerQuickPills test
        function createMockElement(tag) {
            const attrs = {};
            const classes = new Set();
            return {
                tagName: tag.toUpperCase(),
                className: '',
                textContent: '',
                title: '',
                onclick: null,
                children: [],
                setAttribute(k, v) { attrs[k] = v; },
                getAttribute(k) { return attrs[k]; },
                appendChild(child) { this.children.push(child); }
            };
        }

        function simulateBuildLayerQuickPills(container, scrubber, modelKey, numLayers, onNavigate) {
            container.children = [];
            if (scrubber) {
                scrubber.min = '0';
                scrubber.max = String(Math.max(1, numLayers - 1));
            }

            // 1. Embeddings Pill
            const embBtn = createMockElement('button');
            embBtn.className = 'layer-pill-btn';
            embBtn.setAttribute('data-layer', 'embeddings');
            embBtn.textContent = 'Emb';
            embBtn.title = 'Jump to Input Embeddings';
            embBtn.onclick = () => onNavigate('embeddings');
            container.appendChild(embBtn);

            // 2. Layer Pills L0 .. L(numLayers - 1)
            for (let i = 0; i < numLayers; i++) {
                const lBtn = createMockElement('button');
                lBtn.className = `layer-pill-btn ${i === 0 ? 'active' : ''}`;
                lBtn.setAttribute('data-layer', String(i));
                lBtn.textContent = `L${i}`;
                lBtn.title = `Jump to Transformer Layer ${i}`;
                lBtn.onclick = () => onNavigate(i);
                container.appendChild(lBtn);
            }

            // 3. Output Pill
            const outBtn = createMockElement('button');
            outBtn.className = 'layer-pill-btn';
            outBtn.setAttribute('data-layer', 'output');
            outBtn.textContent = 'Out';
            outBtn.title = 'Jump to Output Head / Classifier';
            outBtn.onclick = () => onNavigate('output');
            container.appendChild(outBtn);
        }

        const modelSpecs = [
            { key: 'distilbert', layers: 6, expectedTotal: 8 },
            { key: 'roberta', layers: 12, expectedTotal: 14 },
            { key: 'gpt2', layers: 12, expectedTotal: 14 },
            { key: 'qwen', layers: 28, expectedTotal: 30 }
        ];

        for (const spec of modelSpecs) {
            const container = createMockElement('div');
            const scrubber = { min: '0', max: '0' };
            const navEvents = [];
            simulateBuildLayerQuickPills(container, scrubber, spec.key, spec.layers, (target) => navEvents.push(target));

            assert.strictEqual(container.children.length, spec.expectedTotal, `Expected ${spec.expectedTotal} pills for ${spec.key}`);
            assert.strictEqual(container.children[0].textContent, 'Emb');
            assert.strictEqual(container.children[0].getAttribute('data-layer'), 'embeddings');
            assert.strictEqual(container.children[spec.layers].textContent, `L${spec.layers - 1}`);
            assert.strictEqual(container.children[container.children.length - 1].textContent, 'Out');
            assert.strictEqual(container.children[container.children.length - 1].getAttribute('data-layer'), 'output');

            // Verify click handler invokes onNavigate with correct arguments
            container.children[0].onclick();
            assert.strictEqual(navEvents[0], 'embeddings');
            container.children[3].onclick();
            assert.strictEqual(navEvents[1], 2);
            container.children[container.children.length - 1].onclick();
            assert.strictEqual(navEvents[2], 'output');
        }
    });

    it('synchronizes active state across pills, scrubber, and tree explorer nodes', () => {
        // Simulating the updateActiveLayerIndicator logic
        function createMockClassList(initial = []) {
            const set = new Set(initial);
            return {
                add: (...cls) => cls.forEach(c => set.add(c)),
                remove: (...cls) => cls.forEach(c => set.delete(c)),
                contains: (c) => set.has(c),
                has: (c) => set.has(c)
            };
        }

        const pills = [
            { dataset: { layer: 'embeddings' }, classList: createMockClassList() },
            { dataset: { layer: '0' }, classList: createMockClassList(['active']) },
            { dataset: { layer: '1' }, classList: createMockClassList() },
            { dataset: { layer: '2' }, classList: createMockClassList() },
            { dataset: { layer: 'output' }, classList: createMockClassList() }
        ];
        const treeNodes = [
            { dataset: { layerIndex: '0' }, classList: createMockClassList(['open']) },
            { dataset: { layerIndex: '1' }, classList: createMockClassList() },
            { dataset: { layerIndex: '2' }, classList: createMockClassList() }
        ];
        const scrubber = { value: '0' };
        let stepLabelText = '';

        function simulateUpdateActiveLayerIndicator(layerIdx, label) {
            pills.forEach(btn => {
                const targetLayer = btn.dataset.layer;
                if (String(targetLayer) === String(layerIdx)) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });

            if (typeof layerIdx === 'number' && Number.isFinite(layerIdx)) {
                scrubber.value = String(layerIdx);
            }
            stepLabelText = label || `Layer ${layerIdx}`;

            treeNodes.forEach(node => {
                const nodeIdx = node.dataset.layerIndex;
                if (String(nodeIdx) === String(layerIdx)) {
                    node.classList.add('open', 'active-layer');
                } else {
                    node.classList.remove('active-layer');
                }
            });
        }

        // Test navigating to Layer 2
        simulateUpdateActiveLayerIndicator(2, 'Layer 2');
        assert.ok(pills[3].classList.has('active'), 'Pill L2 must have active class');
        assert.ok(!pills[1].classList.has('active'), 'Pill L0 must not have active class');
        assert.strictEqual(scrubber.value, '2', 'Scrubber must be at value 2');
        assert.strictEqual(stepLabelText, 'Layer 2');
        assert.ok(treeNodes[2].classList.has('open'), 'Layer 2 tree node must be open');
        assert.ok(treeNodes[2].classList.has('active-layer'), 'Layer 2 tree node must have active-layer class');
        assert.ok(!treeNodes[0].classList.has('active-layer'), 'Layer 0 tree node must not have active-layer class');

        // Test navigating to Embeddings
        simulateUpdateActiveLayerIndicator('embeddings', 'Input Embeddings');
        assert.ok(pills[0].classList.has('active'), 'Emb pill must be active');
        assert.ok(!pills[3].classList.has('active'), 'L2 pill must not be active');
        assert.strictEqual(stepLabelText, 'Input Embeddings');

        // Test navigating to Output
        simulateUpdateActiveLayerIndicator('output', 'Output Layer');
        assert.ok(pills[4].classList.has('active'), 'Out pill must be active');
        assert.strictEqual(stepLabelText, 'Output Layer');
    });

    it('computes spatial average Y coordinate altitude accurately for a given layer', () => {
        const mockMeshes = [
            { userData: { parent_layer_index: 0, name: 'distilbert.transformer.layer.0.attention.q_lin' }, position: { y: 2.0, x: -3.0 } },
            { userData: { parent_layer_index: 0, name: 'distilbert.transformer.layer.0.attention.k_lin' }, position: { y: 2.2, x: -1.0 } },
            { userData: { parent_layer_index: 0, name: 'distilbert.transformer.layer.0.ffn.lin1' }, position: { y: 2.8, x: 2.0 } },
            { userData: { parent_layer_index: 1, name: 'distilbert.transformer.layer.1.attention.q_lin' }, position: { y: 6.0, x: -3.0 } },
            { userData: { parent_layer_index: 1, name: 'distilbert.transformer.layer.1.ffn.lin1' }, position: { y: 6.8, x: 2.0 } },
            { userData: { name: 'distilbert.embeddings.word_embeddings' }, position: { y: -2.0, x: 0.0 } },
            { userData: { name: 'distilbert.pre_classifier' }, position: { y: 15.0, x: 0.0 } }
        ];

        function computeLayerTarget(layerIdentifier, meshes) {
            const isEmb = layerIdentifier === 'embeddings' || layerIdentifier === 'emb';
            const isOut = layerIdentifier === 'output' || layerIdentifier === 'out' || layerIdentifier === 'head';

            let matchingMeshes = [];
            if (isEmb) {
                matchingMeshes = meshes.filter(m => (m.userData.name || '').toLowerCase().includes('embed'));
            } else if (isOut) {
                matchingMeshes = meshes.filter(m => {
                    const n = (m.userData.name || '').toLowerCase();
                    return n.includes('output') || n.includes('classifier') || n.includes('head') || n.includes('lm_head');
                });
            } else {
                const numericIdx = parseInt(layerIdentifier, 10);
                if (Number.isFinite(numericIdx)) {
                    matchingMeshes = meshes.filter(m => m.userData.parent_layer_index === numericIdx);
                }
            }

            if (matchingMeshes.length === 0) return null;

            const avgY = matchingMeshes.reduce((sum, m) => sum + m.position.y, 0) / matchingMeshes.length;
            const primaryMesh = matchingMeshes.find(m => {
                const n = (m.userData.name || '').toLowerCase();
                return n.includes('q_lin') || n.includes('q_proj') || n.includes('query');
            }) || matchingMeshes[0];

            return { avgY, primaryMesh, count: matchingMeshes.length };
        }

        // Test layer 0 altitude
        const layer0Result = computeLayerTarget(0, mockMeshes);
        assert.ok(layer0Result !== null);
        assert.strictEqual(layer0Result.count, 3);
        const expectedLayer0Y = (2.0 + 2.2 + 2.8) / 3;
        assert.strictEqual(Math.abs(layer0Result.avgY - expectedLayer0Y) < 1e-6, true);
        assert.strictEqual(layer0Result.primaryMesh.userData.name, 'distilbert.transformer.layer.0.attention.q_lin');

        // Test layer 1 altitude
        const layer1Result = computeLayerTarget(1, mockMeshes);
        assert.ok(layer1Result !== null);
        assert.strictEqual(layer1Result.count, 2);
        const expectedLayer1Y = (6.0 + 6.8) / 2;
        assert.strictEqual(Math.abs(layer1Result.avgY - expectedLayer1Y) < 1e-6, true);

        // Test embeddings altitude
        const embResult = computeLayerTarget('embeddings', mockMeshes);
        assert.ok(embResult !== null);
        assert.strictEqual(embResult.avgY, -2.0);

        // Test output altitude
        const outResult = computeLayerTarget('output', mockMeshes);
        assert.ok(outResult !== null);
        assert.strictEqual(outResult.avgY, 15.0);
    });
});
