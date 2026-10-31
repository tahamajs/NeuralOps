/**
 * tests/schema_validation.test.js
 * ===============================
 * Validates integrity of spatial-data.json, modules-catalog.json, and modules_components.json.
 */

import assert from 'node:assert';
import fs from 'node:fs';
import path from 'node:path';
import test, { describe, it } from 'node:test';

const BASE_DIR = path.resolve('Docs/visualizer');

describe('Data Schema Validation Suite', () => {
    it('validates spatial-data.json models structure and layer specs', () => {
        const raw = fs.readFileSync(path.join(BASE_DIR, 'spatial-data.json'), 'utf8');
        const data = JSON.parse(raw);

        assert.ok(data.models, 'Missing models key');
        const expectedModels = ['distilbert', 'roberta', 'gpt2', 'qwen'];

        for (const mKey of expectedModels) {
            assert.ok(data.models[mKey], `Missing model: ${mKey}`);
            const m = data.models[mKey];
            assert.ok(Array.isArray(m.baseline_nodes), `${mKey} missing baseline_nodes array`);
            assert.ok(Array.isArray(m.compressed_nodes), `${mKey} missing compressed_nodes array`);
            assert.ok(m.baseline_nodes.length > 0, `${mKey} baseline_nodes is empty`);
            assert.ok(m.max_y > 0, `${mKey} max_y must be positive`);

            // Verify first baseline node
            const first = m.baseline_nodes[0];
            assert.ok(first.id, 'Node missing id');
            assert.ok(typeof first.x === 'number', 'Node x must be number');
            assert.ok(typeof first.y === 'number', 'Node y must be number');
            assert.ok(typeof first.width === 'number', 'Node width must be number');
            assert.ok(first.metadata, 'Node missing metadata');
        }
    });

    it('validates modules-catalog.json count and metadata completeness', () => {
        const raw = fs.readFileSync(path.join(BASE_DIR, 'modules-catalog.json'), 'utf8');
        const data = JSON.parse(raw);

        assert.ok(data.modules, 'Missing modules array');
        assert.ok(data.modules.length >= 240, `Module count ${data.modules.length} should be >= 240`);

        data.modules.slice(0, 50).forEach(m => {
            assert.ok(m.id, 'Module missing id');
            assert.ok(m.name, 'Module missing name');
            assert.ok(m.category, 'Module missing category');
            assert.ok(m.description, 'Module missing description');
        });
    });

    it('validates modules_components.json sublayer components and projections', () => {
        const raw = fs.readFileSync(path.join(BASE_DIR, 'modules_components.json'), 'utf8');
        const data = JSON.parse(raw);

        assert.ok(data.modules, 'Missing modules array');
        assert.ok(data.modules.length >= 240, `Module count ${data.modules.length} should be >= 240`);

        const sample = data.modules[0];
        assert.ok(sample.components, 'Missing components in module');
        assert.ok(Array.isArray(sample.components), 'components must be an array');
        assert.ok(sample.model_projections, 'Missing model_projections');
        assert.ok(sample.model_projections.distilbert, 'Missing distilbert projection');
        assert.ok(sample.model_projections.distilbert.savings_ratio > 1.0, 'Savings ratio should exceed 1.0');
    });
});
