/**
 * tests/model_loader.test.js
 * ==========================
 * Unit tests for Hugging Face Model Loader and Configuration Engine.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import {
    isValidHFModelId,
    parseHFConfig,
    buildSpatialSpecFromConfig,
    loadModelFromHF
} from '../src/interactive/model_loader.js';

describe('Hugging Face Model Loader & Spatial Config Engine', () => {
    it('validates model ID format strictly', () => {
        assert.strictEqual(isValidHFModelId('distilbert-base-uncased'), true);
        assert.strictEqual(isValidHFModelId('meta-llama/Llama-2-7b-hf'), true);
        assert.strictEqual(isValidHFModelId('Qwen/Qwen2.5-0.5B'), true);
        assert.strictEqual(isValidHFModelId(''), false);
        assert.strictEqual(isValidHFModelId('bad format$$$'), false);
    });

    it('parses diverse model configs (BERT, GPT, Qwen/LLaMA GQA)', () => {
        // BERT dialect
        const bert = parseHFConfig({
            model_type: 'bert',
            num_hidden_layers: 12,
            hidden_size: 768,
            num_attention_heads: 12
        });
        assert.strictEqual(bert.numLayers, 12);
        assert.strictEqual(bert.hiddenSize, 768);
        assert.strictEqual(bert.isGQA, false);

        // Qwen GQA dialect
        const qwenGQA = parseHFConfig({
            model_type: 'qwen2',
            num_hidden_layers: 24,
            hidden_size: 896,
            num_attention_heads: 14,
            num_key_value_heads: 2
        });
        assert.strictEqual(qwenGQA.isGQA, true);
        assert.strictEqual(qwenGQA.numKVHeads, 2);
        assert.strictEqual(qwenGQA.gqaRatio, 7);
    });

    it('builds 3D spatial specifications accurately from configuration', () => {
        const spec = buildSpatialSpecFromConfig({
            num_hidden_layers: 6,
            hidden_size: 768,
            num_attention_heads: 12
        }, 'test-model');

        assert.strictEqual(spec.modelId, 'test-model');
        assert.strictEqual(spec.numLayers, 6);
        assert.strictEqual(spec.layers.length, 6);
        assert.ok(spec.totalAltitude > 0);
        assert.strictEqual(spec.layers[0].layer_index, 0);
        assert.strictEqual(spec.layers[5].layer_index, 5);
    });

    it('gracefully handles offline fallback in loadModelFromHF', async () => {
        const res = await loadModelFromHF('distilbert-base-uncased');
        assert.ok(res.numLayers >= 6);
        assert.ok(res.layers.length >= 6);
    });
});
