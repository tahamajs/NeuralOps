/**
 * tests/3d_scenes.test.js
 * =======================
 * Unit tests verifying mathematically-accurate 3D scene specifications across all 9 categories (A–I).
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import { buildSceneSpec, getTimelineForCategory } from '../src/3d/scene_router.js';

describe('Mathematically-Accurate 3D Scene Specifications (Categories A–I)', () => {
    it('verifies Category D (Recurrent) strictly excludes S×S square matrix and includes state ribbon', () => {
        const spec = buildSceneSpec('hgrn2', 'D');
        assert.strictEqual(spec.features.hasSquareMatrix, false, 'Category D must NOT have S×S square matrix');
        assert.strictEqual(spec.features.hasStateRibbon, true, 'Category D must have recurrent state ribbon');

        const nodeIds = spec.nodes.map(n => n.id);
        assert.ok(nodeIds.includes('state_ribbon'));
        assert.ok(nodeIds.includes('forget_gate'));
        assert.ok(nodeIds.includes('input_gate'));
        assert.strictEqual(nodeIds.includes('attn_square'), false);
    });

    it('verifies Category E (SVD) contains singular value spectrum, U/V factors, and residual error', () => {
        const spec = buildSceneSpec('a3', 'E');
        assert.strictEqual(spec.features.hasSingularSpectrum, true);
        const nodeIds = spec.nodes.map(n => n.id);
        assert.ok(nodeIds.includes('singular_values'));
        assert.ok(nodeIds.includes('u_factor'));
        assert.ok(nodeIds.includes('v_factor'));
        assert.ok(nodeIds.includes('residual_error'));
    });

    it('verifies Category B (Sparse / Kernel) utilizes associative buffer and excludes S×S matrix', () => {
        const spec = buildSceneSpec('performer', 'B');
        assert.strictEqual(spec.features.hasSquareMatrix, false);
        assert.strictEqual(spec.features.hasAssociativeReorder, true);
        const nodeIds = spec.nodes.map(n => n.id);
        assert.ok(nodeIds.includes('phi_q'));
        assert.ok(nodeIds.includes('phi_k'));
        assert.ok(nodeIds.includes('kv_buffer'));
    });

    it('verifies Category F (MoE) contains top-k router and expert networks', () => {
        const spec = buildSceneSpec('expert_choice_moe', 'F');
        assert.strictEqual(spec.features.hasRouterGating, true);
        const nodeIds = spec.nodes.map(n => n.id);
        assert.ok(nodeIds.includes('moe_router'));
        assert.ok(nodeIds.includes('expert_1'));
        assert.ok(nodeIds.includes('expert_2'));
    });

    it('verifies Category A (Dense) contains full S×S square and Q/K/V projections', () => {
        const spec = buildSceneSpec('full_attention', 'A');
        assert.strictEqual(spec.features.hasSquareMatrix, true);
        const nodeIds = spec.nodes.map(n => n.id);
        assert.ok(nodeIds.includes('q_proj'));
        assert.ok(nodeIds.includes('k_proj'));
        assert.ok(nodeIds.includes('v_proj'));
        assert.ok(nodeIds.includes('attn_square'));
    });

    it('verifies all categories return valid timelines with positive durations', () => {
        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'].forEach(cat => {
            const tl = getTimelineForCategory(cat);
            assert.ok(tl.length >= 3, `Category ${cat} timeline must have at least 3 steps`);
            tl.forEach(step => {
                assert.ok(step.name);
                assert.ok(step.duration > 0, `Step duration in category ${cat} must be positive`);
            });
        });
    });
});
