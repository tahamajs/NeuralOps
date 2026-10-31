/**
 * tests/token_trajectory.test.js
 * ===============================
 * Unit tests verifying token physics and trajectory calculations across 4 model architectures.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import { computeToken3DTrajectory } from '../src/physics/token_trajectory.js';
import { VISUAL_CONFIG } from '../src/constants.js';

const STRIDE = VISUAL_CONFIG.TOKENS.layerStride; // 3.2

describe('Token 3D Trajectory Engine', () => {
    describe('Residual Skip Highway Tokens', () => {
        it('keeps residual tokens at the lateral bypass offset when traversing layers', () => {
            const t = { isResidual: true, lateralNoise: 0 };
            const y = 1.0 + 0.20 * STRIDE; // 20% into Layer 0
            const pos = computeToken3DTrajectory(t, y, false, 'distilbert', 'svd', -4.5);
            assert.strictEqual(pos.sublayer, 'residual');
            assert.ok(Math.abs(pos.x - (-4.5 - 1.85)) < 0.01);
        });

        it('dips into attention residual add (+) at tau ~ 0.40 - 0.48', () => {
            const t = { isResidual: true, lateralNoise: 0 };
            const y = 1.0 + 0.44 * STRIDE;
            const pos = computeToken3DTrajectory(t, y, false, 'distilbert', 'svd', -4.5);
            assert.strictEqual(pos.sublayer, 'residual_add_attn');
            // X position should be moving toward base offset
            assert.ok(pos.x > (-4.5 - 1.85));
        });

        it('dips into FFN residual add (+) at tau ~ 0.88 - 0.98', () => {
            const t = { isResidual: true, lateralNoise: 0 };
            const y = 1.0 + 0.93 * STRIDE;
            const pos = computeToken3DTrajectory(t, y, true, 'distilbert', 'svd', 4.5);
            assert.strictEqual(pos.sublayer, 'residual_add_ffn');
            assert.ok(pos.x < (4.5 + 1.85));
        });
    });

    describe('Qwen GQA Attention Projections', () => {
        it('routes Q-stream to wide query projection offset', () => {
            const t = { isResidual: false, subStream: 'q', headIdx: 0, lateralNoise: 0 };
            const y = 1.0 + 0.30 * STRIDE;
            const pos = computeToken3DTrajectory(t, y, true, 'qwen', 'svd', 4.5);
            assert.strictEqual(pos.sublayer, 'q_proj');
            assert.ok(pos.x < 4.5); // Q offset is negative
        });

        it('routes K-stream to compact key projection bundle', () => {
            const t = { isResidual: false, subStream: 'k', lateralNoise: 0 };
            const y = 1.0 + 0.30 * STRIDE;
            const pos = computeToken3DTrajectory(t, y, true, 'qwen', 'svd', 4.5);
            assert.strictEqual(pos.sublayer, 'k_proj');
            assert.ok(pos.x > 4.5);
        });

        it('routes V-stream to compact value projection bundle', () => {
            const t = { isResidual: false, subStream: 'v', lateralNoise: 0 };
            const y = 1.0 + 0.30 * STRIDE;
            const pos = computeToken3DTrajectory(t, y, true, 'qwen', 'svd', 4.5);
            assert.strictEqual(pos.sublayer, 'v_proj');
            assert.ok(pos.x > 4.5);
        });
    });

    describe('GPT-2 Fused c_attn Projections', () => {
        it('routes tokens through unified fused projection slab', () => {
            const t = { isResidual: false, lateralNoise: 0 };
            const y = 1.0 + 0.30 * STRIDE;
            const pos = computeToken3DTrajectory(t, y, false, 'gpt2', 'svd', -4.5);
            assert.strictEqual(pos.sublayer, 'c_attn');
        });
    });

    describe('SVD Low-Rank Bottleneck Compression', () => {
        it('pinches tokens into rank-r subspace via Factor V', () => {
            const t = { isResidual: false, lateralNoise: 0.5 };
            const y = 1.0 + 0.20 * STRIDE;
            const pos = computeToken3DTrajectory(t, y, true, 'distilbert', 'svd', 4.5);
            assert.strictEqual(pos.sublayer, 'v_factor');
            assert.ok(Math.abs(pos.x - 4.5) < 0.2); // Squeezed narrow
        });

        it('expands tokens back to full hidden dimension via Factor U', () => {
            const t = { isResidual: false, lateralNoise: 0.5 };
            const y = 1.0 + 0.38 * STRIDE;
            const pos = computeToken3DTrajectory(t, y, true, 'distilbert', 'svd', 4.5);
            assert.strictEqual(pos.sublayer, 'u_factor');
            assert.ok(Math.abs(pos.x - 4.5) > 0.2); // Expanded wider
        });
    });

    describe('Mixture-of-Experts 4-Way Routing', () => {
        it('dispatches tokens to router at FFN entry', () => {
            const t = { isResidual: false, lateralNoise: 0, expertTarget: 2 };
            const y = 1.0 + 0.55 * STRIDE;
            const pos = computeToken3DTrajectory(t, y, true, 'distilbert', 'moe', 4.5);
            assert.strictEqual(pos.sublayer, 'moe_router');
            assert.strictEqual(pos.x, 4.5);
        });

        it('fans tokens out to routed expert network', () => {
            const t = { isResidual: false, lateralNoise: 0, expertTarget: 3 };
            const y = 1.0 + 0.75 * STRIDE;
            const pos = computeToken3DTrajectory(t, y, true, 'distilbert', 'moe', 4.5);
            assert.strictEqual(pos.sublayer, 'expert_3');
            // Expert 3 is at (3 - 1.5) * 0.85 = +1.275 offset
            assert.ok(pos.x > 4.5);
        });
    });
});
