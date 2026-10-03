/**
 * tests/lod.test.js
 * =================
 * Unit tests for Adaptive Level of Detail (LOD) Performance Engine.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import {
    getLODLevel,
    applyLOD,
    LOD_THRESHOLDS
} from '../src/core/lod.js';

describe('Adaptive Level of Detail (LOD) Engine', () => {
    it('accurately computes LOD level across camera distances', () => {
        assert.strictEqual(getLODLevel(65.0), 'far');
        assert.strictEqual(getLODLevel(35.0), 'medium');
        assert.strictEqual(getLODLevel(12.0), 'near');
    });

    it('forces coarser LOD when scene node count is very large', () => {
        assert.strictEqual(getLODLevel(10.0, 500), 'far');
        assert.strictEqual(getLODLevel(15.0, 250), 'medium');
    });

    it('filters object visibility based on LOD mode', () => {
        const mockObjects = [
            { visible: true, userData: { isLayerBlock: true } },
            { visible: true, userData: { isStagePlatform: true } },
            { visible: true, userData: { isComponent: true } },
            { visible: true, userData: { isHeadLine: true } },
            { visible: true, userData: { isFineDetail: true } }
        ];

        // Far LOD: only layer blocks and platforms
        const farVisible = applyLOD(mockObjects, 'far');
        assert.strictEqual(farVisible, 2);
        assert.strictEqual(mockObjects[0].visible, true);
        assert.strictEqual(mockObjects[1].visible, true);
        assert.strictEqual(mockObjects[3].visible, false);

        // Near LOD: all 5 visible
        const nearVisible = applyLOD(mockObjects, 'near');
        assert.strictEqual(nearVisible, 5);
    });
});
