/**
 * tests/video_exporter.test.js
 * ============================
 * Unit tests for Feature 1.6: Video Exporter & 360° Orbit Trajectory.
 */

import assert from 'node:assert';
import test, { describe, it } from 'node:test';
import {
    validateVideoExportConfig,
    calculateOrbitFrameCameraPosition,
    generateOrbitKeyframes
} from '../src/export/video_exporter.js';

describe('Feature 1.6: Video Exporter & Orbit Trajectory Engine', () => {
    it('validates video export configuration and frame constraints', () => {
        const config5s = validateVideoExportConfig(5, 30);
        assert.strictEqual(config5s.durationSec, 5);
        assert.strictEqual(config5s.fps, 30);
        assert.strictEqual(config5s.totalFrames, 150);
        assert.strictEqual(config5s.videoBitsPerSecond, 8000000);
        assert.ok(config5s.defaultFilename.includes('5s_orbit'));
        assert.ok(config5s.defaultFilename.endsWith('.webm'));

        const config10s = validateVideoExportConfig(10, 60);
        assert.strictEqual(config10s.durationSec, 10);
        assert.strictEqual(config10s.fps, 60);
        assert.strictEqual(config10s.totalFrames, 600);
    });

    it('computes exact 3D cylindrical coordinates around orbit pivot', () => {
        const center = { x: 0, y: 10, z: 0 };
        const radius = 20.0;
        const height = 5.0;

        // Angle 0: x = 0, z = 20, y = 15
        const pos0 = calculateOrbitFrameCameraPosition(center, radius, height, 0);
        assert.ok(Math.abs(pos0.x - 0) < 1e-4);
        assert.ok(Math.abs(pos0.y - 15) < 1e-4);
        assert.ok(Math.abs(pos0.z - 20) < 1e-4);

        // Angle PI/2: x = 20, z = 0, y = 15
        const posPi2 = calculateOrbitFrameCameraPosition(center, radius, height, Math.PI / 2);
        assert.ok(Math.abs(posPi2.x - 20) < 1e-4);
        assert.ok(Math.abs(posPi2.y - 15) < 1e-4);
        assert.ok(Math.abs(posPi2.z - 0) < 1e-4);

        // Angle PI: x = 0, z = -20, y = 15
        const posPi = calculateOrbitFrameCameraPosition(center, radius, height, Math.PI);
        assert.ok(Math.abs(posPi.x - 0) < 1e-4);
        assert.ok(Math.abs(posPi.y - 15) < 1e-4);
        assert.ok(Math.abs(posPi.z - (-20)) < 1e-4);
    });

    it('generates full 360-degree discrete orbit keyframe sequences', () => {
        const center = { x: 2, y: 5, z: -1 };
        const keyframes = generateOrbitKeyframes(center, 25.0, 10.0, 120);

        assert.strictEqual(keyframes.length, 120);
        assert.strictEqual(keyframes[0].frame, 0);
        assert.strictEqual(keyframes[0].angleRad, 0);

        // Last keyframe must be approaching 2*PI
        const last = keyframes[keyframes.length - 1];
        assert.strictEqual(last.frame, 119);
        assert.ok(last.angleRad > (2 * Math.PI * 0.98));

        // Height must be constant along cylinder
        keyframes.forEach(kf => {
            assert.strictEqual(kf.position.y, 15.0);
            const distXZ = Math.hypot(kf.position.x - center.x, kf.position.z - center.z);
            assert.ok(Math.abs(distXZ - 25.0) < 1e-4);
        });
    });
});
