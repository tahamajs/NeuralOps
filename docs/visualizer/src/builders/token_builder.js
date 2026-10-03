/**
 * builders/token_builder.js
 * =========================
 * InstancedMesh token particle systems and trail visualizers.
 */

import * as THREE from 'three';
import { VISUAL_CONFIG } from '../constants.js';

const CFG = VISUAL_CONFIG.TOKENS;

export const tokenGeometry = new THREE.SphereGeometry(CFG.sphereRadius, CFG.sphereSegments, CFG.sphereSegments);
export const trailGeometry = new THREE.SphereGeometry(CFG.trailRadius, 6, 6);

/**
 * Creates an InstancedMesh system for tokens and their motion trails.
 * @param {Array<object>} tokenDataArray - Array to populate with token simulation state
 * @param {number} baseOffsetX - Column horizontal offset (-4.5 or +4.5)
 * @param {THREE.Material} mat - Main particle material
 * @param {THREE.Material} trailMat - Trail material
 * @param {number} maxGraphHeight - Scene max height
 * @param {number} simulationSpeed - Initial playback speed multiplier
 * @returns {{ main: THREE.InstancedMesh, trail: THREE.InstancedMesh }}
 */
export function createTokenMesh(tokenDataArray, baseOffsetX, mat, trailMat, maxGraphHeight = 16.0, simulationSpeed = 1.0) {
    const mainMesh = new THREE.InstancedMesh(tokenGeometry, mat, CFG.count);
    mainMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mainMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(CFG.count * 3), 3);
    mainMesh.raycast = () => {};

    const trailMesh = new THREE.InstancedMesh(trailGeometry, trailMat, CFG.count);
    trailMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    trailMesh.raycast = () => {};

    const dummy = new THREE.Object3D();
    const defaultColor = new THREE.Color(mat.color);

    for (let i = 0; i < CFG.count; i++) {
        const yInit = Math.random() * maxGraphHeight;
        const expert = Math.floor(Math.random() * 4);
        const isResidual = (i % 4 === 0); // 25% on residual highway
        const subStream = (i % 10 < 7) ? 'q' : ((i % 10 < 8) ? 'k' : 'v'); // GQA distribution
        const swigluStream = (i % 2 === 0) ? 'gate' : 'up'; // SwiGLU bifurcated branches
        const headIdx = i % 12; // 12 Attention heads
        const lateralNoise = (Math.random() - 0.5) * CFG.lateralNoiseRange;

        tokenDataArray.push({
            y: yInit,
            speed: (CFG.baseSpeed + Math.random() * CFG.speedVariance) * simulationSpeed,
            baseX: baseOffsetX,
            lateralOffset: (Math.random() - 0.5) * 1.8,
            lateralNoise: lateralNoise,
            isResidual: isResidual,
            subStream: subStream,
            swigluStream: swigluStream,
            headIdx: headIdx,
            expertTarget: expert,
            pulsePhase: Math.random() * Math.PI * 2
        });

        dummy.position.set(baseOffsetX, yInit, 0);
        dummy.updateMatrix();
        mainMesh.setMatrixAt(i, dummy.matrix);
        trailMesh.setMatrixAt(i, dummy.matrix);
        mainMesh.setColorAt(i, defaultColor);
    }

    if (mainMesh.instanceColor) mainMesh.instanceColor.needsUpdate = true;

    return { main: mainMesh, trail: trailMesh };
}
