/**
 * physics/token_animation.js
 * ===========================
 * Main simulation frame loop for instanced token particles and motion trails.
 */

import * as THREE from 'three';
import { VISUAL_CONFIG, EXPERT_HEX_COLORS } from '../constants.js';
import { computeToken3DTrajectory } from './token_trajectory.js';
import { triggerNodePulse } from './node_pulses.js';

const dummy = new THREE.Object3D();
const dummyTrail = new THREE.Object3D();
const expertThreeColors = EXPERT_HEX_COLORS.map(c => new THREE.Color(c));

/**
 * Updates all token positions, trail matrices, and color tints for one simulation frame.
 * @param {THREE.InstancedMesh} mainMesh - Primary token instanced mesh
 * @param {THREE.InstancedMesh} trailMesh - Motion trail instanced mesh
 * @param {Array<object>} tokenDataArray - Token state pool
 * @param {boolean} isCompressedSide - True if compressed column
 * @param {object} context - Execution context { modelKey, paradigm, maxGraphHeight, simulationSpeed, enablePulses, isPlaying, interactableMeshes, activeSelectedSublayerKey }
 */
export function updateTokens(mainMesh, trailMesh, tokenDataArray, isCompressedSide = false, context = {}) {
    if (!mainMesh || !context.isPlaying) return;
    if (!tokenDataArray || tokenDataArray.length === 0) return;

    const {
        modelKey = 'distilbert',
        paradigm = 'svd',
        maxGraphHeight = 16.0,
        simulationSpeed = 1.0,
        enablePulses = true,
        interactableMeshes = [],
        activeSelectedSublayerKey = null
    } = context;

    const baseOffsetX = isCompressedSide ? VISUAL_CONFIG.LAYOUT.compressedX : VISUAL_CONFIG.LAYOUT.baselineX;
    const defaultColor = isCompressedSide ? new THREE.Color(0x2dd4bf) : new THREE.Color(0x38bdf8);
    const residualColor = new THREE.Color(0x60a5fa);
    const highlightColor = new THREE.Color(0xffffff);

    const count = tokenDataArray.length;

    for (let i = 0; i < count; i++) {
        const t = tokenDataArray[i];

        // 1. Advance vertical progress
        t.y += t.speed * simulationSpeed;
        if (t.y > maxGraphHeight) {
            t.y = 0.2;
            t.lateralNoise = (Math.random() - 0.5) * VISUAL_CONFIG.TOKENS.lateralNoiseRange;
        }

        // 2. Compute 3D position
        const curPos = computeToken3DTrajectory(t, t.y, isCompressedSide, modelKey, paradigm, baseOffsetX);
        const prevY = Math.max(0.1, t.y - t.speed * simulationSpeed * VISUAL_CONFIG.TOKENS.trailMultiplier);
        const prevPos = computeToken3DTrajectory(t, prevY, isCompressedSide, modelKey, paradigm, baseOffsetX);

        // 3. Selection highlight check
        const isSelectedSublayer = activeSelectedSublayerKey && curPos.sublayer &&
            (activeSelectedSublayerKey.includes(curPos.sublayer) || curPos.sublayer.includes(activeSelectedSublayerKey));

        // 4. Update Token Mesh Matrix
        dummy.position.set(curPos.x, t.y, curPos.z);
        let scalePulse = 1.0 + Math.sin(t.y * 3.0 + t.pulsePhase) * 0.15;
        if (isSelectedSublayer) {
            scalePulse *= 2.2;
        }
        dummy.scale.set(scalePulse, scalePulse, scalePulse);
        dummy.updateMatrix();
        mainMesh.setMatrixAt(i, dummy.matrix);

        // 5. Update Motion Trail Mesh Matrix
        dummyTrail.position.set(prevPos.x, prevY, prevPos.z);
        const trailScale = isSelectedSublayer ? 1.4 : VISUAL_CONFIG.TOKENS.trailScale;
        dummyTrail.scale.set(trailScale, trailScale, trailScale);
        dummyTrail.updateMatrix();
        if (trailMesh) trailMesh.setMatrixAt(i, dummyTrail.matrix);

        // 6. Token Color Tinting
        if (mainMesh.instanceColor) {
            if (isSelectedSublayer) {
                mainMesh.setColorAt(i, highlightColor);
            } else if (t.isResidual) {
                mainMesh.setColorAt(i, residualColor);
            } else if (isCompressedSide && modelKey === 'qwen') {
                if (curPos.sublayer === 'q_proj') mainMesh.setColorAt(i, new THREE.Color(0x38bdf8));
                else if (curPos.sublayer === 'k_proj' || curPos.sublayer === 'v_proj') mainMesh.setColorAt(i, new THREE.Color(0xf59e0b));
                else if (curPos.sublayer === 'gate_proj') mainMesh.setColorAt(i, new THREE.Color(0xa855f7));
                else if (curPos.sublayer === 'up_proj') mainMesh.setColorAt(i, new THREE.Color(0x8b5cf6));
                else mainMesh.setColorAt(i, defaultColor);
            } else if (isCompressedSide && paradigm === 'moe') {
                mainMesh.setColorAt(i, expertThreeColors[t.expertTarget || 0]);
            } else {
                mainMesh.setColorAt(i, defaultColor);
            }
        }

        // 7. Node pulse trigger
        if (enablePulses && interactableMeshes.length > 0) {
            const meshCount = interactableMeshes.length;
            for (let mIdx = 0; mIdx < meshCount; mIdx++) {
                const mesh = interactableMeshes[mIdx];
                if (mesh.userData && mesh.userData.isMacro) continue;
                const matchesSide = isCompressedSide ? (mesh.position.x > 0) : (mesh.position.x < 0);
                if (matchesSide) {
                    const dx = Math.abs(mesh.position.x - curPos.x);
                    const dy = Math.abs(mesh.position.y - t.y);
                    if (dx < (mesh.geometry.parameters?.width || 1.0) / 2 && dy < 0.25) {
                        triggerNodePulse(mesh, 400, isCompressedSide ? 'svd_collapse' : 'standard');
                        break;
                    }
                }
            }
        }
    }

    mainMesh.instanceMatrix.needsUpdate = true;
    if (mainMesh.instanceColor) mainMesh.instanceColor.needsUpdate = true;
    if (trailMesh) trailMesh.instanceMatrix.needsUpdate = true;
}
