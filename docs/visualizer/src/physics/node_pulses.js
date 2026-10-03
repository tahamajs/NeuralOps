/**
 * physics/node_pulses.js
 * ======================
 * Tensor Node Activation Pulse animations triggered as tokens pass through matrices.
 */

import * as THREE from 'three';

const activePulses = [];

/**
 * Triggers a momentary emissive brightness flash on a tensor mesh.
 * @param {THREE.Mesh} mesh - Target mesh
 * @param {number} durationMs - Flash duration in milliseconds
 */
export function triggerNodePulse(mesh, durationMs = 320) {
    if (!mesh || !mesh.material) return;

    if (!mesh.userData._pulseBaseEmissive) {
        mesh.userData._pulseBaseEmissive = mesh.material.emissive
            ? mesh.material.emissive.clone()
            : new THREE.Color(0x000000);
        mesh.userData._pulseBaseIntensity = mesh.material.emissiveIntensity || 0;
    }

    mesh.userData._pulseStartTime = performance.now();
    mesh.userData._pulseDuration = durationMs;

    if (!activePulses.includes(mesh)) {
        activePulses.push(mesh);
    }
}

/**
 * Updates active node pulses and decays emissive intensity back to resting levels.
 * @param {number} now - Current timestamp
 */
export function updateNodePulses(now = performance.now()) {
    for (let i = activePulses.length - 1; i >= 0; i--) {
        const mesh = activePulses[i];
        if (!mesh.material) {
            activePulses.splice(i, 1);
            continue;
        }

        const elapsed = now - (mesh.userData._pulseStartTime || now);
        const duration = mesh.userData._pulseDuration || 320;
        const progress = elapsed / duration;

        if (progress >= 1.0) {
            // Restore resting state
            mesh.material.emissiveIntensity = mesh.userData._pulseBaseIntensity || 0;
            activePulses.splice(i, 1);
        } else {
            // Sine flash curve
            const pulseFactor = Math.sin(progress * Math.PI);
            mesh.material.emissiveIntensity = (mesh.userData._pulseBaseIntensity || 0) + pulseFactor * 1.8;
        }
    }
}
