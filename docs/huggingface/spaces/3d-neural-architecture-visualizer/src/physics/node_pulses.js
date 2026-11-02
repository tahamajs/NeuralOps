/**
 * physics/node_pulses.js
 * ======================
 * Tensor Node Activation Pulse animations triggered as tokens pass through matrices.
 * Includes the new "SVD Rank Collapse" animated geometric sequence.
 */

import * as THREE from 'three';

const activePulses = [];

/**
 * Triggers a momentary emissive brightness flash or geometric compression on a tensor mesh.
 * @param {THREE.Mesh} mesh - Target mesh
 * @param {number} durationMs - Flash duration in milliseconds
 * @param {string} pulseType - 'standard' or 'svd_collapse'
 */
export function triggerNodePulse(mesh, durationMs = 320, pulseType = 'standard') {
    if (!mesh || !mesh.material) return;

    if (!mesh.userData._pulseBaseEmissive) {
        mesh.userData._pulseBaseEmissive = mesh.material.emissive
            ? mesh.material.emissive.clone()
            : new THREE.Color(0x000000);
        mesh.userData._pulseBaseIntensity = mesh.material.emissiveIntensity || 0;
    }
    if (!mesh.userData._originalScale) {
        mesh.userData._originalScale = mesh.scale.clone();
    }

    mesh.userData._pulseStartTime = performance.now();
    mesh.userData._pulseDuration = durationMs;
    mesh.userData._pulseType = pulseType;

    if (!activePulses.includes(mesh)) {
        activePulses.push(mesh);
    }
}

/**
 * Updates active node pulses and decays emissive intensity/geometry back to resting levels.
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
            if (mesh.material.emissive) mesh.material.emissive.copy(mesh.userData._pulseBaseEmissive);
            if (mesh.userData._originalScale) mesh.scale.copy(mesh.userData._originalScale);
            activePulses.splice(i, 1);
        } else {
            // Sine flash curve
            const pulseFactor = Math.sin(progress * Math.PI);
            
            if (mesh.userData._pulseType === 'svd_collapse') {
                // SVD collapse animation: color shifts to neon and scale physically squashes
                mesh.material.emissiveIntensity = (mesh.userData._pulseBaseIntensity || 0) + pulseFactor * 2.5;
                const neonColor = new THREE.Color(0x00ffff).lerp(new THREE.Color(0xff00ff), progress);
                if (mesh.material.emissive) mesh.material.emissive.copy(neonColor);
                
                // Squash matrix along X axis to represent low-rank mathematical projection
                const squashFactor = 1.0 - (pulseFactor * 0.6); // 60% compression visually
                mesh.scale.set(
                    mesh.userData._originalScale.x * squashFactor, 
                    mesh.userData._originalScale.y, 
                    mesh.userData._originalScale.z
                );
            } else {
                // Standard pulse
                mesh.material.emissiveIntensity = (mesh.userData._pulseBaseIntensity || 0) + pulseFactor * 1.8;
            }
        }
    }
}
