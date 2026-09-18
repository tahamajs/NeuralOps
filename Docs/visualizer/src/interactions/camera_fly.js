/**
 * interactions/camera_fly.js
 * ===========================
 * Smooth camera transitions, zoom-to-fit, and autofocus interpolations.
 */

import * as THREE from 'three';

let targetPos = null;
let targetLookAt = null;
let isFlying = false;

/**
 * Smoothly interpolates the camera to focus on a target Y altitude and Z distance.
 * @param {THREE.PerspectiveCamera} camera - Active camera
 * @param {object} controls - OrbitControls instance
 * @param {number} targetY - Destination Y altitude
 * @param {number} targetDist - Destination distance
 * @param {number} targetX - Destination center X
 */
export function smoothFlyCamera(camera, controls, targetY, targetDist, targetX = 0) {
    targetPos = new THREE.Vector3(targetX, targetY, targetDist);
    targetLookAt = new THREE.Vector3(targetX, targetY, 0);
    isFlying = true;
}

/**
 * Updates camera position each animation tick if a fly-through transition is active.
 * @param {THREE.PerspectiveCamera} camera - Active camera
 * @param {object} controls - OrbitControls instance
 */
export function updateCameraFlight(camera, controls) {
    if (!isFlying || !targetPos || !targetLookAt) return;

    camera.position.lerp(targetPos, 0.08);
    controls.target.lerp(targetLookAt, 0.08);

    if (camera.position.distanceTo(targetPos) < 0.1 && controls.target.distanceTo(targetLookAt) < 0.1) {
        camera.position.copy(targetPos);
        controls.target.copy(targetLookAt);
        isFlying = false;
        targetPos = null;
        targetLookAt = null;
    }
}
