/**
 * core/camera.js
 * ==============
 * PerspectiveCamera and OrbitControls setup with damping and boundary limits.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VISUAL_CONFIG } from '../constants.js';

/**
 * Initializes the perspective camera and OrbitControls.
 * @param {HTMLCanvasElement} canvas - Rendering canvas
 * @returns {{ camera: THREE.PerspectiveCamera, controls: OrbitControls }}
 */
export function createCameraAndControls(canvas) {
    const aspect = window.innerWidth / window.innerHeight;
    const camera = new THREE.PerspectiveCamera(38, aspect, 0.1, 900);
    camera.position.set(0, 8, 22);

    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.dampingFactor = VISUAL_CONFIG.ANIM.cameraLerp;
    controls.target.set(0, 8, 0);
    controls.maxPolarAngle = Math.PI * 0.88;
    controls.minDistance = 2.5;
    controls.maxDistance = 280;
    controls.screenSpacePanning = true;
    controls.zoomSpeed = 1.25;

    return { camera, controls };
}
