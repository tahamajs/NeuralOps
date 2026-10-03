/**
 * core/lighting.js
 * ================
 * Cinematic studio lighting configuration:
 * - Ambient/Hemisphere ground fill
 * - Directional key light with warm highlights
 * - Cyan and purple edge rim lights for 3D depth perception
 */

import * as THREE from 'three';
import { VISUAL_CONFIG } from '../constants.js';

/**
 * Adds studio lighting to the Three.js scene.
 * @param {THREE.Scene} scene - Target scene
 */
export function setupLighting(scene) {
    const hemiLight = new THREE.HemisphereLight(
        VISUAL_CONFIG.COLORS.hemiSky,
        VISUAL_CONFIG.COLORS.hemiGround,
        2.2
    );
    scene.add(hemiLight);

    const mainLight = new THREE.DirectionalLight(VISUAL_CONFIG.COLORS.mainLight, 2.6);
    mainLight.position.set(12, 32, 20);
    scene.add(mainLight);

    const cyanRim = new THREE.DirectionalLight(VISUAL_CONFIG.COLORS.cyanRim, 1.5);
    cyanRim.position.set(-14, 18, -12);
    scene.add(cyanRim);

    const purpleRim = new THREE.DirectionalLight(VISUAL_CONFIG.COLORS.purpleRim, 1.3);
    purpleRim.position.set(14, 14, -14);
    scene.add(purpleRim);
}
