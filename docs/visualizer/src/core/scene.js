/**
 * core/scene.js
 * =============
 * Scene graph setup, background color, and hierarchical groups.
 */

import * as THREE from 'three';
import { VISUAL_CONFIG } from '../constants.js';

/**
 * Initializes the root Three.js scene and scene groups.
 * @returns {{ scene: THREE.Scene, granularGroup: THREE.Group, macroGroup: THREE.Group, dimensionBadgesGroup: THREE.Group }}
 */
export function createScene() {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(VISUAL_CONFIG.COLORS.sceneBackground);

    const granularGroup = new THREE.Group();
    const macroGroup = new THREE.Group();
    const dimensionBadgesGroup = new THREE.Group();

    scene.add(granularGroup);
    scene.add(macroGroup);
    scene.add(dimensionBadgesGroup);

    return {
        scene,
        granularGroup,
        macroGroup,
        dimensionBadgesGroup
    };
}
