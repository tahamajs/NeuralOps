/**
 * core/materials.js
 * =================
 * Standard PBR materials, wireframe edge styles, token spheres, and dynamic module color caching.
 */

import * as THREE from 'three';
import { VISUAL_CONFIG } from '../constants.js';

const C = VISUAL_CONFIG.COLORS;

export const materials = {
    dense: new THREE.MeshStandardMaterial({
        color: C.denseBase,
        roughness: 0.3,
        metalness: 0.35,
        transparent: true,
        opacity: 0.94
    }),
    svd: new THREE.MeshStandardMaterial({
        color: C.svdCompressed,
        roughness: 0.2,
        metalness: 0.5,
        emissive: C.svdEmissive,
        emissiveIntensity: 0.4
    }),
    moe_router: new THREE.MeshStandardMaterial({
        color: C.moeRouter,
        roughness: 0.25,
        metalness: 0.5,
        emissive: C.moeRouterEmissive,
        emissiveIntensity: 0.4
    }),
    moe_expert: new THREE.MeshStandardMaterial({
        color: C.moeExpert,
        roughness: 0.3,
        metalness: 0.4
    }),
    mla: new THREE.MeshStandardMaterial({
        color: C.mlaLatent,
        roughness: 0.25,
        metalness: 0.45,
        emissive: C.mlaEmissive,
        emissiveIntensity: 0.35
    }),
    slicegpt: new THREE.MeshStandardMaterial({
        color: C.slicegpt,
        roughness: 0.3,
        metalness: 0.4
    }),
    macro_box: new THREE.MeshStandardMaterial({
        color: C.macroBox,
        roughness: 0.2,
        metalness: 0.8,
        transparent: true,
        opacity: 0.18,
        wireframe: false
    })
};

// Edge Outline Wireframe Line Materials
export const edgeMatDense = new THREE.LineBasicMaterial({ color: C.edgeDense, transparent: true, opacity: 0.5 });
export const edgeMatSvd = new THREE.LineBasicMaterial({ color: C.edgeSvd, transparent: true, opacity: 0.85 });
export const edgeMatMoe = new THREE.LineBasicMaterial({ color: C.edgeMoe, transparent: true, opacity: 0.75 });
export const edgeMatMla = new THREE.LineBasicMaterial({ color: C.edgeMla, transparent: true, opacity: 0.85 });
export const edgeMatMacro = new THREE.LineBasicMaterial({ color: C.edgeMacro, transparent: true, opacity: 0.65 });
export const edgeMatResidual = new THREE.LineBasicMaterial({ color: C.edgeResidual, transparent: true, opacity: 0.75 });

// 3D Computational Graph Conduit & Connection Materials
export const conduitMatBaseline = new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.45 });
export const conduitMatCompressed = new THREE.LineBasicMaterial({ color: 0x2dd4bf, transparent: true, opacity: 0.55 });
export const conduitMatMoE = new THREE.LineBasicMaterial({ color: 0xc084fc, transparent: true, opacity: 0.6 });
export const junctionMatResidual = new THREE.MeshBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.85 });
export const portRingMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, side: THREE.DoubleSide, transparent: true, opacity: 0.75 });

// Token Particle Materials
export const tokenMaterialBaseline = new THREE.MeshBasicMaterial({ color: C.tokenBaseline });
export const tokenMaterialCompressed = new THREE.MeshBasicMaterial({ color: C.tokenCompressed });
export const trailMaterialBaseline = new THREE.MeshBasicMaterial({ color: C.trailBaseline, transparent: true, opacity: 0.35 });
export const trailMaterialCompressed = new THREE.MeshBasicMaterial({ color: C.trailCompressed, transparent: true, opacity: 0.35 });

// Custom Dynamic Module Component Material Caches
const customMaterialsCache = new Map();
const customEdgeMaterialsCache = new Map();

/**
 * Gets or creates a cached PBR standard material for custom hex colors.
 * @param {string|null|undefined} hexColor - e.g. "#3acbb8"
 * @returns {THREE.MeshStandardMaterial}
 */
export function getOrCreateCustomMaterial(hexColor) {
    if (!hexColor) return materials.dense;
    if (!customMaterialsCache.has(hexColor)) {
        const c = new THREE.Color(hexColor);
        const mat = new THREE.MeshStandardMaterial({
            color: c,
            roughness: 0.28,
            metalness: 0.35,
            transparent: true,
            opacity: 0.94,
            emissive: c.clone().multiplyScalar(0.25),
            emissiveIntensity: 0.35
        });
        customMaterialsCache.set(hexColor, mat);
    }
    return customMaterialsCache.get(hexColor);
}

/**
 * Gets or creates a cached outline material for custom hex colors.
 * @param {string|null|undefined} hexColor - e.g. "#3acbb8"
 * @returns {THREE.LineBasicMaterial}
 */
export function getOrCreateCustomEdgeMaterial(hexColor) {
    if (!hexColor) return edgeMatDense;
    if (!customEdgeMaterialsCache.has(hexColor)) {
        const c = new THREE.Color(hexColor).addScalar(0.2);
        const mat = new THREE.LineBasicMaterial({
            color: c,
            transparent: true,
            opacity: 0.85
        });
        customEdgeMaterialsCache.set(hexColor, mat);
    }
    return customEdgeMaterialsCache.get(hexColor);
}
