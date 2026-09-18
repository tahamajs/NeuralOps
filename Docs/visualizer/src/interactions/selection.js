/**
 * interactions/selection.js
 * =========================
 * 3D Node Selection, Emissive Radiance Highlighting, and Dynamic PointLight Tracking.
 */

import * as THREE from 'three';

let selectedMeshes = [];
let selectionPointLight = null;
let selectionHaloRing = null;

/**
 * Initializes selection dynamic light fixtures once in the scene.
 * @param {THREE.Scene} scene - Active scene
 */
export function initSelectionLighting(scene) {
    if (!selectionPointLight && scene) {
        selectionPointLight = new THREE.PointLight(0x53ecd9, 0, 14, 1.5);
        scene.add(selectionPointLight);

        const haloGeo = new THREE.RingGeometry(0.8, 0.95, 32);
        const haloMat = new THREE.MeshBasicMaterial({
            color: 0x53ecd9,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0,
            depthWrite: false
        });
        selectionHaloRing = new THREE.Mesh(haloGeo, haloMat);
        selectionHaloRing.visible = false;
        scene.add(selectionHaloRing);
    }
}

/**
 * Clears current selection glow and restores original materials.
 */
export function clearSelectionHighlight() {
    selectedMeshes.forEach(mesh => {
        if (mesh && mesh.userData && mesh.userData._origMaterial) {
            mesh.material.dispose();
            mesh.material = mesh.userData._origMaterial;
            delete mesh.userData._origMaterial;
        }
    });
    selectedMeshes = [];

    if (selectionPointLight) selectionPointLight.intensity = 0;
    if (selectionHaloRing) {
        selectionHaloRing.visible = false;
        selectionHaloRing.material.opacity = 0;
    }
}

/**
 * Highlights a target mesh or part ID with emissive radiance and camera focus.
 * @param {THREE.Mesh|string} target - Mesh or identifier string
 * @param {Array<THREE.Mesh>} interactableMeshes - Pool of selectable meshes
 * @param {object} options - { flyCamera, updateInspector, onSelect }
 * @returns {THREE.Mesh|null} Selected primary mesh
 */
export function selectPartIn3D(target, interactableMeshes = [], options = {}) {
    clearSelectionHighlight();
    if (!target) return null;

    let targetMeshes = [];
    let primaryMesh = null;

    if (target instanceof THREE.Mesh) {
        primaryMesh = target;
        targetMeshes = [target];
    } else if (typeof target === 'string') {
        const targetKey = target.toLowerCase();
        if (targetKey === 'dense' || targetKey === 'baseline') {
            targetMeshes = interactableMeshes.filter(m => m.userData && m.userData.side === 'baseline' && !m.userData.isMacro);
        } else {
            targetMeshes = interactableMeshes.filter(m => {
                const u = m.userData || {};
                const id = (u.id || '').toLowerCase();
                const subType = (u.sub_type || '').toLowerCase();
                const type = (u.type || '').toLowerCase();
                const label = (u.label || '').toLowerCase();
                return id.includes(targetKey) || subType.includes(targetKey) || type === targetKey || label.includes(targetKey);
            });
        }
        if (targetMeshes.length > 0) {
            primaryMesh = targetMeshes.find(m => m.userData && m.userData.side === 'compressed') || targetMeshes[0];
        }
    }

    if (!primaryMesh || targetMeshes.length === 0) return null;
    selectedMeshes = targetMeshes;

    // Determine highlight color
    let hexColor = 0x53ecd9; // default cyan
    if (primaryMesh.userData && primaryMesh.userData.custom_color) {
        hexColor = new THREE.Color(primaryMesh.userData.custom_color).getHex();
    } else if (primaryMesh.userData && primaryMesh.userData.type === 'moe') {
        hexColor = 0xc084fc; // purple
    } else if (primaryMesh.userData && primaryMesh.userData.type === 'mla') {
        hexColor = 0xfbbf24; // amber
    } else if (primaryMesh.userData && primaryMesh.userData.side === 'baseline') {
        hexColor = 0x38bdf8; // sky blue
    }

    selectedMeshes.forEach(mesh => {
        if (mesh.material) {
            if (!mesh.userData._origMaterial) {
                mesh.userData._origMaterial = mesh.material;
            }
            const glowMat = mesh.material.clone();
            glowMat.color.setHex(0xffffff);
            glowMat.emissive.setHex(hexColor);
            glowMat.emissiveIntensity = 2.6;
            glowMat.roughness = 0.05;
            glowMat.metalness = 0.9;
            glowMat.transparent = false;
            glowMat.opacity = 1.0;
            mesh.material = glowMat;
        }
    });

    if (selectionPointLight) {
        selectionPointLight.color.setHex(hexColor);
        selectionPointLight.intensity = 6.0;
        selectionPointLight.position.set(primaryMesh.position.x, primaryMesh.position.y, (primaryMesh.position.z || 0) + 1.2);
    }

    return primaryMesh;
}
