/**
 * builders/macro_builder.js
 * =========================
 * Builds consolidated macro bounding volumes for far-view Level-of-Detail (LOD 0).
 */

import * as THREE from 'three';
import { materials, edgeMatMacro } from '../core/materials.js';
import { createLayerLabelSprite } from './label_builder.js';

/**
 * Builds macro bounding boxes for far camera zoom distances.
 * @param {Array<object>} macroNodes - Macro node definitions from spatialData
 * @param {number} offsetX - Column horizontal offset (-4.5 or +4.5)
 * @param {boolean} isCompressedSide - True if compressed column
 * @param {THREE.Group} macroGroup - Macro scene group
 * @param {Array<THREE.Object3D>} visualObjects - Registry array for lifecycle cleanup
 * @param {Array<THREE.Mesh>} macroMeshes - Array tracking macro meshes
 * @param {Array<THREE.Mesh>} interactableMeshes - Array tracking raycasting interactables
 */
export function buildMacroBlocks(
    macroNodes,
    offsetX,
    isCompressedSide,
    macroGroup,
    visualObjects,
    macroMeshes,
    interactableMeshes
) {
    if (!macroNodes || !macroNodes.length) return;

    macroNodes.forEach(node => {
        const geo = new THREE.BoxGeometry(node.width, node.height, node.depth);
        const mesh = new THREE.Mesh(geo, materials.macro_box);
        mesh.position.set(offsetX, node.y, 0);
        mesh.userData = {
            ...node,
            isMacro: true,
            side: isCompressedSide ? 'compressed' : 'baseline'
        };

        const wireGeo = new THREE.EdgesGeometry(geo);
        const wireLine = new THREE.LineSegments(wireGeo, edgeMatMacro);
        wireLine.raycast = () => {};
        mesh.add(wireLine);

        macroGroup.add(mesh);
        visualObjects.push(mesh);
        macroMeshes.push(mesh);
        interactableMeshes.push(mesh);

        // Macro floating label
        const lbl = node.label || node.name || node.parent_layer_name || 'Layer';
        createLayerLabelSprite(lbl, offsetX, node.y + node.height / 2 + 0.45, true, macroGroup, visualObjects);
    });
}
