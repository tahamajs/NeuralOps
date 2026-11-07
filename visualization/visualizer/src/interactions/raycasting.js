/**
 * interactions/raycasting.js
 * ==========================
 * Raycasting interaction engine for hover tooltips, pointer coordinates, and 3D click selection.
 */

import * as THREE from 'three';

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

/**
 * Resolves the underlying tensor node data from a Raycaster intersection.
 * @param {object} intersection - Three.js intersection result
 * @returns {{ mesh: THREE.Mesh, data: object }|null}
 */
export function resolveNodeFromIntersection(intersection) {
    if (!intersection || !intersection.object) return null;
    let obj = intersection.object;

    while (obj && (!obj.userData || !obj.userData.id) && obj.parent) {
        obj = obj.parent;
    }

    if (obj && obj.userData && obj.userData.id) {
        return { mesh: obj, data: obj.userData };
    }
    return null;
}

/**
 * Binds mouse hover and click events on the WebGL canvas.
 * @param {HTMLCanvasElement} canvas - Target canvas
 * @param {THREE.PerspectiveCamera} camera - Active camera
 * @param {Array<THREE.Mesh>} interactableMeshes - Selectable mesh pool
 * @param {Function} onSelectNode - Callback when node is clicked: (nodeData, isLeft, mesh)
 */
export function setupRaycasting(canvas, camera, interactableMeshes, onSelectNode) {
    const tooltip = document.getElementById('tooltip');

    canvas.addEventListener('pointermove', (event) => {
        const rect = canvas.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(interactableMeshes, false);

        if (intersects.length > 0) {
            const hit = resolveNodeFromIntersection(intersects[0]);
            if (hit && tooltip) {
                canvas.style.cursor = 'pointer';
                const meta = hit.data.metadata || {};
                const shapeStr = meta.shape ? `[${meta.shape.join(' × ')}]` : '';
                tooltip.innerHTML = `
                    <div style="font-weight:700;color:#fff;">${hit.data.label || hit.data.id}</div>
                    <div style="font-size:10px;color:#38bdf8;">${shapeStr} ${meta.params || ''}</div>
                    <div style="font-size:9px;color:#94a3b8;">${hit.data.side === 'compressed' ? '⚡ Compressed' : 'Dense Baseline'}</div>
                `;
                tooltip.style.left = `${event.clientX + 14}px`;
                tooltip.style.top = `${event.clientY + 14}px`;
                tooltip.classList.remove('hidden');
                return;
            }
        }

        canvas.style.cursor = 'default';
        if (tooltip) tooltip.classList.add('hidden');
    });

    canvas.addEventListener('click', (event) => {
        const rect = canvas.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(interactableMeshes, false);

        if (intersects.length > 0) {
            const hit = resolveNodeFromIntersection(intersects[0]);
            if (hit && onSelectNode) {
                const isLeft = (hit.data.side === 'baseline');
                onSelectNode(hit.data, isLeft, hit.mesh);
            }
        }
    });
}
