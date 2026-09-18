/**
 * builders/conduit_builder.js
 * ===========================
 * Generates 3D dataflow wiring, cubic Bezier curves, and residual bypass conduits.
 */

import * as THREE from 'three';
import { edgeMatResidual } from '../core/materials.js';

/**
 * Creates a circular port ring indicating tensor ingress/egress.
 * @param {number} x - Port X
 * @param {number} y - Port Y
 * @param {number} z - Port Z
 * @param {number} colorHex - Hex color
 * @param {THREE.Group} targetGroup - Scene group
 * @param {Array<THREE.Object3D>} visualObjects - Disposal registry
 * @returns {THREE.Mesh}
 */
export function createPortRing(x, y, z, colorHex, targetGroup, visualObjects) {
    const geo = new THREE.RingGeometry(0.035, 0.075, 16);
    const mat = new THREE.MeshBasicMaterial({ color: colorHex, side: THREE.DoubleSide, transparent: true, opacity: 0.85 });
    const ring = new THREE.Mesh(geo, mat);
    ring.rotation.x = Math.PI / 2;
    ring.position.set(x, y, z);
    ring.raycast = () => {};
    if (targetGroup) targetGroup.add(ring);
    if (visualObjects) visualObjects.push(ring);
    return ring;
}

/**
 * Builds a smooth cubic Bezier conduit between two 3D vector points.
 * @param {THREE.Vector3} pStart - Start port
 * @param {THREE.Vector3} pEnd - End port
 * @param {THREE.Material} mat - Line material
 * @param {string} sourceId - Source node ID
 * @param {string} targetId - Target node ID
 * @param {THREE.Group} targetGroup - Scene group
 * @param {Array<THREE.Object3D>} visualObjects - Disposal registry
 * @returns {THREE.Line}
 */
export function buildSmoothConduit(pStart, pEnd, mat, sourceId = '', targetId = '', targetGroup, visualObjects) {
    const pts = [];
    const dx = pEnd.x - pStart.x;
    const dy = pEnd.y - pStart.y;
    const dz = pEnd.z - pStart.z;

    if (Math.abs(dx) < 0.05 && Math.abs(dz) < 0.05) {
        pts.push(pStart);
        pts.push(pEnd);
    } else {
        const curve = new THREE.CubicBezierCurve3(
            pStart,
            new THREE.Vector3(pStart.x, pStart.y + dy * 0.45, pStart.z),
            new THREE.Vector3(pEnd.x, pEnd.y - dy * 0.45, pEnd.z),
            pEnd
        );
        pts.push(...curve.getPoints(12));
    }

    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.Line(geo, mat);
    line.raycast = () => {};
    line.userData = { sourceId, targetId };
    if (targetGroup) targetGroup.add(line);
    if (visualObjects) visualObjects.push(line);
    return line;
}

/**
 * Builds a curved residual bypass conduit curving around attention/FFN sublayers.
 * @param {THREE.Vector3} pStart - Layer entry point
 * @param {THREE.Vector3} pEnd - Addition point (+)
 * @param {number} lateralX - Lateral curve apex X
 * @param {THREE.Material} mat - Line material
 * @param {string} sourceId - Source identifier
 * @param {string} targetId - Target identifier
 * @param {THREE.Group} targetGroup - Scene group
 * @param {Array<THREE.Object3D>} visualObjects - Disposal registry
 * @returns {THREE.Line}
 */
export function buildCurvedBypassConduit(pStart, pEnd, lateralX, mat, sourceId = '', targetId = '', targetGroup, visualObjects) {
    const p1 = new THREE.Vector3(lateralX, pStart.y + 0.15, pStart.z);
    const p2 = new THREE.Vector3(lateralX, pEnd.y - 0.15, pEnd.z);
    const curve = new THREE.CubicBezierCurve3(pStart, p1, p2, pEnd);
    const pts = curve.getPoints(16);

    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.Line(geo, mat);
    line.raycast = () => {};
    line.userData = { sourceId, targetId, isBypass: true };
    if (targetGroup) targetGroup.add(line);
    if (visualObjects) visualObjects.push(line);
    return line;
}

/**
 * Builds an outer bypass arc representing token skip connections.
 * @param {number} yStart - Start Y
 * @param {number} yEnd - End Y
 * @param {number} offsetX - Base X
 * @param {THREE.Group} targetGroup - Scene group
 * @param {Array<THREE.Object3D>} visualObjects - Disposal registry
 */
export function buildResidualBypass(yStart, yEnd, offsetX, targetGroup, visualObjects) {
    const curvePoints = [];
    const bypassX = offsetX + 2.4;
    const steps = 14;

    for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        const y = yStart + t * (yEnd - yStart);
        const bulge = Math.sin(t * Math.PI) * 0.45;
        curvePoints.push(new THREE.Vector3(bypassX + bulge, y, 0));
    }

    const curveGeo = new THREE.BufferGeometry().setFromPoints(curvePoints);
    const line = new THREE.Line(curveGeo, edgeMatResidual);
    line.raycast = () => {};
    if (targetGroup) targetGroup.add(line);
    if (visualObjects) visualObjects.push(line);
}
