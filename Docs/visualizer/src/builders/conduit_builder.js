/**
 * builders/conduit_builder.js
 * ===========================
 * Generates 3D dataflow wiring, cubic Bezier curves, port rings,
 * additive residual junctions (+), and complete layer computational conduits.
 */

import * as THREE from 'three';
import {
    edgeMatResidual,
    conduitMatBaseline,
    conduitMatCompressed,
    conduitMatMoE,
    junctionMatResidual,
    portRingMat
} from '../core/materials.js';
import { anyKey } from '../utils/math.js';

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
export function createPortRing(x, y, z, colorHex = 0x38bdf8, targetGroup, visualObjects) {
    const geo = new THREE.RingGeometry(0.04, 0.08, 16);
    const mat = new THREE.MeshBasicMaterial({
        color: colorHex,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.75
    });
    const ring = new THREE.Mesh(geo, mat);
    ring.rotation.x = Math.PI / 2;
    ring.position.set(x, y, z);
    ring.raycast = () => {};
    if (targetGroup) targetGroup.add(ring);
    if (visualObjects) visualObjects.push(ring);
    return ring;
}

/**
 * Creates a glowing additive junction node (+) indicating residual addition.
 * @param {number} x - Junction X
 * @param {number} y - Junction Y
 * @param {number} z - Junction Z
 * @param {THREE.Group} targetGroup - Scene group
 * @param {Array<THREE.Object3D>} visualObjects - Disposal registry
 * @returns {THREE.Group}
 */
export function createAdditionJunction(x, y, z, targetGroup, visualObjects) {
    const group = new THREE.Group();
    group.position.set(x, y, z);

    // Glowing outer ring
    const ringGeo = new THREE.TorusGeometry(0.12, 0.02, 8, 24);
    const ringMat = junctionMatResidual || new THREE.MeshBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.85 });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = Math.PI / 2;
    group.add(ring);

    // Plus (+) symbol cross bars
    const barGeoH = new THREE.BoxGeometry(0.14, 0.025, 0.02);
    const barGeoV = new THREE.BoxGeometry(0.025, 0.14, 0.02);
    const barMat = new THREE.MeshBasicMaterial({ color: 0xeef8ff });
    const barH = new THREE.Mesh(barGeoH, barMat);
    const barV = new THREE.Mesh(barGeoV, barMat);
    barH.rotation.x = Math.PI / 2;
    barV.rotation.x = Math.PI / 2;
    group.add(barH);
    group.add(barV);

    group.raycast = () => {};
    if (targetGroup) targetGroup.add(group);
    if (visualObjects) visualObjects.push(group);
    return group;
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

    if (Math.abs(dx) < 0.04 && Math.abs(dz) < 0.04) {
        pts.push(pStart);
        pts.push(pEnd);
    } else {
        const curve = new THREE.CubicBezierCurve3(
            pStart,
            new THREE.Vector3(pStart.x, pStart.y + dy * 0.45, pStart.z),
            new THREE.Vector3(pEnd.x, pEnd.y - dy * 0.45, pEnd.z),
            pEnd
        );
        pts.push(...curve.getPoints(14));
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

/**
 * Builds complete 3D dataflow wiring for a single transformer layer:
 * - Intra-layer sequential tensor conduits
 * - Ingress / Egress port rings
 * - Multi-head Q, K, V branching and convergence into W_O
 * - MoE Router fan-out to experts and merge into FFN out
 * - 2-Stage Residual Skip Highway (Attention Add & FFN Add junctions)
 *
 * @param {Array<object>} layerNodes - Nodes belonging to this layer
 * @param {number} worldX - Column X offset (-4.5 or +4.5)
 * @param {boolean} isCompressed - True if compressed column
 * @param {string} modelKey - Active model key
 * @param {string} paradigm - Active compression paradigm
 * @param {THREE.Group} targetGroup - Scene group
 * @param {Array<THREE.Object3D>} visualObjects - Disposal registry
 */
export function buildLayerConduits(
    layerNodes,
    worldX,
    isCompressed,
    modelKey,
    paradigm,
    targetGroup,
    visualObjects
) {
    if (!layerNodes || layerNodes.length === 0) return;

    // Sort nodes vertically from bottom to top
    const sorted = [...layerNodes].sort((a, b) => a.y - b.y);
    const conduitMat = isCompressed ? conduitMatCompressed : conduitMatBaseline;
    const lateralX = worldX + (isCompressed ? 1.85 : -1.85);

    const firstNode = sorted[0];
    const lastNode = sorted[sorted.length - 1];
    const layerEntryY = firstNode.y - (firstNode.height || 0.45) / 2.0;
    const layerExitY = lastNode.y + (lastNode.height || 0.45) / 2.0;

    // 1. Ingress & Egress Ports on individual nodes
    sorted.forEach(node => {
        const halfH = (node.height || 0.45) / 2.0;
        const nodeX = node.x + worldX;
        createPortRing(nodeX, node.y - halfH - 0.01, node.z, isCompressed ? 0x2dd4bf : 0x38bdf8, targetGroup, visualObjects);
        createPortRing(nodeX, node.y + halfH + 0.01, node.z, isCompressed ? 0x2dd4bf : 0x38bdf8, targetGroup, visualObjects);
    });

    // 2. Sequential Intra-Layer Conduits
    for (let i = 0; i < sorted.length - 1; i++) {
        const nA = sorted[i];
        const nB = sorted[i + 1];

        // Skip direct conduit if nA is expert and nB is also expert (they are parallel)
        if (nA.type === 'moe' && nA.sub_type === 'expert' && nB.type === 'moe' && nB.sub_type === 'expert') {
            continue;
        }

        const pA = new THREE.Vector3(nA.x + worldX, nA.y + (nA.height || 0.45) / 2.0, nA.z);
        const pB = new THREE.Vector3(nB.x + worldX, nB.y - (nB.height || 0.45) / 2.0, nB.z);

        buildSmoothConduit(pA, pB, conduitMat, nA.id, nB.id, targetGroup, visualObjects);
    }

    // 3. Q, K, V Branching and Convergence
    const qNodes = sorted.filter(n => anyKey(n.id, ['q_lin', 'q_proj', 'query']));
    const kNodes = sorted.filter(n => anyKey(n.id, ['k_lin', 'k_proj', 'key']));
    const vNodes = sorted.filter(n => anyKey(n.id, ['v_lin', 'v_proj', 'value']));
    const outNodes = sorted.filter(n => anyKey(n.id, ['out_lin', 'o_proj', 'c_proj', 'output']));

    const outNode = outNodes[0] || sorted.find(n => n.sequence_index === 3);

    // 4. MoE Expert Fan-Out and Convergence
    const routerNode = sorted.find(n => n.type === 'moe' && n.sub_type === 'router');
    const expertNodes = sorted.filter(n => n.type === 'moe' && n.sub_type === 'expert');

    if (routerNode && expertNodes.length > 0) {
        const pRouterTop = new THREE.Vector3(routerNode.x + worldX, routerNode.y + (routerNode.height || 0.45) / 2.0, routerNode.z);
        expertNodes.forEach(exp => {
            const pExpBot = new THREE.Vector3(exp.x + worldX, exp.y - (exp.height || 0.45) / 2.0, exp.z);
            buildSmoothConduit(pRouterTop, pExpBot, conduitMatMoE, routerNode.id, exp.id, targetGroup, visualObjects);
        });

        // Expert egress convergence
        const pMerge = new THREE.Vector3(worldX, layerExitY, 0);
        expertNodes.forEach(exp => {
            const pExpTop = new THREE.Vector3(exp.x + worldX, exp.y + (exp.height || 0.45) / 2.0, exp.z);
            buildSmoothConduit(pExpTop, pMerge, conduitMatMoE, exp.id, 'ffn_merge', targetGroup, visualObjects);
        });
    }

    // 5. Residual Highway with 2-Stage Addition Nodes (+)
    // Stage 1: Attention Residual Add Junction (+)
    const attnOutY = outNode ? outNode.y + (outNode.height || 0.45) / 2.0 : (firstNode.y + 1.2);
    const attnAddY = attnOutY + 0.18;
    createAdditionJunction(worldX, attnAddY, 0, targetGroup, visualObjects);

    // Highway Entry -> Lateral Conduit
    const pLayerEntry = new THREE.Vector3(worldX, layerEntryY - 0.1, 0);
    const pHighwayStart = new THREE.Vector3(lateralX, layerEntryY + 0.15, 0);
    buildSmoothConduit(pLayerEntry, pHighwayStart, edgeMatResidual, 'layer_in', 'highway_in', targetGroup, visualObjects);

    // Highway Dip 1 into Attention Add Junction (+)
    const pHighwayAttn = new THREE.Vector3(lateralX, attnAddY, 0);
    const pAttnJunction = new THREE.Vector3(worldX, attnAddY, 0);
    // Vertical lateral segment 1
    buildSmoothConduit(pHighwayStart, pHighwayAttn, edgeMatResidual, 'highway_1', 'highway_attn', targetGroup, visualObjects);
    // Lateral dip into Attention (+)
    buildSmoothConduit(pHighwayAttn, pAttnJunction, edgeMatResidual, 'highway_attn', 'attn_add_node', targetGroup, visualObjects);

    // Stage 2: FFN Residual Add Junction (+)
    const ffnAddY = layerExitY + 0.18;
    createAdditionJunction(worldX, ffnAddY, 0, targetGroup, visualObjects);

    // Highway Dip 2 into FFN Add Junction (+)
    const pHighwayFFN = new THREE.Vector3(lateralX, ffnAddY, 0);
    const pFFNJunction = new THREE.Vector3(worldX, ffnAddY, 0);
    // Vertical lateral segment 2
    buildSmoothConduit(pHighwayAttn, pHighwayFFN, edgeMatResidual, 'highway_attn', 'highway_ffn', targetGroup, visualObjects);
    // Lateral dip into FFN (+)
    buildSmoothConduit(pHighwayFFN, pFFNJunction, edgeMatResidual, 'highway_ffn', 'ffn_add_node', targetGroup, visualObjects);
}

/**
 * Builds inter-layer vertical conduits connecting consecutive transformer layers.
 * @param {Array<object>} layerBounds - Array of { lIdx, minY, maxY } sorted by layer index
 * @param {number} worldX - Column X offset
 * @param {boolean} isCompressed - Column compression flag
 * @param {THREE.Group} targetGroup - Scene group
 * @param {Array<THREE.Object3D>} visualObjects - Disposal registry
 */
export function buildInterLayerConduits(layerBounds, worldX, isCompressed, targetGroup, visualObjects) {
    if (!layerBounds || layerBounds.length === 0) return;

    const conduitMat = isCompressed ? conduitMatCompressed : conduitMatBaseline;

    // Base Embedding to Layer 0 Trunk
    const firstLayer = layerBounds[0];
    if (firstLayer && firstLayer.minY > 0.4) {
        const pEmb = new THREE.Vector3(worldX, 0.45, 0);
        const pL0 = new THREE.Vector3(worldX, firstLayer.minY - 0.2, 0);
        buildSmoothConduit(pEmb, pL0, conduitMat, 'embeddings_out', 'layer_0_in', targetGroup, visualObjects);
        createPortRing(worldX, 0.45, 0, isCompressed ? 0x2dd4bf : 0x38bdf8, targetGroup, visualObjects);
    }

    // Between Layer L exit and Layer L+1 entry
    for (let i = 0; i < layerBounds.length - 1; i++) {
        const cur = layerBounds[i];
        const next = layerBounds[i + 1];

        const exitY = cur.maxY + 0.25;
        const entryY = next.minY - 0.25;

        if (entryY > exitY) {
            const pExit = new THREE.Vector3(worldX, exitY, 0);
            const pEntry = new THREE.Vector3(worldX, entryY, 0);
            buildSmoothConduit(pExit, pEntry, conduitMat, `layer_${cur.lIdx}_exit`, `layer_${next.lIdx}_in`, targetGroup, visualObjects);
        }
    }
}
