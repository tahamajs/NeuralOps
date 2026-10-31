/**
 * builders/granular_builder.js
 * =============================
 * Granular Tensor Node Mesh Builder (LOD 1: Detailed / Near Zoom).
 * Renders individual Q, K, V, FFN, SVD, and MoE tensor boxes with PBR materials,
 * wireframe outlines, attention head partition lines, and layer boundary labels.
 */

import * as THREE from 'three';
import { 
    materials, 
    edgeMatDense, 
    edgeMatSvd, 
    edgeMatMoe, 
    edgeMatMla, 
    getOrCreateCustomMaterial, 
    getOrCreateCustomEdgeMaterial,
    getLayerDepthMaterial,
    getLayerDepthEdgeMaterial
} from '../core/materials.js';
import { createLayerLabelSprite, createTensorBadgeSprite, createLayerLabelSpriteWithDepth } from './label_builder.js';
import { buildResidualBypass, buildLayerConduits, buildInterLayerConduits } from './conduit_builder.js';
import { anyKey } from '../utils/math.js';
import { MODEL_SPECS } from '../constants.js';
import { getLayerDepthProfile } from '../educational/layer_depth_profile.js';

/**
 * Builds an interactive 3D layer stage platform ring in world space.
 */
export function createLayerStageRing(layerIdx, totalLayers, bounds, offsetX, isCompressedSide, targetGroup, visualObjects) {
    const profile = getLayerDepthProfile(layerIdx, null, totalLayers);
    const ringY = bounds.minY - 0.25;
    const ringRadiusX = 2.4;
    const ringRadiusZ = 1.3;

    const curve = new THREE.EllipseCurve(
        0, 0,
        ringRadiusX, ringRadiusZ,
        0, 2 * Math.PI,
        false,
        0
    );
    const points = curve.getPoints(48);
    const ringPoints = points.map(p => new THREE.Vector3(p.x, 0, p.y));
    const ringGeo = new THREE.BufferGeometry().setFromPoints(ringPoints);

    const ringMat = new THREE.LineBasicMaterial({
        color: profile.ringColor,
        transparent: true,
        opacity: 0.35,
        linewidth: 1
    });

    const ringLine = new THREE.LineLoop(ringGeo, ringMat);
    ringLine.position.set(offsetX, ringY, 0);
    ringLine.userData = {
        isLayerStageRing: true,
        parent_layer_index: layerIdx,
        side: isCompressedSide ? 'compressed' : 'baseline'
    };

    if (targetGroup) targetGroup.add(ringLine);
    if (visualObjects) visualObjects.push(ringLine);

    return ringLine;
}

/**
 * Builds individual granular tensor nodes and adds them to the scene.
 * @param {Array<object>} nodes - Node array from spatial data
 * @param {number} offsetX - Column horizontal offset (-4.5 or +4.5)
 * @param {boolean} isCompressedSide - True if compressed column
 * @param {object} context - Shared scene collections { granularGroup, dimensionBadgesGroup, visualObjects, interactableMeshes, svdBottlenecks, moeNodes, layerWidthProfiles }
 * @returns {number} Topmost Y coordinate
 */
export function buildMeshNodes(nodes, offsetX, isCompressedSide, context) {
    const {
        granularGroup,
        dimensionBadgesGroup,
        visualObjects,
        interactableMeshes,
        svdBottlenecks,
        moeNodes,
        layerWidthProfiles
    } = context;

    let topY = 0;
    const layerTracker = new Set();
    const layerMinMax = new Map();
    const modelKey = context.modelKey || 'distilbert';
    const totalLayers = (MODEL_SPECS[modelKey] && MODEL_SPECS[modelKey].num_layers) || 6;

    // SVD bottleneck pairing
    for (let i = 0; i < nodes.length; i++) {
        if (nodes[i].type === 'svd' && nodes[i].sub_type === 'v_factor') {
            const vNode = nodes[i];
            const uNode = nodes[i + 1] && nodes[i + 1].type === 'svd' ? nodes[i + 1] : null;
            if (uNode && svdBottlenecks) {
                svdBottlenecks.push({
                    yStart: vNode.y - 0.4,
                    yNeck: (vNode.y + uNode.y) / 2.0,
                    yEnd: uNode.y + 0.5,
                    rankWidth: vNode.width,
                    fullWidth: uNode.width,
                    worldX: offsetX
                });
            }
        }
    }

    nodes.forEach(node => {
        const lIdx = node.parent_layer_index ?? (node.parent_layer_idx ?? 0);
        let mat = materials.dense;
        let edgeMat = edgeMatDense;

        if (node.custom_color) {
            mat = getOrCreateCustomMaterial(node.custom_color);
            edgeMat = getOrCreateCustomEdgeMaterial(node.custom_color);
        } else if (node.type === 'svd') {
            mat = getLayerDepthMaterial(lIdx, totalLayers, 'svd', isCompressedSide);
            edgeMat = getLayerDepthEdgeMaterial(lIdx, totalLayers);
        } else if (node.type === 'dense' && (node.parent_layer_index !== undefined || node.parent_layer_idx !== undefined)) {
            mat = getLayerDepthMaterial(lIdx, totalLayers, 'dense', isCompressedSide);
            edgeMat = getLayerDepthEdgeMaterial(lIdx, totalLayers);
        } else if (node.type === 'moe' && node.sub_type === 'router') {
            mat = materials.moe_router;
            edgeMat = edgeMatMoe;
        } else if (node.type === 'moe' && node.sub_type === 'expert') {
            mat = materials.moe_expert;
            edgeMat = edgeMatMoe;
        } else if (node.type === 'mla') {
            mat = materials.mla;
            edgeMat = edgeMatMla;
        } else if (node.sub_type === 'sliced_linear') {
            mat = materials.slicegpt;
            edgeMat = edgeMatDense;
        }

        const geo = new THREE.BoxGeometry(node.width, node.height, node.depth);
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(node.x + offsetX, node.y, node.z);
        mesh.userData = {
            ...node,
            side: isCompressedSide ? 'compressed' : 'baseline',
            lastTokenHit: 0
        };

        // Wireframe Outlines
        const wireGeo = new THREE.EdgesGeometry(geo);
        const wireLine = new THREE.LineSegments(wireGeo, edgeMat);
        wireLine.raycast = () => {};
        mesh.add(wireLine);

        // Attention Head Division Lines for Q, K, V
        const isAttn = node.id && anyKey(node.id, ['q_lin', 'k_lin', 'v_lin', 'query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj']);
        if (isAttn && node.width > 0.4) {
            let numHeads = 12;
            const meta = node.metadata || {};
            if (meta.shape && meta.shape.length === 2) {
                const dOut = meta.shape[0];
                if (dOut <= 256) numHeads = Math.max(1, Math.round(dOut / 64)); // e.g. 2 for Qwen GQA KV
                else numHeads = Math.max(1, Math.round(dOut / 64)); // e.g. 14 for Qwen Q
            }
            if (numHeads > 1) {
                const headStep = node.width / numHeads;
                const headPts = [];
                for (let h = 1; h < numHeads; h++) {
                    const xLine = -node.width / 2 + h * headStep;
                    headPts.push(new THREE.Vector3(xLine, -node.height / 2, node.depth / 2 + 0.01));
                    headPts.push(new THREE.Vector3(xLine, node.height / 2, node.depth / 2 + 0.01));
                }
                if (headPts.length > 0) {
                    const headGeo = new THREE.BufferGeometry().setFromPoints(headPts);
                    const headMat = new THREE.LineBasicMaterial({ color: 0x7dd3fc, transparent: true, opacity: 0.25 });
                    const headLines = new THREE.LineSegments(headGeo, headMat);
                    headLines.raycast = () => {};
                    mesh.add(headLines);
                }
            }
        }

        // Floating 3D Dimension Badge (visible in Near Zoom)
        createTensorBadgeSprite(node, offsetX, dimensionBadgesGroup, visualObjects);

        // Record layer width profile for particle stream scaling
        if (layerWidthProfiles) {
            layerWidthProfiles.push({
                yMin: node.y - node.height / 2,
                yMax: node.y + node.height / 2,
                width: node.width
            });
        }

        granularGroup.add(mesh);
        visualObjects.push(mesh);
        interactableMeshes.push(mesh);

        if (node.y > topY) topY = node.y;

        // Track Layer Y bounds for Residual Highways
        if (!layerMinMax.has(lIdx)) {
            layerMinMax.set(lIdx, { minY: node.y, maxY: node.y });
        } else {
            const bounds = layerMinMax.get(lIdx);
            bounds.minY = Math.min(bounds.minY, node.y);
            bounds.maxY = Math.max(bounds.maxY, node.y);
        }

        // Layer Billboard Tag with Rich Depth Specialization Profile
        if (node.parent_layer_name && !layerTracker.has(node.parent_layer_name)) {
            layerTracker.add(node.parent_layer_name);
            const labelX = offsetX + (isCompressedSide ? 3.6 : -3.6);
            createLayerLabelSpriteWithDepth(node.parent_layer_name, lIdx, totalLayers, labelX, node.y, false, granularGroup, visualObjects);
        }

        // MoE connector branching lines
        if (node.type === 'moe' && node.sub_type === 'expert') {
            const lineMat = new THREE.LineBasicMaterial({ color: 0xc895ea, transparent: true, opacity: 0.6 });
            const points = [
                new THREE.Vector3(offsetX, node.y - 1.4, node.z),
                new THREE.Vector3(node.x + offsetX, node.y, node.z)
            ];
            const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
            const line = new THREE.Line(lineGeo, lineMat);
            line.raycast = () => {};
            granularGroup.add(line);
            visualObjects.push(line);
        }

        if (node.type === 'moe' && node.sub_type === 'router' && moeNodes) {
            moeNodes.push({ ...node, worldX: node.x + offsetX });
        }
    });

    // Build 3D Layer Stage Platform Rings
    layerMinMax.forEach((bounds, lIdx) => {
        if (typeof lIdx === 'number' && lIdx !== 999) {
            createLayerStageRing(lIdx, totalLayers, bounds, offsetX, isCompressedSide, granularGroup, visualObjects);
        }
    });

    // Group nodes by layer for structured computational conduits
    const nodesByLayer = new Map();
    nodes.forEach(node => {
        const lIdx = node.parent_layer_index ?? (node.parent_layer_idx ?? 0);
        if (!nodesByLayer.has(lIdx)) nodesByLayer.set(lIdx, []);
        nodesByLayer.get(lIdx).push(node);
    });

    // Build comprehensive intra-layer conduits, port rings, and residual junctions (+)
    nodesByLayer.forEach(layerNodes => {
        buildLayerConduits(
            layerNodes,
            offsetX,
            isCompressedSide,
            context.modelKey || 'distilbert',
            context.paradigm || 'svd',
            granularGroup,
            visualObjects
        );
    });

    // Build inter-layer vertical trunk conduits connecting layers in sequence
    const layerBoundsArray = Array.from(layerMinMax.entries())
        .map(([lIdx, b]) => ({ lIdx, minY: b.minY, maxY: b.maxY }))
        .sort((a, b) => a.lIdx - b.lIdx);
    buildInterLayerConduits(layerBoundsArray, offsetX, isCompressedSide, granularGroup, visualObjects);

    return topY;
}
