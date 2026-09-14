/**
 * app.js
 * ======
 * Live 3D Neural Architecture Visualizer Engine
 * High-Precision Multi-Paradigm & Semantic Zoom Architecture:
 * - Dynamic Model Swapping (DistilBERT, RoBERTa, GPT-2, Qwen3.5)
 * - Dynamic Compression Paradigm Swapping (SVD, MoE, MLA, SliceGPT)
 * - Semantic Zoom & Continuous Level-of-Detail (LOD):
 *     * Far View (Macro): Layers collapse into consolidated architectural blocks
 *     * Near View (Micro): Blocks expand into fine-grained matrix tensors & heads
 * - True Dimensional Proportions (Head partitions, FFN 4x expansion, SVD rank necks)
 * - Residual Skip Connection Highways & Add & Norm Conduits
 * - Physics-Driven Token Dynamics (SVD constriction, MoE 4-way branching, motion trails)
 * - Rich Mathematical & Educational Layer Explanations
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ---------------------------------------------------------------------------
// 1. WebGL Renderer & Dual-Viewport Scene
// ---------------------------------------------------------------------------
const canvas = document.getElementById('webgl-canvas');
const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance'
});

renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2.0));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setScissorTest(true);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.3;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x060a12);

// Studio Lighting
const hemiLight = new THREE.HemisphereLight(0xdbeafe, 0x0f172a, 2.2);
scene.add(hemiLight);

const mainLight = new THREE.DirectionalLight(0xfff7ed, 2.6);
mainLight.position.set(12, 32, 20);
scene.add(mainLight);

const cyanRim = new THREE.DirectionalLight(0x53ecd9, 1.5);
cyanRim.position.set(-14, 18, -12);
scene.add(cyanRim);

const purpleRim = new THREE.DirectionalLight(0xc895ea, 1.3);
purpleRim.position.set(14, 14, -14);
scene.add(purpleRim);

// ---------------------------------------------------------------------------
// 2. Camera Setup & OrbitControls
// ---------------------------------------------------------------------------
const fullAspect = window.innerWidth / window.innerHeight;

const cameraBaseline = new THREE.PerspectiveCamera(38, fullAspect, 0.1, 900);
cameraBaseline.position.set(0, 8, 22);

const controls = new OrbitControls(cameraBaseline, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.target.set(0, 8, 0);
controls.maxPolarAngle = Math.PI * 0.88;
controls.minDistance = 2.5;
controls.maxDistance = 280;
controls.screenSpacePanning = true;
controls.zoomSpeed = 1.25;

const OFFSET_BASELINE = -4.5;
const OFFSET_COMPRESSED = 4.5;

let cameraTargetY = 8.0;
let cameraTargetZ = 22.0;

// ---------------------------------------------------------------------------
// 3. High-Tech Shaders & Materials
// ---------------------------------------------------------------------------
const materials = {
    dense: new THREE.MeshStandardMaterial({
        color: 0x2e4659,
        roughness: 0.3,
        metalness: 0.35,
        transparent: true,
        opacity: 0.94
    }),
    svd: new THREE.MeshStandardMaterial({
        color: 0x3acbb8,
        roughness: 0.2,
        metalness: 0.5,
        emissive: 0x103d36,
        emissiveIntensity: 0.4
    }),
    moe_router: new THREE.MeshStandardMaterial({
        color: 0xb57ee0,
        roughness: 0.25,
        metalness: 0.5,
        emissive: 0x2a143d,
        emissiveIntensity: 0.4
    }),
    moe_expert: new THREE.MeshStandardMaterial({
        color: 0x9f68cb,
        roughness: 0.3,
        metalness: 0.4
    }),
    mla: new THREE.MeshStandardMaterial({
        color: 0xe6a848,
        roughness: 0.25,
        metalness: 0.45,
        emissive: 0x3d280b,
        emissiveIntensity: 0.35
    }),
    slicegpt: new THREE.MeshStandardMaterial({
        color: 0x38bdf8,
        roughness: 0.3,
        metalness: 0.4
    }),
    macro_box: new THREE.MeshStandardMaterial({
        color: 0x172d47,
        roughness: 0.2,
        metalness: 0.8,
        transparent: true,
        opacity: 0.18,
        wireframe: false
    })
};

// Edge Outline Materials
const edgeMatDense = new THREE.LineBasicMaterial({ color: 0x4f748f, transparent: true, opacity: 0.5 });
const edgeMatSvd = new THREE.LineBasicMaterial({ color: 0x53ecd9, transparent: true, opacity: 0.85 });
const edgeMatMoe = new THREE.LineBasicMaterial({ color: 0xc895ea, transparent: true, opacity: 0.75 });
const edgeMatMla = new THREE.LineBasicMaterial({ color: 0xf6be68, transparent: true, opacity: 0.85 });
const edgeMatMacro = new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.65 });
const edgeMatResidual = new THREE.LineBasicMaterial({ color: 0x60a5fa, transparent: true, opacity: 0.6 });

// Custom Dynamic Module Component Materials Cache
const customMaterialsCache = new Map();
const customEdgeMaterialsCache = new Map();

function getOrCreateCustomMaterial(hexColor) {
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

function getOrCreateCustomEdgeMaterial(hexColor) {
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

// ---------------------------------------------------------------------------
// 4. State Management
// ---------------------------------------------------------------------------
let currentModel = 'distilbert';
let currentCompression = 'svd';
let spatialData = null;
let simulationSpeed = 1.0;
let enablePulses = true;
let activeNodeId = null;

// Scene Groups
const granularGroup = new THREE.Group();
const macroGroup = new THREE.Group();
const dimensionBadgesGroup = new THREE.Group();
scene.add(granularGroup);
scene.add(macroGroup);
scene.add(dimensionBadgesGroup);

const interactableMeshes = [];
const visualObjects = [];
const svdBottlenecks = [];
const moeNodes = [];
const macroMeshes = [];
const layerWidthProfilesBaseline = [];
const layerWidthProfilesCompressed = [];

// Instanced Token Simulation State
const NUM_TOKENS = 360;
const tokenGeometry = new THREE.SphereGeometry(0.1, 8, 8);
const trailGeometry = new THREE.SphereGeometry(0.065, 6, 6);

const tokenMaterialBaseline = new THREE.MeshBasicMaterial({ color: 0x9fd2ff });
const tokenMaterialCompressed = new THREE.MeshBasicMaterial({ color: 0x6effeb });
const trailMaterialBaseline = new THREE.MeshBasicMaterial({ color: 0x4a82b8, transparent: true, opacity: 0.35 });
const trailMaterialCompressed = new THREE.MeshBasicMaterial({ color: 0x289e90, transparent: true, opacity: 0.35 });

let tokenMeshBaseline = null;
let tokenMeshCompressed = null;
let trailMeshBaseline = null;
let trailMeshCompressed = null;

const tokenDataBaseline = [];
const tokenDataCompressed = [];
let maxGraphHeight = 16.0;

// MoE Palette
const EXPERT_COLORS = [
    new THREE.Color(0x53ecd9), // Cyan
    new THREE.Color(0xc895ea), // Purple
    new THREE.Color(0xf6be68), // Amber
    new THREE.Color(0xf472b6)  // Rose
];

// ---------------------------------------------------------------------------
// 5. Memory Cleanup & Lifecycle Disposal
// ---------------------------------------------------------------------------
function clearScene() {
    clearSelectionHighlight();
    for (const obj of visualObjects) {
        if (obj.parent) obj.parent.remove(obj);
        if (obj.geometry && obj.geometry !== tokenGeometry && obj.geometry !== trailGeometry) {
            obj.geometry.dispose();
        }
        if (obj.material) {
            const isShared = Object.values(materials).includes(obj.material) ||
                obj.material === edgeMatDense || obj.material === edgeMatSvd || obj.material === edgeMatMoe ||
                obj.material === edgeMatMla || obj.material === edgeMatMacro || obj.material === edgeMatResidual ||
                obj.material === tokenMaterialBaseline || obj.material === tokenMaterialCompressed ||
                obj.material === trailMaterialBaseline || obj.material === trailMaterialCompressed;

            if (!isShared) {
                if (obj.material.map) obj.material.map.dispose();
                obj.material.dispose();
            }
        }
    }
    visualObjects.length = 0;
    interactableMeshes.length = 0;
    macroMeshes.length = 0;
    svdBottlenecks.length = 0;
    moeNodes.length = 0;
    layerWidthProfilesBaseline.length = 0;
    layerWidthProfilesCompressed.length = 0;
    dataflowConduits.length = 0;

    const tokenMeshes = [tokenMeshBaseline, tokenMeshCompressed, trailMeshBaseline, trailMeshCompressed];
    tokenMeshes.forEach(mesh => {
        if (mesh) {
            if (mesh.parent) mesh.parent.remove(mesh);
            mesh.dispose();
        }
    });

    tokenMeshBaseline = null;
    tokenMeshCompressed = null;
    trailMeshBaseline = null;
    trailMeshCompressed = null;

    tokenDataBaseline.length = 0;
    tokenDataCompressed.length = 0;
    maxGraphHeight = 16.0;
}

// ---------------------------------------------------------------------------
// 6. Token Mesh & Trail Builders
// ---------------------------------------------------------------------------
function createTokenMesh(tokenDataArray, baseOffsetX, mat, trailMat) {
    const mainMesh = new THREE.InstancedMesh(tokenGeometry, mat, NUM_TOKENS);
    mainMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mainMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(NUM_TOKENS * 3), 3);
    mainMesh.raycast = () => {};

    const trailMesh = new THREE.InstancedMesh(trailGeometry, trailMat, NUM_TOKENS);
    trailMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    trailMesh.raycast = () => {};

    const dummy = new THREE.Object3D();
    const defaultColor = new THREE.Color(mat.color);

    for (let i = 0; i < NUM_TOKENS; i++) {
        const yInit = Math.random() * maxGraphHeight;
        const expert = Math.floor(Math.random() * 4);
        const isResidual = (i % 4 === 0); // 25% on residual highway (PyTorch: x = x + sublayer(norm(x)))
        const subStream = (i % 10 < 7) ? 'q' : ((i % 10 < 8) ? 'k' : 'v'); // Qwen GQA distribution
        const swigluStream = (i % 2 === 0) ? 'gate' : 'up'; // SwiGLU bifurcated branches
        const headIdx = i % 12; // 12 Attention heads
        const lateralNoise = (Math.random() - 0.5) * 1.4;

        tokenDataArray.push({
            y: yInit,
            speed: (0.026 + Math.random() * 0.034) * simulationSpeed,
            baseX: baseOffsetX,
            lateralOffset: (Math.random() - 0.5) * 1.8,
            lateralNoise: lateralNoise,
            isResidual: isResidual,
            subStream: subStream,
            swigluStream: swigluStream,
            headIdx: headIdx,
            expertTarget: expert,
            pulsePhase: Math.random() * Math.PI * 2
        });

        dummy.position.set(baseOffsetX, yInit, 0);
        dummy.updateMatrix();
        mainMesh.setMatrixAt(i, dummy.matrix);
        trailMesh.setMatrixAt(i, dummy.matrix);
        mainMesh.setColorAt(i, defaultColor);
    }
    if (mainMesh.instanceColor) mainMesh.instanceColor.needsUpdate = true;

    return { main: mainMesh, trail: trailMesh };
}

// ---------------------------------------------------------------------------
// 7. Floating 3D Billboard Labels
// ---------------------------------------------------------------------------
function createLayerLabelSprite(text, worldX, worldY, isMacro = false) {
    const canvas = document.createElement('canvas');
    canvas.width = 300;
    canvas.height = 72;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = isMacro ? 'rgba(10, 24, 44, 0.92)' : 'rgba(7, 14, 24, 0.88)';
    ctx.strokeStyle = isMacro ? '#38bdf8' : '#2b4d70';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(4, 4, 292, 64, 8);
    ctx.fill();
    ctx.stroke();

    ctx.font = isMacro ? 'bold 24px ui-monospace, Consolas, monospace' : 'bold 20px ui-monospace, Consolas, monospace';
    ctx.fillStyle = isMacro ? '#38bdf8' : '#7dd3fc';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 150, 36);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true, opacity: 0.9 });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.position.set(worldX, worldY, 0);
    sprite.scale.set(3.4, 0.82, 1.0);
    sprite.raycast = () => {};

    if (isMacro) {
        macroGroup.add(sprite);
    } else {
        granularGroup.add(sprite);
    }
    visualObjects.push(sprite);
    return sprite;
}

function createTensorBadgeSprite(node, offsetX) {
    const meta = node.metadata || {};
    const shape = meta.shape;
    if (!shape || shape.length !== 2) return null;

    const [dOut, dIn] = shape;
    let badgeText = `[${dOut} × ${dIn}]`;

    if (node.type === 'svd') {
        const r = meta.rank || Math.min(dOut, dIn);
        badgeText = (node.sub_type === 'v_factor') ? `[${dOut}×${dIn}] r=${r}` : `[${dOut}×${dIn}] r=${r}`;
    } else if (node.type === 'moe') {
        badgeText = (node.sub_type === 'router') ? `[${dOut}×${dIn}] Top-1` : `[${dOut}×${dIn}] Exp`;
    } else if (node.type === 'mla') {
        badgeText = `[${dOut}×${dIn}] Latent`;
    } else if (node.sub_type === 'sliced_linear') {
        badgeText = `[${dOut}×${dIn}] Cut 45%`;
    } else if (node.id.includes('k_proj') || node.id.includes('v_proj')) {
        if (dOut <= 256) badgeText = `[${dOut}×${dIn}] 2H GQA`;
        else badgeText = `[${dOut}×${dIn}] ${Math.round(dOut / 64)}H`;
    } else if (node.id.includes('q_proj')) {
        badgeText = `[${dOut}×${dIn}] (${Math.round(dOut / 64)}H)`;
    } else if (dOut >= 3072) {
        badgeText = `[${dOut}×${dIn}] ${(dOut / dIn).toFixed(1)}× FFN`;
    } else if (anyKey(node.id, ['q_lin', 'k_lin', 'v_lin', 'query', 'key', 'value'])) {
        badgeText = `[${dOut}×${dIn}] 12H`;
    }

    const canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');

    const isSvd = (node.type === 'svd');
    const isMoe = (node.type === 'moe');
    const isMla = (node.type === 'mla');
    const isGqa = (node.id.includes('k_proj') || node.id.includes('v_proj'));
    const isSwiglu = (node.id.includes('gate_proj') || node.id.includes('up_proj'));

    const borderColor = isSvd ? '#3acbb8' : (isMoe || isSwiglu ? '#c084fc' : (isGqa || isMla ? '#f59e0b' : '#38bdf8'));
    const textColor = isSvd ? '#5eead4' : (isMoe || isSwiglu ? '#e9d5ff' : (isGqa || isMla ? '#fde68a' : '#7dd3fc'));

    ctx.fillStyle = 'rgba(6, 13, 24, 0.94)';
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.roundRect(4, 4, 312, 56, 8);
    ctx.fill();
    ctx.stroke();

    ctx.font = 'bold 24px ui-monospace, Consolas, monospace';
    ctx.fillStyle = textColor;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(badgeText, 160, 32);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const spriteMat = new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        opacity: 0.0,
        depthWrite: false
    });
    const sprite = new THREE.Sprite(spriteMat);

    // Mount cleanly on front face of 3D box to eliminate any vertical/horizontal text overlap!
    const badgeZ = (node.z || 0) + (node.depth ? node.depth / 2 : 0.60) + 0.05;
    let badgeY = node.y;

    // Stagger vertically if adjacent narrow projections exist (e.g. GQA K and V)
    if (node.id.includes('k_proj')) badgeY -= 0.06;
    if (node.id.includes('v_proj')) badgeY += 0.06;

    sprite.position.set(node.x + offsetX, badgeY, badgeZ);

    // Dynamic width scale bounded by the node width to prevent spillover
    const badgeW = Math.max(0.65, Math.min(1.4, (node.width || 1.2) * 0.92));
    const badgeH = Math.min(0.24, (node.height || 0.42) * 0.62);
    sprite.scale.set(badgeW, badgeH, 1.0);
    sprite.raycast = () => {};

    dimensionBadgesGroup.add(sprite);
    visualObjects.push(sprite);
    return sprite;
}

// ---------------------------------------------------------------------------
// 8. Residual Highway Conduits & Inter-Module Wiring Engine
// ---------------------------------------------------------------------------
function buildResidualBypass(yStart, yEnd, offsetX) {
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
    granularGroup.add(line);
    visualObjects.push(line);
}

// ---------------------------------------------------------------------------
// 8b. Code-Accurate 3D Inter-Module Dataflow Conduits & Wiring Lines
// ---------------------------------------------------------------------------
const dataflowConduits = [];
const conduitMatBaseline = new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.70 });
const conduitMatCompressed = new THREE.LineBasicMaterial({ color: 0x2dd4bf, transparent: true, opacity: 0.85 });
const conduitMatGqaKv = new THREE.LineBasicMaterial({ color: 0xf59e0b, transparent: true, opacity: 0.85 });
const conduitMatSwiglu = new THREE.LineBasicMaterial({ color: 0xa855f7, transparent: true, opacity: 0.85 });
const conduitMatResidualAttn = new THREE.LineBasicMaterial({ color: 0x0284c7, transparent: true, opacity: 0.85 });
const conduitMatResidualFfn = new THREE.LineBasicMaterial({ color: 0x7c3aed, transparent: true, opacity: 0.85 });
const conduitMatInput = new THREE.LineBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.90 });
const conduitMatOutput = new THREE.LineBasicMaterial({ color: 0xf43f5e, transparent: true, opacity: 0.90 });

function createPortRing(x, y, z, colorHex) {
    const geo = new THREE.RingGeometry(0.035, 0.075, 16);
    const mat = new THREE.MeshBasicMaterial({ color: colorHex, side: THREE.DoubleSide, transparent: true, opacity: 0.85 });
    const ring = new THREE.Mesh(geo, mat);
    ring.rotation.x = Math.PI / 2;
    ring.position.set(x, y, z);
    ring.raycast = () => {};
    granularGroup.add(ring);
    visualObjects.push(ring);
    return ring;
}

function buildSmoothConduit(pStart, pEnd, mat, sourceId = '', targetId = '') {
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
    const line = new THREE.Line(geo, mat.clone());
    line.userData = {
        isConduit: true,
        sourceId: sourceId,
        targetId: targetId,
        _origMat: line.material
    };
    line.raycast = () => {};
    granularGroup.add(line);
    visualObjects.push(line);
    dataflowConduits.push(line);
    return line;
}

function buildCurvedBypassConduit(pStart, pEnd, lateralX, mat, sourceId = '', targetId = '') {
    const curve = new THREE.CubicBezierCurve3(
        pStart,
        new THREE.Vector3(lateralX, pStart.y + (pEnd.y - pStart.y) * 0.25, pStart.z),
        new THREE.Vector3(lateralX, pEnd.y - (pEnd.y - pStart.y) * 0.25, pEnd.z),
        pEnd
    );
    const pts = curve.getPoints(16);
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.Line(geo, mat.clone());
    line.userData = {
        isConduit: true,
        isResidualHighway: true,
        sourceId: sourceId,
        targetId: targetId,
        _origMat: line.material
    };
    line.raycast = () => {};
    granularGroup.add(line);
    visualObjects.push(line);
    dataflowConduits.push(line);
    return line;
}

function buildDataflowConduits(nodes, offsetX, isCompressedSide, modelKey, paradigm) {
    if (!nodes || nodes.length === 0) return;

    const baseMat = isCompressedSide ? conduitMatCompressed : conduitMatBaseline;
    const primaryHex = isCompressedSide ? 0x2dd4bf : 0x38bdf8;

    const inputNode = nodes.find(n => n.id && n.id.includes('input_embeddings'));
    const outputNode = nodes.find(n => n.id && n.id.includes('output_head'));

    const layerMap = new Map();
    nodes.forEach(n => {
        if (n.id && (n.id.includes('input_embeddings') || n.id.includes('output_head'))) return;
        const lIdx = (n.parent_layer_idx !== undefined) ? n.parent_layer_idx : (n.parent_layer_index || 0);
        if (!layerMap.has(lIdx)) layerMap.set(lIdx, []);
        layerMap.get(lIdx).push(n);
    });

    const sortedLayers = Array.from(layerMap.keys()).sort((a, b) => a - b);
    let prevLayerTopNode = null;

    // 0. Connect Input Embeddings to Layer 0 Input Terminal
    if (inputNode && sortedLayers.length > 0) {
        const l0Nodes = layerMap.get(sortedLayers[0]);
        if (l0Nodes && l0Nodes.length > 0) {
            const l0First = l0Nodes[0];
            const l0InY = l0First.y - (l0First.height || 0.42) / 2 - 0.25;
            const inPt = new THREE.Vector3(offsetX, inputNode.y + (inputNode.height || 0.38) / 2, 0);
            const l0Pt = new THREE.Vector3(offsetX, l0InY, 0);
            buildSmoothConduit(inPt, l0Pt, conduitMatInput, inputNode.id, l0First.id);
            createPortRing(offsetX, inPt.y, 0, 0x10b981);
        }
    }

    sortedLayers.forEach(lIdx => {
        const lNodes = layerMap.get(lIdx).sort((a, b) => a.y - b.y);
        if (lNodes.length === 0) return;

        const qNodes = lNodes.filter(n => anyKey(n.id, ['q_lin', 'query', 'q_proj']));
        const kNodes = lNodes.filter(n => anyKey(n.id, ['k_lin', 'key', 'k_proj']));
        const vNodes = lNodes.filter(n => anyKey(n.id, ['v_lin', 'value', 'v_proj']));
        const vFactorNodes = lNodes.filter(n => (n.sub_type === 'v_factor'));
        const uFactorNodes = lNodes.filter(n => (n.sub_type === 'u_factor'));
        const attnOutNodes = lNodes.filter(n => anyKey(n.id, ['out_lin', 'output.dense', 'o_proj', 'attn.c_proj']) && !anyKey(n.id, ['mlp', 'ffn', 'intermediate']));
        const ffnUpNodes = lNodes.filter(n => anyKey(n.id, ['lin1', 'intermediate', 'c_fc', 'gate_proj', 'up_proj']));
        const ffnDownNodes = lNodes.filter(n => anyKey(n.id, ['lin2', 'output', 'down_proj', 'mlp.c_proj']) && !anyKey(n.id, ['attention', 'self']));
        const attnAddNode = lNodes.find(n => n.id && n.id.includes('attn.residual_add'));
        const ffnAddNode = lNodes.find(n => n.id && n.id.includes('ffn.residual_add'));

        const firstNode = lNodes[0];
        const lastNode = lNodes[lNodes.length - 1];

        // 1. Layer Input Bus Terminal
        const layerInY = firstNode.y - (firstNode.height || 0.42) / 2 - 0.25;
        const layerInPt = new THREE.Vector3(offsetX, layerInY, 0);
        createPortRing(offsetX, layerInY, 0, primaryHex);

        // Inter-layer link from previous layer
        if (prevLayerTopNode) {
            const prevTopY = prevLayerTopNode.y + (prevLayerTopNode.height || 0.42) / 2 + 0.15;
            buildSmoothConduit(
                new THREE.Vector3(offsetX, prevTopY, 0),
                layerInPt,
                baseMat,
                prevLayerTopNode.id,
                firstNode.id
            );
        }

        // 2. Wiring according to architecture
        if (isCompressedSide && modelKey === 'qwen') {
            // === QWEN GQA & SwiGLU Wiring ===
            const qN = qNodes[0];
            const kN = kNodes[0];
            const vN = vNodes[0];
            const oN = attnOutNodes[0];

            if (qN && kN && vN) {
                const qIn = new THREE.Vector3(qN.x + offsetX, qN.y - qN.height / 2, qN.z || 0);
                const kIn = new THREE.Vector3(kN.x + offsetX, kN.y - kN.height / 2, kN.z || 0);
                const vIn = new THREE.Vector3(vN.x + offsetX, vN.y - vN.height / 2, vN.z || 0);

                buildSmoothConduit(layerInPt, qIn, baseMat, 'input', qN.id);
                buildSmoothConduit(layerInPt, kIn, conduitMatGqaKv, 'input', kN.id);
                buildSmoothConduit(layerInPt, vIn, conduitMatGqaKv, 'input', vN.id);

                createPortRing(qIn.x, qIn.y, qIn.z, 0x38bdf8);
                createPortRing(kIn.x, kIn.y, kIn.z, 0xf59e0b);
                createPortRing(vIn.x, vIn.y, vIn.z, 0xf59e0b);

                if (oN) {
                    const oIn = new THREE.Vector3(oN.x + offsetX, oN.y - oN.height / 2, oN.z || 0);
                    const qOut = new THREE.Vector3(qN.x + offsetX, qN.y + qN.height / 2, qN.z || 0);
                    const kOut = new THREE.Vector3(kN.x + offsetX, kN.y + kN.height / 2, kN.z || 0);
                    const vOut = new THREE.Vector3(vN.x + offsetX, vN.y + vN.height / 2, vN.z || 0);

                    buildSmoothConduit(qOut, oIn, baseMat, qN.id, oN.id);
                    buildSmoothConduit(kOut, oIn, conduitMatGqaKv, kN.id, oN.id);
                    buildSmoothConduit(vOut, oIn, conduitMatGqaKv, vN.id, oN.id);

                    createPortRing(oIn.x, oIn.y, oIn.z, 0x2dd4bf);
                }
            }

            // Residual Addition #1 Link & Skip Highway Cable
            let ffnSourcePt = oN ? new THREE.Vector3(oN.x + offsetX, oN.y + oN.height / 2, oN.z || 0) : layerInPt;
            if (attnAddNode && oN) {
                const addIn = new THREE.Vector3(attnAddNode.x + offsetX, attnAddNode.y - attnAddNode.height / 2, 0);
                const addOut = new THREE.Vector3(attnAddNode.x + offsetX, attnAddNode.y + attnAddNode.height / 2, 0);
                const oOut = new THREE.Vector3(oN.x + offsetX, oN.y + oN.height / 2, oN.z || 0);

                buildSmoothConduit(oOut, addIn, baseMat, oN.id, attnAddNode.id);
                createPortRing(addIn.x, addIn.y, addIn.z, 0x0284c7);

                // Residual Highway #1: Curved bypass from layer input bus (+ X_in)
                buildCurvedBypassConduit(layerInPt, addIn, offsetX - 1.85, conduitMatResidualAttn, 'residual_skip_1', attnAddNode.id);
                createPortRing(addOut.x, addOut.y, addOut.z, 0x0284c7);
                ffnSourcePt = addOut;
            }

            const gateN = lNodes.find(n => n.id.includes('gate_proj'));
            const upN = lNodes.find(n => n.id.includes('up_proj'));
            const downN = lNodes.find(n => n.id.includes('down_proj'));

            if (gateN && upN) {
                const gateIn = new THREE.Vector3(gateN.x + offsetX, gateN.y - gateN.height / 2, gateN.z || 0);
                const upIn = new THREE.Vector3(upN.x + offsetX, upN.y - upN.height / 2, upN.z || 0);

                buildSmoothConduit(ffnSourcePt, gateIn, conduitMatSwiglu, 'ffn_in', gateN.id);
                buildSmoothConduit(ffnSourcePt, upIn, conduitMatSwiglu, 'ffn_in', upN.id);

                createPortRing(gateIn.x, gateIn.y, gateIn.z, 0xa855f7);
                createPortRing(upIn.x, upIn.y, upIn.z, 0x8b5cf6);

                if (downN) {
                    const mulJunction = new THREE.Vector3(offsetX, (gateN.y + gateN.height / 2 + downN.y - downN.height / 2) / 2, 0);
                    const gateOut = new THREE.Vector3(gateN.x + offsetX, gateN.y + gateN.height / 2, gateN.z || 0);
                    const upOut = new THREE.Vector3(upN.x + offsetX, upN.y + upN.height / 2, upN.z || 0);
                    const downIn = new THREE.Vector3(downN.x + offsetX, downN.y - downN.height / 2, downN.z || 0);

                    buildSmoothConduit(gateOut, mulJunction, conduitMatSwiglu, gateN.id, 'mul_junction');
                    buildSmoothConduit(upOut, mulJunction, conduitMatSwiglu, upN.id, 'mul_junction');
                    buildSmoothConduit(mulJunction, downIn, baseMat, 'mul_junction', downN.id);

                    createPortRing(mulJunction.x, mulJunction.y, mulJunction.z, 0xa855f7);
                    createPortRing(downIn.x, downIn.y, downIn.z, 0x2dd4bf);

                    // Residual Addition #2 Link & Skip Highway Cable
                    if (ffnAddNode) {
                        const downOut = new THREE.Vector3(downN.x + offsetX, downN.y + downN.height / 2, 0);
                        const ffnAddIn = new THREE.Vector3(ffnAddNode.x + offsetX, ffnAddNode.y - ffnAddNode.height / 2, 0);
                        const ffnAddOut = new THREE.Vector3(ffnAddNode.x + offsetX, ffnAddNode.y + ffnAddNode.height / 2, 0);

                        buildSmoothConduit(downOut, ffnAddIn, baseMat, downN.id, ffnAddNode.id);
                        createPortRing(ffnAddIn.x, ffnAddIn.y, ffnAddIn.z, 0x7c3aed);

                        // Residual Highway #2: Curved bypass from intermediate token state (+ X_mid)
                        buildCurvedBypassConduit(ffnSourcePt, ffnAddIn, offsetX + 1.85, conduitMatResidualFfn, 'residual_skip_2', ffnAddNode.id);
                        createPortRing(ffnAddOut.x, ffnAddOut.y, ffnAddOut.z, 0x7c3aed);
                    }
                }
            }
        } else if (isCompressedSide && (vFactorNodes.length > 0 || uFactorNodes.length > 0)) {
            // === SVD Low-Rank Bottleneck Wiring ===
            const vN = vFactorNodes[0] || lNodes[0];
            const uN = uFactorNodes[0] || lNodes[1];

            if (vN) {
                const vIn = new THREE.Vector3(vN.x + offsetX, vN.y - vN.height / 2, vN.z || 0);
                buildSmoothConduit(layerInPt, vIn, baseMat, 'input', vN.id);
                createPortRing(vIn.x, vIn.y, vIn.z, 0x3acbb8);
            }

            if (vN && uN) {
                const vOut = new THREE.Vector3(vN.x + offsetX, vN.y + vN.height / 2, vN.z || 0);
                const uIn = new THREE.Vector3(uN.x + offsetX, uN.y - uN.height / 2, uN.z || 0);
                buildSmoothConduit(vOut, uIn, baseMat, vN.id, uN.id);
                createPortRing(vOut.x, vOut.y, vOut.z, 0x3acbb8);
                createPortRing(uIn.x, uIn.y, uIn.z, 0x2dd4bf);
            }

            const postAttn = attnOutNodes[0] || (uN ? uN : null);
            let ffnSourcePt = uN ? new THREE.Vector3(uN.x + offsetX, uN.y + uN.height / 2, uN.z || 0) : layerInPt;

            if (attnAddNode && postAttn) {
                const postOut = new THREE.Vector3(postAttn.x + offsetX, postAttn.y + postAttn.height / 2, 0);
                const addIn = new THREE.Vector3(attnAddNode.x + offsetX, attnAddNode.y - attnAddNode.height / 2, 0);
                const addOut = new THREE.Vector3(attnAddNode.x + offsetX, attnAddNode.y + attnAddNode.height / 2, 0);

                buildSmoothConduit(postOut, addIn, baseMat, postAttn.id, attnAddNode.id);
                createPortRing(addIn.x, addIn.y, addIn.z, 0x0284c7);

                // Residual Highway #1 Bypass (+ X_in)
                buildCurvedBypassConduit(layerInPt, addIn, offsetX - 1.85, conduitMatResidualAttn, 'residual_skip_1', attnAddNode.id);
                createPortRing(addOut.x, addOut.y, addOut.z, 0x0284c7);
                ffnSourcePt = addOut;
            }

            if (ffnUpNodes[0] && ffnDownNodes[0]) {
                const upN = ffnUpNodes[0];
                const downN = ffnDownNodes[0];
                const upIn = new THREE.Vector3(upN.x + offsetX, upN.y - upN.height / 2, 0);
                const upOut = new THREE.Vector3(upN.x + offsetX, upN.y + upN.height / 2, 0);
                const downIn = new THREE.Vector3(downN.x + offsetX, downN.y - downN.height / 2, 0);
                const downOut = new THREE.Vector3(downN.x + offsetX, downN.y + downN.height / 2, 0);

                buildSmoothConduit(ffnSourcePt, upIn, baseMat, 'attn_add', upN.id);
                buildSmoothConduit(upOut, downIn, baseMat, upN.id, downN.id);
                createPortRing(upIn.x, upIn.y, upIn.z, 0x2dd4bf);
                createPortRing(upOut.x, upOut.y, upOut.z, 0x2dd4bf);
                createPortRing(downIn.x, downIn.y, downIn.z, 0x2dd4bf);

                if (ffnAddNode) {
                    const ffnAddIn = new THREE.Vector3(ffnAddNode.x + offsetX, ffnAddNode.y - ffnAddNode.height / 2, 0);
                    const ffnAddOut = new THREE.Vector3(ffnAddNode.x + offsetX, ffnAddNode.y + ffnAddNode.height / 2, 0);

                    buildSmoothConduit(downOut, ffnAddIn, baseMat, downN.id, ffnAddNode.id);
                    createPortRing(ffnAddIn.x, ffnAddIn.y, ffnAddIn.z, 0x7c3aed);

                    // Residual Highway #2 Bypass (+ X_mid)
                    buildCurvedBypassConduit(ffnSourcePt, ffnAddIn, offsetX + 1.85, conduitMatResidualFfn, 'residual_skip_2', ffnAddNode.id);
                    createPortRing(ffnAddOut.x, ffnAddOut.y, ffnAddOut.z, 0x7c3aed);
                }
            }
        } else {
            // === Dense Baseline & General Sequential Conduits ===
            for (let i = 0; i < lNodes.length - 1; i++) {
                const curr = lNodes[i];
                const next = lNodes[i + 1];

                const pStart = new THREE.Vector3(curr.x + offsetX, curr.y + curr.height / 2, curr.z || 0);
                const pEnd = new THREE.Vector3(next.x + offsetX, next.y - next.height / 2, next.z || 0);

                buildSmoothConduit(pStart, pEnd, baseMat, curr.id, next.id);
                createPortRing(pStart.x, pStart.y, pStart.z, primaryHex);
                createPortRing(pEnd.x, pEnd.y, pEnd.z, primaryHex);
            }

            const fIn = new THREE.Vector3(firstNode.x + offsetX, firstNode.y - firstNode.height / 2, firstNode.z || 0);
            buildSmoothConduit(layerInPt, fIn, baseMat, 'input', firstNode.id);
            createPortRing(fIn.x, fIn.y, fIn.z, primaryHex);

            // Add bypass lines for baseline if residual nodes exist
            if (attnAddNode) {
                const addIn = new THREE.Vector3(attnAddNode.x + offsetX, attnAddNode.y - attnAddNode.height / 2, 0);
                buildCurvedBypassConduit(layerInPt, addIn, offsetX - 1.85, conduitMatResidualAttn, 'baseline_skip_1', attnAddNode.id);
                if (ffnAddNode) {
                    const ffnAddIn = new THREE.Vector3(ffnAddNode.x + offsetX, ffnAddNode.y - ffnAddNode.height / 2, 0);
                    const addOut = new THREE.Vector3(attnAddNode.x + offsetX, attnAddNode.y + attnAddNode.height / 2, 0);
                    buildCurvedBypassConduit(addOut, ffnAddIn, offsetX + 1.85, conduitMatResidualFfn, 'baseline_skip_2', ffnAddNode.id);
                }
            }
        }

        prevLayerTopNode = lastNode;
    });

    // 3. Connect Final Layer Top Node to Output Head
    if (outputNode && prevLayerTopNode) {
        const lastTopY = prevLayerTopNode.y + (prevLayerTopNode.height || 0.42) / 2 + 0.15;
        const outBot = new THREE.Vector3(offsetX, outputNode.y - (outputNode.height || 0.38) / 2, 0);
        buildSmoothConduit(new THREE.Vector3(offsetX, lastTopY, 0), outBot, conduitMatOutput, prevLayerTopNode.id, outputNode.id);
        createPortRing(offsetX, outputNode.y - (outputNode.height || 0.38) / 2, 0, 0xf43f5e);
    }
}

// ---------------------------------------------------------------------------
// 9. Macro Block Builder (LOD 0: Overview / Far Zoom)
// ---------------------------------------------------------------------------
function buildMacroBlocks(macroNodes, offsetX, isCompressedSide) {
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
        createLayerLabelSprite(lbl, offsetX, node.y + node.height / 2 + 0.45, true);
    });
}

// ---------------------------------------------------------------------------
// 10. Granular Node Geometry Builder (LOD 1: Detailed / Near Zoom)
// ---------------------------------------------------------------------------
function buildMeshNodes(nodes, offsetX, isCompressedSide) {
    let topY = 0;
    const layerTracker = new Set();
    const layerMinMax = new Map();

    // SVD bottleneck pairing
    for (let i = 0; i < nodes.length; i++) {
        if (nodes[i].type === 'svd' && nodes[i].sub_type === 'v_factor') {
            const vNode = nodes[i];
            const uNode = nodes[i + 1] && nodes[i + 1].type === 'svd' ? nodes[i + 1] : null;
            if (uNode) {
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
        let mat = materials.dense;
        let edgeMat = edgeMatDense;

        if (node.custom_color) {
            mat = getOrCreateCustomMaterial(node.custom_color);
            edgeMat = getOrCreateCustomEdgeMaterial(node.custom_color);
        } else if (node.type === 'svd') {
            mat = materials.svd;
            edgeMat = edgeMatSvd;
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
        const isAttn = anyKey(node.id, ['q_lin', 'k_lin', 'v_lin', 'query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj']);
        if (isAttn && node.width > 0.4) {
            let numHeads = 12;
            const meta = node.metadata || {};
            if (meta.shape && meta.shape.length === 2) {
                const dOut = meta.shape[0];
                if (dOut <= 256) numHeads = Math.max(1, Math.round(dOut / 64)); // e.g. 2 for Qwen GQA
                else numHeads = Math.max(1, Math.round(dOut / 64)); // e.g. 14 for Qwen Q, 12 for RoBERTa
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
        createTensorBadgeSprite(node, offsetX);

        // Record layer width profile for particle stream scaling
        const profileTarget = isCompressedSide ? layerWidthProfilesCompressed : layerWidthProfilesBaseline;
        profileTarget.push({
            yMin: node.y - node.height / 2,
            yMax: node.y + node.height / 2,
            width: node.width
        });

        granularGroup.add(mesh);
        visualObjects.push(mesh);
        interactableMeshes.push(mesh);

        if (node.y > topY) topY = node.y;

        // Track Layer Y bounds for Residual Highways
        const lIdx = node.parent_layer_index ?? 0;
        if (!layerMinMax.has(lIdx)) {
            layerMinMax.set(lIdx, { minY: node.y, maxY: node.y });
        } else {
            const bounds = layerMinMax.get(lIdx);
            bounds.minY = Math.min(bounds.minY, node.y);
            bounds.maxY = Math.max(bounds.maxY, node.y);
        }

        // Layer Billboard Tag
        if (node.parent_layer_name && !layerTracker.has(node.parent_layer_name)) {
            layerTracker.add(node.parent_layer_name);
            const labelX = offsetX + (isCompressedSide ? 3.4 : -3.4);
            createLayerLabelSprite(node.parent_layer_name, labelX, node.y, false);
        }

        // MoE connector branching lines
        if (node.type === 'moe' && node.sub_type === 'expert') {
            const lineMat = new THREE.LineBasicMaterial({
                color: 0xc895ea,
                transparent: true,
                opacity: 0.6
            });
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

        if (node.type === 'moe' && node.sub_type === 'router') {
            moeNodes.push({ ...node, worldX: node.x + offsetX });
        }
    });

    // Build Residual Bypass Conduits per Layer
    layerMinMax.forEach((bounds) => {
        if (bounds.maxY - bounds.minY > 1.5) {
            buildResidualBypass(bounds.minY - 0.2, bounds.maxY + 0.2, offsetX);
        }
    });

    // Build Code-Accurate 3D Inter-Module Wiring Conduits
    buildDataflowConduits(nodes, offsetX, isCompressedSide, currentModel, (activeResearchModule && activeResearchModule.paradigm) || 'svd');

    // Base Grid Ground
    const grid = new THREE.GridHelper(18, 18, 0x1c334d, 0x0c1726);
    grid.position.set(offsetX, 0, 0);
    grid.raycast = () => {};
    scene.add(grid);
    visualObjects.push(grid);

    return topY;
}

function anyKey(str, keys) {
    const s = str.toLowerCase();
    return keys.some(k => s.includes(k));
}

// ---------------------------------------------------------------------------
// 11. Hierarchical Tree Explorer Builder
// ---------------------------------------------------------------------------
function buildHierarchicalTree(modelKey, moduleKey, modelData) {
    const container = document.getElementById('layer-tree-container');
    const countEl = document.getElementById('insp-layer-count');
    if (!container) return;

    container.innerHTML = '';

    const moduleInfo = (modelData.modules && modelData.modules[moduleKey]) || {
        compressed_nodes: modelData.compressed_nodes,
        name: 'SVD Low-Rank'
    };

    const layersMap = new Map();

    const processStream = (nodes, streamType) => {
        nodes.forEach(node => {
            const lIdx = node.parent_layer_index !== null && node.parent_layer_index !== undefined
                ? node.parent_layer_index
                : 999;
            const lName = node.parent_layer_name || (lIdx === 999 ? 'Global Projections' : `Layer ${lIdx}`);

            if (!layersMap.has(lIdx)) {
                layersMap.set(lIdx, {
                    name: lName,
                    index: lIdx,
                    baseline: [],
                    compressed: []
                });
            }
            layersMap.get(lIdx)[streamType].push(node);
        });
    };

    processStream(modelData.baseline_nodes, 'baseline');
    processStream(moduleInfo.compressed_nodes, 'compressed');

    const totalNodes = modelData.baseline_nodes.length + moduleInfo.compressed_nodes.length;
    if (countEl) countEl.innerText = `${totalNodes} modules`;

    const rootEl = document.createElement('div');
    rootEl.className = 'tree-model-root';

    const rootHeader = document.createElement('div');
    rootHeader.className = 'tree-model-header';
    rootHeader.innerHTML = `<span>⚡</span> <span>${modelKey.toUpperCase()} · ${moduleInfo.name}</span>`;
    rootEl.appendChild(rootHeader);

    const sortedLayers = Array.from(layersMap.values()).sort((a, b) => a.index - b.index);

    sortedLayers.forEach((layerGroup, idx) => {
        const layerNode = document.createElement('div');
        layerNode.className = `tree-layer-node ${idx === 0 ? 'open' : ''}`;
        layerNode.dataset.layerIndex = layerGroup.index;

        const layerHeader = document.createElement('div');
        layerHeader.className = 'tree-layer-header';
        layerHeader.innerHTML = `
            <span class="tree-chevron">▶</span>
            <span class="tree-layer-title">${layerGroup.name}</span>
            <small style="color: var(--text-faint); margin-left: auto;">(${layerGroup.baseline.length + layerGroup.compressed.length})</small>
        `;

        layerHeader.addEventListener('click', (e) => {
            e.stopPropagation();
            layerNode.classList.toggle('open');
        });

        const layerContent = document.createElement('div');
        layerContent.className = 'tree-layer-content';

        if (layerGroup.baseline.length > 0) {
            const bGroup = document.createElement('div');
            bGroup.className = 'tree-stream-group';
            bGroup.innerHTML = `<div class="tree-stream-label">Dense Baseline</div>`;
            layerGroup.baseline.sort((a, b) => a.sequence_index - b.sequence_index).forEach(n => {
                bGroup.appendChild(createLeafNode(n, false));
            });
            layerContent.appendChild(bGroup);
        }

        if (layerGroup.compressed.length > 0) {
            const cGroup = document.createElement('div');
            cGroup.className = 'tree-stream-group';
            cGroup.innerHTML = `<div class="tree-stream-label">${moduleInfo.name}</div>`;
            layerGroup.compressed.sort((a, b) => a.sequence_index - b.sequence_index).forEach(n => {
                cGroup.appendChild(createLeafNode(n, true));
            });
            layerContent.appendChild(cGroup);
        }

        layerNode.appendChild(layerHeader);
        layerNode.appendChild(layerContent);
        rootEl.appendChild(layerNode);
    });

    container.appendChild(rootEl);
}

function createLeafNode(node, isCompressed) {
    const leaf = document.createElement('div');
    leaf.className = `tree-leaf ${isCompressed ? 'type-svd' : 'type-dense'}`;
    leaf.dataset.id = node.id;
    leaf.dataset.label = (node.label || node.id).toLowerCase();

    let cleanLabel = node.label || node.id;
    cleanLabel = cleanLabel.replace(/^(transformer|encoder|model)?\.?layer(s)?\.\d+\./, '');
    cleanLabel = cleanLabel.replace(/^h\.\d+\./, '');

    let badgeText = 'Dense';
    if (isCompressed) {
        if (node.sub_type === 'v_factor') badgeText = 'V-Rank';
        else if (node.sub_type === 'u_factor') badgeText = 'U-Exp';
        else if (node.sub_type === 'router') badgeText = 'Router';
        else if (node.sub_type === 'expert') badgeText = 'Expert';
        else if (node.sub_type === 'latent_kv') badgeText = 'Latent';
        else if (node.sub_type === 'sliced_linear') badgeText = 'Sliced';
        else badgeText = 'Comp';
    }

    leaf.innerHTML = `
        <span class="leaf-name" title="${node.label}">${cleanLabel}</span>
        <span class="leaf-badge">${badgeText}</span>
    `;

    leaf.addEventListener('click', (e) => {
        e.stopPropagation();
        focusNodeById(node.id);
    });

    return leaf;
}

function setupTreeFilter() {
    const searchInput = document.getElementById('tree-search-input');
    const clearBtn = document.getElementById('btn-tree-clear');
    if (!searchInput) return;

    const filterTree = () => {
        const query = searchInput.value.trim().toLowerCase();
        const leaves = document.querySelectorAll('.tree-leaf');
        const layerNodes = document.querySelectorAll('.tree-layer-node');

        if (!query) {
            leaves.forEach(l => l.style.display = '');
            layerNodes.forEach(ln => {
                ln.style.display = '';
            });
            return;
        }

        layerNodes.forEach(ln => {
            let hasVisibleLeaf = false;
            const lnLeaves = ln.querySelectorAll('.tree-leaf');
            lnLeaves.forEach(leaf => {
                const label = leaf.dataset.label || '';
                const id = leaf.dataset.id || '';
                const match = label.includes(query) || id.includes(query);
                leaf.style.display = match ? '' : 'none';
                if (match) hasVisibleLeaf = true;
            });

            ln.style.display = hasVisibleLeaf ? '' : 'none';
            if (hasVisibleLeaf) {
                ln.classList.add('open');
            }
        });
    };

    searchInput.addEventListener('input', filterTree);
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            searchInput.value = '';
            filterTree();
            searchInput.focus();
        });
    }
}

// ---------------------------------------------------------------------------
// 12. Canonical Raycasting & Detailed Explanations
// ---------------------------------------------------------------------------
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const tooltip = document.getElementById('tooltip');
const inspector = document.getElementById('inspector');

function resolveNodeFromIntersection(intersection) {
    if (!intersection || !intersection.object) return null;
    let obj = intersection.object;

    while (obj && (!obj.userData || !obj.userData.id) && obj.parent) {
        obj = obj.parent;
    }

    if (obj && obj.userData && obj.userData.id) {
        if (!obj.userData.metadata || obj.userData.metadata.params === undefined) {
            console.error("resolveNodeFromIntersection found object without valid metadata:", obj.name, obj.uuid, obj);
        }
        return { mesh: obj, data: obj.userData };
    }
    return null;
}

function updateInspectorWithNode(node, isLeft) {
    if (!node) return;
    activeNodeId = node.id;

    document.querySelectorAll('.tree-leaf').forEach(leaf => {
        const isActive = (leaf.dataset.id === node.id);
        leaf.classList.toggle('active', isActive);
        if (isActive) {
            const parentLayer = leaf.closest('.tree-layer-node');
            if (parentLayer) parentLayer.classList.add('open');
            leaf.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
    });

    const meta = node.metadata || {};

    const badgeEl = document.getElementById('insp-type-badge');
    if (badgeEl) badgeEl.innerText = (node.type || 'LAYER').toUpperCase();

    const titleEl = document.getElementById('insp-title');
    if (titleEl) titleEl.innerText = node.label || node.id || '—';

    const shapeEl = document.getElementById('insp-shape');
    if (shapeEl) {
        shapeEl.innerText = (meta.shape && Array.isArray(meta.shape))
            ? `[${meta.shape.join(' × ')}]`
            : '—';
    }

    const paramsEl = document.getElementById('insp-params');
    if (paramsEl) {
        paramsEl.innerText = meta.params || (meta.params_raw ? meta.params_raw.toLocaleString() : '—');
    }

    const storageEl = document.getElementById('insp-storage');
    if (storageEl) {
        storageEl.innerText = meta.storage || '—';
    }

    const effEl = document.getElementById('insp-efficiency');
    if (effEl) {
        effEl.innerText = meta.compression || '1.0x Baseline';
    }

    // Comparison Card
    const compBaselineEl = document.getElementById('insp-comp-baseline');
    const compCompressedEl = document.getElementById('insp-comp-compressed');
    const savingsBadgeEl = document.getElementById('insp-savings-badge');

    const denseEq = meta.dense_equivalent_params_formatted || '590.6K';
    if (compBaselineEl) compBaselineEl.innerText = `${denseEq} (Full-Rank)`;
    if (compCompressedEl) compCompressedEl.innerText = `${meta.params || '49.2K'} (${meta.type || 'Comp'})`;

    if (savingsBadgeEl) {
        const ratio = meta.compression_ratio || 1.0;
        if (ratio > 1.0) {
            const pct = Math.round((1 - 1 / ratio) * 100);
            savingsBadgeEl.innerText = `-${pct}% MEMORY`;
            savingsBadgeEl.style.color = '#53ecd9';
        } else {
            savingsBadgeEl.innerText = 'BASELINE 1.0x';
            savingsBadgeEl.style.color = '#8e9fb5';
        }
    }

    // Deep Dimensional & Volumetric Telemetry
    const [dOut, dIn] = (meta.shape && meta.shape.length === 2) ? meta.shape : [768, 768];
    const dimsEl = document.getElementById('insp-tensor-dims');
    if (dimsEl) {
        dimsEl.innerText = (meta.shape && meta.shape.length === 2)
            ? `[${dOut} × ${dIn}]`
            : (meta.shape ? `[${meta.shape.join(' × ')}]` : '—');
    }

    const aspectEl = document.getElementById('insp-aspect-ratio');
    if (aspectEl) {
        if (meta.shape && meta.shape.length === 2) {
            const ratio = (Math.max(dOut, dIn) / Math.max(1, Math.min(dOut, dIn))).toFixed(2);
            aspectEl.innerText = `${ratio} : 1`;
        } else {
            aspectEl.innerText = '1.00 : 1';
        }
    }

    const volEl = document.getElementById('insp-spatial-vol');
    if (volEl) {
        const w = node.width || 1.5;
        const h = node.height || 0.45;
        const d = node.depth || 1.5;
        const vol = (w * h * d).toFixed(2);
        volEl.innerText = `${vol} units³`;
    }

    const headsEl = document.getElementById('insp-heads-info');
    if (headsEl) {
        let headInfo = '—';
        if (node.isMacro) {
            headInfo = `Macro Block (${meta.shape ? meta.shape[0] : 6} Subtensors)`;
        } else if (node.type === 'svd') {
            const r = meta.rank || Math.min(dOut, dIn);
            headInfo = `SVD Rank r=${r} Subspace`;
        } else if (node.type === 'moe') {
            if (node.sub_type === 'router') {
                headInfo = `Top-1 Gating (${meta.num_experts || 4} Experts)`;
            } else {
                headInfo = `Expert ${(meta.expert_idx ?? 0) + 1} of ${meta.num_experts || 4}`;
            }
        } else if (node.type === 'mla') {
            headInfo = `Latent KV (Rank ${meta.rank || 128})`;
        } else if (node.sub_type === 'sliced_linear') {
            headInfo = `SliceGPT Pruned Channel`;
        } else if (node.id.includes('k_proj') || node.id.includes('v_proj')) {
            if (dOut <= 256) {
                headInfo = `2 KV Heads (GQA 7:1 Sharing)`;
            } else {
                headInfo = `${Math.round(dOut / 64)} Heads (64d / head)`;
            }
        } else if (node.id.includes('q_proj')) {
            headInfo = `${Math.round(dOut / 64)} Query Heads (64d)`;
        } else if (anyKey(node.id, ['q_lin', 'k_lin', 'v_lin', 'query', 'key', 'value'])) {
            headInfo = `${Math.round(dOut / 64)} Attention Heads (64d)`;
        } else if (anyKey(node.id, ['intermediate', 'lin1', 'gate_proj', 'up_proj', 'c_fc'])) {
            const expRatio = (dOut / dIn).toFixed(1);
            headInfo = `FFN ${expRatio}× Expansion`;
        } else if (anyKey(node.id, ['out_lin', 'output', 'down_proj', 'c_proj'])) {
            headInfo = `Projection Output Map`;
        } else {
            headInfo = `Dense Linear Map`;
        }
        headsEl.innerText = headInfo;
    }

    // Sync floating .detail-card (Attention Lab / Polymetis style)
    const kickerEl = document.getElementById('detail-kicker');
    const detTitleEl = document.getElementById('detail-title');
    const detDescEl = document.getElementById('detail-desc');
    const fShapeEl = document.getElementById('fact-shape');
    const fParamsEl = document.getElementById('fact-params');
    const fSubspaceEl = document.getElementById('fact-subspace');
    const fDeltaEl = document.getElementById('fact-delta');

    if (kickerEl) kickerEl.innerText = `${(node.type || 'LAYER').toUpperCase()} · ${isLeft ? 'BASELINE' : 'COMPRESSED'}`;
    if (detTitleEl) detTitleEl.innerText = node.label || node.id || 'Layer Component';
    if (detDescEl) {
        let plainDesc = node.label || node.id;
        if (node.type === 'embedding') plainDesc = 'Token sequence ingestion converting discrete IDs to continuous latent representations.';
        else if (node.type === 'residual_add') plainDesc = 'Residual highway addition merging original tokens elementwise with sublayer output.';
        else if (node.type === 'output_head') plainDesc = 'Final unembedding projection mapping hidden states to vocabulary probability logits.';
        else if (node.type === 'svd') plainDesc = `Rank-reduced SVD bottleneck (r=${meta.rank || 64}) projecting into principal subspace.`;
        else if (node.type === 'moe') plainDesc = 'Mixture-of-Experts sparse dynamic routing to specialized subnetwork.';
        else plainDesc = `Linear transformation tensor projecting hidden dimensions [${dOut} × ${dIn}].`;
        detDescEl.innerText = plainDesc;
    }
    if (fShapeEl) fShapeEl.innerText = (meta.shape && Array.isArray(meta.shape)) ? `[${meta.shape.join(' × ')}]` : `[${dOut} × ${dIn}]`;
    if (fParamsEl) fParamsEl.innerText = meta.params || '—';
    if (fSubspaceEl) fSubspaceEl.innerText = meta.rank ? `Rank r=${meta.rank}` : (meta.heads ? `${meta.heads} Heads` : `Dim d=${dIn}`);
    if (fDeltaEl) fDeltaEl.innerText = meta.storage || (meta.compression || '1.0x');

    // Sync floating .compare-card selected node
    const cmpNodeEl = document.getElementById('cmp-selected-node');
    if (cmpNodeEl) cmpNodeEl.innerText = node.id || '—';

    // Rich Architectural Explanation
    const descEl = document.getElementById('insp-desc');
    const noteEl = document.getElementById('insp-edu-note');

    const isBaseline = isLeft;

    if (node.isMacro) {
        if (descEl) {
            descEl.innerHTML = `<strong>Macro Layer Block:</strong> Consolidated enclosing volume for ${node.parent_layer_name}. Contains all ${meta.shape ? meta.shape[0] : 6} constituent sub-tensors with combined footprint of ${meta.params} (${meta.storage}).`;
        }
        if (noteEl) {
            noteEl.innerHTML = `💡 <strong>Semantic Zooming:</strong> Zoom in (scroll up or click "+") to unpack this macro block into individual Self-Attention and Feed-Forward matrix tensors!`;
        }
    } else if (node.type === 'embedding' || node.sub_type === 'input_embedding') {
        if (descEl) {
            descEl.innerHTML = `<strong>Input Tokenization & Embeddings (Tokens In):</strong> Token IDs $[t_1, \\dots, t_S]$ are indexed into vocabulary embedding matrix $W_e \\in \\mathbb{R}^{${dOut} \\times ${dIn}}$. Positional vectors (or RoPE complex rotation matrices $\\mathbf{R}_{\\Theta, m}$) are added to impart sequence order.`;
        }
        if (noteEl) {
            noteEl.innerHTML = `💡 <strong>Tokens Ingestion:</strong> This emerald base plate is where discrete language tokens enter the continuous geometric manifold of the model.`;
        }
    } else if (node.type === 'residual_add' || node.sub_type === 'add_operator') {
        const isAttnAdd = (node.id || '').includes('attn');
        if (descEl) {
            descEl.innerHTML = isAttnAdd
                ? `<strong>Attention Skip Highway (+X_in Addition):</strong> Incoming tokens $X_{\\text{in}}$ bypass the attention sublayer through the skip connection and are added elementwise: $X_{\\text{mid}} = X_{\\text{in}} + \\text{Dropout}(\\text{Attn}(X_{\\text{in}}))$. Prevents vanishing gradients and preserves identity information.`
                : `<strong>FFN Skip Highway (+X_mid Addition):</strong> Intermediate tokens $X_{\\text{mid}}$ bypass the feed-forward network and are added elementwise: $X_{\\text{layer}} = X_{\\text{mid}} + \\text{Dropout}(\\text{FFN}(X_{\\text{mid}}))$. Combines contextual features with non-linear factual memory.`;
        }
        if (noteEl) {
            noteEl.innerHTML = isAttnAdd
                ? `💡 <strong>Highway Addition #1:</strong> Trace the curved lateral bypass cable — notice how raw token identities are injected directly into the contextual delta!`
                : `💡 <strong>Highway Addition #2:</strong> Trace the second lateral bypass cable — notice how the accumulated token state is merged with the FFN factual output!`;
        }
    } else if (node.type === 'output_head' || node.sub_type === 'lm_head') {
        if (descEl) {
            descEl.innerHTML = `<strong>Output LM Head & Next-Token Distribution (Logits Out):</strong> The final normalized token states $X^{(L)} \\in \\mathbb{R}^{S \\times ${dIn}}$ are projected across the vocabulary $\\mathbb{R}^{${dOut}}$ via unembedding matrix $W_{\\text{vocab}}^T$, yielding logits $\\hat{y} = \\text{Softmax}(\\text{Norm}(X^{(L)}) W_{\\text{vocab}}^T)$.`;
        }
        if (noteEl) {
            noteEl.innerHTML = `💡 <strong>Next-Token Sampling:</strong> This rose top slab is where continuous token representations are converted into real probability distributions over the dictionary!`;
        }
    } else if (node.type === 'svd') {
        const rank = meta.rank || 64;
        if (descEl) {
            descEl.innerHTML = `<strong>Low-Rank Factorization ($W \\approx U \\cdot V$):</strong> Decomposes dense linear map $\\mathbb{R}^{${dIn}} \\to \\mathbb{R}^{${dOut}}$ into projection factor $V \\in \\mathbb{R}^{${rank} \\times ${dIn}}$ and expansion factor $U \\in \\mathbb{R}^{${dOut} \\times ${rank}}$. Parameter count drops from $${(dIn * dOut).toLocaleString()}$ to $${((dIn + dOut) * rank).toLocaleString()}$.`;
        }
        if (noteEl) {
            noteEl.innerHTML = `💡 <strong>Physical SVD Neck:</strong> Look at the 3D geometry — notice how the $V$-factor is razor-thin along the rank axis ($r=${rank}$), physically forcing token representations to compress into the principal eigenspace!`;
        }
    } else if (node.type === 'moe') {
        if (node.sub_type === 'router') {
            if (descEl) {
                descEl.innerHTML = `<strong>MoE Gating Router:</strong> Computes routing logits $G(x) = \\text{Softmax}(\\text{TopK}(x \\cdot W_g))$. Dispatches each token selectively to the most specialized expert subnetwork.`;
            }
            if (noteEl) {
                noteEl.innerHTML = `💡 <strong>Sparse Computation:</strong> Only the top-$1$ expert is evaluated per token, keeping FLOPs low while multiplying total parameter capacity!`;
            }
        } else {
            if (descEl) {
                descEl.innerHTML = `<strong>Sparse Expert Module:</strong> Independent sub-network specialized for a subset of semantic tokens. Evaluates active representations with shape $[${dOut} \\times ${dIn}]$.`;
            }
            if (noteEl) {
                noteEl.innerHTML = `💡 <strong>Parallel Experts:</strong> Look at the 4 horizontally spaced expert blocks — watch how the token stream branches into distinct colors as tokens enter their routed expert!`;
            }
        }
    } else if (node.type === 'mla') {
        if (descEl) {
            descEl.innerHTML = `<strong>Multi-Head Latent Attention (MLA):</strong> Low-rank Key-Value compression. Projects input into a compact latent vector $c^{KV} \\in \\mathbb{R}^{${meta.rank || 128}}$, slashing KV cache memory consumption during autoregressive generation by up to $8\\times$.`;
        }
        if (noteEl) {
            noteEl.innerHTML = `💡 <strong>Inference KV Bottleneck:</strong> By storing only the latent vector $c^{KV}$ in GPU VRAM instead of full $K, V$ heads, throughput jumps drastically for long contexts!`;
        }
    } else if (node.sub_type === 'sliced_linear') {
        if (descEl) {
            descEl.innerHTML = `<strong>SliceGPT Structured Pruning:</strong> Orthogonally rotates representations to concentrate energy, then removes $45\\%$ of redundant channels ($[${dOut} \\times ${dIn}]$). Yields direct hardware acceleration with no sparse kernel overhead.`;
        }
        if (noteEl) {
            noteEl.innerHTML = `💡 <strong>Dense Slice:</strong> Notice the visibly smaller matrix volume in 3D — memory and compute are reduced quadratically!`;
        }
    } else {
        if (descEl) {
            descEl.innerHTML = `<strong>Full-Rank Dense Projection:</strong> Operates at complete dimensional capacity $\\mathbb{R}^{${dIn}} \\to \\mathbb{R}^{${dOut}}$ with $O(N^2)$ quadratic token memory cost.`;
        }
        if (noteEl) {
            noteEl.innerHTML = `💡 <strong>Dense Stream:</strong> Representations maintain an unconstrained parallel path through all hidden channels.`;
        }
    }

    inspector.classList.remove('hidden');
}

// ---------------------------------------------------------------------------
// 12. Component & Node Illumination Engine
// ---------------------------------------------------------------------------
let selectedMeshes = [];
let selectionHighlightBox = null;
let selectionPointLight = null;
let selectionDimmedMeshes = [];
let activeSelectedSublayerKey = '';

function initSelectionLighting() {
    if (!selectionPointLight) {
        selectionPointLight = new THREE.PointLight(0x53ecd9, 0, 30);
        selectionPointLight.castShadow = false;
        scene.add(selectionPointLight);
    }
}

function clearSelectionHighlight() {
    activeSelectedSublayerKey = '';

    // 1. Restore original materials on selected meshes
    selectedMeshes.forEach(mesh => {
        if (mesh.userData && mesh.userData._origMaterial) {
            mesh.material = mesh.userData._origMaterial;
            delete mesh.userData._origMaterial;
        }
    });
    selectedMeshes = [];

    // 2. Restore dimmed meshes
    selectionDimmedMeshes.forEach(mesh => {
        if (mesh.userData && mesh.userData._origOpacity !== undefined) {
            if (mesh.material) {
                mesh.material.opacity = mesh.userData._origOpacity;
                mesh.material.transparent = mesh.userData._origTransparent;
            }
            delete mesh.userData._origOpacity;
            delete mesh.userData._origTransparent;
        }
    });
    selectionDimmedMeshes = [];

    // 3. Remove 3D highlight box helper
    if (selectionHighlightBox) {
        if (selectionHighlightBox.parent) selectionHighlightBox.parent.remove(selectionHighlightBox);
        if (selectionHighlightBox.geometry) selectionHighlightBox.geometry.dispose();
        selectionHighlightBox = null;
    }

    // 4. Extinguish selection light
    if (selectionPointLight) {
        selectionPointLight.intensity = 0;
    }

    // 5. Clean up UI active classes
    document.querySelectorAll('.component-pipe-item.active-highlight').forEach(el => {
        el.classList.remove('active-highlight');
    });
    document.querySelectorAll('.tf-step.step-active-glow').forEach(el => {
        el.classList.remove('step-active-glow');
    });

    // 6. Restore dataflow conduits
    dataflowConduits.forEach(c => {
        if (c.userData && c.userData._origMat) {
            c.material.opacity = c.userData._origMat.opacity;
            c.material.color.copy(c.userData._origMat.color);
        }
    });
}

function syncUiPartHighlights(key) {
    if (!key) return;
    const k = key.toLowerCase();

    // Sync component pipeline items
    document.querySelectorAll('.component-pipe-item').forEach(item => {
        const compId = (item.dataset.compId || '').toLowerCase();
        const text = item.innerText.toLowerCase();
        const isMatch = (compId && (k.includes(compId) || compId.includes(k))) || text.includes(k);
        item.classList.toggle('active-highlight', isMatch);
        if (isMatch) {
            item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    });

    // Sync structural transformation steps
    document.querySelectorAll('.tf-step').forEach(step => {
        const stepKey = (step.dataset.stepKey || '').toLowerCase();
        const text = step.innerText.toLowerCase();
        const isMatch = (stepKey && (k.includes(stepKey) || stepKey.includes(k))) || text.includes(k);
        step.classList.toggle('step-active-glow', isMatch);
    });
}

function selectPartIn3D(target, options = {}) {
    initSelectionLighting();
    clearSelectionHighlight();
    if (!target) return;

    let targetMeshes = [];
    let primaryMesh = null;
    let targetKey = '';

    if (target instanceof THREE.Mesh) {
        primaryMesh = target;
        targetMeshes = [target];
        const u = target.userData || {};
        targetKey = u.id || u.sub_type || u.type || '';
    } else if (typeof target === 'string') {
        targetKey = target.toLowerCase();
        // If targetKey is 'dense', match baseline side
        if (targetKey === 'dense' || targetKey === 'baseline') {
            targetMeshes = interactableMeshes.filter(m => m.userData && m.userData.side === 'baseline' && !m.userData.isMacro);
        } else {
            // Match against interactableMeshes by id, sub_type, type, or label
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

    if (!primaryMesh || targetMeshes.length === 0) return;

    selectedMeshes = targetMeshes;

    // Determine highlight color
    let hexColor = 0x53ecd9; // default cyan neon
    if (primaryMesh.userData && primaryMesh.userData.custom_color) {
        hexColor = new THREE.Color(primaryMesh.userData.custom_color).getHex();
    } else if (primaryMesh.userData && primaryMesh.userData.type === 'moe') {
        hexColor = 0xc084fc; // purple neon
    } else if (primaryMesh.userData && primaryMesh.userData.type === 'mla') {
        hexColor = 0xfbbf24; // amber neon
    } else if (primaryMesh.userData && primaryMesh.userData.side === 'baseline') {
        hexColor = 0x38bdf8; // sky blue
    }

    // 1. Fully illuminate selected mesh(es) with maximum emissive radiance
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

    // 2. Position dynamic 3D PointLight right in front of primary mesh
    if (selectionPointLight) {
        selectionPointLight.color.setHex(hexColor);
        selectionPointLight.intensity = 6.0;
        selectionPointLight.position.set(
            primaryMesh.position.x,
            primaryMesh.position.y,
            primaryMesh.position.z + 2.5
        );
    }

    // 3. Attach glowing 3D BoxHelper cage around primary mesh
    selectionHighlightBox = new THREE.BoxHelper(primaryMesh, hexColor);
    if (selectionHighlightBox.material) {
        selectionHighlightBox.material.depthTest = false;
        selectionHighlightBox.material.transparent = true;
        selectionHighlightBox.material.opacity = 1.0;
    }
    scene.add(selectionHighlightBox);

    // 4. Focus Isolation Dimming: dim non-selected meshes so target shines radiantly
    interactableMeshes.forEach(m => {
        if (!selectedMeshes.includes(m) && m.material) {
            if (m.userData._origOpacity === undefined) {
                m.userData._origOpacity = m.material.opacity !== undefined ? m.material.opacity : 1.0;
                m.userData._origTransparent = m.material.transparent !== undefined ? m.material.transparent : false;
            }
            m.material.transparent = true;
            m.material.opacity = 0.22;
            selectionDimmedMeshes.push(m);
        }
    });

    // 4b. Conduit Illumination: brighten lines connecting to selected mesh
    dataflowConduits.forEach(c => {
        const u = c.userData || {};
        const isConnected = (u.sourceId && (u.sourceId.toLowerCase().includes(targetKey) || targetKey.includes(u.sourceId.toLowerCase()))) ||
                            (u.targetId && (u.targetId.toLowerCase().includes(targetKey) || targetKey.includes(u.targetId.toLowerCase())));
        if (isConnected) {
            c.material.opacity = 1.0;
            c.material.color.setHex(0xffffff);
        } else if (c.material) {
            c.material.opacity = 0.18;
        }
    });

    // 5. Smooth camera glide
    if (options.flyCamera !== false) {
        smoothFlyCamera(primaryMesh.position.y, 11.0, primaryMesh.position.x);
    }

    // 6. Update Inspector Drawer with node telemetry
    if (options.updateInspector !== false && primaryMesh.userData) {
        updateInspectorWithNode(primaryMesh.userData, primaryMesh.position.x < 0);
    }

    // 6b. Update Mathematical Formulation with the exact clicked component formula
    const activeComps = getModelPipelineComponents(activeResearchModule, currentModel);
    const matchedComp = activeComps.find(c => {
        if (!c || !c.id) return false;
        const cid = c.id.toLowerCase();
        const tk = targetKey.toLowerCase();
        return cid.includes(tk) || tk.includes(cid);
    });
    if (matchedComp && matchedComp.math_latex) {
        renderLatexFormula(activeResearchModule, matchedComp.math_latex);
    } else if (primaryMesh.userData && primaryMesh.userData.math_latex) {
        renderLatexFormula(activeResearchModule, primaryMesh.userData.math_latex);
    } else {
        renderLatexFormula(activeResearchModule);
    }

    // 7. Sync UI Highlights
    activeSelectedSublayerKey = targetKey.toLowerCase();
    syncUiPartHighlights(targetKey);
}

window.selectPartIn3D = selectPartIn3D;
window.clearSelectionHighlight = clearSelectionHighlight;
window.focusNodeById = function(nodeId) {
    selectPartIn3D(nodeId, { flyCamera: true, updateInspector: true });
};

function handleInteraction(clientX, clientY, isClick = false) {
    const isLeft = clientX < window.innerWidth / 2;
    mouse.x = (clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, cameraBaseline);

    // Raycast visible interactable objects
    const hits = raycaster.intersectObjects(interactableMeshes, false);
    if (hits.length > 0) {
        const resolved = resolveNodeFromIntersection(hits[0]);
        if (!resolved || !resolved.data) return;
        const node = resolved.data;
        const meta = node.metadata || {};

        tooltip.innerHTML = `
            <strong>${node.label || node.id}</strong>
            <div>Type: ${meta.type || node.type}</div>
            <div>Shape: ${meta.shape ? `[${meta.shape.join(' × ')}]` : '—'}</div>
            <div>Params: ${meta.params || '—'} · Storage: ${meta.storage || '—'}</div>
            <div>Efficiency: <span style="color:#53ecd9">${meta.compression || '1.0x Baseline'}</span></div>
        `;
        tooltip.style.left = `${clientX}px`;
        tooltip.style.top = `${clientY}px`;
        tooltip.classList.remove('hidden');

        if (isClick) {
            selectPartIn3D(resolved.mesh, { flyCamera: false, updateInspector: true });
            if (node.isMacro) {
                smoothFlyCamera(node.y, 14.0);
            }
        }
    } else {
        tooltip.classList.add('hidden');
        if (isClick) {
            clearSelectionHighlight();
        }
    }
}

function handleDoubleClick(clientX, clientY) {
    const isLeft = clientX < window.innerWidth / 2;
    mouse.x = (clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, cameraBaseline);

    const hits = raycaster.intersectObjects(interactableMeshes, false);
    if (hits.length > 0) {
        const resolved = resolveNodeFromIntersection(hits[0]);
        if (resolved && resolved.data) {
            const node = resolved.data;
            updateInspectorWithNode(node, isLeft);
            smoothFlyCamera(node.y, 11.0);
        }
    }
}

// ---------------------------------------------------------------------------
// 13. Global HUD Update
// ---------------------------------------------------------------------------
function updateComparisonHUD(modelKey, moduleKey, modelData) {
    const nameEl = document.getElementById('hud-model-name');
    const baseParamsEl = document.getElementById('hud-base-params');
    const compParamsEl = document.getElementById('hud-comp-params');
    const ratioEl = document.getElementById('hud-ratio');
    const savedEl = document.getElementById('hud-saved');

    const mod = (modelData.modules && modelData.modules[moduleKey]) || modelData;

    if (nameEl) nameEl.innerText = `${modelKey.toUpperCase()} · ${moduleKey.toUpperCase()}`;
    if (baseParamsEl) baseParamsEl.innerText = modelData.total_baseline_params_formatted || '—';
    if (compParamsEl) compParamsEl.innerText = mod.total_compressed_params_formatted || '—';
    if (ratioEl) ratioEl.innerText = mod.overall_compression_ratio ? `${mod.overall_compression_ratio.toFixed(2)}x` : '—';
    if (savedEl) savedEl.innerText = mod.storage_saved_formatted || '—';

    // Sync floating .compare-card
    const cmpModelEl = document.getElementById('cmp-model-name');
    const cmpBaseEl = document.getElementById('cmp-baseline-params');
    const cmpCompEl = document.getElementById('cmp-compressed-params');
    const cmpRatioEl = document.getElementById('cmp-ratio');
    const cmpStorageEl = document.getElementById('cmp-storage');

    if (cmpModelEl) cmpModelEl.innerText = modelKey.toUpperCase();
    if (cmpBaseEl) cmpBaseEl.innerText = modelData.total_baseline_params_formatted || '—';
    if (cmpCompEl) cmpCompEl.innerText = mod.total_compressed_params_formatted || '—';
    if (cmpRatioEl) {
        const ratio = mod.overall_compression_ratio || 1.0;
        const pct = Math.round((1 - 1 / Math.max(0.001, ratio)) * 100);
        cmpRatioEl.innerText = `${ratio.toFixed(2)}x (-${pct}%)`;
    }
    if (cmpStorageEl) cmpStorageEl.innerText = mod.storage_saved_formatted || '—';
}

// ---------------------------------------------------------------------------
// 14. Dynamic Model & Compression Loading
// ---------------------------------------------------------------------------
function loadModelAndModule(modelName, moduleName) {
    if (!spatialData || !spatialData.models[modelName]) return;

    currentModel = modelName;
    if (moduleName) currentCompression = moduleName;

    // Update Model Tabs UI
    document.querySelectorAll('.model-tab').forEach(tab => {
        const isActive = (tab.dataset.model === modelName);
        tab.dataset.active = isActive ? "true" : "false";
        tab.setAttribute('aria-selected', isActive ? "true" : "false");
    });

    // Update Compression Tabs UI
    document.querySelectorAll('.comp-tab').forEach(tab => {
        const isActive = (tab.dataset.module === moduleName);
        tab.classList.toggle('active', isActive);
        tab.setAttribute('aria-selected', isActive ? "true" : "false");
    });

    if (activeResearchModule) {
        applyResearchModule(activeResearchModule);
        return;
    }

    clearScene();

    const modelData = spatialData.models[modelName];
    const moduleData = (modelData.modules && modelData.modules[moduleName]) || {
        compressed_nodes: modelData.compressed_nodes,
        macro_nodes: modelData.macro_nodes || [],
        name: 'SVD Low-Rank'
    };

    // Viewport Titles
    const leftTitle = document.getElementById('left-panel-title');
    const rightTitle = document.getElementById('right-panel-title');
    if (leftTitle) leftTitle.innerText = `${modelName.toUpperCase()} (Dense Baseline)`;
    if (rightTitle) rightTitle.innerText = `${modelName.toUpperCase()} (${moduleData.name})`;

    // Build Dense Baseline Side
    const h1 = buildMeshNodes(getAugmentedBaselineNodes(modelName), OFFSET_BASELINE, false);
    if (modelData.baseline_macro_nodes) {
        buildMacroBlocks(modelData.baseline_macro_nodes, OFFSET_BASELINE, false);
    }

    // Build Compressed Side
    const h2 = buildMeshNodes(moduleData.compressed_nodes, OFFSET_COMPRESSED, true);
    if (moduleData.macro_nodes) {
        buildMacroBlocks(moduleData.macro_nodes, OFFSET_COMPRESSED, true);
    }

    maxGraphHeight = Math.max(h1, h2, modelData.max_y || 16.0);

    // Build Token Streams & Motion Trails
    const baseStreams = createTokenMesh(tokenDataBaseline, OFFSET_BASELINE, tokenMaterialBaseline, trailMaterialBaseline);
    tokenMeshBaseline = baseStreams.main;
    trailMeshBaseline = baseStreams.trail;
    scene.add(tokenMeshBaseline);
    scene.add(trailMeshBaseline);

    const compStreams = createTokenMesh(tokenDataCompressed, OFFSET_COMPRESSED, tokenMaterialCompressed, trailMaterialCompressed);
    tokenMeshCompressed = compStreams.main;
    trailMeshCompressed = compStreams.trail;
    scene.add(tokenMeshCompressed);
    scene.add(trailMeshCompressed);

    // Smooth Adaptive Camera Framing
    smoothFlyCamera(maxGraphHeight * 0.5, Math.max(18.0, maxGraphHeight * 1.35));

    // Build Tree & HUD
    buildHierarchicalTree(modelName, moduleName, modelData);
    updateComparisonHUD(modelName, moduleName, modelData);
    renderTokenLifecycle(activeResearchModule || { id: moduleName, name: moduleData.name }, modelName);
    renderComponentPipeline(activeResearchModule || { id: moduleName, name: moduleData.name }, modelName);
    renderStructuralTransformation(activeResearchModule || { id: moduleName, name: moduleData.name }, modelName);
    renderModelsMatrix(activeResearchModule || { id: moduleName, name: moduleData.name });
    renderLatexFormula(activeResearchModule || { id: moduleName, name: moduleData.name });

    const labelEl = document.getElementById('active-mode-label');
    if (labelEl) labelEl.textContent = `${modelName.toUpperCase()} · ${moduleData.name}`;

    const tokensEl = document.getElementById('active-tokens-count');
    if (tokensEl) tokensEl.textContent = `${NUM_TOKENS} Flowing Tokens`;
}

// ---------------------------------------------------------------------------
// 15. Continuous Semantic Zooming / Level-of-Detail (LOD)
// ---------------------------------------------------------------------------
function updateSemanticLOD() {
    const camDist = cameraBaseline.position.distanceTo(controls.target);

    // Macro Threshold: > 30 units away
    // Micro Threshold: < 18 units close
    let macroOpacity = 0.0;
    let granularOpacity = 0.95;

    if (camDist > 40) {
        macroOpacity = 0.35;
        granularOpacity = 0.88;
        macroGroup.visible = true;
    } else if (camDist > 25) {
        const factor = (camDist - 25) / 15; // 0 to 1
        macroOpacity = factor * 0.32;
        granularOpacity = 0.95 - factor * 0.07;
        macroGroup.visible = true;
    } else {
        macroOpacity = 0.0;
        granularOpacity = 0.95;
        macroGroup.visible = false;
    }

    materials.macro_box.opacity = macroOpacity;
    materials.dense.opacity = granularOpacity;
    materials.svd.opacity = granularOpacity;
    materials.moe_router.opacity = granularOpacity;
    materials.moe_expert.opacity = granularOpacity;
    materials.mla.opacity = granularOpacity;
    customMaterialsCache.forEach(mat => {
        mat.opacity = granularOpacity;
    });

    // Floating Dimension Badges in Near Zoom (appear when camDist < 25)
    if (camDist < 25) {
        dimensionBadgesGroup.visible = true;
        const badgeAlpha = Math.min(1.0, Math.max(0.0, (25 - camDist) / 7.0));
        dimensionBadgesGroup.children.forEach(s => {
            if (s.material) s.material.opacity = badgeAlpha * 0.95;
        });
    } else {
        dimensionBadgesGroup.visible = false;
    }

    // Update active LOD button state
    document.querySelectorAll('.btn-lod').forEach(btn => {
        if (camDist > 32 && btn.dataset.lod === 'macro') btn.classList.add('active');
        else if (camDist <= 32 && camDist >= 18 && btn.dataset.lod === 'mid') btn.classList.add('active');
        else if (camDist < 18 && btn.dataset.lod === 'micro') btn.classList.add('active');
        else btn.classList.remove('active');
    });
}

// ---------------------------------------------------------------------------
// 16. Code-Accurate Computational Graph Trajectory Engine
// ---------------------------------------------------------------------------
const dummy = new THREE.Object3D();
const dummyTrail = new THREE.Object3D();

function computeToken3DTrajectory(t, y, isCompressed, modelKey, paradigm, baseOffsetX) {
    const LAYER_STRIDE = 3.2;
    const yRel = Math.max(0, y - 1.0);
    const layerIdx = Math.floor(yRel / LAYER_STRIDE);
    const tau = (yRel % LAYER_STRIDE) / LAYER_STRIDE; // 0.0 to 1.0 within the layer
    const yBase = 1.0 + layerIdx * LAYER_STRIDE;

    let posX = baseOffsetX;
    let posZ = 0;
    let activeSublayer = '';

    // -------------------------------------------------------------
    // BRANCH A: Residual Highway Stream (25% of tokens in PyTorch: residual = x)
    // -------------------------------------------------------------
    if (t.isResidual) {
        posX = baseOffsetX - 0.85 + (t.lateralNoise || 0) * 0.12;
        posZ = -0.75 + (t.lateralNoise || 0) * 0.12;

        if (tau >= 0.42 && tau <= 0.48) {
            const bridgeProgress = (tau - 0.42) / 0.06;
            const dip = Math.sin(bridgeProgress * Math.PI);
            posX += dip * 0.45;
            posZ += dip * 0.45;
            activeSublayer = 'residual_add_attn';
        } else if (tau >= 0.92 && tau <= 0.98) {
            const bridgeProgress = (tau - 0.92) / 0.06;
            const dip = Math.sin(bridgeProgress * Math.PI);
            posX += dip * 0.45;
            posZ += dip * 0.45;
            activeSublayer = 'residual_add_ffn';
        } else {
            activeSublayer = 'residual';
        }
        return { x: posX, z: posZ, sublayer: activeSublayer };
    }

    // -------------------------------------------------------------
    // BRANCH B: Active Neural Transform Sublayer Stream (75% of tokens)
    // -------------------------------------------------------------
    if (isCompressed) {
        if (modelKey === 'qwen') {
            // === QWEN2.5 (GQA Attention + SwiGLU FFN) ===
            if (tau < 0.12) {
                posX = baseOffsetX + (t.lateralNoise || 0) * 0.3;
                posZ = 0;
                activeSublayer = 'norm';
            } else if (tau < 0.46) {
                // GQA: q_proj (x=-0.7), k_proj (x=-0.15), v_proj (x=+0.20) -> o_proj (x=+0.75)
                const stream = t.subStream || 'q';
                const prog = (tau - 0.12) / 0.34;
                if (prog < 0.72) {
                    if (stream === 'q') {
                        posX = baseOffsetX - 0.70 + (t.lateralNoise || 0) * 0.45;
                        posZ = -0.1 + (t.lateralNoise || 0) * 0.15;
                        activeSublayer = 'q_proj';
                    } else if (stream === 'k') {
                        posX = baseOffsetX - 0.15 + (t.lateralNoise || 0) * 0.10;
                        posZ = 0.25;
                        activeSublayer = 'k_proj';
                    } else {
                        posX = baseOffsetX + 0.20 + (t.lateralNoise || 0) * 0.10;
                        posZ = 0.25;
                        activeSublayer = 'v_proj';
                    }
                } else {
                    const mProg = (prog - 0.72) / 0.28;
                    const startX = (stream === 'q') ? (baseOffsetX - 0.70) : ((stream === 'k') ? (baseOffsetX - 0.15) : (baseOffsetX + 0.20));
                    posX = THREE.MathUtils.lerp(startX, baseOffsetX + 0.75, mProg);
                    posZ = 0;
                    activeSublayer = 'o_proj';
                }
            } else if (tau < 0.52) {
                const prog = (tau - 0.46) / 0.06;
                posX = THREE.MathUtils.lerp(baseOffsetX + 0.75, baseOffsetX, prog);
                posZ = 0;
                activeSublayer = 'residual_merge';
            } else if (tau < 0.76) {
                // SwiGLU: gate_proj (x=-0.60) & up_proj (x=+0.60)
                const isGate = (t.swigluStream === 'gate');
                const targetSubX = isGate ? (baseOffsetX - 0.60) : (baseOffsetX + 0.60);
                const prog = (tau - 0.52) / 0.24;
                posX = THREE.MathUtils.lerp(baseOffsetX, targetSubX, Math.min(1.0, prog * 2.0)) + (t.lateralNoise || 0) * 0.22;
                posZ = (t.lateralNoise || 0) * 0.18;
                activeSublayer = isGate ? 'gate_proj' : 'up_proj';
            } else if (tau < 0.90) {
                // Multiplicative silu(gate) * up converges into down_proj at center
                const prog = (tau - 0.76) / 0.14;
                const isGate = (t.swigluStream === 'gate');
                const fromX = isGate ? (baseOffsetX - 0.60) : (baseOffsetX + 0.60);
                posX = THREE.MathUtils.lerp(fromX, baseOffsetX, prog) + (t.lateralNoise || 0) * (1 - prog) * 0.2;
                posZ = 0;
                activeSublayer = 'down_proj';
            } else {
                posX = baseOffsetX + (t.lateralNoise || 0) * 0.15;
                posZ = 0;
                activeSublayer = 'output_norm';
            }
        } else if (modelKey === 'gpt2') {
            // === GPT-2 (Fused Conv1D c_attn + Conv1D c_fc) ===
            if (tau < 0.12) {
                posX = baseOffsetX + (t.lateralNoise || 0) * 0.3;
                posZ = 0;
                activeSublayer = 'norm';
            } else if (tau < 0.25) {
                const prog = (tau - 0.12) / 0.13;
                posX = THREE.MathUtils.lerp(baseOffsetX, baseOffsetX - 0.50, prog);
                posZ = (t.lateralNoise || 0) * 0.95;
                activeSublayer = 'v_factor';
            } else if (tau < 0.38) {
                const prog = (tau - 0.25) / 0.13;
                posX = THREE.MathUtils.lerp(baseOffsetX - 0.50, baseOffsetX + 0.50, prog);
                posZ = (t.lateralNoise || 0) * 0.25;
                activeSublayer = 'u_factor';
            } else if (tau < 0.48) {
                const prog = (tau - 0.38) / 0.10;
                posX = THREE.MathUtils.lerp(baseOffsetX + 0.50, baseOffsetX, prog);
                posZ = 0;
                activeSublayer = 'c_proj';
            } else if (tau < 0.72) {
                const prog = (tau - 0.48) / 0.24;
                posX = THREE.MathUtils.lerp(baseOffsetX, baseOffsetX - 0.50, Math.min(1.0, prog * 2.0)) + (t.lateralNoise || 0) * 0.35;
                posZ = 0;
                activeSublayer = 'c_fc';
            } else if (tau < 0.88) {
                const prog = (tau - 0.72) / 0.16;
                posX = THREE.MathUtils.lerp(baseOffsetX - 0.50, baseOffsetX + 0.50, prog) + (t.lateralNoise || 0) * 0.20;
                posZ = 0;
                activeSublayer = 'c_proj';
            } else {
                const prog = (tau - 0.88) / 0.12;
                posX = THREE.MathUtils.lerp(baseOffsetX + 0.50, baseOffsetX, prog);
                posZ = 0;
                activeSublayer = 'output_norm';
            }
        } else if (paradigm === 'moe') {
            // === MIXTURE OF EXPERTS ===
            if (tau < 0.15) {
                posX = baseOffsetX + (t.lateralNoise || 0) * 0.3;
                posZ = 0;
                activeSublayer = 'norm';
            } else if (tau < 0.28) {
                posX = baseOffsetX;
                posZ = 0;
                activeSublayer = 'router';
            } else if (tau < 0.44) {
                const expTarget = (t.expertTarget || 0);
                const expX = baseOffsetX + (expTarget - 1.5) * 0.75;
                const prog = (tau - 0.28) / 0.16;
                posX = THREE.MathUtils.lerp(baseOffsetX, expX, Math.sin(prog * Math.PI * 0.5));
                posZ = 0;
                activeSublayer = 'expert';
            } else if (tau < 0.52) {
                const expTarget = (t.expertTarget || 0);
                const expX = baseOffsetX + (expTarget - 1.5) * 0.75;
                const prog = (tau - 0.44) / 0.08;
                posX = THREE.MathUtils.lerp(expX, baseOffsetX, prog);
                posZ = 0;
                activeSublayer = 'expert_merge';
            } else if (tau < 0.80) {
                posX = baseOffsetX + Math.sin((tau - 0.52) * 10) * 0.35;
                posZ = 0;
                activeSublayer = 'ffn';
            } else {
                posX = baseOffsetX;
                posZ = 0;
                activeSublayer = 'output_norm';
            }
        } else {
            // === SVD LOW-RANK (DistilBERT & RoBERTa) ===
            if (tau < 0.12) {
                posX = baseOffsetX + (t.lateralNoise || 0) * 0.3;
                posZ = 0;
                activeSublayer = 'norm';
            } else if (tau < 0.26) {
                const prog = (tau - 0.12) / 0.14;
                const pinch = 1.0 - 0.78 * Math.sin(prog * Math.PI); // pinches into rank-64 funnel!
                posX = THREE.MathUtils.lerp(baseOffsetX, baseOffsetX - 0.45, prog) + (t.lateralNoise || 0) * 0.14 * pinch;
                posZ = (t.lateralNoise || 0) * 0.28 * pinch;
                activeSublayer = 'v_factor';
            } else if (tau < 0.40) {
                const prog = (tau - 0.26) / 0.14;
                const expand = 0.25 + 0.75 * Math.sin(prog * Math.PI); // expands out to hidden size!
                posX = THREE.MathUtils.lerp(baseOffsetX - 0.45, baseOffsetX + 0.45, prog) + (t.lateralNoise || 0) * 0.38 * expand;
                posZ = (t.lateralNoise || 0) * 0.20;
                activeSublayer = 'u_factor';
            } else if (tau < 0.50) {
                const prog = (tau - 0.40) / 0.10;
                posX = THREE.MathUtils.lerp(baseOffsetX + 0.45, baseOffsetX, prog);
                posZ = 0;
                activeSublayer = 'residual_merge';
            } else if (tau < 0.72) {
                const prog = (tau - 0.50) / 0.22;
                posX = THREE.MathUtils.lerp(baseOffsetX, baseOffsetX - 0.50, Math.min(1.0, prog * 1.8)) + (t.lateralNoise || 0) * 0.45;
                posZ = 0;
                activeSublayer = 'intermediate';
            } else if (tau < 0.88) {
                const prog = (tau - 0.72) / 0.16;
                posX = THREE.MathUtils.lerp(baseOffsetX - 0.50, baseOffsetX + 0.50, prog) + (t.lateralNoise || 0) * 0.22;
                posZ = 0;
                activeSublayer = 'output_dense';
            } else {
                const prog = (tau - 0.88) / 0.12;
                posX = THREE.MathUtils.lerp(baseOffsetX + 0.50, baseOffsetX, prog);
                posZ = 0;
                activeSublayer = 'output_norm';
            }
        }
    } else {
        // === DENSE BASELINE (Left Side, baseOffsetX = -4.5) ===
        if (tau < 0.12) {
            posX = baseOffsetX + (t.lateralNoise || 0) * 0.35;
            posZ = 0;
            activeSublayer = 'dense_norm';
        } else if (tau < 0.45) {
            const headOffset = ((t.headIdx || 0) - 5.5) / 6.0 * 0.70;
            const prog = (tau - 0.12) / 0.33;
            posX = baseOffsetX + headOffset * Math.sin(prog * Math.PI) + (t.lateralNoise || 0) * 0.08;
            posZ = 0;
            activeSublayer = 'dense_attn';
        } else if (tau < 0.52) {
            const prog = (tau - 0.45) / 0.07;
            posX = baseOffsetX + (t.lateralNoise || 0) * (1 - prog) * 0.25;
            posZ = 0;
            activeSublayer = 'dense_attn_add';
        } else if (tau < 0.72) {
            const prog = (tau - 0.52) / 0.20;
            posX = baseOffsetX - 0.55 * Math.sin(prog * Math.PI * 0.5) + (t.lateralNoise || 0) * 0.55;
            posZ = 0;
            activeSublayer = 'dense_intermediate';
        } else if (tau < 0.88) {
            const prog = (tau - 0.72) / 0.16;
            posX = THREE.MathUtils.lerp(baseOffsetX - 0.55, baseOffsetX + 0.55, prog) + (t.lateralNoise || 0) * 0.25;
            posZ = 0;
            activeSublayer = 'dense_output';
        } else {
            const prog = (tau - 0.88) / 0.12;
            posX = THREE.MathUtils.lerp(baseOffsetX + 0.55, baseOffsetX, prog);
            posZ = 0;
            activeSublayer = 'dense_ffn_add';
        }
    }

    return { x: posX, z: posZ, sublayer: activeSublayer };
}

function updateTokens(mainMesh, trailMesh, tokenDataArray, isCompressedSide = false) {
    if (!mainMesh || !isPlaying) return;
    if (tokenDataArray.length === 0) return;

    const now = performance.now();
    const modelKey = currentModel;
    const paradigm = (activeResearchModule && activeResearchModule.paradigm) || 'svd';
    const baseOffsetX = isCompressedSide ? OFFSET_COMPRESSED : OFFSET_BASELINE;
    const defaultColor = isCompressedSide ? new THREE.Color(0x2dd4bf) : new THREE.Color(0x38bdf8);
    const residualColor = new THREE.Color(0x60a5fa);
    const highlightColor = new THREE.Color(0xffffff);

    for (let i = 0; i < NUM_TOKENS; i++) {
        const t = tokenDataArray[i];

        // 1. Advance token height
        t.y += t.speed * simulationSpeed;
        if (t.y > maxGraphHeight) {
            t.y = 0.2;
            t.lateralNoise = (Math.random() - 0.5) * 1.4;
        }

        // 2. Compute exact 3D position from architectural computational graph
        const curPos = computeToken3DTrajectory(t, t.y, isCompressedSide, modelKey, paradigm, baseOffsetX);
        const prevY = Math.max(0.1, t.y - t.speed * simulationSpeed * 1.8);
        const prevPos = computeToken3DTrajectory(t, prevY, isCompressedSide, modelKey, paradigm, baseOffsetX);

        // 3. Check if active sublayer matches user selection
        const isSelectedSublayer = activeSelectedSublayerKey && curPos.sublayer &&
            (activeSelectedSublayerKey.includes(curPos.sublayer) || curPos.sublayer.includes(activeSelectedSublayerKey));

        // 4. Update Token Scale and Position
        dummy.position.set(curPos.x, t.y, curPos.z);
        let scalePulse = 1.0 + Math.sin(t.y * 3.0 + t.pulsePhase) * 0.15;
        if (isSelectedSublayer) {
            scalePulse *= 2.2;
        }
        dummy.scale.set(scalePulse, scalePulse, scalePulse);
        dummy.updateMatrix();
        mainMesh.setMatrixAt(i, dummy.matrix);

        // 5. Update Motion Trail
        dummyTrail.position.set(prevPos.x, prevY, prevPos.z);
        const trailScale = isSelectedSublayer ? 1.4 : 0.7;
        dummyTrail.scale.set(trailScale, trailScale, trailScale);
        dummyTrail.updateMatrix();
        if (trailMesh) trailMesh.setMatrixAt(i, dummyTrail.matrix);

        // 6. Token Color Tinting
        if (mainMesh.instanceColor) {
            if (isSelectedSublayer) {
                mainMesh.setColorAt(i, highlightColor);
            } else if (t.isResidual) {
                mainMesh.setColorAt(i, residualColor);
            } else if (isCompressedSide && modelKey === 'qwen') {
                if (curPos.sublayer === 'q_proj') mainMesh.setColorAt(i, new THREE.Color(0x38bdf8));
                else if (curPos.sublayer === 'k_proj' || curPos.sublayer === 'v_proj') mainMesh.setColorAt(i, new THREE.Color(0xf59e0b));
                else if (curPos.sublayer === 'gate_proj') mainMesh.setColorAt(i, new THREE.Color(0xa855f7));
                else if (curPos.sublayer === 'up_proj') mainMesh.setColorAt(i, new THREE.Color(0x8b5cf6));
                else mainMesh.setColorAt(i, defaultColor);
            } else if (isCompressedSide && paradigm === 'moe') {
                mainMesh.setColorAt(i, EXPERT_COLORS[t.expertTarget || 0]);
            } else {
                mainMesh.setColorAt(i, defaultColor);
            }
        }

        // 7. Trigger Mesh Activation Pulse when Token enters its geometry
        if (enablePulses) {
            for (let mIdx = 0; mIdx < interactableMeshes.length; mIdx++) {
                const mesh = interactableMeshes[mIdx];
                if (mesh.userData.isMacro) continue;
                const matchesSide = isCompressedSide ? (mesh.position.x > 0) : (mesh.position.x < 0);
                if (matchesSide) {
                    const dx = Math.abs(mesh.position.x - curPos.x);
                    const dy = Math.abs(mesh.position.y - t.y);
                    const dz = Math.abs(mesh.position.z - curPos.z);
                    const hw = (mesh.geometry.parameters ? mesh.geometry.parameters.width : 1.0) / 2 + 0.15;
                    const hh = (mesh.geometry.parameters ? mesh.geometry.parameters.height : 0.4) / 2 + 0.15;
                    const hd = (mesh.geometry.parameters ? mesh.geometry.parameters.depth : 1.0) / 2 + 0.15;

                    if (dx <= hw && dy <= hh && dz <= hd) {
                        mesh.userData.lastTokenHit = now;
                    }
                }
            }
        }
    }

    mainMesh.instanceMatrix.needsUpdate = true;
    if (mainMesh.instanceColor) mainMesh.instanceColor.needsUpdate = true;
    if (trailMesh) trailMesh.instanceMatrix.needsUpdate = true;
}

// ---------------------------------------------------------------------------
// 17. Node Activation Pulse Animation
// ---------------------------------------------------------------------------
function updateNodePulses(now) {
    if (!enablePulses) return;

    for (let i = 0; i < interactableMeshes.length; i++) {
        const mesh = interactableMeshes[i];
        if (mesh.userData.isMacro) continue;
        const elapsed = now - (mesh.userData.lastTokenHit || 0);

        if (elapsed < 320) {
            const norm = elapsed / 320;
            const wave = Math.sin(norm * Math.PI);

            if (mesh.userData.type === 'svd') {
                mesh.material.emissiveIntensity = 0.4 + wave * 0.9;
            } else if (mesh.userData.type === 'mla') {
                mesh.material.emissiveIntensity = 0.35 + wave * 0.8;
            } else {
                mesh.scale.set(
                    1.0 + wave * 0.04,
                    1.0 + wave * 0.04,
                    1.0 + wave * 0.04
                );
            }
        } else {
            if (mesh.userData.type === 'svd') {
                mesh.material.emissiveIntensity = 0.4;
            } else if (mesh.userData.type === 'mla') {
                mesh.material.emissiveIntensity = 0.35;
            }
            mesh.scale.set(1.0, 1.0, 1.0);
        }
    }
}

// ---------------------------------------------------------------------------
// 18. UI Listeners & Controls Wiring
// ---------------------------------------------------------------------------
// Model Selector
document.querySelectorAll('.model-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        loadModelAndModule(tab.dataset.model, currentCompression);
    });
});

// Compression Paradigm Selector
document.querySelectorAll('.comp-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        loadModelAndModule(currentModel, tab.dataset.module);
    });
});

// Animation Speed
document.querySelectorAll('.btn-speed').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.btn-speed').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        simulationSpeed = parseFloat(btn.dataset.speed);
    });
});

// Pulse Toggle
document.getElementById('toggle-pulse').addEventListener('change', e => {
    enablePulses = e.target.checked;
});

// ---------------------------------------------------------------------------
// Precision Camera Navigation & Zoom Controller
// ---------------------------------------------------------------------------
let isCameraAnimating = false;
const animTargetLookAt = new THREE.Vector3();
const animTargetCamPos = new THREE.Vector3();

function smoothFlyCamera(targetLookY, targetDist, targetLookX = null) {
    const isInspectorOpen = inspector && !inspector.classList.contains('hidden') && window.innerWidth > 1024;
    let finalX = isInspectorOpen ? 1.5 : 0;
    if (targetLookX !== null && targetLookX !== undefined) {
        finalX = targetLookX + (isInspectorOpen ? 1.8 : 0);
    }

    animTargetLookAt.set(finalX, targetLookY, 0);
    const offset = new THREE.Vector3().subVectors(cameraBaseline.position, controls.target);
    if (offset.lengthSq() < 0.1) offset.set(0, 4, 20);
    offset.normalize().multiplyScalar(targetDist);
    animTargetCamPos.copy(animTargetLookAt).add(offset);
    isCameraAnimating = true;
}

// Immediately yield full tactile control to user whenever they interact with wheel or touch
controls.addEventListener('start', () => {
    isCameraAnimating = false;
});
canvas.addEventListener('wheel', () => {
    isCameraAnimating = false;
}, { passive: true });

// Semantic Zoom LOD Buttons
document.querySelectorAll('.btn-lod').forEach(btn => {
    btn.addEventListener('click', () => {
        const lod = btn.dataset.lod;
        if (lod === 'macro') {
            smoothFlyCamera(maxGraphHeight * 0.5, Math.max(38.0, maxGraphHeight * 1.55));
        } else if (lod === 'mid') {
            smoothFlyCamera(controls.target.y, 22.0);
        } else if (lod === 'micro') {
            smoothFlyCamera(controls.target.y, 8.5);
        }
    });
});

const zoomInBtn = document.getElementById('btn-zoom-in');
const zoomOutBtn = document.getElementById('btn-zoom-out');
if (zoomInBtn) {
    zoomInBtn.addEventListener('click', () => {
        const curDist = cameraBaseline.position.distanceTo(controls.target);
        smoothFlyCamera(controls.target.y, Math.max(controls.minDistance, curDist * 0.72));
    });
}
if (zoomOutBtn) {
    zoomOutBtn.addEventListener('click', () => {
        const curDist = cameraBaseline.position.distanceTo(controls.target);
        smoothFlyCamera(controls.target.y, Math.min(controls.maxDistance, curDist * 1.38));
    });
}

// Mouse Interactivity
window.addEventListener('mousemove', e => handleInteraction(e.clientX, e.clientY, false));
window.addEventListener('click', e => {
    if (!inspector.contains(e.target) && !e.target.closest('.topbar') &&
        !e.target.closest('.model-selector') && !e.target.closest('.compression-selector') &&
        !e.target.closest('.edu-controls') && !e.target.closest('.comparison-hud') &&
        !e.target.closest('#module-modal')) {
        handleInteraction(e.clientX, e.clientY, true);
    }
});

window.addEventListener('dblclick', e => {
    if (!inspector.contains(e.target) && !e.target.closest('.topbar') &&
        !e.target.closest('.model-selector') && !e.target.closest('.compression-selector') &&
        !e.target.closest('.edu-controls') && !e.target.closest('.comparison-hud') &&
        !e.target.closest('#module-modal')) {
        handleDoubleClick(e.clientX, e.clientY);
    }
});

document.getElementById('btn-close-insp').addEventListener('click', () => {
    inspector.classList.add('hidden');
});

let isPlaying = true;
const playBtn = document.getElementById('btn-play-pause');
playBtn.addEventListener('click', () => {
    isPlaying = !isPlaying;
    playBtn.innerHTML = isPlaying
        ? '<span class="btn-icon">⏸</span> Pause Flow'
        : '<span class="btn-icon">▶</span> Play Flow';
});

document.getElementById('btn-reset-cam').addEventListener('click', () => {
    smoothFlyCamera(maxGraphHeight * 0.5, Math.max(16.0, maxGraphHeight * 1.35));
});

window.addEventListener('resize', () => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize(w, h);

    cameraBaseline.aspect = w / h;
    cameraBaseline.updateProjectionMatrix();
});

// ---------------------------------------------------------------------------
// 19. Render Loop
// ---------------------------------------------------------------------------

// SVG Leader Lines (3D to HTML Pipeline)
function updateScreenSpaceGuides() {
    const svgEl = document.getElementById('label-guides');
    const pipeItems = document.querySelectorAll('.component-pipe-item');
    if (!svgEl || pipeItems.length === 0) return;

    // We will dynamically create paths
    let pathsHTML = '';
    const width = window.innerWidth;
    const height = window.innerHeight;

    // Ensure SVG covers the screen
    svgEl.style.position = 'absolute';
    svgEl.style.top = '0';
    svgEl.style.left = '0';
    svgEl.style.width = '100%';
    svgEl.style.height = '100%';
    svgEl.style.pointerEvents = 'none';
    svgEl.style.zIndex = '10';

    pipeItems.forEach(item => {
        const compId = item.getAttribute('data-comp-id');
        if (!compId) return;
        
        let targetKey = compId.toLowerCase();
        let targetMesh = interactableMeshes.find(m => {
            const u = m.userData;
            if (!u || u.side !== 'baseline') return false;
            return (u.id && u.id.toLowerCase().includes(targetKey)) ||
                   (u.sub_type && u.sub_type.toLowerCase().includes(targetKey)) ||
                   (u.type && u.type.toLowerCase().includes(targetKey)) ||
                   (u.label && u.label.toLowerCase().includes(targetKey));
        });
        
        // If no direct match, try a looser match or skip
        if (!targetMesh) return;

        // Project 3D position to 2D screen space
        const pos3D = targetMesh.position.clone();
        pos3D.project(cameraBaseline);
        
        // Check if behind camera
        if (pos3D.z > 1) return;

        const screenX = (pos3D.x *  .5 + .5) * width;
        const screenY = (pos3D.y * -.5 + .5) * height;

        // Get target HTML element position
        const rect = item.getBoundingClientRect();
        const targetX = rect.left;
        const targetY = rect.top + rect.height / 2;

        // Extract color from CSS variable or default
        const color = item.style.getPropertyValue('--cpi-color') || '#2dd4bf';

        // Draw curved SVG path
        // M: move to 3D point, C: cubic bezier curve to HTML element
        const cp1x = screenX + (targetX - screenX) * 0.4;
        const cp1y = screenY;
        const cp2x = screenX + (targetX - screenX) * 0.6;
        const cp2y = targetY;
        
        pathsHTML += `<path d="M ${screenX} ${screenY} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${targetX} ${targetY}" fill="none" stroke="${color}" stroke-width="1.5" stroke-dasharray="4 4" opacity="0.6"></path>`;
        
        // Add a dot at the 3D mesh location
        pathsHTML += `<circle cx="${screenX}" cy="${screenY}" r="3" fill="${color}" opacity="0.8" />`;
    });

    svgEl.innerHTML = pathsHTML;
}

function render(time) {
    requestAnimationFrame(render);

    // Smooth Camera Transition (only active when flying programmatically)
    if (isCameraAnimating) {
        controls.target.lerp(animTargetLookAt, 0.08);
        cameraBaseline.position.lerp(animTargetCamPos, 0.08);

        if (controls.target.distanceTo(animTargetLookAt) < 0.05 && cameraBaseline.position.distanceTo(animTargetCamPos) < 0.05) {
            controls.target.copy(animTargetLookAt);
            cameraBaseline.position.copy(animTargetCamPos);
            isCameraAnimating = false;
        }
    }

    // Always update controls smoothly without fighting the user
    controls.update();

    // Continuous Semantic Zoom LOD Interpolation
    updateSemanticLOD();

    // Update Physics Tokens & Trails
    updateTokens(tokenMeshBaseline, trailMeshBaseline, tokenDataBaseline, false);
    updateTokens(tokenMeshCompressed, trailMeshCompressed, tokenDataCompressed, true);
    updateScreenSpaceGuides();

    // Update Layer Pulses
    updateNodePulses(performance.now());

    // Update Selection Illumination Pulse
    if (selectedMeshes.length > 0) {
        const t = performance.now() * 0.006;
        const pulse = (Math.sin(t) + 1) * 0.5; // 0 to 1
        if (selectionPointLight) {
            selectionPointLight.intensity = 4.5 + pulse * 3.5;
        }
        selectedMeshes.forEach(m => {
            if (m.material && m.material.emissiveIntensity !== undefined) {
                m.material.emissiveIntensity = 2.0 + pulse * 1.5;
            }
        });
        if (selectionHighlightBox && selectionHighlightBox.material) {
            selectionHighlightBox.material.opacity = 0.6 + pulse * 0.4;
        }
    }

    const width = window.innerWidth;
    const height = window.innerHeight;

    renderer.setViewport(0, 0, width, height);
    renderer.setScissor(0, 0, width, height);
    renderer.render(scene, cameraBaseline);
    updateFPS();
}

// ---------------------------------------------------------------------------
// 20. 240+ Research Module Repository Management
// ---------------------------------------------------------------------------
let modulesCatalog = [];
let activeResearchModule = null;

function initModuleCatalog() {
    const modal = document.getElementById('module-modal');
    const openBtn = document.getElementById('btn-open-catalog');
    const browseBtn = document.getElementById('btn-browse-all-modules');
    const closeBtn = document.getElementById('btn-close-modal');
    const searchInput = document.getElementById('modal-search-input');
    const catPills = document.querySelectorAll('.cat-pill');

    let currentCat = 'all';
    let currentSearch = '';

    const openModal = () => {
        if (modal) {
            modal.classList.remove('hidden');
            if (searchInput) {
                searchInput.value = '';
                searchInput.focus();
            }
            renderModuleCards('all', '');
        }
    };

    const closeModal = () => {
        if (modal) modal.classList.add('hidden');
    };

    if (openBtn) openBtn.addEventListener('click', openModal);
    if (browseBtn) browseBtn.addEventListener('click', openModal);
    if (closeBtn) closeBtn.addEventListener('click', closeModal);

    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeModal();
        });
    }

    // Category filtering
    catPills.forEach(pill => {
        pill.addEventListener('click', () => {
            catPills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            currentCat = pill.dataset.cat;
            renderModuleCards(currentCat, currentSearch);
        });
    });

    // Real-time search
    if (searchInput) {
        searchInput.addEventListener('input', () => {
            currentSearch = searchInput.value.trim().toLowerCase();
            renderModuleCards(currentCat, currentSearch);
        });
    }
}

function renderModuleCards(category = 'all', searchQuery = '') {
    const grid = document.getElementById('modal-module-grid');
    const countEl = document.getElementById('modal-module-count');
    if (!grid) return;

    grid.innerHTML = '';

    const filtered = modulesCatalog.filter(m => {
        const matchesCat = (category === 'all' || m.category === category);
        const text = (m.id + ' ' + m.name + ' ' + m.description + ' ' + m.category + ' ' + (m.math || '')).toLowerCase();
        const matchesSearch = !searchQuery || text.includes(searchQuery);
        return matchesCat && matchesSearch;
    });

    if (countEl) countEl.innerText = `${filtered.length} of ${modulesCatalog.length}`;

    if (filtered.length === 0) {
        grid.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; padding: 40px; color: var(--text-faint);">No research modules found matching "${searchQuery}".</div>`;
        return;
    }

    filtered.forEach(mod => {
        const card = document.createElement('div');
        card.className = 'module-card';

        const mathSnippet = mod.math
            ? `<div class="card-math" title="${mod.math}">$$${mod.math.slice(0, 75)}${mod.math.length > 75 ? '...' : ''}$$</div>`
            : '';

        card.innerHTML = `
            <div class="card-top">
                <div class="card-badge-row">
                    <span class="card-category">${mod.icon || '⚡'} ${mod.category}</span>
                    <span class="card-type">${mod.type}</span>
                </div>
                <h4 class="card-title">${mod.name}</h4>
                <p class="card-desc">${mod.description}</p>
                ${mathSnippet}
            </div>
            <div class="card-bottom">
                <span class="card-file" title="${mod.file}">${mod.id}.py</span>
                <button class="btn-apply-module" data-id="${mod.id}">Apply to 3D</button>
            </div>
        `;

        const applyBtn = card.querySelector('.btn-apply-module');
        applyBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            applyResearchModule(mod);
            const modal = document.getElementById('module-modal');
            if (modal) modal.classList.add('hidden');
        });

        card.addEventListener('click', () => {
            applyResearchModule(mod);
            const modal = document.getElementById('module-modal');
            if (modal) modal.classList.add('hidden');
        });

        grid.appendChild(card);
    });
}

let modulesComponentsMap = {};

function populateQuickSelectDropdown() {
    const select = document.getElementById('quick-module-select');
    if (!select || !modulesCatalog || modulesCatalog.length === 0) return;

    select.innerHTML = '<option value="" disabled selected>⚡ Quick Pick (241 Modules)...</option>';

    // Group modules by category
    const groups = {};
    modulesCatalog.forEach(mod => {
        const cat = mod.category || 'Other';
        if (!groups[cat]) groups[cat] = [];
        groups[cat].push(mod);
    });

    const catIcons = {
        'Low-Rank SVD & Factorization': '⚡',
        'Mixture of Experts & Routing': '🔀',
        'Latent & KV-Cache Compression': '🧠',
        'Structured Pruning & Channel Slicing': '✂️',
        'Linear Attention & State-Space (SSM)': '🐍',
        'Quantization & Mixed Precision': '💎',
        'Dynamic & Efficient Attention': '🌟'
    };

    const orderedCats = [
        'Low-Rank SVD & Factorization',
        'Mixture of Experts & Routing',
        'Latent & KV-Cache Compression',
        'Structured Pruning & Channel Slicing',
        'Linear Attention & State-Space (SSM)',
        'Quantization & Mixed Precision',
        'Dynamic & Efficient Attention'
    ];

    orderedCats.forEach(cat => {
        if (!groups[cat] || groups[cat].length === 0) return;
        const icon = catIcons[cat] || '📦';
        const optgroup = document.createElement('optgroup');
        optgroup.label = `${icon} ${cat} (${groups[cat].length})`;

        groups[cat].sort((a, b) => a.name.localeCompare(b.name)).forEach(mod => {
            const opt = document.createElement('option');
            opt.value = mod.id;
            opt.textContent = `${mod.name.slice(0, 32)} (${mod.id})`;
            optgroup.appendChild(opt);
        });

        select.appendChild(optgroup);
    });

    select.addEventListener('change', (e) => {
        const id = e.target.value;
        if (!id) return;
        const mod = modulesComponentsMap[id] || modulesCatalog.find(m => m.id === id);
        if (mod) {
            applyResearchModule(mod);
        }
    });
}

const VOCAB_SIZES = {
    qwen: 151936,
    gpt2: 50257,
    roberta: 50265,
    distilbert: 30522
};

function getAugmentedBaselineNodes(modelName) {
    const modelData = spatialData && spatialData.models && spatialData.models[modelName];
    if (!modelData) return [];
    
    const originalNodes = modelData.baseline_nodes || [];
    if (originalNodes.some(n => n.id && n.id.includes('input_embeddings'))) {
        return originalNodes;
    }
    
    const vocabSize = VOCAB_SIZES[modelName] || 30522;
    const dModel = (MODEL_SPECS[modelName] && MODEL_SPECS[modelName].hidden_size) || (modelName === 'qwen' ? 896 : 768);
    const augmented = [];

    // 1. Base Input Embeddings Slab
    augmented.push({
        id: 'baseline.input_embeddings',
        name: 'INPUT EMBEDDINGS (Tokens In)',
        parent_layer: 'Input Stage',
        parent_layer_idx: -1,
        parent_layer_name: 'Input Stage',
        parent_layer_index: -1,
        type: 'embedding',
        sub_type: 'input_embedding',
        role: 'Token ID Lookup + Position Encodings',
        shape_transform: `[${vocabSize} × ${dModel}]`,
        math_latex: modelName === 'qwen'
            ? 'X^{(0)} = W_e[T], \\quad \\mathbf{R}_{\\Theta, m}^d \\text{ applied to } Q, K'
            : 'X^{(0)} = W_e[T] + W_{\\text{pos}}[P]',
        custom_color: '#10b981',
        width: 3.4,
        height: 0.38,
        depth: 2.2,
        x: 0,
        y: 0.35,
        z: 0,
        metadata: {
            shape: [vocabSize, dModel],
            params: `${((vocabSize * dModel) / 1e6).toFixed(1)}M`,
            params_raw: vocabSize * dModel,
            storage: `${((vocabSize * dModel * 2) / 1e6).toFixed(1)} MB`,
            role: 'Tokens Ingestion Highway',
            type: 'Token + Position Embedding Table',
            compression: '1.0x (Baseline)'
        }
    });

    // 2. Group baseline layer nodes and insert residual addition operator nodes
    const layerMap = new Map();
    originalNodes.forEach(n => {
        const lIdx = (n.parent_layer_index !== undefined) ? n.parent_layer_index : (n.parent_layer_idx || 0);
        if (!layerMap.has(lIdx)) layerMap.set(lIdx, []);
        layerMap.get(lIdx).push(n);
    });

    let topY = 0;
    const sortedLayers = Array.from(layerMap.keys()).sort((a, b) => a - b);
    sortedLayers.forEach(lIdx => {
        const lNodes = layerMap.get(lIdx).sort((a, b) => a.y - b.y);
        const attnOut = lNodes.find(n => anyKey(n.id, ['out_lin', 'output.dense', 'o_proj', 'attn.c_proj']) && !anyKey(n.id, ['mlp', 'ffn', 'intermediate']));
        const ffnLast = lNodes.find(n => anyKey(n.id, ['lin2', 'output', 'down_proj', 'mlp.c_proj']) && !anyKey(n.id, ['attention', 'self']));

        let insertedAttnAdd = false;
        let insertedFfnAdd = false;

        lNodes.forEach(node => {
            augmented.push(node);
            if (node.y > topY) topY = node.y;

            if (attnOut && node.id === attnOut.id && !insertedAttnAdd) {
                insertedAttnAdd = true;
                const addY = Number((node.y + 0.28).toFixed(3));
                augmented.push({
                    id: `baseline.layer.${lIdx}.attn.residual_add`,
                    name: `(+) Layer ${lIdx} Attn Residual Add (+X)`,
                    parent_layer: node.parent_layer_name || `Layer ${lIdx}`,
                    parent_layer_idx: lIdx,
                    parent_layer_name: node.parent_layer_name || `Layer ${lIdx}`,
                    parent_layer_index: lIdx,
                    type: 'residual_add',
                    sub_type: 'add_operator',
                    role: 'Skip Highway Addition #1 (+ X_in)',
                    shape_transform: `[Seq × ${dModel}]`,
                    math_latex: 'X_{\\text{mid}} = X_{\\text{in}} + \\text{Dropout}(\\text{Attn}(X_{\\text{in}}))',
                    custom_color: '#0284c7',
                    width: 1.9,
                    height: 0.22,
                    depth: 0.9,
                    x: 0,
                    y: addY,
                    z: 0,
                    metadata: {
                        shape: ['Seq', dModel],
                        params: '0 (Op)',
                        params_raw: 0,
                        storage: '0 MB (In-place)',
                        role: 'Residual Context Accumulation Highway (+X_in)',
                        type: 'Residual Addition #1',
                        compression: '1.0x (Baseline)'
                    }
                });
                if (addY > topY) topY = addY;
            }

            if (ffnLast && node.id === ffnLast.id && !insertedFfnAdd) {
                insertedFfnAdd = true;
                const addY = Number((node.y + 0.28).toFixed(3));
                augmented.push({
                    id: `baseline.layer.${lIdx}.ffn.residual_add`,
                    name: `(+) Layer ${lIdx} FFN Residual Add (+X_mid)`,
                    parent_layer: node.parent_layer_name || `Layer ${lIdx}`,
                    parent_layer_idx: lIdx,
                    parent_layer_name: node.parent_layer_name || `Layer ${lIdx}`,
                    parent_layer_index: lIdx,
                    type: 'residual_add',
                    sub_type: 'add_operator',
                    role: 'Skip Highway Addition #2 (+ X_mid)',
                    shape_transform: `[Seq × ${dModel}]`,
                    math_latex: 'X_{\\text{out}} = X_{\\text{mid}} + \\text{Dropout}(\\text{FFN}(X_{\\text{mid}}))',
                    custom_color: '#7c3aed',
                    width: 1.9,
                    height: 0.22,
                    depth: 0.9,
                    x: 0,
                    y: addY,
                    z: 0,
                    metadata: {
                        shape: ['Seq', dModel],
                        params: '0 (Op)',
                        params_raw: 0,
                        storage: '0 MB (In-place)',
                        role: 'Residual Knowledge Accumulation Highway (+X_mid)',
                        type: 'Residual Addition #2',
                        compression: '1.0x (Baseline)'
                    }
                });
                if (addY > topY) topY = addY;
            }
        });
    });

    // 3. Top Output Head Slab
    const topYOutput = Number((topY + 0.65).toFixed(3));
    augmented.push({
        id: 'baseline.output_head',
        name: 'OUTPUT LM HEAD (Logits Out)',
        parent_layer: 'Output Stage',
        parent_layer_idx: sortedLayers.length,
        parent_layer_name: 'Output Stage',
        parent_layer_index: sortedLayers.length,
        type: 'output_head',
        sub_type: 'lm_head',
        role: 'LayerNorm / RMSNorm + Logits Vocabulary Projection',
        shape_transform: `[${dModel} × ${vocabSize}]`,
        math_latex: modelName === 'qwen'
            ? '\\hat{y} = \\text{Softmax}\\left(\\text{RMSNorm}(X^{(L)}) W_{\\text{vocab}}^T\\right)'
            : '\\hat{y} = \\text{Softmax}\\left(\\text{LayerNorm}(X^{(L)}) W_{\\text{vocab}}^T\\right)',
        custom_color: '#f43f5e',
        width: 3.4,
        height: 0.38,
        depth: 2.2,
        x: 0,
        y: topYOutput,
        z: 0,
        metadata: {
            shape: [dModel, vocabSize],
            params: `${((dModel * vocabSize) / 1e6).toFixed(1)}M`,
            params_raw: dModel * vocabSize,
            storage: `${((dModel * vocabSize * 2) / 1e6).toFixed(1)} MB`,
            role: 'Logits Out Prediction Highway',
            type: 'Unembedding Vocabulary Projection',
            compression: '1.0x (Baseline)'
        }
    });

    return augmented;
}

function synthesizeModule3DNodes(moduleMeta, modelName) {
    const modelData = spatialData.models[modelName];
    if (!modelData) return null;

    const numLayers = modelData.capped_num_layers || 6;
    const proj = (moduleMeta.model_projections && moduleMeta.model_projections[modelName]) || {
        hidden_size: modelName === 'qwen' ? 896 : 768,
        intermediate_size: modelName === 'qwen' ? 4864 : 3072,
        savings_ratio: 4.0
    };
    const dModel = proj.hidden_size || (modelName === 'qwen' ? 896 : 768);
    const dFfn = proj.intermediate_size || (modelName === 'qwen' ? 4864 : 3072);
    const vocabSize = VOCAB_SIZES[modelName] || 30522;

    const components = moduleMeta.components && moduleMeta.components.length > 0
        ? moduleMeta.components
        : [
            { id: 'v_factor', name: 'Low-Rank Projection', role: 'Bottleneck Subspace', shape_transform: '[r × d]', color: '#3acbb8' },
            { id: 'u_factor', name: 'Low-Rank Expansion', role: 'Output Expansion', shape_transform: '[d × r]', color: '#2dd4bf' }
        ];

    const cNodes = [];
    const macroNodes = [];
    const LAYER_STRIDE = 3.2;

    // Stage 1: Explicit 3D Input Tokenization & Embedding Stage Base Slab
    cNodes.push({
        id: 'compressed.input_embeddings',
        name: 'INPUT EMBEDDINGS (Tokens In)',
        parent_layer: 'Input Stage',
        parent_layer_idx: -1,
        parent_layer_name: 'Input Stage',
        parent_layer_index: -1,
        type: 'embedding',
        sub_type: 'input_embedding',
        role: 'Token ID Lookup + Position Encodings',
        shape_transform: `[${vocabSize} × ${dModel}]`,
        math_latex: modelName === 'qwen'
            ? 'X^{(0)} = W_e[T], \\quad \\mathbf{R}_{\\Theta, m}^d \\text{ applied to } Q, K'
            : 'X^{(0)} = W_e[T] + W_{\\text{pos}}[P]',
        module_id: moduleMeta.id,
        custom_color: '#10b981',
        width: 3.4,
        height: 0.38,
        depth: 2.2,
        x: 0,
        y: 0.35,
        z: 0,
        metadata: {
            shape: [vocabSize, dModel],
            params: `${((vocabSize * dModel) / 1e6).toFixed(1)}M`,
            params_raw: vocabSize * dModel,
            storage: `${((vocabSize * dModel * 2) / 1e6).toFixed(1)} MB`,
            role: 'Tokens Ingestion Highway',
            type: 'Token + Position Embedding Table'
        }
    });

    for (let l = 0; l < numLayers; l++) {
        const yBase = 1.0 + l * LAYER_STRIDE;
        const layerName = modelName === 'qwen' ? `Layer ${l * 2}` : `Layer ${l}`;

        // 1. Attention Sublayer Components (Tailored to authentic model architecture)
        if (modelName === 'gpt2') {
            // GPT-2: Fused Conv1D c_attn (768 x 2304) and c_proj
            cNodes.push({
                id: `compressed.layer.${l}.v_factor`,
                name: 'attn.c_attn.v_factor (r=64)',
                parent_layer: layerName,
                parent_layer_idx: l,
                type: moduleMeta.paradigm || 'svd',
                sub_type: 'v_factor',
                role: 'Fused QKV Low-Rank Projection',
                shape_transform: `[64 × ${dModel}]`,
                math_latex: 'x V_{c\\_attn}',
                module_id: moduleMeta.id,
                custom_color: '#3acbb8',
                width: 0.35,
                height: 0.45,
                depth: 2.2, // Distinct wide packed tensor!
                x: -0.5,
                y: Number((yBase + 0.30).toFixed(3)),
                z: 0,
                metadata: {
                    shape: [64, dModel],
                    params: 64 * dModel,
                    storage: 'FP16',
                    role: 'Fused QKV Low-Rank Projection'
                }
            });

            cNodes.push({
                id: `compressed.layer.${l}.u_factor`,
                name: 'attn.c_attn.u_factor (r=64)',
                parent_layer: layerName,
                parent_layer_idx: l,
                type: moduleMeta.paradigm || 'svd',
                sub_type: 'u_factor',
                role: 'Fused QKV Low-Rank Expansion',
                shape_transform: `[2304 × 64]`,
                math_latex: 'h U_{c\\_attn}',
                module_id: moduleMeta.id,
                custom_color: '#2dd4bf',
                width: 2.2, // Wide expansion!
                height: 0.45,
                depth: 0.35,
                x: 0.5,
                y: Number((yBase + 0.70).toFixed(3)),
                z: 0,
                metadata: {
                    shape: [2304, 64],
                    params: 2304 * 64,
                    storage: 'FP16',
                    role: 'Fused QKV Low-Rank Expansion'
                }
            });

            cNodes.push({
                id: `compressed.layer.${l}.c_proj`,
                name: 'attn.c_proj',
                parent_layer: layerName,
                parent_layer_idx: l,
                type: moduleMeta.paradigm || 'svd',
                sub_type: 'output_proj',
                role: 'Attention Output Projection',
                shape_transform: `[${dModel} × ${dModel}]`,
                math_latex: 'x W_{proj} + b',
                module_id: moduleMeta.id,
                custom_color: '#5eead4',
                width: 1.2,
                height: 0.38,
                depth: 1.2,
                x: 0.0,
                y: Number((yBase + 1.10).toFixed(3)),
                z: 0,
                metadata: {
                    shape: [dModel, dModel],
                    params: dModel * dModel,
                    storage: 'FP16',
                    role: 'Attention Output Projection'
                }
            });
        } else if (modelName === 'qwen') {
            // Qwen: Grouped-Query Attention (GQA) with 14 Q heads and 2 KV heads!
            cNodes.push({
                id: `compressed.layer.${l}.q_proj`,
                name: 'self_attn.q_proj (14 Heads)',
                parent_layer: layerName,
                parent_layer_idx: l,
                type: moduleMeta.paradigm || 'svd',
                sub_type: 'q_proj',
                role: '14 Query Heads Low-Rank',
                shape_transform: `[${dModel} × ${dModel}]`,
                math_latex: 'x W_q',
                module_id: moduleMeta.id,
                custom_color: '#38bdf8',
                width: 1.6,
                height: 0.42,
                depth: 0.45,
                x: -0.7,
                y: Number((yBase + 0.25).toFixed(3)),
                z: 0,
                metadata: {
                    shape: [dModel, dModel],
                    params: dModel * dModel,
                    storage: 'FP16',
                    role: '14 Query Heads'
                }
            });

            cNodes.push({
                id: `compressed.layer.${l}.k_proj`,
                name: 'self_attn.k_proj (2 KV Heads)',
                parent_layer: layerName,
                parent_layer_idx: l,
                type: moduleMeta.paradigm || 'svd',
                sub_type: 'k_proj',
                role: 'GQA 2 Key Heads (7:1 Sharing)',
                shape_transform: `[128 × ${dModel}]`,
                math_latex: 'x W_k',
                module_id: moduleMeta.id,
                custom_color: '#f59e0b',
                width: 0.28, // Distinct thin GQA slice!
                height: 0.42,
                depth: 1.5,
                x: -0.15,
                y: Number((yBase + 0.55).toFixed(3)),
                z: 0,
                metadata: {
                    shape: [128, dModel],
                    params: 128 * dModel,
                    storage: 'FP16',
                    role: '2 Key Heads (GQA)'
                }
            });

            cNodes.push({
                id: `compressed.layer.${l}.v_proj`,
                name: 'self_attn.v_proj (2 KV Heads)',
                parent_layer: layerName,
                parent_layer_idx: l,
                type: moduleMeta.paradigm || 'svd',
                sub_type: 'v_proj',
                role: 'GQA 2 Value Heads (7:1 Sharing)',
                shape_transform: `[128 × ${dModel}]`,
                math_latex: 'x W_v',
                module_id: moduleMeta.id,
                custom_color: '#d97706',
                width: 0.28, // Distinct thin GQA slice!
                height: 0.42,
                depth: 1.5,
                x: 0.20,
                y: Number((yBase + 0.85).toFixed(3)),
                z: 0,
                metadata: {
                    shape: [128, dModel],
                    params: 128 * dModel,
                    storage: 'FP16',
                    role: '2 Value Heads (GQA)'
                }
            });

            cNodes.push({
                id: `compressed.layer.${l}.o_proj`,
                name: 'self_attn.o_proj (Output)',
                parent_layer: layerName,
                parent_layer_idx: l,
                type: moduleMeta.paradigm || 'svd',
                sub_type: 'o_proj',
                role: 'Attention Output Projection',
                shape_transform: `[${dModel} × ${dModel}]`,
                math_latex: 'x W_o',
                module_id: moduleMeta.id,
                custom_color: '#2dd4bf',
                width: 1.6,
                height: 0.42,
                depth: 0.45,
                x: 0.75,
                y: Number((yBase + 1.15).toFixed(3)),
                z: 0,
                metadata: {
                    shape: [dModel, dModel],
                    params: dModel * dModel,
                    storage: 'FP16',
                    role: 'Output Projection'
                }
            });
        } else {
            // DistilBERT & RoBERTa: Standard factorized Attention Sublayer
            const nComps = components.length;
            const compSpacing = Math.min(0.7, 2.4 / Math.max(1, nComps));
            const totalCompWidth = (nComps - 1) * compSpacing;
            const startX = -totalCompWidth / 2;

            components.forEach((comp, idx) => {
                let width = 1.1;
                let depth = 1.1;
                let height = 0.42;
                let yOffset = 0.45;
                let xPos = startX + idx * compSpacing;
                let zPos = 0;

                const cid = comp.id.toLowerCase();
                const role = (comp.role || '').toLowerCase();

                if (cid.includes('v_factor') || cid.includes('v_proj') || role.includes('rank') || role.includes('bottleneck')) {
                    width = 0.35;
                    depth = 1.4;
                    yOffset = 0.30;
                    xPos = -0.45;
                } else if (cid.includes('u_factor') || cid.includes('u_proj') || role.includes('expansion')) {
                    width = 1.4;
                    depth = 0.35;
                    yOffset = 0.65;
                    xPos = 0.45;
                } else if (cid.includes('adapter') || cid.includes('lora') || role.includes('residual')) {
                    width = 0.75;
                    depth = 0.75;
                    height = 0.3;
                    yOffset = 0.48;
                    zPos = 0.8;
                } else if (cid.includes('router') || role.includes('router')) {
                    width = 1.5;
                    depth = 0.45;
                    height = 0.32;
                    yOffset = 0.25;
                    xPos = 0.0;
                } else if (cid.includes('expert') || role.includes('expert')) {
                    width = 0.55;
                    depth = 0.75;
                    height = 0.45;
                    yOffset = 0.70;
                } else if (cid.includes('ssm') || role.includes('state')) {
                    width = 0.9;
                    depth = 0.9;
                    height = 0.45;
                    yOffset = 0.60;
                } else if (cid.includes('kv') || role.includes('kv') || role.includes('latent')) {
                    width = 0.5;
                    depth = 0.5;
                    height = 0.4;
                    yOffset = 0.40;
                }

                cNodes.push({
                    id: `compressed.layer.${l}.${comp.id}`,
                    name: comp.name,
                    parent_layer: layerName,
                    parent_layer_idx: l,
                    type: moduleMeta.paradigm || 'svd',
                    sub_type: comp.id,
                    role: comp.role,
                    shape_transform: comp.shape_transform,
                    math_latex: moduleMeta.math_latex || moduleMeta.math || '',
                    module_id: moduleMeta.id,
                    custom_color: comp.color || '#2dd4bf',
                    width: Number(width.toFixed(3)),
                    height: Number(height.toFixed(3)),
                    depth: Number(depth.toFixed(3)),
                    x: Number(xPos.toFixed(3)),
                    y: Number((yBase + yOffset).toFixed(3)),
                    z: Number(zPos.toFixed(3)),
                    metadata: {
                        shape: [dModel, dModel],
                        params: Math.round(dModel * dModel / (moduleMeta.paradigm === 'svd' ? 8 : 4)),
                        storage: 'FP16',
                        role: comp.role,
                        shape_transform: comp.shape_transform
                    }
                });
            });
        }

        // 1b. Attention Residual Addition (+ X_in) Skip Highway Operator Node
        cNodes.push({
            id: `compressed.layer.${l}.attn.residual_add`,
            name: `(+) Layer ${l} Attn Residual Add (+X)`,
            parent_layer: layerName,
            parent_layer_idx: l,
            parent_layer_name: layerName,
            parent_layer_index: l,
            type: 'residual_add',
            sub_type: 'add_operator',
            role: 'Skip Highway Addition #1 (+ X_in)',
            shape_transform: `[Seq × ${dModel}]`,
            math_latex: 'X_{\\text{mid}} = X_{\\text{in}} + \\text{Dropout}(\\text{Attn}(X_{\\text{in}}))',
            module_id: moduleMeta.id,
            custom_color: '#0284c7',
            width: 1.9,
            height: 0.22,
            depth: 0.9,
            x: 0,
            y: Number((yBase + 1.40).toFixed(3)),
            z: 0,
            metadata: {
                shape: ['Seq', dModel],
                params: '0 (Op)',
                params_raw: 0,
                storage: '0 MB (In-place)',
                role: 'Residual Context Accumulation Highway (+X_in)',
                type: 'Residual Addition #1'
            }
        });

        // 2. FFN Sublayer (Tailored to Model Architecture!)
        if (modelName === 'qwen') {
            // SwiGLU Gated MLP with 3 distinct matrices!
            cNodes.push({
                id: `compressed.layer.${l}.mlp.gate_proj`,
                name: 'mlp.gate_proj (SiLU)',
                parent_layer: layerName,
                parent_layer_idx: l,
                type: moduleMeta.paradigm || 'svd',
                sub_type: 'gate_proj',
                custom_color: '#a855f7',
                role: 'SwiGLU Gating Projection',
                shape_transform: `[${dFfn} × ${dModel}]`,
                math_latex: '\\text{SiLU}(x W_{gate})',
                module_id: moduleMeta.id,
                width: 1.4,
                height: 0.42,
                depth: 1.0,
                x: -0.6,
                y: Number((yBase + 1.65).toFixed(3)),
                z: 0,
                metadata: {
                    shape: [dFfn, dModel],
                    params: dFfn * dModel,
                    storage: 'FP16',
                    role: 'SwiGLU Gating Projection'
                }
            });

            cNodes.push({
                id: `compressed.layer.${l}.mlp.up_proj`,
                name: 'mlp.up_proj',
                parent_layer: layerName,
                parent_layer_idx: l,
                type: moduleMeta.paradigm || 'svd',
                sub_type: 'up_proj',
                custom_color: '#8b5cf6',
                role: 'SwiGLU Up Projection',
                shape_transform: `[${dFfn} × ${dModel}]`,
                math_latex: 'x W_{up}',
                module_id: moduleMeta.id,
                width: 1.4,
                height: 0.42,
                depth: 1.0,
                x: 0.6,
                y: Number((yBase + 1.65).toFixed(3)),
                z: 0,
                metadata: {
                    shape: [dFfn, dModel],
                    params: dFfn * dModel,
                    storage: 'FP16',
                    role: 'SwiGLU Up Projection'
                }
            });

            cNodes.push({
                id: `compressed.layer.${l}.mlp.down_proj`,
                name: 'mlp.down_proj',
                parent_layer: layerName,
                parent_layer_idx: l,
                type: moduleMeta.paradigm || 'svd',
                sub_type: 'down_proj',
                custom_color: '#6366f1',
                role: 'SwiGLU Down Output Projection',
                shape_transform: `[${dModel} × ${dFfn}]`,
                math_latex: '( \\text{SiLU}(x W_g) \\odot x W_u ) W_d',
                module_id: moduleMeta.id,
                width: 1.1,
                height: 0.42,
                depth: 1.5,
                x: 0.0,
                y: Number((yBase + 2.25).toFixed(3)),
                z: 0,
                metadata: {
                    shape: [dModel, dFfn],
                    params: dModel * dFfn,
                    storage: 'FP16',
                    role: 'SwiGLU Down Output Projection'
                }
            });
        } else if (modelName === 'gpt2') {
            // Conv1D FFN with c_fc and c_proj
            cNodes.push({
                id: `compressed.layer.${l}.mlp.c_fc`,
                name: 'mlp.c_fc',
                parent_layer: layerName,
                parent_layer_idx: l,
                type: moduleMeta.paradigm || 'svd',
                sub_type: 'c_fc',
                custom_color: '#38bdf8',
                role: 'Conv1D FFN Intermediate Expansion',
                shape_transform: `[${dFfn} × ${dModel}]`,
                math_latex: '\\text{GELU}(x W + b)',
                module_id: moduleMeta.id,
                width: 1.5,
                height: 0.45,
                depth: 1.2,
                x: 0,
                y: Number((yBase + 1.65).toFixed(3)),
                z: 0,
                metadata: {
                    shape: [dFfn, dModel],
                    params: dFfn * dModel,
                    storage: 'FP16',
                    role: 'Conv1D FFN Intermediate Expansion'
                }
            });

            cNodes.push({
                id: `compressed.layer.${l}.mlp.c_proj`,
                name: 'mlp.c_proj',
                parent_layer: layerName,
                parent_layer_idx: l,
                type: moduleMeta.paradigm || 'svd',
                sub_type: 'c_proj',
                custom_color: '#0ea5e9',
                role: 'Conv1D FFN Output Projection',
                shape_transform: `[${dModel} × ${dFfn}]`,
                math_latex: 'x W + b',
                module_id: moduleMeta.id,
                width: 1.2,
                height: 0.45,
                depth: 1.5,
                x: 0,
                y: Number((yBase + 2.25).toFixed(3)),
                z: 0,
                metadata: {
                    shape: [dModel, dFfn],
                    params: dModel * dFfn,
                    storage: 'FP16',
                    role: 'Conv1D FFN Output Projection'
                }
            });
        } else if (modelName === 'roberta') {
            // RoBERTa: intermediate.dense and output.dense
            cNodes.push({
                id: `compressed.layer.${l}.intermediate.dense`,
                name: 'intermediate.dense',
                parent_layer: layerName,
                parent_layer_idx: l,
                type: moduleMeta.paradigm || 'svd',
                sub_type: 'intermediate',
                custom_color: '#38bdf8',
                role: 'Dense Expansion (GELU)',
                shape_transform: `[${dFfn} × ${dModel}]`,
                math_latex: '\\text{GELU}(x W + b)',
                module_id: moduleMeta.id,
                width: 1.4,
                height: 0.45,
                depth: 1.2,
                x: 0,
                y: Number((yBase + 1.65).toFixed(3)),
                z: 0,
                metadata: {
                    shape: [dFfn, dModel],
                    params: dFfn * dModel,
                    storage: 'FP16',
                    role: 'Dense Expansion (GELU)'
                }
            });

            cNodes.push({
                id: `compressed.layer.${l}.output.dense`,
                name: 'output.dense',
                parent_layer: layerName,
                parent_layer_idx: l,
                type: moduleMeta.paradigm || 'svd',
                sub_type: 'output',
                custom_color: '#0ea5e9',
                role: 'LayerNorm Dense Output Map',
                shape_transform: `[${dModel} × ${dFfn}]`,
                math_latex: 'x W + b',
                module_id: moduleMeta.id,
                width: 1.2,
                height: 0.45,
                depth: 1.4,
                x: 0,
                y: Number((yBase + 2.25).toFixed(3)),
                z: 0,
                metadata: {
                    shape: [dModel, dFfn],
                    params: dModel * dFfn,
                    storage: 'FP16',
                    role: 'Dense Output Map'
                }
            });
        } else {
            // DistilBERT: lin1 and lin2
            cNodes.push({
                id: `compressed.layer.${l}.ffn.lin1`,
                name: 'ffn.lin1',
                parent_layer: layerName,
                parent_layer_idx: l,
                type: moduleMeta.paradigm || 'svd',
                sub_type: 'ffn_up',
                custom_color: '#38bdf8',
                role: 'Feed-Forward Intermediate (GELU)',
                shape_transform: `[${dFfn} × ${dModel}]`,
                math_latex: '\\text{GELU}(x W_1 + b_1)',
                module_id: moduleMeta.id,
                width: 1.4,
                height: 0.45,
                depth: 1.2,
                x: 0,
                y: Number((yBase + 1.65).toFixed(3)),
                z: 0,
                metadata: {
                    shape: [dFfn, dModel],
                    params: dFfn * dModel,
                    storage: 'FP16',
                    role: 'Feed-Forward Intermediate Expansion'
                }
            });

            cNodes.push({
                id: `compressed.layer.${l}.ffn.lin2`,
                name: 'ffn.lin2',
                parent_layer: layerName,
                parent_layer_idx: l,
                type: moduleMeta.paradigm || 'svd',
                sub_type: 'ffn_down',
                custom_color: '#0ea5e9',
                role: 'Feed-Forward Output Projection',
                shape_transform: `[${dModel} × ${dFfn}]`,
                math_latex: 'x W_2 + b_2',
                module_id: moduleMeta.id,
                width: 1.2,
                height: 0.45,
                depth: 1.4,
                x: 0,
                y: Number((yBase + 2.25).toFixed(3)),
                z: 0,
                metadata: {
                    shape: [dModel, dFfn],
                    params: dModel * dFfn,
                    storage: 'FP16',
                    role: 'Feed-Forward Output Projection'
                }
            });
        }

        // 2b. FFN Residual Addition (+ X_mid) Skip Highway Operator Node
        cNodes.push({
            id: `compressed.layer.${l}.ffn.residual_add`,
            name: `(+) Layer ${l} FFN Residual Add (+X_mid)`,
            parent_layer: layerName,
            parent_layer_idx: l,
            parent_layer_name: layerName,
            parent_layer_index: l,
            type: 'residual_add',
            sub_type: 'add_operator',
            role: 'Skip Highway Addition #2 (+ X_mid)',
            shape_transform: `[Seq × ${dModel}]`,
            math_latex: 'X_{\\text{out}} = X_{\\text{mid}} + \\text{Dropout}(\\text{FFN}(X_{\\text{mid}}))',
            module_id: moduleMeta.id,
            custom_color: '#7c3aed',
            width: 1.9,
            height: 0.22,
            depth: 0.9,
            x: 0,
            y: Number((yBase + 2.70).toFixed(3)),
            z: 0,
            metadata: {
                shape: ['Seq', dModel],
                params: '0 (Op)',
                params_raw: 0,
                storage: '0 MB (In-place)',
                role: 'Residual Knowledge Accumulation Highway (+X_mid)',
                type: 'Residual Addition #2'
            }
        });

        // 3. Macro Bounding Block for Layer
        macroNodes.push({
            id: `macro.compressed.layer.${l}`,
            name: `${layerName} (${moduleMeta.name.slice(0, 18)})`,
            parent_layer: layerName,
            parent_layer_idx: l,
            parent_layer_name: layerName,
            isMacro: true,
            width: modelName === 'qwen' ? 4.2 : 3.8,
            height: 2.7,
            depth: modelName === 'qwen' ? 3.8 : 3.4,
            x: 0,
            y: Number((yBase + 1.35).toFixed(3)),
            z: 0,
            metadata: {
                shape: [components.length + (modelName === 'qwen' ? 3 : 2)],
                params: `${((proj.compressed_layer_params || proj.compressed_layer_p || 800000) / 1e6).toFixed(2)}M`,
                storage: 'FP16'
            }
        });
    }

    // Stage 7: Explicit 3D Output LM Head & Softmax Vocabulary Projection Slab
    const topYOutput = Number((1.0 + (numLayers - 1) * LAYER_STRIDE + 3.25).toFixed(3));
    cNodes.push({
        id: 'compressed.output_head',
        name: 'OUTPUT LM HEAD (Logits Out)',
        parent_layer: 'Output Stage',
        parent_layer_idx: numLayers,
        parent_layer_name: 'Output Stage',
        parent_layer_index: numLayers,
        type: 'output_head',
        sub_type: 'lm_head',
        role: 'RMSNorm / LayerNorm + Logits Vocabulary Projection',
        shape_transform: `[${dModel} × ${vocabSize}]`,
        math_latex: modelName === 'qwen'
            ? '\\hat{y} = \\text{Softmax}\\left(\\text{RMSNorm}(X^{(L)}) W_{\\text{vocab}}^T\\right)'
            : '\\hat{y} = \\text{Softmax}\\left(\\text{LayerNorm}(X^{(L)}) W_{\\text{vocab}}^T\\right)',
        module_id: moduleMeta.id,
        custom_color: '#f43f5e',
        width: 3.4,
        height: 0.38,
        depth: 2.2,
        x: 0,
        y: topYOutput,
        z: 0,
        metadata: {
            shape: [dModel, vocabSize],
            params: `${((dModel * vocabSize) / 1e6).toFixed(1)}M`,
            params_raw: dModel * vocabSize,
            storage: `${((dModel * vocabSize * 2) / 1e6).toFixed(1)} MB`,
            role: 'Logits Out Prediction Highway',
            type: 'Unembedding Vocabulary Projection'
        }
    });

    return { cNodes, macroNodes };
}

// ---------------------------------------------------------------------------
// 20. Architectural Matrix, Pipeline & Math Rendering
// ---------------------------------------------------------------------------
const MODEL_SPECS = {
    distilbert: { 
        name: 'DistilBERT-Base', 
        family: 'Encoder', 
        num_layers: 6, 
        hidden_size: 768, 
        intermediate_size: 3072, 
        base_layer_params: 7077888, 
        base_total: 66301440,
        descr: '6 Layers · 768d · 12 Heads · Distilled Encoder'
    },
    roberta: { 
        name: 'RoBERTa-Base', 
        family: 'Encoder', 
        num_layers: 12, 
        hidden_size: 768, 
        intermediate_size: 3072, 
        base_layer_params: 7077888, 
        base_total: 123933696,
        descr: '12 Layers · 768d · 12 Heads · Bidirectional Encoder'
    },
    gpt2: { 
        name: 'GPT-2-Small', 
        family: 'Causal Decoder', 
        num_layers: 12, 
        hidden_size: 768, 
        intermediate_size: 3072, 
        base_layer_params: 7077888, 
        base_total: 124318464,
        descr: '12 Layers · 768d · Fused c_attn Conv1D · Causal LM'
    },
    qwen: { 
        name: 'Qwen2.5-0.5B', 
        family: 'GQA Decoder (SLM)', 
        num_layers: 24, 
        hidden_size: 896, 
        intermediate_size: 4864, 
        base_layer_params: 14909440, 
        base_total: 494008320,
        descr: '24 Layers · 896d · GQA (14Q:2KV) · SwiGLU MLP'
    }
};

function formatParamCount(num) {
    if (!num && num !== 0) return '—';
    if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
    return String(num);
}

function formatByteCount(bytes) {
    if (!bytes && bytes !== 0) return '—';
    if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + ' GB';
    if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + ' MB';
    if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + ' KB';
    return bytes + ' B';
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function renderStructuralTransformation(moduleMeta, modelKey) {
    const cardEl = document.getElementById('transformation-card');
    if (!cardEl) return;

    const spec = MODEL_SPECS[modelKey] || MODEL_SPECS.distilbert;
    const proj = (moduleMeta.model_projections && moduleMeta.model_projections[modelKey]) || {};
    const paradigm = moduleMeta.paradigm || 'svd';

    const d = spec.hidden_size;
    const dFfn = spec.intermediate_size;
    const r = 64;

    const diagramEl = document.getElementById('tf-diagram-body');
    const tableEl = document.getElementById('tf-table-body');
    const explEl = document.getElementById('tf-explanation-text');
    const badgeEl = document.getElementById('tf-paradigm-badge');

    if (badgeEl) badgeEl.innerText = `${spec.name.toUpperCase()} · ${(moduleMeta.category || paradigm).toUpperCase()}`;

    const layerBaseP = proj.baseline_layer_params || spec.base_layer_params;
    const layerCompP = proj.compressed_layer_params || proj.compressed_layer_p || Math.round(layerBaseP / (proj.savings_ratio || 4.0));
    const ratio = proj.savings_ratio || (layerBaseP / Math.max(1, layerCompP));
    const savedBytes = proj.storage_saved_formatted || formatByteCount(proj.storage_saved_bytes || Math.max(0, (spec.base_total - (proj.total_compressed_params || spec.base_total / ratio)) * 2));

    if (paradigm === 'svd') {
        let denseShape = `[${d} × ${d}]`;
        let denseP = d * d;
        let vShape = `[${r} × ${d}]`;
        let vP = r * d;
        let uShape = `[${d} × ${r}]`;
        let uP = d * r;

        if (modelKey === 'gpt2') {
            denseShape = `[${d} × ${d * 3}] (Fused QKV)`;
            denseP = d * (d * 3);
            vShape = `[${r} × ${d * 3}]`;
            vP = r * (d * 3);
            uShape = `[${d} × ${r}]`;
            uP = d * r;
        } else if (modelKey === 'qwen') {
            denseShape = `[${d} × ${dFfn}] (SwiGLU)`;
            denseP = d * dFfn;
            vShape = `[${r} × ${dFfn}]`;
            vP = r * dFfn;
            uShape = `[${d} × ${r}]`;
            uP = d * r;
        }

        if (diagramEl) {
            diagramEl.innerHTML = `
                <div class="tf-step dense" data-step-key="dense" onclick="selectPartIn3D('dense')" title="Click to illuminate Dense Layer in 3D">
                    <span class="tf-step-name">Dense Layer</span>
                    <span class="tf-step-shape">${denseShape}</span>
                    <span class="tf-step-param">${formatParamCount(denseP)}</span>
                </div>
                <span class="tf-arrow">➔ SVD ➔</span>
                <div class="tf-step compressed" data-step-key="v_factor" onclick="selectPartIn3D('v_factor')" title="Click to illuminate Factor V in 3D">
                    <span class="tf-step-name">Factor V (Proj)</span>
                    <span class="tf-step-shape">${vShape}</span>
                    <span class="tf-step-param">${formatParamCount(vP)}</span>
                </div>
                <span class="tf-arrow">➔ Squeeze ➔</span>
                <div class="tf-step compressed" data-step-key="u_factor" onclick="selectPartIn3D('u_factor')" title="Click to illuminate Factor U in 3D">
                    <span class="tf-step-name">Factor U (Exp)</span>
                    <span class="tf-step-shape">${uShape}</span>
                    <span class="tf-step-param">${formatParamCount(uP)}</span>
                </div>
            `;
        }
        if (tableEl) {
            tableEl.innerHTML = `
                <tr>
                    <td>Architecture</td>
                    <td>Dense Monolithic Layer</td>
                    <td class="highlight">Factorized Bottleneck (r=${r})</td>
                    <td class="highlight">${ratio.toFixed(1)}x Smaller</td>
                </tr>
                <tr>
                    <td>Per-Layer Params</td>
                    <td>${formatParamCount(layerBaseP)}</td>
                    <td class="highlight">${formatParamCount(layerCompP)}</td>
                    <td class="highlight">-${Math.round((1 - 1 / ratio) * 100)}%</td>
                </tr>
                <tr>
                    <td>Active Subspace</td>
                    <td>Full Rank (${d})</td>
                    <td class="highlight">Rank r=${r}</td>
                    <td class="highlight">-${Math.round((1 - r / d) * 100)}% Subspace</td>
                </tr>
                <tr>
                    <td>Model Storage Saved</td>
                    <td>${formatByteCount(spec.base_total * 2)}</td>
                    <td class="highlight">${savedBytes} Saved</td>
                    <td class="highlight">High Efficiency</td>
                </tr>
            `;
        }
        if (explEl) {
            explEl.innerHTML = `
                <strong>Structural Transformation (${spec.name}):</strong> Full-rank weight matrices are decomposed into an orthogonal eigenspace projection (Factor V) and expansion reconstruction (Factor U). Hidden representations funnel strictly through rank $r=${r}$, shedding high-frequency parameter noise while compressing layer weights by <strong>${ratio.toFixed(2)}x</strong>.
            `;
        }
    } else if (paradigm === 'mla') {
        const dLatent = Math.round(d / 4);
        if (diagramEl) {
            diagramEl.innerHTML = `
                <div class="tf-step dense" data-step-key="dense" onclick="selectPartIn3D('dense')" title="Click to illuminate Full KV in 3D">
                    <span class="tf-step-name">Full KV Matrix</span>
                    <span class="tf-step-shape">[${d} × 2·${d}]</span>
                    <span class="tf-step-param">Full Context Heads</span>
                </div>
                <span class="tf-arrow">➔ Compress ➔</span>
                <div class="tf-step compressed" data-step-key="mla" onclick="selectPartIn3D('mla')" title="Click to illuminate Latent KV in 3D">
                    <span class="tf-step-name">Latent c_t^KV</span>
                    <span class="tf-step-shape">[${dLatent}] Latent</span>
                    <span class="tf-step-param">Compressed KV</span>
                </div>
                <span class="tf-arrow">➔ RoPE ➔</span>
                <div class="tf-step compressed" data-step-key="mla" onclick="selectPartIn3D('mla')" title="Click to illuminate Decoupled KV in 3D">
                    <span class="tf-step-name">Decoupled KV</span>
                    <span class="tf-step-shape">Content + Position</span>
                    <span class="tf-step-param">Cached Latent</span>
                </div>
            `;
        }
        if (tableEl) {
            tableEl.innerHTML = `
                <tr>
                    <td>KV Cache Footprint</td>
                    <td>100% Uncompressed KV</td>
                    <td class="highlight">Latent Subspace (d_c=${dLatent})</td>
                    <td class="highlight">5.3x Smaller Cache</td>
                </tr>
                <tr>
                    <td>Layer Parameters</td>
                    <td>${formatParamCount(layerBaseP)}</td>
                    <td class="highlight">${formatParamCount(layerCompP)}</td>
                    <td class="highlight">-${Math.round((1 - 1 / ratio) * 100)}%</td>
                </tr>
                <tr>
                    <td>Attention Mechanism</td>
                    <td>Multi-Head Attention</td>
                    <td class="highlight">Multi-Head Latent Attention</td>
                    <td class="highlight">Low-Rank Latents</td>
                </tr>
            `;
        }
        if (explEl) {
            explEl.innerHTML = `
                <strong>Structural Transformation (${spec.name}):</strong> Compresses Key-Value activations into a low-rank latent vector $c_t^{KV} \\in \\mathbb{R}^{${dLatent}}$. During autoregressive generation, only this compact latent vector is cached in memory, eliminating KV cache bottlenecks while decoupled rotary embeddings preserve token position fidelity.
            `;
        }
    } else if (paradigm === 'moe') {
        if (diagramEl) {
            diagramEl.innerHTML = `
                <div class="tf-step dense" data-step-key="dense" onclick="selectPartIn3D('dense')" title="Click to illuminate Dense FFN in 3D">
                    <span class="tf-step-name">Dense FFN</span>
                    <span class="tf-step-shape">[${dFfn} × ${d}]</span>
                    <span class="tf-step-param">${formatParamCount(2 * d * dFfn)}</span>
                </div>
                <span class="tf-arrow">➔ Route ➔</span>
                <div class="tf-step compressed" data-step-key="router" onclick="selectPartIn3D('router')" title="Click to illuminate MoE Router in 3D">
                    <span class="tf-step-name">Router Gate</span>
                    <span class="tf-step-shape">[4 × ${d}]</span>
                    <span class="tf-step-param">Top-2 Gating</span>
                </div>
                <span class="tf-arrow">➔ Dispatch ➔</span>
                <div class="tf-step compressed" data-step-key="expert" onclick="selectPartIn3D('expert')" title="Click to illuminate Sparse Experts in 3D">
                    <span class="tf-step-name">4 Sparse Experts</span>
                    <span class="tf-step-shape">4 × [${dFfn} × ${d}]</span>
                    <span class="tf-step-param">2 Active / Token</span>
                </div>
            `;
        }
        if (tableEl) {
            tableEl.innerHTML = `
                <tr>
                    <td>FFN Activation</td>
                    <td>Dense Monolithic</td>
                    <td class="highlight">Top-2 Expert Routing</td>
                    <td class="highlight">2 of 4 Active</td>
                </tr>
                <tr>
                    <td>Layer Footprint</td>
                    <td>${formatParamCount(layerBaseP)}</td>
                    <td class="highlight">${formatParamCount(layerCompP)}</td>
                    <td class="highlight">${ratio.toFixed(1)}x FLOP Efficiency</td>
                </tr>
                <tr>
                    <td>Active Subnet / Token</td>
                    <td>100% of Weights</td>
                    <td class="highlight">50% Sparse Active</td>
                    <td class="highlight">-50% Compute</td>
                </tr>
            `;
        }
        if (explEl) {
            explEl.innerHTML = `
                <strong>Structural Transformation (${spec.name}):</strong> Replaces monolithic feed-forward blocks with 4 specialized expert subnetworks coordinated by a learned gating router. For each incoming token, only the top-2 experts are activated, doubling representation capacity while keeping per-token FLOPs fixed.
            `;
        }
    } else if (paradigm === 'ssm') {
        if (diagramEl) {
            diagramEl.innerHTML = `
                <div class="tf-step dense" data-step-key="dense" onclick="selectPartIn3D('dense')" title="Click to illuminate Dense Attention in 3D">
                    <span class="tf-step-name">Dense Attn</span>
                    <span class="tf-step-shape">O(N²) Softmax</span>
                    <span class="tf-step-param">Quadratic Attn</span>
                </div>
                <span class="tf-arrow">➔ SSM ➔</span>
                <div class="tf-step compressed" data-step-key="ssm" onclick="selectPartIn3D('ssm')" title="Click to illuminate Conv1D Mixer in 3D">
                    <span class="tf-step-name">Conv1D Mixer</span>
                    <span class="tf-step-shape">[${d} × 4]</span>
                    <span class="tf-step-param">Local Mixing</span>
                </div>
                <span class="tf-arrow">➔ Scan ➔</span>
                <div class="tf-step compressed" data-step-key="ssm" onclick="selectPartIn3D('ssm')" title="Click to illuminate Selective Scan in 3D">
                    <span class="tf-step-name">Selective Scan</span>
                    <span class="tf-step-shape">h_t = A·h + B·x</span>
                    <span class="tf-step-param">Linear O(N)</span>
                </div>
            `;
        }
        if (tableEl) {
            tableEl.innerHTML = `
                <tr>
                    <td>Sequence Scaling</td>
                    <td>O(N²) Quadratic</td>
                    <td class="highlight">O(N) Linear Scan</td>
                    <td class="highlight">Infinite Context</td>
                </tr>
                <tr>
                    <td>KV Memory State</td>
                    <td>Grows with Context</td>
                    <td class="highlight">O(1) Fixed Hidden State</td>
                    <td class="highlight">Zero KV Cache</td>
                </tr>
                <tr>
                    <td>Layer Footprint</td>
                    <td>${formatParamCount(layerBaseP)}</td>
                    <td class="highlight">${formatParamCount(layerCompP)}</td>
                    <td class="highlight">${ratio.toFixed(1)}x Efficiency</td>
                </tr>
            `;
        }
        if (explEl) {
            explEl.innerHTML = `
                <strong>Structural Transformation (${spec.name}):</strong> Replaces quadratic $O(N^2)$ dot-product attention with a continuous State Space Model (SSM) featuring input-dependent selective discretization $(\\Delta, B, C)$. State updates run as associative scans with fixed $O(1)$ recurrent memory footprint.
            `;
        }
    } else if (paradigm === 'slicegpt') {
        if (diagramEl) {
            diagramEl.innerHTML = `
                <div class="tf-step dense" data-step-key="dense" onclick="selectPartIn3D('dense')" title="Click to illuminate Full Matrix in 3D">
                    <span class="tf-step-name">Full Dense</span>
                    <span class="tf-step-shape">[${d} × ${d}]</span>
                    <span class="tf-step-param">100% Channels</span>
                </div>
                <span class="tf-arrow">➔ Ortho Q ➔</span>
                <div class="tf-step compressed" data-step-key="sliced_linear" onclick="selectPartIn3D('sliced_linear')" title="Click to illuminate PCA Basis in 3D">
                    <span class="tf-step-name">PCA Basis Q</span>
                    <span class="tf-step-shape">[${d} × ${d}]</span>
                    <span class="tf-step-param">Energy Rotation</span>
                </div>
                <span class="tf-arrow">➔ Slice ➔</span>
                <div class="tf-step compressed" data-step-key="sliced_linear" onclick="selectPartIn3D('sliced_linear')" title="Click to illuminate Sliced Channels in 3D">
                    <span class="tf-step-name">Sliced Matrix</span>
                    <span class="tf-step-shape">[${Math.round(d * 0.55)} × ${d}]</span>
                    <span class="tf-step-param">-45% Columns Deleted</span>
                </div>
            `;
        }
        if (tableEl) {
            tableEl.innerHTML = `
                <tr>
                    <td>Matrix Width</td>
                    <td>${d} Channels</td>
                    <td class="highlight">${Math.round(d * 0.55)} Channels</td>
                    <td class="highlight">-45% Sliced</td>
                </tr>
                <tr>
                    <td>Layer Parameters</td>
                    <td>${formatParamCount(layerBaseP)}</td>
                    <td class="highlight">${formatParamCount(layerCompP)}</td>
                    <td class="highlight">-${Math.round((1 - 1 / ratio) * 100)}% Memory</td>
                </tr>
                <tr>
                    <td>Speedup Hardware</td>
                    <td>Standard Dense BLAS</td>
                    <td class="highlight">Dense Small BLAS</td>
                    <td class="highlight">Native 1.8x Speedup</td>
                </tr>
            `;
        }
        if (explEl) {
            explEl.innerHTML = `
                <strong>Structural Transformation (${spec.name}):</strong> Computes principal variance axes via orthogonal transformation $Q$, concentrating token information into the leading 55% of channels, then permanently slices away the least-significant 45% of matrix dimensions without requiring sparse GEMM kernels.
            `;
        }
    } else if (paradigm === 'quant') {
        if (diagramEl) {
            diagramEl.innerHTML = `
                <div class="tf-step dense" data-step-key="dense" onclick="selectPartIn3D('dense')" title="Click to illuminate FP16 Layer in 3D">
                    <span class="tf-step-name">FP16 Weight</span>
                    <span class="tf-step-shape">[${d} × ${d}]</span>
                    <span class="tf-step-param">16-bit Float (2B/param)</span>
                </div>
                <span class="tf-arrow">➔ Quant ➔</span>
                <div class="tf-step compressed" data-step-key="quant" onclick="selectPartIn3D('quant')" title="Click to illuminate Quantized Scale in 3D">
                    <span class="tf-step-name">Scale & Zero</span>
                    <span class="tf-step-shape">s · (W - z)</span>
                    <span class="tf-step-param">Grid Calibration</span>
                </div>
                <span class="tf-arrow">➔ INT/Ternary ➔</span>
                <div class="tf-step compressed" data-step-key="quant" onclick="selectPartIn3D('quant')" title="Click to illuminate Low-Bit Tensor in 3D">
                    <span class="tf-step-name">Low-Bit Tensor</span>
                    <span class="tf-step-shape">[${d} × ${d}] Low-Bit</span>
                    <span class="tf-step-param">INT4 / 1.58-bit</span>
                </div>
            `;
        }
        if (tableEl) {
            tableEl.innerHTML = `
                <tr>
                    <td>Storage Precision</td>
                    <td>FP16 (16-bit)</td>
                    <td class="highlight">INT4 / 1.58-bit Ternary</td>
                    <td class="highlight">4x to 8x Density</td>
                </tr>
                <tr>
                    <td>Layer Memory Footprint</td>
                    <td>${formatByteCount(layerBaseP * 2)}</td>
                    <td class="highlight">${formatByteCount(Math.round(layerBaseP * 0.5))}</td>
                    <td class="highlight">-75% Memory Traffic</td>
                </tr>
                <tr>
                    <td>Compute Engine</td>
                    <td>FP16 Multipliers</td>
                    <td class="highlight">Integer SIMD / Adders</td>
                    <td class="highlight">High Efficiency</td>
                </tr>
            `;
        }
        if (explEl) {
            explEl.innerHTML = `
                <strong>Structural Transformation (${spec.name}):</strong> Quantizes weight parameters from 16-bit floating point down to discrete low-bit representations (INT4 or 1.58-bit ternary {-1, 0, +1}). Dramatically reduces memory bandwidth pressure and allows integer execution on Tensor Cores.
            `;
        }
    } else if (paradigm === 'linear') {
        if (diagramEl) {
            diagramEl.innerHTML = `
                <div class="tf-step dense" data-step-key="dense" onclick="selectPartIn3D('dense')" title="Click to illuminate Softmax Attention in 3D">
                    <span class="tf-step-name">Softmax Attn</span>
                    <span class="tf-step-shape">Softmax(QKᵀ/√d)V</span>
                    <span class="tf-step-param">O(N²) Quadratic</span>
                </div>
                <span class="tf-arrow">➔ Kernel ➔</span>
                <div class="tf-step compressed" data-step-key="linear" onclick="selectPartIn3D('linear')" title="Click to illuminate Feature Map in 3D">
                    <span class="tf-step-name">Feature Map φ</span>
                    <span class="tf-step-shape">φ(x) = elu(x) + 1</span>
                    <span class="tf-step-param">Non-Negative Map</span>
                </div>
                <span class="tf-arrow">➔ Associate ➔</span>
                <div class="tf-step compressed" data-step-key="linear" onclick="selectPartIn3D('linear')" title="Click to illuminate Linear Kernel in 3D">
                    <span class="tf-step-name">Linear Kernel</span>
                    <span class="tf-step-shape">φ(Q) · (φ(K)ᵀ·V)</span>
                    <span class="tf-step-param">O(N) Linear</span>
                </div>
            `;
        }
        if (tableEl) {
            tableEl.innerHTML = `
                <tr>
                    <td>Sequence Complexity</td>
                    <td>O(N²) Quadratic</td>
                    <td class="highlight">O(N) Linear Token Scaling</td>
                    <td class="highlight">Linear Speedup</td>
                </tr>
                <tr>
                    <td>Layer Parameters</td>
                    <td>${formatParamCount(layerBaseP)}</td>
                    <td class="highlight">${formatParamCount(layerCompP)}</td>
                    <td class="highlight">${ratio.toFixed(1)}x Efficiency</td>
                </tr>
                <tr>
                    <td>Attention Matrix State</td>
                    <td>N × N Intermediate Map</td>
                    <td class="highlight">d × d Cumulative State</td>
                    <td class="highlight">Constant Hidden Size</td>
                </tr>
            `;
        }
        if (explEl) {
            explEl.innerHTML = `
                <strong>Structural Transformation (${spec.name}):</strong> Reorders dot-product matrix multiplication by expressing softmax attention as associative feature kernels $\\phi(Q) (\\phi(K)^T V)$. Sequence complexity collapses from $O(N^2)$ to $O(N)$, unlocking inference at arbitrary context lengths.
            `;
        }
    } else {
        if (diagramEl) {
            diagramEl.innerHTML = `
                <div class="tf-step dense" data-step-key="dense" onclick="selectPartIn3D('dense')" title="Click to illuminate Dense Layer in 3D">
                    <span class="tf-step-name">Dense Layer</span>
                    <span class="tf-step-shape">[${d} × ${d}]</span>
                    <span class="tf-step-param">FP16 Weights</span>
                </div>
                <span class="tf-arrow">➔ Comp ➔</span>
                <div class="tf-step compressed" data-step-key="compressed" onclick="selectPartIn3D('compressed')" title="Click to illuminate Optimized Op in 3D">
                    <span class="tf-step-name">Optimized Op</span>
                    <span class="tf-step-shape">[${d} × ${d}]</span>
                    <span class="tf-step-param">${ratio.toFixed(1)}x Efficiency</span>
                </div>
            `;
        }
        if (tableEl) {
            tableEl.innerHTML = `
                <tr>
                    <td>Compression Ratio</td>
                    <td>1.0x Baseline</td>
                    <td class="highlight">${ratio.toFixed(1)}x</td>
                    <td class="highlight">High Efficiency</td>
                </tr>
                <tr>
                    <td>Layer Footprint</td>
                    <td>${formatParamCount(layerBaseP)}</td>
                    <td class="highlight">${formatParamCount(layerCompP)}</td>
                    <td class="highlight">Saved</td>
                </tr>
            `;
        }
        if (explEl) {
            explEl.innerHTML = `
                <strong>Structural Transformation (${spec.name}):</strong> Applies specialized tensor optimization yielding ${ratio.toFixed(1)}x parameter and memory reduction while preserving semantic attention fidelity.
            `;
        }
    }
}

function renderModelsMatrix(moduleMeta) {
    const gridEl = document.getElementById('models-matrix-grid');
    if (!gridEl) return;

    const projections = moduleMeta.model_projections || {};
    const modelKeys = ['distilbert', 'roberta', 'gpt2', 'qwen'];

    gridEl.innerHTML = modelKeys.map(key => {
        const spec = MODEL_SPECS[key];
        const proj = projections[key] || {};

        const displayName = proj.model_name || spec.name;
        const family = proj.family || spec.family;
        const layers = proj.num_layers || spec.num_layers;
        const hidden = proj.hidden_size || spec.hidden_size;

        const savingsRatio = proj.savings_ratio || 4.0;
        const baseTotal = proj.total_baseline_params || spec.base_total;
        const compTotal = proj.total_compressed_params || Math.round(baseTotal / savingsRatio);
        const savedFormatted = proj.storage_saved_formatted || formatByteCount(proj.storage_saved_bytes || Math.max(0, (baseTotal - compTotal) * 2));
        const layerComp = proj.compressed_layer_params || proj.compressed_layer_p || Math.round(compTotal / layers);

        const isActive = (key === currentModel);

        const bmark = (proj.benchmark) || (moduleMeta.verified_benchmark && moduleMeta.verified_benchmark.models && moduleMeta.verified_benchmark.models[key]);
        let bmarkHtml = '';
        if (bmark && bmark.status) {
            const isPass = bmark.status === 'PASS';
            const isPassLow = bmark.status === 'PASS_LOW';
            const statusClass = isPass ? 'status-pass' : (isPassLow ? 'status-pass-low' : 'status-skip');
            const accStr = (bmark.accuracy_pct !== null && bmark.accuracy_pct !== undefined) ? `${Number(bmark.accuracy_pct).toFixed(1)}%` : (isPass ? 'PASS' : bmark.status);
            const vramStr = (bmark.peak_vram_mb !== null && bmark.peak_vram_mb !== undefined) ? `${Math.round(bmark.peak_vram_mb)}MB` : null;
            const speedStr = (bmark.throughput_tokens_sec !== null && bmark.throughput_tokens_sec !== undefined) ? `${Math.round(bmark.throughput_tokens_sec / 1000)}k t/s` : null;

            bmarkHtml = `
                <div class="mmc-benchmark-row">
                    <span class="mmc-bench-badge ${statusClass}">${bmark.status}</span>
                    <span class="mmc-bench-acc">Acc: <b>${accStr}</b></span>
                    ${vramStr ? `<span class="mmc-bench-metric">VRAM: <b>${vramStr}</b></span>` : ''}
                    ${speedStr ? `<span class="mmc-bench-metric">Spd: <b>${speedStr}</b></span>` : ''}
                </div>
            `;
        }

        // Build SVG spark-bars for benchmark metrics
        let sparkHtml = '';
        if (bmark && bmark.accuracy_pct != null) {
            const acc = Math.min(100, Math.max(0, bmark.accuracy_pct));
            const vram = bmark.peak_vram_mb ? Math.min(100, bmark.peak_vram_mb / 10) : 0;
            const spd = bmark.throughput_tokens_sec ? Math.min(100, bmark.throughput_tokens_sec / 1200) : 0;
            const accColor = acc >= 75 ? '#34d399' : acc >= 50 ? '#fbbf24' : '#f87171';
            sparkHtml = `
                <div class="mmc-spark-section">
                    <div class="mmc-spark-row">
                        <span class="mmc-spark-label">Acc</span>
                        <div class="mmc-spark-bar-bg"><div class="mmc-spark-bar" style="width:${acc}%;background:${accColor}"></div></div>
                        <span class="mmc-spark-val" style="color:${accColor}">${acc.toFixed(1)}%</span>
                    </div>
                    <div class="mmc-spark-row">
                        <span class="mmc-spark-label">VRAM</span>
                        <div class="mmc-spark-bar-bg"><div class="mmc-spark-bar" style="width:${vram}%;background:#818cf8"></div></div>
                        <span class="mmc-spark-val">${bmark.peak_vram_mb ? Math.round(bmark.peak_vram_mb)+'MB' : '—'}</span>
                    </div>
                    <div class="mmc-spark-row">
                        <span class="mmc-spark-label">Spd</span>
                        <div class="mmc-spark-bar-bg"><div class="mmc-spark-bar" style="width:${spd}%;background:#f59e0b"></div></div>
                        <span class="mmc-spark-val">${bmark.throughput_tokens_sec ? Math.round(bmark.throughput_tokens_sec/1000)+'k/s' : '—'}</span>
                    </div>
                </div>
            `;
        }

        return `
            <div class="model-matrix-card ${isActive ? 'active' : ''}" data-model="${key}" onclick="switchModelFromMatrix('${key}')" title="Click to view 3D architecture on ${displayName}">
                <div class="mmc-header">
                    <span class="mmc-name">${displayName}</span>
                    <span class="mmc-ratio-badge">${Number(savingsRatio).toFixed(1)}x</span>
                </div>
                <div class="mmc-spec">${family} · ${layers}L · d=${hidden}</div>
                <div class="mmc-stats">
                    <span>Base: <b>${formatParamCount(baseTotal)}</b></span>
                    <span>Comp: <b>${formatParamCount(compTotal)}</b></span>
                </div>
                <div class="mmc-stats">
                    <span>Saved: <b class="mmc-saved">${savedFormatted}</b></span>
                    <span>/Layer: <b>${formatParamCount(layerComp)}</b></span>
                </div>
                ${bmarkHtml}
                ${sparkHtml}
            </div>
        `;
    }).join('');
}

window.switchModelFromMatrix = function(modelKey) {
    if (modelKey === currentModel) return;
    loadModelAndModule(modelKey, currentCompression);
};

function getModelPipelineComponents(moduleMeta, modelKey = currentModel) {
    const proj = (moduleMeta && moduleMeta.model_projections && moduleMeta.model_projections[modelKey]) || {};
    const dModel = proj.hidden_size || (modelKey === 'qwen' ? 896 : (modelKey === 'gpt2' ? 768 : 768));
    const dFfn = proj.intermediate_size || (modelKey === 'qwen' ? 4864 : (modelKey === 'gpt2' ? 3072 : 3072));
    const vocabSize = VOCAB_SIZES[modelKey] || 30522;
    
    const isAttn = (moduleMeta && (moduleMeta.category || "").toLowerCase().includes("attention")) || (moduleMeta && (moduleMeta.type || "").toLowerCase().includes("attention"));
    
    let pipeline = [];
    
    pipeline.push({
        id: 'input_embeddings', name: 'INPUT EMBEDDINGS', role: 'Token IDs -> Continuous Vectors', path: 'Stage 1: Token Ingestion',
        shape_transform: `[${vocabSize} × ${dModel}]`, math_latex: 'X^{(0)} = W_{\\text{embed}}[T]', color: '#10b981', params: `${((vocabSize * dModel) / 1e6).toFixed(1)}M`, stage: 'Input'
    });

    const groupCustom = (comps, stageName, pathName) => {
        let result = [];
        let qkvGroup = [];
        let gateUpGroup = [];
        
        for (let i = 0; i < comps.length; i++) {
            let c = { ...comps[i], stage: stageName, path: comps[i].path || pathName, math_latex: comps[i].math_latex || '\\text{Custom Component}' };
            let id = (c.id || "").toLowerCase();
            
            if ((id.startsWith('q_') || id.startsWith('k_') || id.startsWith('v_')) && !id.includes('factor')) {
                qkvGroup.push(c);
                let nextId = (i < comps.length - 1) ? (comps[i+1].id || "").toLowerCase() : "";
                if (!((nextId.startsWith('q_') || nextId.startsWith('k_') || nextId.startsWith('v_')) && !nextId.includes('factor'))) {
                    if (qkvGroup.length > 1) result.push(qkvGroup);
                    else result.push(qkvGroup[0]);
                    qkvGroup = [];
                }
            } else if (id.startsWith('gate_') || id.startsWith('up_')) {
                gateUpGroup.push(c);
                let nextId = (i < comps.length - 1) ? (comps[i+1].id || "").toLowerCase() : "";
                if (!(nextId.startsWith('gate_') || nextId.startsWith('up_'))) {
                    if (gateUpGroup.length > 1) result.push(gateUpGroup);
                    else result.push(gateUpGroup[0]);
                    gateUpGroup = [];
                }
            } else {
                result.push(c);
            }
        }
        return result;
    };

    if (isAttn && moduleMeta && moduleMeta.components && moduleMeta.components.length > 0) {
        pipeline.push(...groupCustom(moduleMeta.components, 'Attention', 'Path A: Attention Mechanism'));
    } else if (modelKey === 'qwen') {
        pipeline.push([
            { id: 'q_proj', name: 'Q Proj', role: 'Queries', path: 'Path A', shape_transform: `[${dModel} × ${dModel}]`, math_latex: 'Q = X W_q', color: '#38bdf8', params: `${((dModel * dModel) / 1000).toFixed(0)}K`, stage: 'Attention' },
            { id: 'k_proj', name: 'K Proj', role: 'Keys', path: 'Path A', shape_transform: `[128 × ${dModel}]`, math_latex: 'K = X W_k', color: '#f59e0b', params: `${((128 * dModel) / 1000).toFixed(0)}K`, stage: 'Attention' },
            { id: 'v_proj', name: 'V Proj', role: 'Values', path: 'Path A', shape_transform: `[128 × ${dModel}]`, math_latex: 'V = X W_v', color: '#d97706', params: `${((128 * dModel) / 1000).toFixed(0)}K`, stage: 'Attention' }
        ]);
        pipeline.push({ id: 'o_proj', name: 'O Proj', role: 'Output', path: 'Convergence', shape_transform: `[${dModel} × ${dModel}]`, math_latex: 'Out = \\text{Softmax}(Q K^T) V W_o', color: '#2dd4bf', params: `${((dModel * dModel) / 1000).toFixed(0)}K`, stage: 'Attention' });
    } else if (modelKey === 'gpt2') {
        pipeline.push([
            { id: 'c_attn', name: 'c_attn', role: 'Fused QKV', path: 'Path A', shape_transform: `[3×${dModel} × ${dModel}]`, math_latex: '[Q,K,V] = X W_{c\\_attn}', color: '#38bdf8', params: `${((3*dModel * dModel) / 1000).toFixed(0)}K`, stage: 'Attention' }
        ]);
        pipeline.push({ id: 'c_proj', name: 'c_proj', role: 'Output', path: 'Convergence', shape_transform: `[${dModel} × ${dModel}]`, math_latex: 'Out = \\text{Softmax}(Q K^T) V W_{proj}', color: '#2dd4bf', params: `${((dModel * dModel) / 1000).toFixed(0)}K`, stage: 'Attention' });
    } else {
        pipeline.push([
            { id: 'qkv_proj', name: 'QKV Proj', role: 'QKV Projections', path: 'Path A', shape_transform: `[3×${dModel} × ${dModel}]`, math_latex: '[Q,K,V] = X W', color: '#38bdf8', params: `${((3*dModel * dModel) / 1000).toFixed(0)}K`, stage: 'Attention' }
        ]);
        pipeline.push({ id: 'out_proj', name: 'Output Proj', role: 'Output', path: 'Convergence', shape_transform: `[${dModel} × ${dModel}]`, math_latex: 'Out = \\text{Softmax}(Q K^T) V W_o', color: '#2dd4bf', params: `${((dModel * dModel) / 1000).toFixed(0)}K`, stage: 'Attention' });
    }

    pipeline.push({ id: 'attn.residual_add', name: '(+) Residual Add', role: 'Skip Connection', path: 'Highway #1', shape_transform: `[Seq × ${dModel}]`, math_latex: 'X_{\\text{mid}} = X_{\\text{in}} + \\text{Attn}(X_{\\text{in}})', color: '#0284c7', params: '0 (Op)', stage: 'Residual' });

    if (!isAttn && moduleMeta && moduleMeta.components && moduleMeta.components.length > 0) {
        pipeline.push(...groupCustom(moduleMeta.components, 'FFN/MLP', 'Path B: MLP Mechanism'));
    } else if (modelKey === 'qwen') {
        pipeline.push([
            { id: 'gate_proj', name: 'Gate Proj', role: 'SwiGLU Gate', path: 'Path B', shape_transform: `[${dFfn} × ${dModel}]`, math_latex: 'G = \\text{SiLU}(X W_{gate})', color: '#a855f7', params: `${((dFfn * dModel) / 1e6).toFixed(1)}M`, stage: 'MLP' },
            { id: 'up_proj', name: 'Up Proj', role: 'SwiGLU Up', path: 'Path B', shape_transform: `[${dFfn} × ${dModel}]`, math_latex: 'U = X W_{up}', color: '#8b5cf6', params: `${((dFfn * dModel) / 1e6).toFixed(1)}M`, stage: 'MLP' }
        ]);
        pipeline.push({ id: 'down_proj', name: 'Down Proj', role: 'SwiGLU Down', path: 'Convergence', shape_transform: `[${dModel} × ${dFfn}]`, math_latex: 'Y = (G \\odot U) W_{down}', color: '#5eead4', params: `${((dModel * dFfn) / 1e6).toFixed(1)}M`, stage: 'MLP' });
    } else {
        pipeline.push([
            { id: 'ffn_up', name: 'FFN Up', role: 'Expansion', path: 'Path B', shape_transform: `[${dFfn} × ${dModel}]`, math_latex: 'H = \\text{GELU}(X W_1)', color: '#38bdf8', params: `${((dFfn * dModel) / 1e6).toFixed(1)}M`, stage: 'MLP' }
        ]);
        pipeline.push({ id: 'ffn_down', name: 'FFN Down', role: 'Contraction', path: 'Convergence', shape_transform: `[${dModel} × ${dFfn}]`, math_latex: 'Y = H W_2', color: '#60a5fa', params: `${((dModel * dFfn) / 1e6).toFixed(1)}M`, stage: 'MLP' });
    }

    pipeline.push({ id: 'ffn.residual_add', name: '(+) Residual Add', role: 'Skip Connection', path: 'Highway #2', shape_transform: `[Seq × ${dModel}]`, math_latex: 'X_{\\text{out}} = X_{\\text{mid}} + \\text{FFN}(X_{\\text{mid}})', color: '#7c3aed', params: '0 (Op)', stage: 'Residual' });
    pipeline.push({ id: 'output_head', name: 'OUTPUT LM HEAD', role: 'Vocabulary Logits', path: 'Stage 7: Output', shape_transform: `[${dModel} × ${vocabSize}]`, math_latex: '\\hat{y} = \\text{Softmax}(X W_{\\text{vocab}}^T)', color: '#f43f5e', params: `${((dModel * vocabSize) / 1e6).toFixed(1)}M`, stage: 'Output' });

    return pipeline;
}

function renderTokenLifecycle(moduleMeta, modelKey = currentModel) {
    const listEl = document.getElementById('lifecycle-steps-list');
    if (!listEl) return;

    const proj = (moduleMeta && moduleMeta.model_projections && moduleMeta.model_projections[modelKey]) || {};
    const dModel = proj.hidden_size || (modelKey === 'qwen' ? 896 : 768);
    const dFfn = proj.intermediate_size || (modelKey === 'qwen' ? 4864 : 3072);
    const vocabSize = VOCAB_SIZES[modelKey] || 30522;
    
    const isAttn = (moduleMeta && (moduleMeta.category || "").toLowerCase().includes("attention")) || (moduleMeta && (moduleMeta.type || "").toLowerCase().includes("attention"));
    const hasCustom = (moduleMeta && moduleMeta.components && moduleMeta.components.length > 0);

    const steps = [
        {
            num: 1,
            title: 'Token Ingestion & Position Encoding',
            badge: 'INPUT',
            badgeColor: '#10b981',
            badgeBg: 'rgba(16, 185, 129, 0.2)',
            enters: `Token IDs [t_1, ..., t_S] ∈ {1..${vocabSize.toLocaleString()}}`,
            added: modelKey === 'qwen' 
                ? 'RoPE Rotary Position Frequencies (Θ_i) injected at Q & K' 
                : `+ W_pos [Positional Vector] ∈ ℝ^{Seq × ${dModel}}`,
            exits: `Continuous Hidden States X^{(0)} ∈ ℝ^{Seq × ${dModel}}`,
            math: modelKey === 'qwen' 
                ? 'X^{(0)} = W_e[T], \\quad \\mathbf{q}_m = \\mathbf{R}_{\\Theta, m}^d (W_q x_m), \\; \\mathbf{k}_n = \\mathbf{R}_{\\Theta, n}^d (W_k x_n)' 
                : 'X^{(0)} = W_{\\text{embed}}[T] + W_{\\text{pos}}[P] \\in \\mathbb{R}^{S \\times ' + dModel + '}',
            target: 'input_embeddings'
        },
        {
            num: 2,
            title: modelKey === 'qwen' ? 'RMS Normalization (Pre-Attn)' : 'Layer Normalization (Pre-Attn)',
            badge: 'NORM',
            badgeColor: '#38bdf8',
            badgeBg: 'rgba(56, 189, 248, 0.2)',
            enters: `Incoming Tokens X ∈ ℝ^{Seq × ${dModel}}`,
            added: 'Variance Normalization & Scale γ',
            exits: `Normalized Tokens X̃ ∈ ℝ^{Seq × ${dModel}}`,
            math: modelKey === 'qwen'
                ? '\\tilde{X} = \\frac{X}{\\text{RMS}(X)} \\odot \\gamma, \\quad \\text{RMS}(X) = \\sqrt{\\frac{1}{d}\\sum x_i^2 + \\epsilon}'
                : '\\tilde{X} = \\frac{X - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\odot \\gamma + \\beta',
            target: modelKey === 'qwen' ? 'q_proj' : 'v_factor'
        },
        {
            num: 3,
            title: (isAttn && hasCustom) ? moduleMeta.name : (modelKey === 'qwen' ? 'Grouped-Query Attention Contextualization' : 'Self-Attention Contextualization'),
            badge: (isAttn && hasCustom) ? 'CUSTOM ATTN' : 'ATTENTION',
            badgeColor: '#2dd4bf',
            badgeBg: 'rgba(45, 212, 191, 0.2)',
            enters: `Normalized Tokens X̃ ∈ ℝ^{Seq × ${dModel}}`,
            added: (isAttn && hasCustom) ? moduleMeta.components.map(c => c.name).join(' ➔ ') : 'Context Interaction Matrix: Softmax(Q K^T / √d_k) · V',
            exits: `Attention Context Delta O ∈ ℝ^{Seq × ${dModel}}`,
            math: (isAttn && hasCustom && moduleMeta.math_latex) ? moduleMeta.math_latex : '\\text{Attn}(X) = \\text{Softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}} + M\\right) V \\cdot W_o',
            target: (isAttn && hasCustom) ? moduleMeta.components[0].id : (modelKey === 'qwen' ? 'o_proj' : (modelKey === 'gpt2' ? 'c_proj' : 'out_lin'))
        },
        {
            num: 4,
            title: '(+) Attention Residual Skip Highway',
            badge: '(+) ADD #1',
            badgeColor: '#0284c7',
            badgeBg: 'rgba(2, 132, 199, 0.25)',
            enters: `Original Tokens X_in + Context Delta O`,
            added: '+ X_in (Unmodified incoming tokens added via bypass highway)',
            exits: `Accumulated Tokens X_mid = X_in + Dropout(O)`,
            math: 'X_{\\text{mid}} = X_{\\text{in}} + \\text{Dropout}\\left(\\text{Attn}(X_{\\text{in}})\\right)',
            target: 'attn.residual_add'
        },
        {
            num: 5,
            title: (!isAttn && hasCustom) ? moduleMeta.name : (modelKey === 'qwen' ? 'SwiGLU Gated Feed-Forward Network' : 'Feed-Forward Knowledge Retrieval (GELU)'),
            badge: (!isAttn && hasCustom) ? 'CUSTOM FFN' : 'FFN',
            badgeColor: '#a855f7',
            badgeBg: 'rgba(168, 85, 247, 0.2)',
            enters: `Contextualized Tokens X_mid ∈ ℝ^{Seq × ${dModel}}`,
            added: (!isAttn && hasCustom) ? moduleMeta.components.map(c => c.name).join(' ➔ ') : (modelKey === 'qwen' 
                ? `Bifurcated Non-Linear Gating: SiLU(X W_g) ⊙ (X W_u) across ${dFfn} intermediate dims` 
                : `4× Dimension Expansion GELU(X W_1 + b_1) across ${dFfn} hidden dims`),
            exits: `Factual Knowledge Delta Y_ffn ∈ ℝ^{Seq × ${dModel}}`,
            math: (!isAttn && hasCustom && moduleMeta.math_latex) ? moduleMeta.math_latex : (modelKey === 'qwen'
                ? 'Y_{\\text{ffn}} = \\left( \\text{SiLU}(X_{\\text{mid}} W_g) \\odot (X_{\\text{mid}} W_u) \\right) W_d'
                : 'Y_{\\text{ffn}} = \\text{GELU}(X_{\\text{mid}} W_1 + b_1) W_2 + b_2'),
            target: (!isAttn && hasCustom) ? moduleMeta.components[0].id : (modelKey === 'qwen' ? 'gate_proj' : (modelKey === 'gpt2' ? 'c_fc' : 'intermediate'))
        },
        {
            num: 6,
            title: '(+) FFN Residual Skip Highway',
            badge: '(+) ADD #2',
            badgeColor: '#7c3aed',
            badgeBg: 'rgba(124, 58, 237, 0.25)',
            enters: `Intermediate Tokens X_mid + FFN Knowledge Delta Y_ffn`,
            added: '+ X_mid (Intermediate tokens added via bypass highway)',
            exits: `Layer Output Tokens X_layer = X_mid + Dropout(Y_ffn)`,
            math: 'X_{\\text{layer}} = X_{\\text{mid}} + \\text{Dropout}\\left(\\text{FFN}(X_{\\text{mid}})\\right)',
            target: 'ffn.residual_add'
        },
        {
            num: 7,
            title: 'Output LM Head & Next-Token Softmax',
            badge: 'OUTPUT',
            badgeColor: '#f43f5e',
            badgeBg: 'rgba(244, 63, 94, 0.2)',
            enters: `Final Hidden Representation X^{(L)} ∈ ℝ^{Seq × ${dModel}}`,
            added: `Final Norm + Unembedding Projection W_vocab^T ∈ ℝ^{${dModel} × ${vocabSize}}`,
            exits: `Probability Distribution P(t_{S+1}) ∈ [0, 1]^{${vocabSize.toLocaleString()}}`,
            math: '\\hat{y} = \\text{Softmax}\\left(\\text{Norm}(X^{(L)}) W_{\\text{vocab}}^T\\right) \\in \\mathbb{R}^{S \\times V}',
            target: 'output_head'
        }
    ];

    listEl.innerHTML = steps.map(s => `
        <div class="lifecycle-step-item" data-step-target="${s.target}" onclick="selectPartIn3D('${s.target}')" title="Click to illuminate ${s.title} in 3D">
            <div class="lsi-header">
                <div class="lsi-title-group">
                    <span class="lsi-num">0${s.num}</span>
                    <span class="lsi-title">${s.title}</span>
                </div>
                <span class="lsi-badge" style="color: ${s.badgeColor}; background: ${s.badgeBg}; border: 1px solid ${s.badgeColor};">${s.badge}</span>
            </div>
            <div class="lsi-flow">
                <span class="lsi-in">In: ${escapeHtml(s.enters)}</span>
                <span class="lsi-arrow">➔</span>
                <span class="lsi-added">Added: ${escapeHtml(s.added)}</span>
                <span class="lsi-arrow">➔</span>
                <span class="lsi-out">Out: ${escapeHtml(s.exits)}</span>
            </div>
            <div class="lsi-math" data-math="${encodeURIComponent(s.math)}">$$ ${s.math} $$</div>
        </div>
    `).join('');

    if (window.katex) {
        listEl.querySelectorAll('.lsi-math').forEach(el => {
            const raw = decodeURIComponent(el.getAttribute('data-math') || '');
            if (raw) {
                try { katex.render(raw, el, { displayMode: false, throwOnError: false }); } catch(e) {}
            }
        });
    }
}


// Pipeline item click handler — expose globally
window.focusPipelineComponent = function(compId) {
    // Focus in 3D
    if (window.selectPartIn3D) selectPartIn3D(compId, { flyCamera: true, updateInspector: true });
    // Highlight pipeline item
    document.querySelectorAll('.component-pipe-item').forEach(el => {
        el.classList.toggle('cpi-highlighted', el.getAttribute('data-comp-id') === compId);
    });
    // Scroll lifecycle step if matching
    document.querySelectorAll('.lifecycle-step-item').forEach(el => {
        const tgt = el.getAttribute('data-step-target');
        if (tgt && compId.includes(tgt)) el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
};

function renderComponentPipeline(moduleMeta, modelKey = currentModel) {
    const listEl = document.getElementById('component-pipeline-list');
    const badgeEl = document.getElementById('component-count-badge');
    if (!listEl) return;

    const comps = getModelPipelineComponents(moduleMeta, modelKey);
    let totalBlocks = 0;
    
    // Flatten for counting and highlighting logic
    const flatComps = [];
    comps.forEach(c => {
        if (Array.isArray(c)) { totalBlocks += c.length; flatComps.push(...c); }
        else { totalBlocks += 1; flatComps.push(c); }
    });

    if (badgeEl) badgeEl.innerText = `${totalBlocks} BLOCKS`;

    if (totalBlocks === 0) {
        listEl.innerHTML = `<div style="font-size: 10px; color: var(--text-muted); padding: 6px;">Single Unified Linear Projection</div>`;
        return;
    }

    const renderItem = (c) => `
        <div class="component-pipe-item ${Array.isArray(c) ? 'nested-item' : ''}" data-comp-id="${c.id}" onclick="selectPartIn3D('${c.id}')" title="Click to illuminate ${c.name} in 3D & view mathematical specs" style="--cpi-color: ${c.color || '#2dd4bf'};">
            <div class="cpi-left">
                <span class="cpi-swatch" style="background: ${c.color || '#2dd4bf'};"></span>
                <div class="cpi-info">
                    <div class="cpi-name-row">
                        <span class="cpi-name">${c.name}</span>
                        <span class="cpi-path-tag">${c.path}</span>
                    </div>
                    <span class="cpi-role">${c.role}</span>
                    <span class="cpi-math-preview" data-raw-math="${encodeURIComponent(c.math_latex || '')}">$$ ${c.math_latex || ''} $$</span>
                </div>
            </div>
            <div class="cpi-right">
                <span class="cpi-shape">${c.shape_transform || ''}</span>
                <span class="cpi-params">${c.params || ''}</span>
            </div>
        </div>
    `;

    listEl.innerHTML = comps.map(group => {
        if (Array.isArray(group)) {
            return `<div class="pipeline-parallel-group">
                <div class="parallel-indicator"></div>
                ${group.map(c => renderItem(c)).join('')}
            </div>`;
        } else {
            return renderItem(group);
        }
    }).join('');

    if (window.katex) {
        listEl.querySelectorAll('.cpi-math-preview').forEach(el => {
            const raw = decodeURIComponent(el.getAttribute('data-raw-math') || '');
            if (raw) {
                try { katex.render(raw, el, { displayMode: false, throwOnError: false }); } catch(e) {}
            }
        });
    }

    if (activeSelectedSublayerKey) {
        syncUiPartHighlights(activeSelectedSublayerKey);
    }
}

window.focusComponentIn3D = function(compId) {
    selectPartIn3D(compId, { flyCamera: true, updateInspector: true });
};

function renderLatexFormula(moduleMeta, formulaOverride = null) {
    const bodyEl = document.getElementById('formula-latex-body');
    if (!bodyEl) return;
    const math = formulaOverride || (moduleMeta && (moduleMeta.math_latex || moduleMeta.math)) || 'W_{comp} = \\mathcal{C}(W_{dense})';

    // Add copy button
    const copyBtn = document.createElement('button');
    copyBtn.className = 'formula-copy-btn';
    copyBtn.title = 'Copy LaTeX formula';
    copyBtn.innerHTML = '⎘ Copy LaTeX';
    copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(math).then(() => {
            copyBtn.innerHTML = '✓ Copied!';
            setTimeout(() => { copyBtn.innerHTML = '⎘ Copy LaTeX'; }, 1500);
        }).catch(() => {
            copyBtn.innerHTML = '⎘ Copy LaTeX';
        });
    });
    bodyEl.appendChild(copyBtn);

    if (window.katex) {
        try {
            katex.render(math, bodyEl, { displayMode: true, throwOnError: false });
            return;
        } catch (e) {
            console.warn('KaTeX render error:', e);
        }
    }

    bodyEl.innerHTML = `<div style="font-size: 11px; font-weight: 600; color: #38bdf8; font-family: var(--font-mono); line-height: 1.5; word-break: break-all; text-align: center; padding: 4px 0;">$$ ${escapeHtml(math)} $$</div>`;
}

function updateModelSelectorTabs(moduleMeta) {
    const projections = (moduleMeta && moduleMeta.model_projections) || {};
    const tabs = document.querySelectorAll('.model-tab');
    tabs.forEach(tab => {
        const mKey = tab.dataset.model;
        const spec = MODEL_SPECS[mKey];
        if (!spec) return;
        const proj = projections[mKey];
        const ratio = proj ? Number(proj.savings_ratio).toFixed(1) + 'x' : '';
        const ratioBadge = ratio ? `<span class="tab-ratio">${ratio}</span>` : '';
        tab.innerHTML = `${spec.name.replace('-Base', '').replace('-Small', '')}${ratioBadge}`;
    });
}

function applyResearchModule(moduleMeta) {
    if (!spatialData || !spatialData.models[currentModel]) return;

    // Retrieve full module with components from map if available
    if (modulesComponentsMap[moduleMeta.id]) {
        moduleMeta = { ...modulesComponentsMap[moduleMeta.id], ...moduleMeta };
    }
    activeResearchModule = moduleMeta;

    // 1. Sync dropdown selection
    const select = document.getElementById('quick-module-select');
    if (select && select.value !== moduleMeta.id) {
        select.value = moduleMeta.id;
    }

    // 2. Synthesize exact 3D nodes for this module on currentModel
    const synthesized = synthesizeModule3DNodes(moduleMeta, currentModel);
    const modelData = spatialData.models[currentModel];

    clearScene();

    // Rebuild baseline side
    const h1 = buildMeshNodes(getAugmentedBaselineNodes(currentModel), OFFSET_BASELINE, false);
    if (modelData.baseline_macro_nodes) {
        buildMacroBlocks(modelData.baseline_macro_nodes, OFFSET_BASELINE, false);
    }

    // Rebuild compressed side with customized module nodes
    let h2 = 0;
    if (synthesized) {
        h2 = buildMeshNodes(synthesized.cNodes, OFFSET_COMPRESSED, true);
        buildMacroBlocks(synthesized.macroNodes, OFFSET_COMPRESSED, true);
    } else {
        const defaultMod = (modelData.modules && modelData.modules[moduleMeta.paradigm || 'svd']) || modelData;
        h2 = buildMeshNodes(defaultMod.compressed_nodes, OFFSET_COMPRESSED, true);
        if (defaultMod.macro_nodes) buildMacroBlocks(defaultMod.macro_nodes, OFFSET_COMPRESSED, true);
    }

    maxGraphHeight = Math.max(h1, h2, modelData.max_y || 16.0);

    // Rebuild token streams
    const baseStreams = createTokenMesh(tokenDataBaseline, OFFSET_BASELINE, tokenMaterialBaseline, trailMaterialBaseline);
    tokenMeshBaseline = baseStreams.main;
    trailMeshBaseline = baseStreams.trail;
    scene.add(tokenMeshBaseline);
    scene.add(trailMeshBaseline);

    const compStreams = createTokenMesh(tokenDataCompressed, OFFSET_COMPRESSED, tokenMaterialCompressed, trailMaterialCompressed);
    tokenMeshCompressed = compStreams.main;
    trailMeshCompressed = compStreams.trail;
    scene.add(tokenMeshCompressed);
    scene.add(trailMeshCompressed);

    // Smoothly fly camera to center both models
    smoothFlyCamera(maxGraphHeight * 0.5, Math.max(18.0, maxGraphHeight * 1.35));

    // Update Headings & Viewport Titles (for both Baseline and Compressed!)
    const spec = MODEL_SPECS[currentModel] || { name: currentModel.toUpperCase(), descr: 'Dense Baseline' };

    const leftTitle = document.getElementById('left-panel-title');
    if (leftTitle) {
        leftTitle.innerText = `${spec.name.toUpperCase()} (Dense Baseline)`;
    }

    const rightTitle = document.getElementById('right-panel-title');
    if (rightTitle) {
        rightTitle.innerText = `${spec.name.toUpperCase()} (${moduleMeta.name.slice(0, 26)})`;
    }

    const labelEl = document.getElementById('active-mode-label');
    if (labelEl) {
        labelEl.textContent = `${spec.name.toUpperCase()} · ${moduleMeta.id.toUpperCase()}`;
    }

    // Update Comparison HUD with exact module projections
    const proj = (moduleMeta.model_projections && moduleMeta.model_projections[currentModel]) || {};
    const dModel = proj.hidden_size || spec.hidden_size;
    const baseP = proj.total_baseline_params || spec.base_total;
    const compP = proj.total_compressed_params || Math.round(baseP / (proj.savings_ratio || 4.0));
    const ratio = proj.savings_ratio || (baseP / Math.max(1, compP));
    const savedBytes = proj.storage_saved_bytes || Math.max(0, (baseP - compP) * 2);

    const layerBaseP = proj.baseline_layer_params || spec.base_layer_params;
    const layerCompP = proj.compressed_layer_p || proj.compressed_layer_params || Math.round(layerBaseP / ratio);

    const hudComp = document.getElementById('hud-compressed-params');
    if (hudComp) hudComp.innerText = formatParamCount(compP);

    const hudRatio = document.getElementById('hud-ratio-badge');
    if (hudRatio) hudRatio.innerText = `${Number(ratio).toFixed(1)}x`;

    const hudSaved = document.getElementById('hud-storage-saved');
    if (hudSaved) {
        hudSaved.innerText = proj.storage_saved_formatted || formatByteCount(savedBytes);
    }

    const nameEl = document.getElementById('hud-model-name');
    if (nameEl) nameEl.innerText = `${spec.name.toUpperCase()} · ${moduleMeta.id.slice(0, 16).toUpperCase()}`;

    // Update Inspector Header & Primary Stats
    const badgeEl = document.getElementById('insp-type-badge');
    if (badgeEl) badgeEl.innerText = (moduleMeta.category || 'Attention Module').toUpperCase();

    const titleEl = document.getElementById('insp-title');
    if (titleEl) titleEl.innerText = moduleMeta.name;

    const effEl = document.getElementById('insp-efficiency');
    if (effEl) effEl.innerText = `${Number(ratio).toFixed(2)}x`;

    const paramsEl = document.getElementById('insp-params');
    if (paramsEl) paramsEl.innerText = formatParamCount(compP);

    const storageEl = document.getElementById('insp-storage');
    if (storageEl) storageEl.innerText = proj.storage_saved_formatted || formatByteCount(savedBytes);

    const compBaseEl = document.getElementById('insp-comp-baseline');
    if (compBaseEl) compBaseEl.innerText = `${formatParamCount(baseP)} (${formatParamCount(layerBaseP)}/L)`;

    const compCompEl = document.getElementById('insp-comp-compressed');
    if (compCompEl) compCompEl.innerText = `${formatParamCount(compP)} (${formatParamCount(layerCompP)}/L)`;

    const compSavingsBadge = document.getElementById('insp-savings-badge');
    if (compSavingsBadge) compSavingsBadge.innerText = `-${Math.round((1 - 1 / Math.max(1, ratio)) * 100)}%`;

    const descEl = document.getElementById('insp-desc');
    if (descEl) {
        descEl.innerHTML = `
            <strong>${moduleMeta.name}</strong><br>
            <span style="color: var(--col-svd); font-size: 10px; font-family: var(--font-mono);">${moduleMeta.file}</span><br>
            <p style="margin-top: 6px; font-size: 11px; line-height: 1.4; color: #cbd5e1;">${moduleMeta.description}</p>
        `;
    }

    const noteEl = document.getElementById('insp-edu-note');
    if (noteEl) {
        noteEl.innerHTML = `
            💡 <strong>Dynamic Architectural Synthesis:</strong><br>
            Executing on <strong>${spec.name}</strong> (${spec.descr}).<br>
            Achieving <strong>${Number(ratio).toFixed(2)}x compression</strong> (${formatParamCount(layerBaseP)} → ${formatParamCount(layerCompP)} per layer).
        `;
    }

    const dimsEl = document.getElementById('insp-tensor-dims');
    if (dimsEl) {
        if (currentModel === 'qwen') dimsEl.innerText = `[QWEN 896×896 · GQA 14:2 · SwiGLU 4864]`;
        else if (currentModel === 'gpt2') dimsEl.innerText = `[GPT-2 768×2304 Fused · FFN 3072]`;
        else if (currentModel === 'roberta') dimsEl.innerText = `[ROBERTA 768×768 · FFN 3072 · 12H]`;
        else dimsEl.innerText = `[DISTILBERT 768×768 · FFN 3072 · 12H]`;
    }

    const aspectEl = document.getElementById('insp-aspect-ratio');
    if (aspectEl) aspectEl.innerText = `${Number(ratio).toFixed(2)}x Saved`;

    const volEl = document.getElementById('insp-spatial-vol');
    if (volEl) volEl.innerText = `${(moduleMeta.components || []).length} Blocks`;

    const headsEl = document.getElementById('insp-heads-info');
    if (headsEl) headsEl.innerText = spec.descr;

    // Render Matrix, Pipeline, Formula & Tab badges
    renderModelsMatrix(moduleMeta);
    renderTokenLifecycle(moduleMeta, currentModel);
    renderComponentPipeline(moduleMeta, currentModel);
    renderLatexFormula(moduleMeta);
    renderStructuralTransformation(moduleMeta, currentModel);
    updateModelSelectorTabs(moduleMeta);
    // Param breakdown & comparison pin
    if (window.renderParamBreakdownTable) renderParamBreakdownTable(moduleMeta, currentModel);
    if (typeof renderComparisonPin === 'function' && _pinnedModule) renderComparisonPin();

    // Update global module counter badge in header
    const modCountBadge = document.getElementById('hud-active-module-badge');
    if (modCountBadge) {
        modCountBadge.textContent = moduleMeta.name.slice(0, 18).toUpperCase();
        modCountBadge.style.opacity = '1';
    }

    // Permalink: update URL hash with current module+model
    try {
        const newHash = `#module=${encodeURIComponent(moduleMeta.id)}&model=${currentModel}`;
        history.replaceState(null, '', newHash);
    } catch(e) {}

    inspector.classList.remove('hidden');
}

// ---------------------------------------------------------------------------
// 21. Bootstrap
// ---------------------------------------------------------------------------

// ─────────────────────────────────────────────────────────────────────────────
// MEGA ENHANCEMENT — Added Features
// ─────────────────────────────────────────────────────────────────────────────

// ── FPS Counter ──────────────────────────────────────────────────────────────
let _fpsFrames = 0, _fpsLast = performance.now(), _fpsValue = 0;
function updateFPS() {
    _fpsFrames++;
    const now = performance.now();
    if (now - _fpsLast >= 1000) {
        _fpsValue = _fpsFrames;
        _fpsFrames = 0;
        _fpsLast = now;
        const el = document.getElementById('fps-counter');
        if (el) {
            el.textContent = `${_fpsValue} fps`;
            el.style.color = _fpsValue >= 50 ? '#34d399' : _fpsValue >= 30 ? '#fbbf24' : '#f87171';
        }
    }
}

// ── Auto-Tour Camera Mode ────────────────────────────────────────────────────
let _tourActive = false;
let _tourStep = 0;
let _tourTimer = null;
const TOUR_STEPS = [
    { y: 2,  dist: 20, label: '01 — Input Embeddings',      target: null },
    { y: 5,  dist: 18, label: '02 — Q · K · V Projections', target: 'q_lin' },
    { y: 8,  dist: 16, label: '03 — Attention Output',       target: 'out_lin' },
    { y: 10, dist: 16, label: '04 — Residual Add #1',        target: null },
    { y: 13, dist: 18, label: '05 — FFN Up Projection',      target: 'lin1' },
    { y: 16, dist: 18, label: '06 — FFN Down Projection',    target: 'lin2' },
    { y: 20, dist: 22, label: '07 — Output LM Head',         target: null },
    { y: 12, dist: 35, label: '08 — Full Architecture View', target: null, fullFit: true },
];

function startTour() {
    _tourActive = true;
    _tourStep = 0;
    const btn = document.getElementById('btn-auto-tour');
    if (btn) { btn.innerHTML = '⏹ Stop'; btn.classList.add('active'); }
    const lbl = document.getElementById('tour-step-label');
    if (lbl) { lbl.style.display = 'block'; lbl.style.opacity = '1'; }
    runTourStep();
}
function stopTour() {
    _tourActive = false;
    clearTimeout(_tourTimer);
    const btn = document.getElementById('btn-auto-tour');
    if (btn) { btn.innerHTML = '▶ Tour'; btn.classList.remove('active'); }
    const lbl = document.getElementById('tour-step-label');
    if (lbl) lbl.style.opacity = '0';
    // Clear active lifecycle highlight
    document.querySelectorAll('.lifecycle-step-item').forEach(el => el.classList.remove('active-step'));
}
function runTourStep() {
    if (!_tourActive) return;
    const step = TOUR_STEPS[_tourStep % TOUR_STEPS.length];

    // Camera fly
    if (step.fullFit) {
        smoothFlyCamera(maxGraphHeight * 0.5, Math.max(18, maxGraphHeight * 1.35));
    } else {
        smoothFlyCamera(step.y, step.dist);
    }

    // Safe mesh selection — guard with try/catch and check interactableMeshes
    if (step.target && interactableMeshes.length > 0) {
        try {
            selectPartIn3D(step.target, { flyCamera: false, updateInspector: false });
        } catch(e) { /* ignore if mesh not found */ }
    }

    // Update tour step label
    const lbl = document.getElementById('tour-step-label');
    if (lbl) { lbl.textContent = step.label; lbl.style.opacity = '1'; }

    // Highlight matching lifecycle step
    const idx = _tourStep % 7;
    document.querySelectorAll('.lifecycle-step-item').forEach((el, i) => {
        el.classList.toggle('active-step', i === idx);
    });

    _tourStep++;
    const isLast = (_tourStep >= TOUR_STEPS.length);
    _tourTimer = setTimeout(() => {
        if (isLast) stopTour();
        else runTourStep();
    }, 3200);
}
window.toggleTour = function() {
    if (_tourActive) stopTour(); else startTour();
};

// ── Benchmark CSV Export ─────────────────────────────────────────────────────
window.exportBenchmarkCSV = function() {
    if (!activeResearchModule) return;
    const m = activeResearchModule;
    const rows = [['Model', 'Status', 'Accuracy%', 'F1', 'VRAM_MB', 'Throughput_t/s', 'Savings_x', 'Storage_Saved']];
    const keys = ['distilbert', 'roberta', 'gpt2', 'qwen'];
    keys.forEach(k => {
        const proj = (m.model_projections || {})[k] || {};
        const bm = proj.benchmark || {};
        rows.push([
            proj.model_name || k,
            bm.status || '—',
            bm.accuracy_pct != null ? bm.accuracy_pct.toFixed(2) : '—',
            bm.f1_score != null ? bm.f1_score.toFixed(4) : '—',
            bm.peak_vram_mb != null ? Math.round(bm.peak_vram_mb) : '—',
            bm.throughput_tokens_sec != null ? Math.round(bm.throughput_tokens_sec) : '—',
            proj.savings_ratio != null ? proj.savings_ratio.toFixed(2) : '—',
            proj.storage_saved_formatted || '—'
        ]);
    });
    const csv = rows.map(r => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${m.id}_benchmarks.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
};

// ── Color Legend Overlay ─────────────────────────────────────────────────────
function buildColorLegend() {
    const el = document.getElementById('color-legend');
    if (!el) return;
    const entries = [
        { color: '#2e4659', label: 'Dense Baseline', desc: 'Full-rank weight matrix (O(N²))' },
        { color: '#3acbb8', label: 'SVD Compressed', desc: 'Low-rank factorized U·V pair' },
        { color: '#b57ee0', label: 'MoE Router', desc: 'Gating network — token dispatch' },
        { color: '#9f68cb', label: 'MoE Expert', desc: 'Sparse expert sub-network' },
        { color: '#e6a848', label: 'MLA Latent', desc: 'KV-cache latent bottleneck' },
        { color: '#38bdf8', label: 'SliceGPT', desc: 'Orthogonally pruned channel slice' },
        { color: '#10b981', label: 'Embedding', desc: 'Token ID → continuous vector' },
        { color: '#f43f5e', label: 'LM Head', desc: 'Logit projection → vocabulary' },
        { color: '#60a5fa', label: 'Residual Add', desc: 'Skip-connection + operator' },
    ];
    el.innerHTML = `
        <div class="legend-title">🎨 Node Color Legend</div>
        ${entries.map(e => `
            <div class="legend-row">
                <span class="legend-swatch" style="background:${e.color}"></span>
                <div class="legend-info">
                    <span class="legend-label">${e.label}</span>
                    <span class="legend-desc">${e.desc}</span>
                </div>
            </div>
        `).join('')}
    `;
}

// ── Parameter Breakdown Table ─────────────────────────────────────────────────
window.renderParamBreakdownTable = function(moduleMeta, modelKey) {
    const el = document.getElementById('param-breakdown-table');
    if (!el || !moduleMeta) return;
    const proj = (moduleMeta.model_projections || {})[modelKey || currentModel] || {};
    const dModel = proj.hidden_size || 768;
    const dFfn = proj.intermediate_size || 3072;
    const comps = moduleMeta.components || [];

    let rows = '';
    if (comps.length > 0) {
        comps.forEach(c => {
            // Parse shape_transform to compute param count
            const shapeStr = c.shape_transform || '';
            const paramEst = (() => {
                const m = shapeStr.match(/\[(\d+)\s*[×x]\s*(\d+)\]/);
                if (m) return parseInt(m[1]) * parseInt(m[2]);
                if (shapeStr.includes('d_out') && shapeStr.includes('d_in')) return dModel * dModel;
                if (shapeStr.includes('d_in')) return 64 * dModel;
                if (shapeStr.includes('d_out')) return dModel * 64;
                return null;
            })();
            rows += `<tr>
                <td><span class="pbt-swatch" style="background:${c.color||'#2dd4bf'}"></span>${c.name}</td>
                <td class="pbt-mono">${shapeStr}</td>
                <td class="pbt-mono">${paramEst ? formatParamCount(paramEst) : '—'}</td>
                <td class="pbt-role">${c.role || '—'}</td>
            </tr>`;
        });
    } else {
        // generic rows from pipeline
        const pipeline = getModelPipelineComponents(moduleMeta, modelKey || currentModel);
        pipeline.flat().forEach(c => {
            rows += `<tr>
                <td><span class="pbt-swatch" style="background:${c.color||'#2dd4bf'}"></span>${c.name}</td>
                <td class="pbt-mono">${c.shape_transform || '—'}</td>
                <td class="pbt-mono">${c.params || '—'}</td>
                <td class="pbt-role">${c.role || '—'}</td>
            </tr>`;
        });
    }

    el.innerHTML = `
        <div class="pbt-title">📐 Component Parameter Breakdown</div>
        <table class="pbt-table">
            <thead><tr>
                <th>Component</th><th>Shape</th><th>Params</th><th>Role</th>
            </tr></thead>
            <tbody>${rows}</tbody>
        </table>
        <button class="formula-copy-btn" onclick="exportBenchmarkCSV()" style="margin-top:8px;">⬇ Export CSV</button>
    `;
};

// ── Module Comparison Pin ─────────────────────────────────────────────────────
let _pinnedModule = null;
window.pinCurrentModule = function() {
    if (!activeResearchModule) return;
    _pinnedModule = { ...activeResearchModule };
    const el = document.getElementById('pinned-module-name');
    if (el) el.textContent = _pinnedModule.name;
    const section = document.getElementById('comparison-pin-section');
    if (section) section.style.display = 'block';
    renderComparisonPin();
};
function renderComparisonPin() {
    if (!_pinnedModule || !activeResearchModule) return;
    const el = document.getElementById('comparison-pin-content');
    if (!el) return;
    const A = _pinnedModule, B = activeResearchModule;
    const modelKeys = ['distilbert', 'roberta', 'gpt2', 'qwen'];
    let rows = modelKeys.map(k => {
        const pA = ((A.model_projections || {})[k] || {});
        const pB = ((B.model_projections || {})[k] || {});
        const ratioA = pA.savings_ratio || '—';
        const ratioB = pB.savings_ratio || '—';
        const winner = (typeof ratioA === 'number' && typeof ratioB === 'number') ?
            (ratioA > ratioB ? 'A' : ratioB > ratioA ? 'B' : '=') : '?';
        return `<tr>
            <td>${k.toUpperCase()}</td>
            <td class="${winner==='A'?'cp-win':''}">${typeof ratioA==='number'?ratioA.toFixed(1)+'x':ratioA}</td>
            <td class="${winner==='B'?'cp-win':''}">${typeof ratioB==='number'?ratioB.toFixed(1)+'x':ratioB}</td>
        </tr>`;
    }).join('');

    el.innerHTML = `
        <table class="pbt-table">
            <thead><tr><th>Model</th><th title="${A.name}">${(A.name||'').slice(0,14)}…</th><th title="${B.name}">${(B.name||'').slice(0,14)}…</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>
        <button class="formula-copy-btn" style="margin-top:6px;" onclick="document.getElementById('comparison-pin-section').style.display='none'; _pinnedModule=null;">✕ Clear Pin</button>
    `;
}

// ── Hook renderParamBreakdown & renderComparisonPin into applyResearchModule ──
const _origApply = window._origApplyResearchModule;
// We patch at call site below

// ── Color Legend auto-build ───────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    buildColorLegend();

    const tourBtn = document.getElementById('btn-auto-tour');
    if (tourBtn) tourBtn.addEventListener('click', () => window.toggleTour());

    const legendToggle = document.getElementById('btn-legend-toggle');
    const legendPanel = document.getElementById('color-legend');
    if (legendToggle && legendPanel) {
        legendToggle.addEventListener('click', () => {
            const visible = legendPanel.style.display !== 'none';
            legendPanel.style.display = visible ? 'none' : 'block';
            legendToggle.textContent = visible ? '🎨 Legend' : '✕ Legend';
        });
    }

    const pinBtn = document.getElementById('btn-pin-module');
    if (pinBtn) pinBtn.addEventListener('click', () => window.pinCurrentModule());
});


async function bootstrap() {
    setupTreeFilter();
    initModuleCatalog();

    // 1. Load Module Components & Catalog (241 Modules)
    if (window.__EMBEDDED_MODULES_COMPONENTS__ && window.__EMBEDDED_MODULES_COMPONENTS__.modules) {
        modulesCatalog = window.__EMBEDDED_MODULES_COMPONENTS__.modules;
        modulesCatalog.forEach(m => { modulesComponentsMap[m.id] = m; });
        renderModuleCards('all', '');
        populateQuickSelectDropdown();
    } else {
        try {
            const compRes = await fetch('modules_components.json');
            const compJson = await compRes.json();
            modulesCatalog = compJson.modules || [];
            modulesCatalog.forEach(m => { modulesComponentsMap[m.id] = m; });
            renderModuleCards('all', '');
            populateQuickSelectDropdown();
        } catch (err) {
            console.warn('Could not load modules_components.json:', err);
            if (window.__EMBEDDED_MODULES_CATALOG__ && window.__EMBEDDED_MODULES_CATALOG__.modules) {
                modulesCatalog = window.__EMBEDDED_MODULES_CATALOG__.modules;
                modulesCatalog.forEach(m => { modulesComponentsMap[m.id] = m; });
                renderModuleCards('all', '');
                populateQuickSelectDropdown();
            }
        }
    }

    // 2. Load 3D Spatial Geometry
    if (window.__EMBEDDED_SPATIAL_DATA__) {
        spatialData = window.__EMBEDDED_SPATIAL_DATA__;
        loadModelAndModule('distilbert', 'svd');
        if (modulesCatalog.length > 0) {
            const initialMod = modulesCatalog.find(m => m.id === 'SOAFusedAttention') || modulesCatalog[0];
            if (initialMod) applyResearchModule(initialMod);
        }
        render(0);
        return;
    }

    try {
        const res = await fetch('spatial-data.json');
        spatialData = await res.json();

        // Parse hash permalink
        let initialModelKey = 'distilbert';
        let initialModuleId = 'SOAFusedAttention';
        try {
            const hash = window.location.hash;
            const modMatch = hash.match(/module=([^&]+)/);
            const mdlMatch = hash.match(/model=([^&]+)/);
            if (modMatch) initialModuleId = decodeURIComponent(modMatch[1]);
            if (mdlMatch) initialModelKey = decodeURIComponent(mdlMatch[1]);
        } catch(e) {}

        loadModelAndModule(initialModelKey, 'svd');
        if (modulesCatalog.length > 0) {
            const initialMod = modulesCatalog.find(m => m.id === initialModuleId) || modulesCatalog.find(m => m.id === 'SOAFusedAttention') || modulesCatalog[0];
            if (initialMod) applyResearchModule(initialMod);
        }
        render(0);
    } catch (err) {
        console.error('Failed to load spatial-data.json:', err);
    }
}

window.__CAMERA__ = cameraBaseline;
window.__CONTROLS__ = controls;
window.__SCENE__ = scene;
window.__GRANULAR__ = granularGroup;
window.__MACRO__ = macroGroup;
window.__MATERIALS__ = materials;

bootstrap();



// ---------------------------------------------------------------------------
// 22. Polymetis / Attention Lab UI Wiring
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
    // 2D/3D View Toggle
    const btn3d = document.getElementById('btn-view-3d');
    const btn2d = document.getElementById('btn-view-2d');
    if (btn3d && btn2d) {
        btn3d.addEventListener('click', () => {
            btn3d.classList.add('active');
            btn2d.classList.remove('active');
            btn3d.setAttribute('aria-selected', 'true');
            btn2d.setAttribute('aria-selected', 'false');
            
            // Switch to 3D perspective
            controls.enableRotate = true;
            controls.minAzimuthAngle = -Infinity;
            controls.maxAzimuthAngle = Infinity;
            controls.minPolarAngle = 0;
            controls.maxPolarAngle = Math.PI;
            controls.enableZoom = true;
            smoothFlyCamera(10, 25, 0); // 3D View // Restore typical 3D angle
        });
        
        btn2d.addEventListener('click', () => {
            btn2d.classList.add('active');
            btn3d.classList.remove('active');
            btn2d.setAttribute('aria-selected', 'true');
            btn3d.setAttribute('aria-selected', 'false');
            
            // Switch to 2D Orthographic-like locked perspective
            controls.enableRotate = false;
            // Align camera flat against the network
            smoothFlyCamera(15, 40, 0); // 2D Front View
            // Lock horizontal/vertical angles for 2D mode
            controls.minAzimuthAngle = 0;
            controls.maxAzimuthAngle = 0;
            controls.minPolarAngle = Math.PI / 2;
            controls.maxPolarAngle = Math.PI / 2;
        });
    }

    // Mechanism Navigation Buttons
    const mechButtons = document.querySelectorAll('.mechanism-button');
    mechButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            mechButtons.forEach(b => {
                b.classList.remove('active');
                b.setAttribute('aria-selected', 'false');
            });
            btn.classList.add('active');
            btn.setAttribute('aria-selected', 'true');
            
            const mModel = btn.dataset.model;
            const mModule = btn.dataset.module || currentCompression;
            if (mModel) {
                if (window.switchModelFromMatrix) {
                    window.switchModelFromMatrix(mModel);
                } else {
                    currentModel = mModel;
                    currentCompression = mModule;
                    loadModelAndModule(currentModel, currentCompression);
                }
            }
        });
    });

    // Control Dock (Play/Pause, Speed, Scrubber)
    const selSpeed = document.getElementById('select-speed');
    const scrubber = document.getElementById('timeline-scrubber');
    const timeCode = document.getElementById('timeline-step');
    
    // Play/Pause - sync with existing playBtn listener
    const playBtnDock = document.getElementById('btn-play-pause');
    if (playBtnDock) {
        // Remove old inline text, refresh icon on click
        const refreshPlayIcon = () => {
            playBtnDock.innerHTML = isPlaying
                ? '<span class="btn-icon">⏸</span> Pause Flow'
                : '<span class="btn-icon">▶</span> Play Flow';
            playBtnDock.classList.toggle('paused', !isPlaying);
        };
        // Override default click with proper state sync
        playBtnDock.addEventListener('click', () => {
            isPlaying = !isPlaying;
            refreshPlayIcon();
        });
        refreshPlayIcon();
    }
    
    if (selSpeed) {
        selSpeed.addEventListener('change', (e) => {
            simulationSpeed = parseFloat(e.target.value);
            // visual feedback
            selSpeed.style.color = '#53ecd9';
            setTimeout(() => { selSpeed.style.color = ''; }, 500);
        });
    }
    
    if (scrubber && timeCode) {
        // Build lifecycle stage labels dynamically
        const getStageTargets = () => {
            if (!activeResearchModule) return ['q_lin', 'q_lin', 'q_lin', 'out_lin', 'lin1', 'lin2', 'lin2'];
            const model = currentModel;
            if (model === 'qwen') return ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'];
            if (model === 'gpt2') return ['c_attn', 'c_attn', 'c_attn', 'c_proj', 'c_fc', 'c_fc', 'c_proj'];
            return ['q_lin', 'k_lin', 'v_lin', 'out_lin', 'lin1', 'lin2', 'lin2'];
        };
        const stageLabels = ['01 Input', '02 Norm', '03 Attention', '04 +Residual', '05 FFN', '06 +Residual', '07 Output'];
        
        scrubber.addEventListener('input', (e) => {
            const step = parseInt(e.target.value, 10);
            timeCode.innerText = stageLabels[step] || `Step ${step + 1}/7`;
            const stages = getStageTargets();
            const target = stages[step];
            if (target && window.selectPartIn3D) {
                window.selectPartIn3D(target, { flyCamera: true, updateInspector: false });
            }
            // Highlight the matching lifecycle step item
            document.querySelectorAll('.lifecycle-step-item').forEach((el, i) => {
                el.classList.toggle('active-step', i === step);
            });
        });
    }

    // ---- Keyboard shortcuts ----
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT') return;
        switch(e.code) {
            case 'Space':
                e.preventDefault();
                isPlaying = !isPlaying;
                const pb = document.getElementById('btn-play-pause');
                if (pb) pb.innerHTML = isPlaying ? '<span class="btn-icon">⏸</span> Pause Flow' : '<span class="btn-icon">▶</span> Play Flow';
                break;
            case 'KeyR':
                smoothFlyCamera(maxGraphHeight * 0.5, Math.max(18.0, maxGraphHeight * 1.35));
                break;
            case 'Digit1': case 'Numpad1': selectPartIn3D('q_lin', { flyCamera: true }); break;
            case 'Digit2': case 'Numpad2': selectPartIn3D('k_lin', { flyCamera: true }); break;
            case 'Digit3': case 'Numpad3': selectPartIn3D('v_lin', { flyCamera: true }); break;
            case 'Digit4': case 'Numpad4': selectPartIn3D('out_lin', { flyCamera: true }); break;
            case 'Digit5': case 'Numpad5': selectPartIn3D('lin1', { flyCamera: true }); break;
            case 'Digit6': case 'Numpad6': selectPartIn3D('lin2', { flyCamera: true }); break;
            case 'KeyM':
                const modal = document.getElementById('module-modal');
                if (modal) modal.classList.toggle('hidden');
                break;
            case 'Escape':
                const mod = document.getElementById('module-modal');
                if (mod) mod.classList.add('hidden');
                break;
        }
    });
    
    // ---- Keyboard shortcut tooltip ----
    const shortcutHint = document.createElement('div');
    shortcutHint.id = 'shortcut-hint';
    shortcutHint.innerHTML = `
        <div class="shortcut-row"><kbd>Space</kbd> Play/Pause</div>
        <div class="shortcut-row"><kbd>R</kbd> Reset Camera</div>
        <div class="shortcut-row"><kbd>M</kbd> Module Library</div>
        <div class="shortcut-row"><kbd>1-6</kbd> Jump to Layer</div>
        <div class="shortcut-row"><kbd>Esc</kbd> Close Modal</div>
    `;
    document.body.appendChild(shortcutHint);
    
    let shortcutTimer;
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT') return;
        shortcutHint.classList.add('visible');
        clearTimeout(shortcutTimer);
        shortcutTimer = setTimeout(() => shortcutHint.classList.remove('visible'), 2000);
    });
});
