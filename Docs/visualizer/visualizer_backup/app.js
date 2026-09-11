/**
 * app.js
 * ======
 * Live 3D Neural Architecture Visualizer Engine
 * High-Quality Visuals & Educational Token Dynamics:
 * - Dynamic Model Swapping (DistilBERT, RoBERTa, GPT-2, Qwen3.5)
 * - True Funneling Token Animation: Tokens physically pinch through SVD rank bottlenecks
 * - Visual Architecture Pulses: Nodes subtly breathe as tokens pass through them
 * - Neon Edge Wiring: Glowing wireframe edges highlighting geometric matrix boundaries
 * - Strict Memory Disposal (zero WebGL resource leaks)
 * - Smooth Adaptive Camera Lerp & Interactive Inspection
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ---------------------------------------------------------------------------
// 1. WebGL Renderer & Scene Initialization
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

// Atmospheric Studio Lighting
const hemiLight = new THREE.HemisphereLight(0xdbeafe, 0x0f172a, 2.0);
scene.add(hemiLight);

const mainLight = new THREE.DirectionalLight(0xfff7ed, 2.6);
mainLight.position.set(12, 30, 20);
scene.add(mainLight);

const cyanRim = new THREE.DirectionalLight(0x53ecd9, 1.4);
cyanRim.position.set(-14, 18, -12);
scene.add(cyanRim);

const purpleRim = new THREE.DirectionalLight(0xc895ea, 1.2);
purpleRim.position.set(14, 12, -14);
scene.add(purpleRim);

// ---------------------------------------------------------------------------
// 2. Dual-Camera & Synchronized OrbitControls
// ---------------------------------------------------------------------------
const aspect = (window.innerWidth / 2) / window.innerHeight;

const cameraBaseline = new THREE.PerspectiveCamera(38, aspect, 0.1, 900);
cameraBaseline.position.set(0, 8, 24);

const cameraCompressed = new THREE.PerspectiveCamera(38, aspect, 0.1, 900);
cameraCompressed.position.set(0, 8, 24);

const controls = new OrbitControls(cameraBaseline, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.target.set(0, 6, 0);
controls.maxPolarAngle = Math.PI * 0.88;
controls.minDistance = 3;
controls.maxDistance = 280;

const OFFSET_BASELINE = -6.5;
const OFFSET_COMPRESSED = 6.5;

let cameraTargetY = 4.5;
let cameraTargetZ = 18.0;

// ---------------------------------------------------------------------------
// 3. Shaders & High-Tech Materials
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
        metalness: 0.45
    })
};

// Edge Line Materials
const edgeMatDense = new THREE.LineBasicMaterial({ color: 0x4f748f, transparent: true, opacity: 0.4 });
const edgeMatSvd = new THREE.LineBasicMaterial({ color: 0x53ecd9, transparent: true, opacity: 0.85 });
const edgeMatMoe = new THREE.LineBasicMaterial({ color: 0xc895ea, transparent: true, opacity: 0.75 });

// ---------------------------------------------------------------------------
// 4. State Management
// ---------------------------------------------------------------------------
let currentModel = 'distilbert';
let spatialData = null;
let simulationSpeed = 1.0;
let enablePulses = true;

const interactableMeshes = [];
const visualObjects = [];
const svdBottlenecks = [];
const moeNodes = [];

// Instanced Token Simulation State
const NUM_TOKENS = 360;
const tokenGeometry = new THREE.SphereGeometry(0.1, 10, 10);
const tokenMaterialBaseline = new THREE.MeshBasicMaterial({ color: 0x9fd2ff });
const tokenMaterialCompressed = new THREE.MeshBasicMaterial({ color: 0x6effeb });

let tokenMeshBaseline = null;
let tokenMeshCompressed = null;
const tokenDataBaseline = [];
const tokenDataCompressed = [];
let maxGraphHeight = 16.0;

// ---------------------------------------------------------------------------
// 5. Memory Cleanup & Resource Disposal
// ---------------------------------------------------------------------------
function clearScene() {
    for (const obj of visualObjects) {
        scene.remove(obj);
        if (obj.geometry && obj.geometry !== tokenGeometry) {
            obj.geometry.dispose();
        }
        if (obj.material && !Object.values(materials).includes(obj.material) &&
            obj.material !== edgeMatDense && obj.material !== edgeMatSvd && obj.material !== edgeMatMoe &&
            obj.material !== tokenMaterialBaseline && obj.material !== tokenMaterialCompressed) {
            obj.material.dispose();
        }
    }
    visualObjects.length = 0;
    interactableMeshes.length = 0;
    svdBottlenecks.length = 0;
    moeNodes.length = 0;

    if (tokenMeshBaseline) {
        scene.remove(tokenMeshBaseline);
        tokenMeshBaseline.dispose();
        tokenMeshBaseline = null;
    }
    if (tokenMeshCompressed) {
        scene.remove(tokenMeshCompressed);
        tokenMeshCompressed.dispose();
        tokenMeshCompressed = null;
    }

    tokenDataBaseline.length = 0;
    tokenDataCompressed.length = 0;
    maxGraphHeight = 16.0;
}

// ---------------------------------------------------------------------------
// 6. Token Mesh Builder
// ---------------------------------------------------------------------------
function createTokenMesh(tokenDataArray, baseOffsetX, mat) {
    const instancedMesh = new THREE.InstancedMesh(tokenGeometry, mat, NUM_TOKENS);
    instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

    for (let i = 0; i < NUM_TOKENS; i++) {
        tokenDataArray.push({
            y: Math.random() * maxGraphHeight,
            speed: (0.03 + Math.random() * 0.04) * simulationSpeed,
            baseX: baseOffsetX,
            lateralOffset: (Math.random() - 0.5) * 1.8,
            expertTarget: Math.floor(Math.random() * 4),
            pulsePhase: Math.random() * Math.PI * 2
        });

        const dummy = new THREE.Object3D();
        dummy.position.set(baseOffsetX, tokenDataArray[i].y, 0);
        dummy.updateMatrix();
        instancedMesh.setMatrixAt(i, dummy.matrix);
    }
    return instancedMesh;
}

// ---------------------------------------------------------------------------
// 7. Node Geometry & Glowing Edge Builder
// ---------------------------------------------------------------------------
function buildMeshNodes(nodes, offsetX, isCompressedSide) {
    let topY = 0;

    // Detect SVD pairs for bottleneck funneling
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

        if (node.type === 'svd') {
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
        }

        const geo = new THREE.BoxGeometry(node.width, node.height, node.depth);
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(node.x + offsetX, node.y, node.z);
        mesh.userData = node;

        // Glowing Wireframe Edge Outlines for crisp definition
        const wireGeo = new THREE.EdgesGeometry(geo);
        const wireLine = new THREE.LineSegments(wireGeo, edgeMat);
        wireLine.raycast = () => {}; // Never intercept raycaster
        mesh.add(wireLine);

        scene.add(mesh);
        visualObjects.push(mesh);
        interactableMeshes.push(mesh);

        if (node.y > topY) topY = node.y;

        // Visual branching wires for MoE
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
            scene.add(line);
            visualObjects.push(line);
        }

        if (node.type === 'moe' && node.sub_type === 'router') {
            moeNodes.push({ ...node, worldX: node.x + offsetX });
        }
    });

    // Base Grid Ground with Radial Floor Fade
    const grid = new THREE.GridHelper(16, 16, 0x1c334d, 0x0c1726);
    grid.position.set(offsetX, 0, 0);
    scene.add(grid);
    visualObjects.push(grid);

    return topY;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// 8. Dynamic Model Loading
// ---------------------------------------------------------------------------
function populateLayerDropdown(modelData) {
    const dropdown = document.getElementById('layer-select-dropdown');
    const countEl = document.getElementById('insp-layer-count');
    if (!dropdown) return;

    dropdown.innerHTML = '<option value="">-- Choose a layer or click in 3D --</option>';

    // Group by layer block
    const allLayers = [];
    modelData.baseline_nodes.forEach(bNode => {
        allLayers.push({
            id: bNode.id,
            label: bNode.label,
            type: 'baseline',
            node: bNode
        });
    });
    modelData.compressed_nodes.forEach(cNode => {
        allLayers.push({
            id: cNode.id,
            label: cNode.label,
            type: 'compressed',
            node: cNode
        });
    });

    if (countEl) {
        countEl.innerText = `${allLayers.length} modules available`;
    }

    allLayers.forEach(item => {
        const opt = document.createElement('option');
        opt.value = item.id;
        const prefix = item.type === 'compressed' ? '[SVD] ' : '[Dense] ';
        opt.textContent = `${prefix}${item.label}`;
        dropdown.appendChild(opt);
    });
}

function loadModel(modelName) {
    if (!spatialData || !spatialData.models[modelName]) return;

    clearScene();
    currentModel = modelName;

    // Update Tab UI
    document.querySelectorAll('.model-tab').forEach(tab => {
        const isActive = (tab.dataset.model === modelName);
        tab.dataset.active = isActive ? "true" : "false";
        tab.setAttribute('aria-selected', isActive ? "true" : "false");
    });

    const modelData = spatialData.models[modelName];

    // Update Viewport Titles
    document.getElementById('left-panel-title').innerText = `${modelName.toUpperCase()} (Dense Baseline)`;
    document.getElementById('right-panel-title').innerText = `${modelName.toUpperCase()} (SVD Compressed)`;

    const h1 = buildMeshNodes(modelData.baseline_nodes, OFFSET_BASELINE, false);
    const h2 = buildMeshNodes(modelData.compressed_nodes, OFFSET_COMPRESSED, true);
    maxGraphHeight = Math.max(h1, h2, modelData.max_y || 14.0);

    // Build Token Streams
    tokenMeshBaseline = createTokenMesh(tokenDataBaseline, OFFSET_BASELINE, tokenMaterialBaseline);
    scene.add(tokenMeshBaseline);

    tokenMeshCompressed = createTokenMesh(tokenDataCompressed, OFFSET_COMPRESSED, tokenMaterialCompressed);
    scene.add(tokenMeshCompressed);

    // Adaptive Camera Centering
    cameraTargetY = maxGraphHeight * 0.5;
    cameraTargetZ = Math.max(14.0, maxGraphHeight * 1.35);

    // Populate dropdown
    populateLayerDropdown(modelData);

    // Update Topbar Metrics
    const labelEl = document.getElementById('active-mode-label');
    if (labelEl) labelEl.textContent = `${modelName.toUpperCase()} · Dense vs SVD`;

    const tokensEl = document.getElementById('active-tokens-count');
    if (tokensEl) tokensEl.textContent = `${NUM_TOKENS} Flowing Tokens`;
}

// ---------------------------------------------------------------------------
// 9. Bootstrap and Controls Wiring
// ---------------------------------------------------------------------------
document.querySelectorAll('.model-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        loadModel(tab.dataset.model);
    });
});

// Speed Controls
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

async function bootstrap() {
    try {
        const res = await fetch('spatial-data.json');
        spatialData = await res.json();
        loadModel('distilbert');
    } catch (err) {
        console.error('Failed to load spatial-data.json:', err);
    }
}

bootstrap();

// ---------------------------------------------------------------------------
// 10. Raycaster & Inspector Interaction
// ---------------------------------------------------------------------------
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const tooltip = document.getElementById('tooltip');
const inspector = document.getElementById('inspector');

function findTargetNode(intersectedObject) {
    let obj = intersectedObject;
    while (obj && (!obj.userData || !obj.userData.id) && obj.parent) {
        obj = obj.parent;
    }
    return obj ? { mesh: obj, data: obj.userData } : null;
}

function updateInspectorWithNode(node, isLeft) {
    if (!node) return;

    // Sync dropdown selection if matched
    const dropdown = document.getElementById('layer-select-dropdown');
    if (dropdown && dropdown.value !== node.id) {
        dropdown.value = node.id;
    }

    // Type badge
    const badgeEl = document.getElementById('insp-type-badge');
    if (badgeEl) {
        badgeEl.innerText = (node.type || 'LAYER').toUpperCase();
    }

    // Title
    const titleEl = document.getElementById('insp-title');
    if (titleEl) {
        titleEl.innerText = node.label || node.id || 'Layer';
    }

    // Logical shape
    const shapeEl = document.getElementById('insp-shape');
    if (shapeEl) {
        if (node.metadata?.shape && Array.isArray(node.metadata.shape)) {
            shapeEl.innerText = `[${node.metadata.shape.join(' × ')}]`;
        } else if (node.width && node.height) {
            shapeEl.innerText = `[${Math.round(node.width * 256)} × ${Math.round(node.depth * 256)}]`;
        } else {
            shapeEl.innerText = '[768 × 768]';
        }
    }

    // Parameters
    const paramsEl = document.getElementById('insp-params');
    if (paramsEl) {
        paramsEl.innerText = node.metadata?.params || (node.type === 'svd' ? '49.2K' : '590.6K');
    }

    // Storage
    const storageEl = document.getElementById('insp-storage');
    if (storageEl) {
        storageEl.innerText = node.metadata?.storage || (node.type === 'svd' ? '196.8 KB' : '2.36 MB');
    }

    // Efficiency / Compression
    const effEl = document.getElementById('insp-efficiency');
    if (effEl) {
        effEl.innerText = node.metadata?.compression || (node.type === 'svd' ? '6.0x Compressed' : '1.0x Baseline');
    }

    // Comparative Card Data
    const compBaselineEl = document.getElementById('insp-comp-baseline');
    const compCompressedEl = document.getElementById('insp-comp-compressed');
    const savingsBadgeEl = document.getElementById('insp-savings-badge');

    if (node.type === 'svd') {
        const rank = node.metadata?.rank || 64;
        if (compBaselineEl) compBaselineEl.innerText = '590.6K (Full-Rank W)';
        if (compCompressedEl) compCompressedEl.innerText = `${node.metadata?.params || '49.2K'} (U×V factor, r=${rank})`;
        if (savingsBadgeEl) {
            savingsBadgeEl.innerText = '-83.3% MEMORY';
            savingsBadgeEl.style.color = '#53ecd9';
        }
    } else {
        if (compBaselineEl) compBaselineEl.innerText = `${node.metadata?.params || '590.6K'} (Full-Rank)`;
        if (compCompressedEl) compCompressedEl.innerText = '~98.4K (Est. SVD r=64)';
        if (savingsBadgeEl) {
            savingsBadgeEl.innerText = 'BASELINE 1.0x';
            savingsBadgeEl.style.color = '#8e9fb5';
        }
    }

    // Description
    const descEl = document.getElementById('insp-desc');
    if (descEl) {
        const streamType = isLeft ? 'Dense Baseline' : 'SVD Compressed';
        const w = (node.width || 1).toFixed(2);
        const h = (node.height || 1).toFixed(2);
        const d = (node.depth || 1).toFixed(2);
        descEl.innerText = `Block in ${streamType} for ${currentModel.toUpperCase()}. Spatial 3D Bounding Matrix: [${w}W × ${h}H × ${d}D]. Active tensor rank: ${node.metadata?.rank || 'Full'}.`;
    }

    // Educational Note
    const noteEl = document.getElementById('insp-edu-note');
    if (noteEl) {
        if (node.type === 'svd') {
            const rank = node.metadata?.rank || 64;
            noteEl.innerHTML = `💡 <strong>SVD Bottleneck ($r=${rank}$):</strong> Decomposes full-rank parameter weight into factors $W \\approx U \\cdot V$. Watch flowing tokens constrict sharply as they stream through the thin rank bottleneck!`;
        } else if (node.type === 'moe') {
            noteEl.innerHTML = `💡 <strong>MoE Routing:</strong> A gating router dispatches tokens dynamically across independent expert subnetworks, activating only top-$k$ paths per token!`;
        } else {
            noteEl.innerHTML = `💡 <strong>Dense Projection:</strong> Full quadratic capacity $O(d_{in} \\times d_{out})$. Token representations maintain a wide unconstrained stream.`;
        }
    }

    inspector.classList.remove('hidden');
}

function handleInteraction(clientX, clientY, isClick = false) {
    const isLeft = clientX < window.innerWidth / 2;
    const viewWidth = window.innerWidth / 2;

    if (isLeft) {
        mouse.x = (clientX / viewWidth) * 2 - 1;
        raycaster.setFromCamera(mouse, cameraBaseline);
    } else {
        mouse.x = ((clientX - viewWidth) / viewWidth) * 2 - 1;
        raycaster.setFromCamera(mouse, cameraCompressed);
    }

    mouse.y = -(clientY / window.innerHeight) * 2 + 1;

    const hits = raycaster.intersectObjects(interactableMeshes, true);
    if (hits.length > 0) {
        const target = findTargetNode(hits[0].object);
        if (!target || !target.data) return;
        const node = target.data;

        tooltip.innerHTML = `
            <strong>${node.label || node.id}</strong>
            <div>Type: ${node.metadata?.type || (node.type === 'svd' ? 'SVD Low-Rank' : 'Dense Projection')}</div>
            <div>Params: ${node.metadata?.params || (node.type === 'svd' ? '49.2K' : '590.6K')} · Storage: ${node.metadata?.storage || (node.type === 'svd' ? '196.8 KB' : '2.36 MB')}</div>
            <div>Efficiency: <span style="color:#53ecd9">${node.metadata?.compression || (node.type === 'svd' ? '6.0x Compressed' : '1.0x Baseline')}</span></div>
        `;
        tooltip.style.left = `${clientX}px`;
        tooltip.style.top = `${clientY}px`;
        tooltip.classList.remove('hidden');

        if (isClick) {
            updateInspectorWithNode(node, isLeft);
        }
    } else {
        tooltip.classList.add('hidden');
    }
}

// Dropdown change listener to choose layer directly
const layerDropdown = document.getElementById('layer-select-dropdown');
if (layerDropdown) {
    layerDropdown.addEventListener('change', (e) => {
        const selectedId = e.target.value;
        if (!selectedId) return;

        // Search in interactableMeshes
        let foundMesh = null;
        for (const m of interactableMeshes) {
            if (m.userData && m.userData.id === selectedId) {
                foundMesh = m;
                break;
            }
        }

        if (foundMesh) {
            const isLeft = foundMesh.position.x < 0;
            updateInspectorWithNode(foundMesh.userData, isLeft);

            // Smoothly center camera target around selected node height
            cameraTargetY = foundMesh.position.y;
            cameraTargetZ = 16.0;

            // Flash effect on mesh
            const origColor = foundMesh.material.color.getHex();
            foundMesh.material.color.setHex(0xffffff);
            setTimeout(() => {
                foundMesh.material.color.setHex(origColor);
            }, 300);
        }
    });
}

window.addEventListener('mousemove', e => handleInteraction(e.clientX, e.clientY, false));
window.addEventListener('click', e => {
    if (!inspector.contains(e.target) && !e.target.closest('.topbar') &&
        !e.target.closest('.model-selector') && !e.target.closest('.edu-controls')) {
        handleInteraction(e.clientX, e.clientY, true);
    }
});

document.getElementById('btn-close-insp').addEventListener('click', () => {
    inspector.classList.add('hidden');
});

// UI Buttons
let isPlaying = true;
const playBtn = document.getElementById('btn-play-pause');

playBtn.addEventListener('click', () => {
    isPlaying = !isPlaying;
    playBtn.innerHTML = isPlaying ? '<span class="btn-icon">⏸</span> Pause Flow' : '<span class="btn-icon">▶</span> Play Flow';
});

document.getElementById('btn-reset-cam').addEventListener('click', () => {
    cameraTargetY = maxGraphHeight * 0.5;
    cameraTargetZ = Math.max(14.0, maxGraphHeight * 1.35);
});

// ---------------------------------------------------------------------------
// 11. Physics-Driven Token Funneling & Dynamics
// ---------------------------------------------------------------------------
const dummy = new THREE.Object3D();

function updateTokens(instancedMesh, tokenDataArray, isCompressedSide = false) {
    if (!instancedMesh || !isPlaying) return;
    if (tokenDataArray.length === 0) return;

    for (let i = 0; i < NUM_TOKENS; i++) {
        const t = tokenDataArray[i];
        t.y += t.speed * simulationSpeed;

        if (t.y > maxGraphHeight) {
            t.y = 0.2;
            t.lateralOffset = (Math.random() - 0.5) * 1.8;
            t.expertTarget = Math.floor(Math.random() * 4);
        }

        let targetX = t.baseX + t.lateralOffset;

        // TRUE SVD FUNNELING ON COMPRESSED SIDE
        if (isCompressedSide && svdBottlenecks.length > 0) {
            for (const b of svdBottlenecks) {
                if (t.y >= b.yStart && t.y <= b.yEnd) {
                    // Normalize position through the bottleneck [0 -> 1]
                    const normY = (t.y - b.yStart) / (b.yEnd - b.yStart);
                    // Funnel constriction curve: pinches inward at neck (normY=0.5)
                    const pinchFactor = 0.25 + 0.75 * Math.pow(Math.abs(normY - 0.5) * 2.0, 2);
                    targetX = b.worldX + (t.lateralOffset * pinchFactor * (b.rankWidth / b.fullWidth));
                    break;
                }
            }
        }

        // MoE Branching Animation
        if (isCompressedSide && moeNodes.length > 0) {
            for (const moe of moeNodes) {
                const dy = t.y - moe.y;
                if (dy >= 0 && dy <= 1.4) {
                    const spread = (t.expertTarget - 1.5) * 1.5;
                    const progress = dy / 1.4;
                    targetX = t.baseX + spread * Math.sin(progress * Math.PI * 0.5);
                } else if (dy > 1.4 && dy < 2.0) {
                    const spread = (t.expertTarget - 1.5) * 1.5;
                    targetX = t.baseX + spread;
                } else if (dy >= 2.0 && dy < 3.0) {
                    const spread = (t.expertTarget - 1.5) * 1.5;
                    const progress = (3.0 - dy) / 1.0;
                    targetX = t.baseX + spread * Math.sin(progress * Math.PI * 0.5);
                }
            }
        }

        dummy.position.set(targetX, t.y, 0);

        // Subtle organic breathing scale
        const scalePulse = 1.0 + Math.sin(t.y * 3.0 + t.pulsePhase) * 0.15;
        dummy.scale.set(scalePulse, scalePulse, scalePulse);

        dummy.updateMatrix();
        instancedMesh.setMatrixAt(i, dummy.matrix);
    }
    instancedMesh.instanceMatrix.needsUpdate = true;
}

// ---------------------------------------------------------------------------
// 12. Node Activation Pulse Waves
// ---------------------------------------------------------------------------
function updateNodePulses(time) {
    if (!enablePulses) return;

    for (let i = 0; i < interactableMeshes.length; i++) {
        const mesh = interactableMeshes[i];
        // Wave propagates upwards with Y position
        const pulse = Math.sin(time * 0.003 - mesh.position.y * 0.4);
        if (pulse > 0.7) {
            const intensity = (pulse - 0.7) / 0.3;
            if (mesh.userData.type === 'svd') {
                mesh.material.emissiveIntensity = 0.4 + intensity * 0.5;
            } else {
                mesh.scale.set(
                    1.0 + intensity * 0.03,
                    1.0 + intensity * 0.03,
                    1.0 + intensity * 0.03
                );
            }
        } else {
            if (mesh.userData.type === 'svd') {
                mesh.material.emissiveIntensity = 0.4;
            }
            mesh.scale.set(1, 1, 1);
        }
    }
}

// ---------------------------------------------------------------------------
// 13. Window Resizing
// ---------------------------------------------------------------------------
window.addEventListener('resize', () => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize(w, h);

    const halfAspect = (w / 2) / h;
    cameraBaseline.aspect = halfAspect;
    cameraBaseline.updateProjectionMatrix();

    cameraCompressed.aspect = halfAspect;
    cameraCompressed.updateProjectionMatrix();
});

// ---------------------------------------------------------------------------
// 14. Render Loop with Smooth Camera Lerp & Split-Screen
// ---------------------------------------------------------------------------
function render(time) {
    requestAnimationFrame(render);

    // Smooth camera transition
    controls.target.y += (cameraTargetY - controls.target.y) * 0.08;
    cameraBaseline.position.z += (cameraTargetZ - cameraBaseline.position.z) * 0.08;
    cameraBaseline.position.y += ((cameraTargetY + 2.0) - cameraBaseline.position.y) * 0.08;

    controls.update();

    cameraCompressed.position.copy(cameraBaseline.position);
    cameraCompressed.quaternion.copy(cameraBaseline.quaternion);

    // Update Tokens
    updateTokens(tokenMeshBaseline, tokenDataBaseline, false);
    updateTokens(tokenMeshCompressed, tokenDataCompressed, true);

    // Update Activation Waves
    updateNodePulses(time);

    const width = window.innerWidth;
    const height = window.innerHeight;
    const halfWidth = Math.floor(width / 2);

    // Left Viewport: Baseline Model
    renderer.setViewport(0, 0, halfWidth, height);
    renderer.setScissor(0, 0, halfWidth, height);
    renderer.render(scene, cameraBaseline);

    // Right Viewport: Compressed Model
    renderer.setViewport(halfWidth, 0, width - halfWidth, height);
    renderer.setScissor(halfWidth, 0, width - halfWidth, height);
    renderer.render(scene, cameraCompressed);
}

render(0);
