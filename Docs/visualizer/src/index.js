/**
 * index.js
 * ========
 * Primary Application Entrypoint for 3D Neural Architecture Visualizer.
 * Coordinates rendering, data pipelines, simulation loop, and user interactivity.
 */

import * as THREE from 'three';
import { VISUAL_CONFIG, MODEL_SPECS, TOUR_STEPS } from './constants.js';
import { store } from './state/store.js';
import { createRenderer, setupResizeHandler } from './core/renderer.js';
import { createScene } from './core/scene.js';
import { createCameraAndControls } from './core/camera.js';
import { setupLighting } from './core/lighting.js';
import {
    materials,
    edgeMatMacro,
    edgeMatDense,
    tokenMaterialBaseline,
    tokenMaterialCompressed,
    trailMaterialBaseline,
    trailMaterialCompressed
} from './core/materials.js';
import { loadSpatialData } from './data/spatial_loader.js';
import { loadModuleCatalog, loadModuleComponents } from './data/module_loader.js';
import { buildMeshNodes } from './builders/granular_builder.js';
import { buildMacroBlocks } from './builders/macro_builder.js';
import { createTokenMesh } from './builders/token_builder.js';
import { smoothFlyCamera, updateCameraFlight } from './interactions/camera_fly.js';
import { initSelectionLighting, selectPartIn3D, clearSelectionHighlight, animateSelectionPulse } from './interactions/selection.js';
import { setupRaycasting } from './interactions/raycasting.js';
import { updateTokens } from './physics/token_animation.js';
import { updateNodePulses } from './physics/node_pulses.js';
import { updateComparisonHUD } from './ui/hud.js';
import { updateInspectorWithNode } from './ui/inspector.js';
import { buildHierarchicalTree } from './ui/tree_explorer.js';
import { initModuleCatalog } from './ui/module_catalog.js';
import { initDocModal } from './educational/doc_modal.js';
import { togglePresentMode } from './educational/present_mode.js';
import { openLearnModal } from './educational/learn_modal.js';
import { openModelMatrixDashboard } from './educational/model_matrix_dashboard.js';
import { openAttentionHeatmapExplorer } from './interactive/attention_heatmap.js';
import { openTokenPredictionPlayground } from './interactive/token_prediction.js';
import { openCompressionSimulator } from './interactive/compression_simulator.js';
import { openModelRaceModal } from './interactive/model_race.js';
import { openVideoExportModal } from './export/video_exporter.js';
import { copyShareablePermalink, syncUrlHash, parseHashToState, applyHashStateToVisualizer } from './interactions/permalink.js';
import { setupControls } from './ui/controls.js';
import { perfMonitor } from './performance/monitor.js';

// Application Registry Collections
const visualObjects = [];
const interactableMeshes = [];
const macroMeshes = [];
const svdBottlenecks = [];
const moeNodes = [];
const layerWidthProfilesBaseline = [];
const layerWidthProfilesCompressed = [];

const tokenDataBaseline = [];
const tokenDataCompressed = [];
let tokenMeshBaseline = null;
let tokenMeshCompressed = null;
let trailMeshBaseline = null;
let trailMeshCompressed = null;

let maxGraphHeight = 16.0;
let activeSelectedSublayerKey = null;

// Tour state
let tourActive = false;
let tourStep = 0;
let tourTimer = null;

/**
 * Main application initialization.
 */
export async function bootstrapApp() {
    const canvas = document.getElementById('webgl-canvas');
    if (!canvas) return;

    // 1. Initialize Core Three.js Subsystems
    const renderer = createRenderer(canvas);
    const { scene, granularGroup, macroGroup, dimensionBadgesGroup } = createScene();
    const { camera, controls } = createCameraAndControls(canvas);
    setupLighting(scene);
    setupResizeHandler(renderer, camera);
    initSelectionLighting(scene);

    // 2. Load Datasets
    let spatialData = null;
    let catalogModules = [];
    let modulesComponents = null;

    try {
        [spatialData, catalogModules, modulesComponents] = await Promise.all([
            loadSpatialData(),
            loadModuleCatalog(),
            loadModuleComponents()
        ]);
        store.set({ spatialData, modulesCatalog: catalogModules, modulesComponents });
    } catch (err) {
        console.error('[bootstrapApp] Dataset initialization error:', err);
    }

    // 3. Clear and Rebuild 3D Scene for Active Model
    function rebuildSceneForModel(modelKey, moduleKey = 'svd') {
        clearSelectionHighlight();

        // Dispose previous visual objects
        for (const obj of visualObjects) {
            if (obj.parent) obj.parent.remove(obj);
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material && !Object.values(materials).includes(obj.material)) {
                if (obj.material.map) obj.material.map.dispose();
                obj.material.dispose();
            }
        }

        visualObjects.length = 0;
        interactableMeshes.length = 0;
        macroMeshes.length = 0;
        svdBottlenecks.length = 0;
        moeNodes.length = 0;
        layerWidthProfilesBaseline.length = 0;
        layerWidthProfilesCompressed.length = 0;

        granularGroup.clear();
        macroGroup.clear();
        dimensionBadgesGroup.clear();

        tokenDataBaseline.length = 0;
        tokenDataCompressed.length = 0;

        if (!spatialData || !spatialData.models[modelKey]) return;
        const modelData = spatialData.models[modelKey];
        maxGraphHeight = modelData.max_y || 16.0;
        window.maxGraphHeight = maxGraphHeight;

        const contextBase = {
            granularGroup,
            dimensionBadgesGroup,
            visualObjects,
            interactableMeshes,
            svdBottlenecks,
            moeNodes,
            layerWidthProfiles: layerWidthProfilesBaseline
        };

        const contextComp = {
            granularGroup,
            dimensionBadgesGroup,
            visualObjects,
            interactableMeshes,
            svdBottlenecks,
            moeNodes,
            layerWidthProfiles: layerWidthProfilesCompressed
        };

        // Build Granular Nodes
        buildMeshNodes(modelData.baseline_nodes, VISUAL_CONFIG.LAYOUT.baselineX, false, contextBase);
        const compNodes = (modelData.modules && modelData.modules[moduleKey] && modelData.modules[moduleKey].compressed_nodes) || modelData.compressed_nodes;
        buildMeshNodes(compNodes, VISUAL_CONFIG.LAYOUT.compressedX, true, contextComp);

        // Build Macro Blocks (LOD 0)
        if (modelData.baseline_macro_nodes) {
            buildMacroBlocks(modelData.baseline_macro_nodes, VISUAL_CONFIG.LAYOUT.baselineX, false, macroGroup, visualObjects, macroMeshes, interactableMeshes);
        }
        const compMacro = (modelData.modules && modelData.modules[moduleKey] && modelData.modules[moduleKey].macro_nodes) || modelData.macro_nodes;
        if (compMacro) {
            buildMacroBlocks(compMacro, VISUAL_CONFIG.LAYOUT.compressedX, true, macroGroup, visualObjects, macroMeshes, interactableMeshes);
        }

        // Build Token Meshes
        const tokensBase = createTokenMesh(tokenDataBaseline, VISUAL_CONFIG.LAYOUT.baselineX, tokenMaterialBaseline, trailMaterialBaseline, maxGraphHeight, store.get('simulationSpeed'));
        const tokensComp = createTokenMesh(tokenDataCompressed, VISUAL_CONFIG.LAYOUT.compressedX, tokenMaterialCompressed, trailMaterialCompressed, maxGraphHeight, store.get('simulationSpeed'));

        tokenMeshBaseline = tokensBase.main;
        trailMeshBaseline = tokensBase.trail;
        tokenMeshCompressed = tokensComp.main;
        trailMeshCompressed = tokensComp.trail;

        granularGroup.add(tokenMeshBaseline);
        granularGroup.add(trailMeshBaseline);
        granularGroup.add(tokenMeshCompressed);
        granularGroup.add(trailMeshCompressed);

        visualObjects.push(tokenMeshBaseline, trailMeshBaseline, tokenMeshCompressed, trailMeshCompressed);

        // Update Tree and HUD
        buildHierarchicalTree(modelKey, moduleKey, modelData, (nodeId) => {
            selectPartIn3D(nodeId, interactableMeshes);
            const foundMesh = interactableMeshes.find(m => m.userData && m.userData.id === nodeId);
            if (foundMesh) {
                updateInspectorWithNode(foundMesh.userData, foundMesh.userData.side === 'baseline', store.get('activeResearchModule'), modelKey);
            }
        });

        updateComparisonHUD(modelKey, moduleKey, modelData);
    }

    // 4. Raycasting & Node Interactivity
    setupRaycasting(canvas, camera, interactableMeshes, (nodeData, isLeft, mesh) => {
        selectPartIn3D(mesh, interactableMeshes);
        updateInspectorWithNode(nodeData, isLeft, store.get('activeResearchModule'), store.get('currentModel'));
    });

    // 5. Controls & User Events
    setupControls({
        onPlayPause: () => {
            const next = !store.get('isPlaying');
            store.set({ isPlaying: next });
            return next;
        },
        onSpeedChange: (speed) => {
            store.set({ simulationSpeed: speed });
        },
        onScrub: (step) => {
            const targetY = 1.0 + step * VISUAL_CONFIG.TOKENS.layerStride;
            smoothFlyCamera(camera, controls, targetY, 18.0);
        },
        onSwitchView: (viewMode) => {
            if (viewMode === '2d') {
                smoothFlyCamera(camera, controls, maxGraphHeight * 0.5, 32.0, 0);
            } else {
                smoothFlyCamera(camera, controls, maxGraphHeight * 0.5, 22.0, 4.0);
            }
        },
        onResetCamera: () => {
            smoothFlyCamera(camera, controls, maxGraphHeight * 0.5, Math.max(18, maxGraphHeight * 1.35), 0);
        }
    });

    // 6. Module Catalog Initialization
    initModuleCatalog(catalogModules, (chosenModule) => {
        store.set({ activeResearchModule: chosenModule });
        rebuildSceneForModel(store.get('currentModel'), chosenModule.paradigm || 'svd');
        const inspector = document.getElementById('inspector');
        if (inspector) inspector.classList.remove('hidden');
    });

    // 6b. Research Paper & Markdown Document Reader Modal
    initDocModal((chosenModule) => {
        store.set({ activeResearchModule: chosenModule });
        rebuildSceneForModel(store.get('currentModel'), chosenModule.paradigm || 'svd');
        const inspector = document.getElementById('inspector');
        if (inspector) inspector.classList.remove('hidden');
    });

    // 7. Mechanism Nav Bar Tab Bindings
    document.querySelectorAll('.mechanism-button').forEach(btn => {
        btn.addEventListener('click', () => {
            const mModel = btn.dataset.model || 'distilbert';
            const mModule = btn.dataset.module || 'svd';
            store.set({ currentModel: mModel, currentCompression: mModule });
            rebuildSceneForModel(mModel, mModule);
        });
    });

    // 8. Global Window Exports for Inline HTML Handlers
    window.smoothFlyCamera = (y, dist, x = 0) => smoothFlyCamera(camera, controls, y, dist, x);
    window.smoothZoomBy = (factor) => smoothZoomBy(camera, controls, factor);

    const zoomInBtn = document.getElementById('btn-zoom-in');
    const zoomOutBtn = document.getElementById('btn-zoom-out');
    if (zoomInBtn) zoomInBtn.addEventListener('click', () => smoothZoomBy(camera, controls, 0.78));
    if (zoomOutBtn) zoomOutBtn.addEventListener('click', () => smoothZoomBy(camera, controls, 1.28));

    const presentBtn = document.getElementById('btn-present-mode');
    if (presentBtn) presentBtn.addEventListener('click', () => togglePresentMode({ camera, controls, store, rebuildSceneForModel }));

    const learnBtn = document.getElementById('btn-open-learn');
    if (learnBtn) learnBtn.addEventListener('click', () => openLearnModal());

    const matrixBtn = document.getElementById('btn-open-model-matrix');
    if (matrixBtn) matrixBtn.addEventListener('click', () => openModelMatrixDashboard((m) => {
        store.set({ currentModel: m });
        rebuildSceneForModel(m, (store.get('activeResearchModule') && store.get('activeResearchModule').paradigm) || 'svd');
    }));

    const heatmapBtn = document.getElementById('btn-open-heatmap');
    if (heatmapBtn) heatmapBtn.addEventListener('click', () => openAttentionHeatmapExplorer());

    const predictionBtn = document.getElementById('btn-open-prediction');
    if (predictionBtn) predictionBtn.addEventListener('click', () => openTokenPredictionPlayground());

    const compressionBtn = document.getElementById('btn-open-compression');
    if (compressionBtn) compressionBtn.addEventListener('click', () => {
        const curMod = store.get('activeResearchModule');
        const initRank = (curMod && curMod.rank) || 64;
        openCompressionSimulator({
            initialRank: initRank,
            hiddenDim: 768,
            numLayers: 12,
            onApplyTo3D: (newRank) => {
                const scaleFactor = Math.max(0.25, Math.min(2.2, newRank / 64));
                interactableMeshes.forEach(mesh => {
                    if (mesh.userData && (mesh.userData.type === 'svd' || mesh.userData.sub_type === 'v_factor' || mesh.userData.sub_type === 'u_factor')) {
                        mesh.scale.set(scaleFactor, 1.0, scaleFactor);
                    }
                });
                updateComparisonHUD(store.get('currentModel'), store.get('activeResearchModule') || { paradigm: 'svd', rank: newRank });
            }
        });
    });

    const raceBtn = document.getElementById('btn-open-race');
    if (raceBtn) raceBtn.addEventListener('click', () => openModelRaceModal());

    const shareBtn = document.getElementById('btn-share');
    if (shareBtn) {
        shareBtn.addEventListener('click', () => {
            const curMod = store.get('activeResearchModule');
            const inspectorEl = document.getElementById('inspector');
            const state = {
                model: store.get('currentModel'),
                module: curMod ? (curMod.id || curMod.name) : undefined,
                view: controls.enableRotate === false ? '2d' : '3d',
                rank: curMod && curMod.rank ? curMod.rank : undefined,
                inspector: inspectorEl ? !inspectorEl.classList.contains('hidden') : false
            };
            copyShareablePermalink(state);
            syncUrlHash(state);
        });
    }

    const videoBtn = document.getElementById('btn-record-video');
    if (videoBtn) {
        videoBtn.addEventListener('click', () => {
            openVideoExportModal({
                canvas: renderer.domElement,
                camera,
                controls
            });
        });
    }

    document.addEventListener('keydown', (e) => {
        if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
        if (e.key === '+' || e.key === '=') {
            smoothZoomBy(camera, controls, 0.8);
        } else if (e.key === '-' || e.key === '_') {
            smoothZoomBy(camera, controls, 1.25);
        } else if (e.key === 'p' || e.key === 'P') {
            togglePresentMode({ camera, controls, store, rebuildSceneForModel });
        } else if (e.key === 'l' || e.key === 'L') {
            openLearnModal();
        } else if (e.key === 'h' || e.key === 'H') {
            openAttentionHeatmapExplorer();
        } else if (e.key === 't' || e.key === 'T') {
            openTokenPredictionPlayground();
        } else if (e.key === 'c' || e.key === 'C') {
            if (compressionBtn) compressionBtn.click();
        } else if (e.key === 'v' || e.key === 'V') {
            if (videoBtn) videoBtn.click();
        }
    });

    window.selectPartIn3D = (target) => {
        const mesh = selectPartIn3D(target, interactableMeshes);
        if (mesh) {
            updateInspectorWithNode(mesh.userData, mesh.userData.side === 'baseline', store.get('activeResearchModule'), store.get('currentModel'));
            smoothFlyCamera(camera, controls, mesh.position.y, 8.5, mesh.position.x);
        }
    };
    window.focusPipelineComponent = (compId) => {
        window.selectPartIn3D(compId);
    };
    window.toggleTour = () => {
        tourActive = !tourActive;
        const tourBtn = document.getElementById('btn-auto-tour');
        const tourLabel = document.getElementById('tour-step-label');

        if (!tourActive) {
            clearTimeout(tourTimer);
            if (tourBtn) tourBtn.innerHTML = '▶ Tour';
            if (tourLabel) tourLabel.style.display = 'none';
            return;
        }

        if (tourBtn) tourBtn.innerHTML = '⏹ Stop';
        tourStep = 0;

        function runStep() {
            if (!tourActive) return;
            const s = TOUR_STEPS[tourStep % TOUR_STEPS.length];
            if (tourLabel) {
                tourLabel.style.display = 'block';
                tourLabel.classList.add('show');
                tourLabel.innerHTML = `<strong>${s.label}</strong><div class="tour-desc">${s.desc || ''}</div>`;
            }
            if (s.target) {
                const mesh = selectPartIn3D(s.target, interactableMeshes);
                if (mesh) {
                    updateInspectorWithNode(mesh.userData, mesh.userData.side === 'baseline', store.get('activeResearchModule'), store.get('currentModel'));
                }
            } else if (s.fullFit) {
                clearSelectionHighlight();
            }
            smoothFlyCamera(camera, controls, s.y, s.dist);
            tourStep++;
            tourTimer = setTimeout(runStep, VISUAL_CONFIG.ANIM.tourStepDuration);
        }
        runStep();
    };

    // Initial Scene Build & URL Query Parameters Auto-Loader (?module=...&model=...)
    const urlParams = new URLSearchParams(window.location.search);
    const paramModule = urlParams.get('module');
    const paramModel = urlParams.get('model');

    if (paramModel && ['distilbert', 'roberta', 'gpt2', 'qwen'].includes(paramModel)) {
        store.set({ currentModel: paramModel });
    }

    if (paramModule && catalogModules.length > 0) {
        const found = catalogModules.find(m => m.id.toLowerCase() === paramModule.toLowerCase() || (m.name && m.name.toLowerCase().includes(paramModule.toLowerCase())));
        if (found) {
            store.set({ activeResearchModule: found, currentCompression: found.paradigm || 'svd' });
            const insp = document.getElementById('inspector');
            if (insp) insp.classList.remove('hidden');
        }
    }

    rebuildSceneForModel(store.get('currentModel'), store.get('currentCompression'));
    smoothFlyCamera(camera, controls, maxGraphHeight * 0.5, Math.max(18, maxGraphHeight * 1.35));

    // Restore deep-linked state from URL hash if present
    if (typeof window !== 'undefined' && window.location.hash) {
        const parsed = parseHashToState(window.location.hash);
        applyHashStateToVisualizer(parsed, {
            store,
            catalogModules,
            rebuildSceneForModel,
            smoothFlyCamera: (cam, ctrl, y, d, x) => smoothFlyCamera(cam, ctrl, y, d, x),
            camera,
            controls
        });
    }

    // 9. Main Animation / Simulation Frame Loop
    function animate(timestamp) {
        requestAnimationFrame(animate);

        perfMonitor.tick(timestamp);
        controls.update();
        updateCameraFlight(camera, controls);
        animateSelectionPulse(timestamp);

        // Continuous Semantic LOD based on camera distance
        const camDist = camera.position.distanceTo(controls.target);
        if (camDist > VISUAL_CONFIG.ZOOM.macroThreshold) {
            macroGroup.visible = true;
            granularGroup.visible = false;
            dimensionBadgesGroup.visible = false;
        } else if (camDist > VISUAL_CONFIG.ZOOM.transitionStart) {
            macroGroup.visible = true;
            granularGroup.visible = true;
            dimensionBadgesGroup.visible = false;
        } else {
            macroGroup.visible = false;
            granularGroup.visible = true;
            dimensionBadgesGroup.visible = true;
        }

        // Particle Physics Simulation
        const simContext = {
            modelKey: store.get('currentModel'),
            paradigm: (store.get('activeResearchModule') && store.get('activeResearchModule').paradigm) || store.get('currentCompression') || 'svd',
            maxGraphHeight,
            simulationSpeed: store.get('simulationSpeed'),
            enablePulses: store.get('enablePulses'),
            isPlaying: store.get('isPlaying'),
            interactableMeshes,
            activeSelectedSublayerKey
        };

        if (store.get('isPlaying')) {
            updateTokens(tokenMeshBaseline, trailMeshBaseline, tokenDataBaseline, false, simContext);
            updateTokens(tokenMeshCompressed, trailMeshCompressed, tokenDataCompressed, true, simContext);
            updateNodePulses(timestamp);
        }

        // Dual-Viewport Scissor Render
        const w = window.innerWidth;
        const h = window.innerHeight;

        renderer.setViewport(0, 0, w, h);
        renderer.setScissor(0, 0, w, h);
        renderer.render(scene, camera);
    }

    requestAnimationFrame(animate);
}
