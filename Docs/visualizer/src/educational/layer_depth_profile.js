/**
 * educational/layer_depth_profile.js
 * ==================================
 * Layer-by-Layer Depth Specialization & Telemetry Engine.
 * Formulates the theoretical and empirical differences across Transformer depth:
 * 1. Normalized Relative Depth (tau in [0.0, 1.0])
 * 2. 4-Stage Architectural Progression (Lexical -> Syntactic -> Disambiguation -> Semantic Output)
 * 3. Attention Entropy Transition (Diffuse -> Inductive -> Sharp)
 * 4. Singular Value Spectral Decay & Optimal Low-Rank (r*) Allocation
 * 5. Residual Stream Highway Dominance vs Sublayer Delta Evolution
 * 6. FFN Activation Sparsity Scaling
 * 7. Depth-Aware 3D PBR Materials, Stage Platform Halos, and Layer Badges
 */

import * as THREE from 'three';
import { MODEL_SPECS } from '../constants.js';

// Depth Phase Palettes (PBR colors and glowing rims)
export const LAYER_PHASES = [
    {
        name: 'Stage 1 · Morpho-Syntactic & Local Receptive Field',
        shortTag: 'Lexical / Local Syntax',
        colorHex: '#06b6d4',      // Cyan
        colorRgb: [6, 182, 212],
        accentBorder: '#22d3ee',
        bgTint: 'rgba(6, 182, 212, 0.12)',
        materialColor: 0x164e63,  // Deep cyan-slate
        edgeColor: 0x22d3ee,
        ringColor: 0x06b6d4,
        tauMax: 0.25
    },
    {
        name: 'Stage 2 · Syntactic Dependencies & Multi-Hop Induction',
        shortTag: 'Syntactic Induction',
        colorHex: '#0284c7',      // Sky Azure
        colorRgb: [2, 132, 199],
        accentBorder: '#38bdf8',
        bgTint: 'rgba(2, 132, 199, 0.12)',
        materialColor: 0x1e3a8a,  // Deep azure
        edgeColor: 0x38bdf8,
        ringColor: 0x0284c7,
        tauMax: 0.50
    },
    {
        name: 'Stage 3 · Contextual Disambiguation & Semantic Binding',
        shortTag: 'Semantic Binding',
        colorHex: '#6366f1',      // Indigo / Royal Violet
        colorRgb: [99, 102, 241],
        accentBorder: '#818cf8',
        bgTint: 'rgba(99, 102, 241, 0.12)',
        materialColor: 0x312e81,  // Deep indigo
        edgeColor: 0x818cf8,
        ringColor: 0x6366f1,
        tauMax: 0.75
    },
    {
        name: 'Stage 4 · High-Level Abstraction & Logit Shaping',
        shortTag: 'Semantic Output',
        colorHex: '#a855f7',      // Amethyst / Purple
        colorRgb: [168, 85, 247],
        accentBorder: '#c084fc',
        bgTint: 'rgba(168, 85, 247, 0.12)',
        materialColor: 0x581c87,  // Deep purple
        edgeColor: 0xc084fc,
        ringColor: 0xa855f7,
        tauMax: 1.00
    }
];

// Material Caching per Layer Depth & Type
const layerDepthMaterialCache = new Map();
const layerDepthEdgeCache = new Map();

/**
 * Computes the full Layer Depth Profile for a specific layer index in a model.
 * @param {number|string} layerOrIdx - Layer index or identifier ('embeddings', 'output')
 * @param {string} currentModel - Model key (distilbert, roberta, gpt2, qwen)
 * @param {number|null} customTotalLayers - Optional explicit layer count
 * @returns {object} Layer depth profile
 */
export function getLayerDepthProfile(layerOrIdx, currentModel = 'distilbert', customTotalLayers = null) {
    const spec = MODEL_SPECS[currentModel] || MODEL_SPECS.distilbert;
    const totalLayers = customTotalLayers || spec.num_layers || 6;

    let layerIdx = 0;
    let isSpecial = false;
    let specialType = '';

    if (typeof layerOrIdx === 'string') {
        const lower = layerOrIdx.toLowerCase();
        if (lower.includes('embed')) {
            isSpecial = true;
            specialType = 'embeddings';
            layerIdx = 0;
        } else if (lower.includes('output') || lower.includes('head') || lower.includes('class')) {
            isSpecial = true;
            specialType = 'output';
            layerIdx = totalLayers - 1;
        } else {
            const parsed = parseInt(layerOrIdx, 10);
            layerIdx = Number.isFinite(parsed) ? parsed : 0;
        }
    } else if (typeof layerOrIdx === 'number') {
        layerIdx = Math.max(0, Math.min(totalLayers - 1, layerOrIdx));
    }

    const normalizedDepth = totalLayers > 1 ? layerIdx / (totalLayers - 1) : 0;

    // Resolve Phase
    let phase = LAYER_PHASES[0];
    let phaseIndex = 0;
    for (let i = 0; i < LAYER_PHASES.length; i++) {
        if (normalizedDepth <= LAYER_PHASES[i].tauMax || i === LAYER_PHASES.length - 1) {
            phase = LAYER_PHASES[i];
            phaseIndex = i;
            break;
        }
    }

    // Mathematical calculations across layer depth
    // 1. Attention Entropy: H(A_l) in nats
    // Early layers: ~3.12 nats (diffuse), Late layers: ~1.05 nats (sharp)
    const attentionEntropy = +(1.05 + 2.07 * Math.pow(1 - normalizedDepth, 1.35)).toFixed(2);
    const entropyPercent = Math.round((attentionEntropy / 3.12) * 100);

    // 2. Singular Value Decay Ratio: sigma_1 / sigma_r
    // Middle layers experience fastest decay (highest compressibility)
    const midParabola = 4 * normalizedDepth * (1 - normalizedDepth); // Peaks at 1.0 when tau=0.5
    const singularDecayRatio = +(6.2 + midParabola * 28.3).toFixed(1);

    // 3. Recommended Optimal Low-Rank Allocation (r*)
    let optimalSvdRank = 64;
    if (normalizedDepth < 0.20) optimalSvdRank = 64;
    else if (normalizedDepth < 0.45) optimalSvdRank = 32;
    else if (normalizedDepth < 0.75) optimalSvdRank = 24;
    else optimalSvdRank = 48;

    // 4. Residual Highway Stream Accumulation
    // alpha_res = ||x_res|| / (||x_res|| + ||delta_layer||)
    const residualAccumulationRatio = +(0.54 + 0.39 * Math.pow(normalizedDepth, 0.8)).toFixed(2);
    const residualPercent = Math.round(residualAccumulationRatio * 100);
    const layerDeltaPercent = 100 - residualPercent;

    // 5. Activation Sparsity (GELU/ReLU zero rate)
    const activationSparsityPercent = Math.round(15 + 33 * normalizedDepth);

    // 6. Linguistic & Transformer Representation Role
    let linguisticRole = '';
    let patternSummary = '';
    if (phaseIndex === 0) {
        linguisticRole = 'Extracts local n-gram token collocations, punctuation delimiters, character-level morphology, and calibrates absolute/rotary positional embeddings.';
        patternSummary = 'Diffuse Receptive Field (broad attention over local tokens and boundary markers)';
    } else if (phaseIndex === 1) {
        linguisticRole = 'Builds hierarchical syntactic trees, verb-argument dependencies, and multi-hop Induction Heads that track pattern repetition ([A][B] ... [A] -> [B]).';
        patternSummary = 'Structured Syntactic Links (vertical delimiter lines and diagonal lookahead)';
    } else if (phaseIndex === 2) {
        linguisticRole = 'Resolves polysemous words, binds anaphoric pronouns to antecedents across multi-sentence spans, and tracks entity state machines.';
        patternSummary = 'Relational Clusters & Coreference (selective entity-to-entity attention)';
    } else {
        linguisticRole = 'Synthesizes global semantic representations, suppresses contextually irrelevant tokens, and aligns residual vectors with the unembedding output vocabulary.';
        patternSummary = 'Sharp Semantic Focusing (high attention concentration on predictive tokens and task hubs)';
    }

    return {
        layerIndex: layerIdx,
        totalLayers,
        modelKey: currentModel,
        modelName: spec.name,
        normalizedDepth,
        phaseIndex,
        phaseName: phase.name,
        shortTag: phase.shortTag,
        themeColor: phase.colorHex,
        themeRgb: phase.colorRgb,
        accentBorder: phase.accentBorder,
        bgTint: phase.bgTint,
        ringColor: phase.ringColor,
        isSpecial,
        specialType,
        attentionEntropy,
        entropyPercent,
        singularDecayRatio,
        optimalSvdRank,
        residualPercent,
        layerDeltaPercent,
        activationSparsityPercent,
        linguisticRole,
        patternSummary,
        mathFormulaResidual: '\\mathbf{x}_{l} = \\mathbf{x}_{l-1} + \\mathbf{W}_{O,l} \\left( \\operatorname{Softmax}\\left(\\frac{\\mathbf{Q}_l \\mathbf{K}_l^\\top}{\\sqrt{d_k}}\\right) \\mathbf{V}_l \\right) + \\operatorname{FFN}_l(\\dots)',
        mathFormulaEntropy: '\\mathcal{H}(A_l) = -\\frac{1}{S}\\sum_{i=1}^{S}\\sum_{j=1}^{S} A_{l,ij} \\ln A_{l,ij}'
    };
}

/**
 * Gets or creates a depth-shifted standard PBR material for 3D tensor meshes.
 * @param {number} layerIdx - Layer index
 * @param {number} totalLayers - Total layers in model
 * @param {string} type - Node type ('dense', 'svd', etc.)
 * @param {boolean} isCompressedSide - True if compressed column
 * @returns {THREE.MeshStandardMaterial}
 */
export function getLayerDepthMaterial(layerIdx, totalLayers, type = 'dense', isCompressedSide = false) {
    const key = `${layerIdx}_${totalLayers}_${type}_${isCompressedSide}`;
    if (layerDepthMaterialCache.has(key)) {
        return layerDepthMaterialCache.get(key);
    }

    const profile = getLayerDepthProfile(layerIdx, null, totalLayers);
    let mat;

    if (type === 'dense' && !isCompressedSide) {
        // Base dense material tinted with subtle layer phase hue
        const c = new THREE.Color(LAYER_PHASES[profile.phaseIndex].materialColor);
        mat = new THREE.MeshStandardMaterial({
            color: c,
            roughness: 0.28,
            metalness: 0.38,
            transparent: true,
            opacity: 0.94,
            emissive: c.clone().multiplyScalar(0.2),
            emissiveIntensity: 0.25
        });
    } else if (type === 'svd') {
        // Compressed SVD preserves purple/teal identity but shifts emissive glow with layer depth
        const baseColor = new THREE.Color(0x3acbb8);
        const emissiveColor = new THREE.Color(LAYER_PHASES[profile.phaseIndex].colorHex).multiplyScalar(0.4);
        mat = new THREE.MeshStandardMaterial({
            color: baseColor,
            roughness: 0.22,
            metalness: 0.52,
            emissive: emissiveColor,
            emissiveIntensity: 0.45
        });
    } else {
        const c = new THREE.Color(LAYER_PHASES[profile.phaseIndex].colorHex);
        mat = new THREE.MeshStandardMaterial({
            color: c,
            roughness: 0.3,
            metalness: 0.4
        });
    }

    layerDepthMaterialCache.set(key, mat);
    return mat;
}

/**
 * Gets or creates a depth-shifted edge outline wireframe material.
 * @param {number} layerIdx - Layer index
 * @param {number} totalLayers - Total layers in model
 * @returns {THREE.LineBasicMaterial}
 */
export function getLayerDepthEdgeMaterial(layerIdx, totalLayers) {
    const key = `${layerIdx}_${totalLayers}_edge`;
    if (layerDepthEdgeCache.has(key)) {
        return layerDepthEdgeCache.get(key);
    }

    const profile = getLayerDepthProfile(layerIdx, null, totalLayers);
    const edgeColor = LAYER_PHASES[profile.phaseIndex].edgeColor;
    const mat = new THREE.LineBasicMaterial({
        color: edgeColor,
        transparent: true,
        opacity: 0.70
    });

    layerDepthEdgeCache.set(key, mat);
    return mat;
}

/**
 * Builds an interactive 3D layer stage platform ring in world space.
 * @param {number} layerIdx - Layer index
 * @param {number} totalLayers - Total layers in model
 * @param {object} bounds - { minY, maxY } bounds of the layer
 * @param {number} offsetX - Column X position (-4.5 or +4.5)
 * @param {boolean} isCompressedSide - True if compressed column
 * @param {THREE.Group} targetGroup - Scene group to attach to
 * @param {Array<THREE.Object3D>} visualObjects - Registry array for disposal
 * @returns {THREE.Mesh}
 */
export function createLayerStageRing(layerIdx, totalLayers, bounds, offsetX, isCompressedSide, targetGroup, visualObjects) {
    const profile = getLayerDepthProfile(layerIdx, null, totalLayers);
    const ringY = bounds.minY - 0.25;
    const ringRadiusX = 2.4;
    const ringRadiusZ = 1.3;

    // Elliptical ring geometry
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
 * Creates an upgraded 3D Billboard Sprite displaying complete layer depth metadata.
 * @param {string} text - Title text
 * @param {number} layerIdx - Layer index
 * @param {number} totalLayers - Total layers
 * @param {number} worldX - World X position
 * @param {number} worldY - World Y position
 * @param {boolean} isMacro - Macro mode
 * @param {THREE.Group} targetGroup - Scene group
 * @param {Array<THREE.Object3D>} visualObjects - Disposal registry
 * @returns {THREE.Sprite}
 */
export function createLayerLabelSpriteWithDepth(text, layerIdx, totalLayers, worldX, worldY, isMacro = false, targetGroup, visualObjects) {
    const profile = getLayerDepthProfile(layerIdx, null, totalLayers);
    const canvas = document.createElement('canvas');
    canvas.width = 340;
    canvas.height = 92;
    const ctx = canvas.getContext('2d');

    // Background Card
    ctx.fillStyle = isMacro ? 'rgba(8, 18, 36, 0.94)' : 'rgba(5, 12, 22, 0.90)';
    ctx.strokeStyle = profile.themeColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(4, 4, 332, 84, 8);
    ctx.fill();
    ctx.stroke();

    // Stage Pip & Header
    ctx.fillStyle = profile.themeColor;
    ctx.beginPath();
    ctx.arc(22, 28, 6, 0, Math.PI * 2);
    ctx.fill();

    ctx.font = 'bold 20px ui-monospace, Consolas, monospace';
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 36, 28);

    // Subtitle Tag
    ctx.font = 'bold 12px ui-monospace, Consolas, monospace';
    ctx.fillStyle = profile.themeColor;
    ctx.fillText(profile.shortTag.toUpperCase(), 36, 50);

    // Telemetry Footnote
    ctx.font = '11px ui-monospace, Consolas, monospace';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.65)';
    ctx.fillText(`Ent: ${profile.attentionEntropy} nats  |  Rec: r=${profile.optimalSvdRank}  |  Res: ${profile.residualPercent}%`, 18, 72);

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true, opacity: 0.94 });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.position.set(worldX, worldY, 0);
    sprite.scale.set(3.8, 1.03, 1.0);
    sprite.raycast = () => {};

    if (targetGroup) targetGroup.add(sprite);
    if (visualObjects) visualObjects.push(sprite);
    return sprite;
}

/**
 * Renders the complete interactive Layer Depth Telemetry Card into the Inspector Drawer.
 * @param {object|number|string} nodeOrLayerIdx - Selected node or layer identifier
 * @param {string} currentModel - Active model key
 * @param {string} containerId - Target HTML element ID
 */
export function renderLayerDepthProfile(nodeOrLayerIdx, currentModel = 'distilbert', containerId = 'layer-depth-profile-card') {
    let container = document.getElementById(containerId);
    if (!container) {
        // Create container and insert right after attention-deep-dive-card or inspector stat grid
        const parent = document.querySelector('.drawer-body');
        if (!parent) return;
        container = document.createElement('div');
        container.id = containerId;
        container.className = 'layer-depth-card';
        container.style.marginTop = '12px';

        const anchor = document.getElementById('attention-deep-dive-card') || document.getElementById('insp-comp-card');
        if (anchor && anchor.nextSibling) {
            parent.insertBefore(container, anchor.nextSibling);
        } else {
            parent.appendChild(container);
        }
    }

    let layerIdx = 0;
    if (typeof nodeOrLayerIdx === 'object' && nodeOrLayerIdx !== null) {
        layerIdx = nodeOrLayerIdx.parent_layer_index ?? (nodeOrLayerIdx.parent_layer_idx ?? 0);
    } else {
        layerIdx = nodeOrLayerIdx;
    }

    const profile = getLayerDepthProfile(layerIdx, currentModel);

    // Build interactive mini-timeline pills for all layers
    const layerPillsHtml = Array.from({ length: profile.totalLayers }).map((_, idx) => {
        const p = getLayerDepthProfile(idx, currentModel, profile.totalLayers);
        const isActive = idx === profile.layerIndex;
        return `
            <button 
                class="depth-step-btn ${isActive ? 'active' : ''}" 
                onclick="window.navigateToLayer && window.navigateToLayer(${idx})"
                style="
                    flex: 1;
                    padding: 4px 2px;
                    font-size: 9px;
                    font-weight: 700;
                    font-family: var(--font-mono);
                    border-radius: 4px;
                    background: ${isActive ? p.themeColor : 'rgba(255, 255, 255, 0.05)'};
                    color: ${isActive ? '#040e1d' : 'var(--text-muted)'};
                    border: 1px solid ${isActive ? p.accentBorder : 'rgba(255, 255, 255, 0.08)'};
                    cursor: pointer;
                    transition: all 0.15s ease;
                "
                title="${p.phaseName} (Ent: ${p.attentionEntropy} nats)"
            >
                L${idx}
            </button>
        `;
    }).join('');

    container.innerHTML = `
        <div style="
            background: rgba(7, 14, 26, 0.85);
            border: 1px solid ${profile.accentBorder};
            border-radius: 8px;
            padding: 12px;
            box-shadow: 0 4px 20px ${profile.bgTint};
        ">
            <!-- Stage Header -->
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="
                        display:inline-block;
                        width:10px;
                        height:10px;
                        border-radius:50%;
                        background:${profile.themeColor};
                        box-shadow:0 0 8px ${profile.themeColor};
                    "></span>
                    <strong style="font-size:11px; color:#ffffff; letter-spacing:0.5px; text-transform:uppercase;">
                        LAYER ${profile.layerIndex} OF ${profile.totalLayers} · DEPTH PROFILE
                    </strong>
                </div>
                <span class="badge" style="
                    background:${profile.bgTint};
                    color:${profile.accentBorder};
                    border:1px solid ${profile.accentBorder};
                    font-size:9px;
                    font-weight:700;
                    padding:2px 8px;
                    border-radius:4px;
                ">
                    τ = ${(profile.normalizedDepth * 100).toFixed(0)}% DEPTH
                </span>
            </div>

            <!-- Mini Layer Stepper Strip -->
            <div style="display:flex; gap:3px; margin-bottom:12px;">
                ${layerPillsHtml}
            </div>

            <!-- Stage Title & Functional Role -->
            <div style="
                background: ${profile.bgTint};
                border-left: 3px solid ${profile.themeColor};
                padding: 8px 10px;
                border-radius: 0 6px 6px 0;
                margin-bottom: 12px;
            ">
                <div style="font-size:11px; font-weight:700; color:${profile.accentBorder}; margin-bottom:3px;">
                    ${profile.phaseName}
                </div>
                <div style="font-size:11px; line-height:1.5; color:var(--text-main);">
                    ${profile.linguisticRole}
                </div>
            </div>

            <!-- 4 Live Telemetry Gauges -->
            <div style="display:grid; grid-template-columns: repeat(2, 1fr); gap:8px; margin-bottom:12px;">
                <!-- 1. Attention Entropy -->
                <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:8px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <small style="font-size:9px; color:var(--text-faint);">ATTN ENTROPY $\\mathcal{H}$</small>
                        <strong style="font-size:11px; color:${profile.accentBorder}; font-family:var(--font-mono);">${profile.attentionEntropy} nats</strong>
                    </div>
                    <div style="height:4px; background:rgba(255,255,255,0.1); border-radius:2px; overflow:hidden;">
                        <div style="width:${profile.entropyPercent}%; height:100%; background:linear-gradient(90deg, #38bdf8, ${profile.themeColor});"></div>
                    </div>
                    <div style="font-size:9px; color:var(--text-faint); margin-top:4px;">
                        ${profile.normalizedDepth < 0.3 ? 'Diffuse / Local' : (profile.normalizedDepth < 0.7 ? 'Structured Induction' : 'Sharp / Focused')}
                    </div>
                </div>

                <!-- 2. SVD Compressibility & Optimal Rank -->
                <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:8px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <small style="font-size:9px; color:var(--text-faint);">SPECTRUM DECAY</small>
                        <strong style="font-size:11px; color:#34d399; font-family:var(--font-mono);">${profile.singularDecayRatio}×</strong>
                    </div>
                    <div style="height:4px; background:rgba(255,255,255,0.1); border-radius:2px; overflow:hidden;">
                        <div style="width:${Math.min(100, Math.round((profile.singularDecayRatio / 35) * 100))}%; height:100%; background:#10b981;"></div>
                    </div>
                    <div style="font-size:9px; color:var(--text-faint); margin-top:4px;">
                        Optimal Low-Rank: <strong style="color:#34d399;">r* = ${profile.optimalSvdRank}</strong>
                    </div>
                </div>

                <!-- 3. Residual Stream Accumulation -->
                <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:8px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <small style="font-size:9px; color:var(--text-faint);">RESIDUAL RATIO</small>
                        <strong style="font-size:11px; color:#f6be68; font-family:var(--font-mono);">${profile.residualPercent}% Res</strong>
                    </div>
                    <div style="height:4px; background:rgba(255,255,255,0.1); border-radius:2px; overflow:hidden; display:flex;">
                        <div style="width:${profile.residualPercent}%; height:100%; background:#f59e0b;"></div>
                        <div style="width:${profile.layerDeltaPercent}%; height:100%; background:#38bdf8;"></div>
                    </div>
                    <div style="font-size:9px; color:var(--text-faint); margin-top:4px;">
                        ${profile.residualPercent}% Trunk / ${profile.layerDeltaPercent}% Sublayer $\\Delta$
                    </div>
                </div>

                <!-- 4. Activation Sparsity -->
                <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:6px; padding:8px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <small style="font-size:9px; color:var(--text-faint);">FFN SPARSITY</small>
                        <strong style="font-size:11px; color:#e879f9; font-family:var(--font-mono);">${profile.activationSparsityPercent}% Zero</strong>
                    </div>
                    <div style="height:4px; background:rgba(255,255,255,0.1); border-radius:2px; overflow:hidden;">
                        <div style="width:${profile.activationSparsityPercent}%; height:100%; background:#c084fc;"></div>
                    </div>
                    <div style="font-size:9px; color:var(--text-faint); margin-top:4px;">
                        Intermediate Neurons Inactive
                    </div>
                </div>
            </div>

            <!-- Mathematical Formulation Callout -->
            <div style="
                background: rgba(6, 13, 24, 0.7);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 6px;
                padding: 8px 10px;
                font-size: 11px;
            ">
                <div style="color:var(--text-faint); font-size:10px; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px;">
                    Residual Highway Formulation at Layer ${profile.layerIndex}:
                </div>
                <div class="katex-math-render" style="text-align:center; color:#e0f2fe; padding:4px 0; font-size:11px; font-family:var(--font-mono);">
                    $$ \\mathbf{x}_{${profile.layerIndex + 1}} = \\mathbf{x}_{${profile.layerIndex}} + \\text{Attn}_{${profile.layerIndex}}(\\mathbf{x}_{${profile.layerIndex}}) + \\text{FFN}_{${profile.layerIndex}}(\\dots) $$
                </div>
            </div>
        </div>
    `;

    // Render KaTeX if available
    if (window.katex) {
        container.querySelectorAll('.katex-math-render').forEach(el => {
            const raw = el.textContent.replace(/\$\$/g, '').trim();
            try {
                el.innerHTML = window.katex.renderToString(raw, { displayMode: true, throwOnError: false });
            } catch (e) {
                // Fallback kept
            }
        });
    }
}
