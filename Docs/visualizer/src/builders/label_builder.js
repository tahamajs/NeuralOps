/**
 * builders/label_builder.js
 * =========================
 * 3D Dynamic Billboard Canvas Sprites:
 * - Layer Title badges (Macro and Granular)
 * - Matrix dimension & tensor shape badges ([768 × 768], GQA heads, rank bottleneck)
 */

import * as THREE from 'three';
import { anyKey } from '../utils/math.js';
import { getLayerDepthProfile } from '../educational/layer_depth_profile.js';

/**
 * Creates a canvas sprite for layer headers.
 * @param {string} text - Title text
 * @param {number} worldX - World X position
 * @param {number} worldY - World Y position
 * @param {boolean} isMacro - Whether sprite belongs to Macro group
 * @param {THREE.Group} targetGroup - Group to attach to
 * @param {Array<THREE.Object3D>} visualObjects - Registry array for disposal
 * @returns {THREE.Sprite}
 */
export function createLayerLabelSprite(text, worldX, worldY, isMacro = false, targetGroup, visualObjects) {
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

    if (targetGroup) targetGroup.add(sprite);
    if (visualObjects) visualObjects.push(sprite);
    return sprite;
}

/**
 * Creates a floating dimensional badge attached to tensor nodes.
 * @param {object} node - Tensor node
 * @param {number} offsetX - Column horizontal offset
 * @param {THREE.Group} targetGroup - Group to attach to
 * @param {Array<THREE.Object3D>} visualObjects - Registry array for disposal
 * @returns {THREE.Sprite|null}
 */
export function createTensorBadgeSprite(node, offsetX, targetGroup, visualObjects) {
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
    } else if (node.id && (node.id.includes('k_proj') || node.id.includes('v_proj'))) {
        if (dOut <= 256) badgeText = `[${dOut}×${dIn}] 2H GQA`;
        else badgeText = `[${dOut}×${dIn}] ${Math.round(dOut / 64)}H`;
    } else if (node.id && node.id.includes('q_proj')) {
        badgeText = `[${dOut}×${dIn}] (${Math.round(dOut / 64)}H)`;
    } else if (dOut >= 3072) {
        badgeText = `[${dOut}×${dIn}] ${(dOut / dIn).toFixed(1)}× FFN`;
    } else if (node.id && anyKey(node.id, ['q_lin', 'k_lin', 'v_lin', 'query', 'key', 'value'])) {
        badgeText = `[${dOut}×${dIn}] 12H`;
    }

    const canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');

    const isSvd = (node.type === 'svd');
    const isMoe = (node.type === 'moe');
    const isMla = (node.type === 'mla');
    const isGqa = node.id && (node.id.includes('k_proj') || node.id.includes('v_proj'));
    const isSwiglu = node.id && (node.id.includes('gate_proj') || node.id.includes('up_proj'));

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
        opacity: 0.95,
        depthWrite: false
    });
    const sprite = new THREE.Sprite(spriteMat);

    const badgeZ = (node.z || 0) + (node.depth ? node.depth / 2 : 0.60) + 0.05;
    let badgeY = node.y;

    if (node.id && node.id.includes('k_proj')) badgeY -= 0.06;
    if (node.id && node.id.includes('v_proj')) badgeY += 0.06;

    sprite.position.set(node.x + offsetX, badgeY, badgeZ);

    const badgeW = Math.max(0.65, Math.min(1.4, (node.width || 1.2) * 0.92));
    const badgeH = Math.min(0.24, (node.height || 0.42) * 0.62);
    sprite.scale.set(badgeW, badgeH, 1.0);
    sprite.raycast = () => {};

    if (targetGroup) targetGroup.add(sprite);
    if (visualObjects) visualObjects.push(sprite);
    return sprite;
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

