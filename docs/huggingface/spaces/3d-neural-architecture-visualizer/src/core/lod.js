/**
 * core/lod.js
 * ===========
 * Adaptive Level of Detail (LOD) & Scalable Scene Complexity Manager.
 * Dynamically modulates 3D mesh visibility, outline resolution, and sublayer
 * particle streams based on camera distance to maintain 60 FPS across all devices.
 */

export const LOD_THRESHOLDS = {
    FAR_DISTANCE: 55.0,
    MEDIUM_DISTANCE: 22.0
};

/**
 * Determines LOD level ('far', 'medium', 'near') based on camera Euclidean distance.
 * @param {number} cameraDistance - Distance from camera to active target
 * @param {number} [totalNodes=100] - Total active scene meshes
 * @returns {'far'|'medium'|'near'} LOD mode
 */
export function getLODLevel(cameraDistance, totalNodes = 100) {
    if (cameraDistance > LOD_THRESHOLDS.FAR_DISTANCE || totalNodes > 400) {
        return 'far';
    }
    if (cameraDistance > LOD_THRESHOLDS.MEDIUM_DISTANCE || totalNodes > 200) {
        return 'medium';
    }
    return 'near';
}

/**
 * Applies LOD visibility rules to a list of visual scene objects.
 * @param {Array<Object>} visualObjects - Array of Three.js objects or mock mesh items
 * @param {'far'|'medium'|'near'} level - Target LOD level
 * @returns {number} Count of visible objects
 */
export function applyLOD(visualObjects, level) {
    if (!visualObjects || !Array.isArray(visualObjects)) return 0;
    let visibleCount = 0;

    visualObjects.forEach(obj => {
        if (!obj) return;
        const uData = obj.userData || {};

        switch (level) {
            case 'far':
                // Only show macro layer block boundaries and stage platforms
                obj.visible = Boolean(uData.isLayerBlock || uData.isStagePlatform || uData.type === 'macro');
                break;

            case 'medium':
                // Show macro layer blocks and major component projections (hide fine head division lines)
                obj.visible = !uData.isFineDetail && !uData.isHeadLine && !uData.isDimensionBadge;
                break;

            case 'near':
            default:
                // Show everything (all sublayers, head lines, dimension badges)
                obj.visible = true;
                break;
        }

        if (obj.visible) visibleCount++;
    });

    return visibleCount;
}
