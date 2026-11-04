/**
 * 3d/visual_language.js
 * =====================
 * Canonical Spatial Language, Chromatic Palette, and FLOP-Proportional Timing Engine
 * for 3D Neural Architecture Visualizations across All 9 Architectural Paradigms.
 */

export const COLORS = {
    input: 0x3b82f6,           // Blue
    query: 0xf59e0b,           // Amber
    key: 0x10b981,             // Emerald
    value: 0x8b5cf6,           // Violet
    output: 0xef4444,          // Crimson
    attention: 0x06b6d4,       // Cyan
    state: 0xec4899,           // Fuchsia/Pink (Recurrent State Ribbon)
    gate: 0xfacc15,            // Yellow (Gates f_t, i_t)
    expert: 0xa855f7,          // Purple
    router: 0xf97316,          // Orange
    zero: 0x1e293b,            // Dark Slate (Pruned / Zeroed)
    reconstructed: 0x22d3ee,   // Bright Cyan
    error_positive: 0xf43f5e,  // Rose
    error_negative: 0x3b82f6   // Blue
};

export const DIMENSIONS = {
    sequence_position: 1.0,
    feature_dim: 0.01,
    layer_gap: 3.0,
    matrix_scale: 0.5
};

/**
 * Calculates animation step duration proportional to operation FLOPs.
 * @param {Object} step - Step with flops property
 * @param {number} totalFlops - Total FLOPs across all steps
 * @returns {number} Duration in seconds [0.3s .. 3.0s]
 */
export function stepDuration(step, totalFlops) {
    const baseDuration = 1.0;
    const minDuration = 0.3;
    const maxDuration = 3.0;
    if (!totalFlops || totalFlops <= 0) return baseDuration;
    const ratio = (step.flops || 1) / totalFlops;
    return Math.max(minDuration, Math.min(maxDuration, ratio * 5.0));
}
