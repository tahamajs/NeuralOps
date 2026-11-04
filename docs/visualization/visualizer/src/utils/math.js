/**
 * utils/math.js
 * =============
 * Mathematical helpers for string matching, clamping, linear interpolation, and geometry.
 */

/**
 * Checks if a string contains any of the provided search keywords (case-insensitive).
 * @param {string|null|undefined} str - Target string
 * @param {string[]} keywords - List of search tokens
 * @returns {boolean}
 */
export function anyKey(str, keywords) {
    if (!str || !keywords || !keywords.length) return false;
    const lower = String(str).toLowerCase();
    return keywords.some(k => lower.includes(k.toLowerCase()));
}

/**
 * Clamps a numeric value within [min, max].
 * @param {number} val - Input value
 * @param {number} min - Lower bound
 * @param {number} max - Upper bound
 * @returns {number}
 */
export function clamp(val, min, max) {
    return Math.max(min, Math.min(max, val));
}

/**
 * Linear interpolation between a and b.
 * @param {number} a - Start value
 * @param {number} b - End value
 * @param {number} t - Progress factor [0, 1]
 * @returns {number}
 */
export function lerp(a, b, t) {
    return a + (b - a) * t;
}
