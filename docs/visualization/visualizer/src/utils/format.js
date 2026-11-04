/**
 * utils/format.js
 * ===============
 * Formatting helpers for metric numbers, parameter counts, memory bytes, and percentages.
 */

/**
 * Formats a raw parameter count into human-readable engineering units (K, M, B).
 * @param {number|null|undefined} num - Raw integer parameter count
 * @returns {string} Formatted string, e.g. "590.6K", "66.36M", "0.50B"
 */
export function formatParamCount(num) {
    if (!num && num !== 0) return '—';
    if (num >= 1e9) return (num / 1e9).toFixed(2) + 'B';
    if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
    if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
    return String(num);
}

/**
 * Formats byte counts into storage units (KB, MB, GB).
 * @param {number|null|undefined} bytes - Raw byte integer
 * @returns {string} Formatted string, e.g. "148.6 MB", "2.36 MB"
 */
export function formatByteCount(bytes) {
    if (!bytes && bytes !== 0) return '—';
    if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + ' GB';
    if (bytes >= 1e6) return (bytes / 1e6).toFixed(1) + ' MB';
    if (bytes >= 1e3) return (bytes / 1e3).toFixed(1) + ' KB';
    return bytes + ' B';
}

/**
 * Formats a compression ratio into a clean display string.
 * @param {number|null|undefined} ratio - Reduction factor (e.g. 2.27)
 * @returns {string} e.g. "2.27x (-56.0%)"
 */
export function formatSavingsRatio(ratio) {
    if (!ratio || ratio <= 1.0) return '1.0x (Baseline)';
    const pct = ((1.0 - (1.0 / ratio)) * 100).toFixed(1);
    return `${ratio.toFixed(2)}x (-${pct}%)`;
}
