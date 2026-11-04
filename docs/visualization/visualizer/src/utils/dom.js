/**
 * utils/dom.js
 * ============
 * Safe DOM querying, HTML string escaping, and UI feedback utilities.
 */

/**
 * Escapes unsafe characters for direct innerHTML insertion to prevent XSS.
 * @param {string|null|undefined} str - Raw string
 * @returns {string} Sanitized string
 */
export function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

/**
 * Safely finds an element by ID, returning null without throwing.
 * @param {string} id - Element DOM ID
 * @returns {HTMLElement|null}
 */
export function safeGetElement(id) {
    return document.getElementById(id);
}

/**
 * Displays a transient floating toast notification.
 * @param {string} message - Text or emoji message
 * @param {number} durationMs - Milliseconds before auto-removal
 */
export function showToast(message, durationMs = 2500) {
    const toast = document.createElement('div');
    toast.className = 'permalink-copied';
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), durationMs);
}
