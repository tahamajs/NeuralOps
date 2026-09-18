/**
 * educational/formula_renderer.js
 * ================================
 * High-definition LaTeX formula rendering using KaTeX with clipboard copy support.
 */

import { escapeHtml } from '../utils/dom.js';

/**
 * Renders a mathematical equation using KaTeX, with graceful plain-text fallback.
 * @param {object|null} moduleMeta - Module metadata containing math or math_latex
 * @param {string|null} formulaOverride - Optional explicit formula string
 */
export function renderLatexFormula(moduleMeta, formulaOverride = null) {
    const bodyEl = document.getElementById('formula-latex-body');
    if (!bodyEl) return;
    bodyEl.innerHTML = '';

    const math = formulaOverride || (moduleMeta && (moduleMeta.math_latex || moduleMeta.math)) || 'W_{comp} = \\mathcal{C}(W_{dense})';

    // Copy Button
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

    const katexTarget = document.createElement('div');
    bodyEl.appendChild(katexTarget);

    if (window.katex) {
        try {
            window.katex.render(math, katexTarget, { displayMode: true, throwOnError: false });
            return;
        } catch (e) {
            console.warn('[formula_renderer] KaTeX render error:', e);
        }
    }

    katexTarget.innerHTML = `<div style="font-size: 11px; font-weight: 600; color: #38bdf8; font-family: var(--font-mono); line-height: 1.5; word-break: break-all; text-align: center; padding: 4px 0;">$$ ${escapeHtml(math)} $$</div>`;
}
