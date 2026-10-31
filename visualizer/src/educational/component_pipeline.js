/**
 * educational/component_pipeline.js
 * =================================
 * Architectural Component Pipeline & Sublayer Breakdown Generator.
 * Breaks down each module into logical blocks with mathematical KaTeX equations,
 * dimension transformations, and 3D node cross-highlighting.
 */

import { escapeHtml } from '../utils/dom.js';

/**
 * Renders the component pipeline list in the deep inspector drawer.
 * @param {object} moduleMeta - Active module metadata
 * @param {string} modelKey - Selected model ('distilbert' | 'roberta' | 'gpt2' | 'qwen')
 */
export function renderComponentPipeline(moduleMeta, modelKey = 'distilbert') {
    const listEl = document.getElementById('component-pipeline-list');
    const badgeEl = document.getElementById('component-count-badge');
    if (!listEl) return;

    const components = (moduleMeta && moduleMeta.components) ? moduleMeta.components : [];
    if (badgeEl) badgeEl.innerText = `${components.length} BLOCKS`;

    if (!components.length) {
        listEl.innerHTML = '<div style="font-size:11px;color:var(--text-faint);padding:6px 0;">Standard monolithic linear projection.</div>';
        return;
    }

    listEl.innerHTML = components.map((c, i) => `
        <div class="component-pipe-item" data-comp-id="${c.id}" onclick="window.focusPipelineComponent && window.focusPipelineComponent('${c.id}')" title="Click to inspect in 3D">
            <div class="cpi-header">
                <span class="cpi-idx">0${i + 1}</span>
                <strong class="cpi-name" style="color: ${c.color || 'var(--cyan)'};">${escapeHtml(c.name)}</strong>
                <span class="cpi-shape">${escapeHtml(c.shape_transform || '—')}</span>
            </div>
            <p class="cpi-role" style="font-size:10px; color:var(--text-faint); margin:4px 0 2px 0;">${escapeHtml(c.role || '')}</p>
            ${c.math_latex ? `
                <div style="margin-top:6px; padding:4px 8px; background:rgba(6,13,24,0.65); border-radius:5px; border:1px solid rgba(56,189,248,0.2);">
                    <div class="cpi-math" style="font-size:11px; color:#38bdf8; text-align:center;">$$ ${c.math_latex} $$</div>
                </div>
            ` : ''}
        </div>
    `).join('');

    if (window.katex) {
        listEl.querySelectorAll('.cpi-math').forEach(el => {
            const raw = el.textContent.replace(/\$\$/g, '').trim();
            if (raw) {
                try { window.katex.render(raw, el, { displayMode: true, throwOnError: false }); } catch (e) {}
            }
        });
    }
}
