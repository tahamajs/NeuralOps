/**
 * ui/module_catalog.js
 * ====================
 * Research Repository Modal & Module Catalog Browser (240+ Modules).
 */

import { escapeHtml } from '../utils/dom.js';

let catalogModules = [];
let activeCategory = 'all';
let activeSort = 'accuracy';
let activeSearchQuery = '';

/**
 * Initializes the module library modal and populates cards.
 * @param {Array<object>} modules - Module catalog array
 * @param {Function} onSelectModule - Callback when a module card is picked
 */
export function initModuleCatalog(modules, onSelectModule) {
    catalogModules = modules || [];

    const modal = document.getElementById('module-modal');
    const openBtn = document.getElementById('btn-open-catalog');
    const closeBtn = document.getElementById('btn-close-modal');
    const searchInput = document.getElementById('modal-search-input');
    const countEl = document.getElementById('modal-module-count');

    if (countEl) countEl.innerText = catalogModules.length;

    if (openBtn && modal) {
        openBtn.addEventListener('click', () => {
            modal.classList.remove('hidden');
            renderModuleCards(onSelectModule);
        });
    }

    if (closeBtn && modal) {
        closeBtn.addEventListener('click', () => {
            modal.classList.add('hidden');
        });
    }

    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            activeSearchQuery = e.target.value.trim().toLowerCase();
            renderModuleCards(onSelectModule);
        });
    }

    // Category pills
    document.querySelectorAll('.cat-pill').forEach(pill => {
        pill.addEventListener('click', () => {
            document.querySelectorAll('.cat-pill').forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            activeCategory = pill.dataset.cat;
            renderModuleCards(onSelectModule);
        });
    });

    // Sort buttons
    document.querySelectorAll('.sort-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeSort = btn.dataset.sort;
            renderModuleCards(onSelectModule);
        });
    });

    // Quick Select Dropdown synchronization
    const quickSelect = document.getElementById('quick-module-select');
    if (quickSelect) {
        quickSelect.innerHTML = '<option value="" disabled selected>⚡ Quick Pick (241 Modules)...</option>';
        catalogModules.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.id;
            opt.textContent = `${m.icon || '⚡'} ${m.name || m.id}`;
            quickSelect.appendChild(opt);
        });
        quickSelect.addEventListener('change', (e) => {
            const chosen = catalogModules.find(m => m.id === e.target.value);
            if (chosen && onSelectModule) onSelectModule(chosen);
        });
    }
}

/**
 * Renders filtered and sorted module cards into the gallery grid.
 * @param {Function} onSelectModule - Callback when a card is selected
 */
export function renderModuleCards(onSelectModule) {
    const grid = document.getElementById('modal-module-grid');
    if (!grid) return;

    let filtered = catalogModules.filter(m => {
        if (activeCategory !== 'all' && m.category !== activeCategory) return false;
        if (activeSearchQuery) {
            const haystack = `${m.name} ${m.id} ${m.category} ${m.description || ''} ${m.file || ''}`.toLowerCase();
            if (!haystack.includes(activeSearchQuery)) return false;
        }
        return true;
    });

    // Sorting
    filtered.sort((a, b) => {
        if (activeSort === 'accuracy') {
            const accA = (a.verified_benchmark && a.verified_benchmark.accuracy_pct) || 0;
            const accB = (b.verified_benchmark && b.verified_benchmark.accuracy_pct) || 0;
            return accB - accA;
        } else if (activeSort === 'savings') {
            const rA = (a.model_projections && a.model_projections.distilbert && a.model_projections.distilbert.savings_ratio) || 1.0;
            const rB = (b.model_projections && b.model_projections.distilbert && b.model_projections.distilbert.savings_ratio) || 1.0;
            return rB - rA;
        } else if (activeSort === 'name') {
            return (a.name || a.id).localeCompare(b.name || b.id);
        }
        return (a.category || '').localeCompare(b.category || '');
    });

    grid.innerHTML = filtered.map(m => {
        const bench = m.verified_benchmark || {};
        const isPass = bench.status === 'PASS';
        return `
            <div class="modal-card" data-module-id="${m.id}">
                <div class="modal-card-head">
                    <span class="mc-icon">${m.icon || '⚡'}</span>
                    <strong>${escapeHtml(m.name || m.id)}</strong>
                    <span class="badge ${isPass ? 'badge-pass' : 'badge-neutral'}">${bench.status || 'VERIFIED'}</span>
                </div>
                <p class="mc-desc">${escapeHtml(m.description || '')}</p>
                <div class="mc-footer">
                    <span class="mc-cat">${escapeHtml(m.category || '')}</span>
                    <div style="display:flex;gap:6px;">
                        <button class="round-button btn-view-doc" data-module-id="${m.id}" style="padding:4px 8px;font-size:10px;background:rgba(56,189,248,0.15);color:#38bdf8;border:1px solid rgba(56,189,248,0.4);">📖 Doc</button>
                        <button class="round-button btn-apply" data-module-id="${m.id}" style="padding:4px 8px;font-size:10px;">Apply</button>
                    </div>
                </div>
            </div>
        `;
    }).join('');

    // Bind card click triggers & Doc reader triggers
    grid.querySelectorAll('.btn-view-doc').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const modId = btn.dataset.moduleId;
            const target = catalogModules.find(m => m.id === modId);
            if (target && window.openModuleDoc) {
                window.openModuleDoc(target);
            }
        });
    });

    grid.querySelectorAll('.modal-card').forEach(card => {
        card.addEventListener('click', (e) => {
            const modId = card.dataset.moduleId;
            const target = catalogModules.find(m => m.id === modId);
            if (target && onSelectModule) {
                onSelectModule(target);
                const modal = document.getElementById('module-modal');
                if (modal) modal.classList.add('hidden');
            }
        });
    });
}
