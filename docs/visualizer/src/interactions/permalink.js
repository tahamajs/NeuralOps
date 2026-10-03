/**
 * interactions/permalink.js
 * =========================
 * Deep-linking and Shareable Permalink Manager.
 * Serializes and deserializes application state to/from URL hash:
 * #model=roberta&module=svd_linear_attention&view=3d&layer=3
 * Enables instant deep-linking, camera flight to layer, and inspector restoration.
 */

const VALID_MODELS = ['distilbert', 'roberta', 'gpt2', 'qwen'];
const VALID_VIEWS = ['3d', '2d'];

/**
 * Parses URL hash into a sanitized state object.
 * @param {string} hashStr - Raw hash string (e.g. '#model=roberta&layer=3')
 * @returns {object} Parsed state
 */
export function parseHashToState(hashStr = '') {
    const raw = (hashStr || '').replace(/^#/, '').trim();
    if (!raw) return {};

    const params = new URLSearchParams(raw);
    const result = {};

    if (params.has('model')) {
        const m = params.get('model').toLowerCase();
        if (VALID_MODELS.includes(m)) result.model = m;
    }

    if (params.has('module')) {
        result.module = params.get('module').trim();
    }

    if (params.has('view')) {
        const v = params.get('view').toLowerCase();
        if (VALID_VIEWS.includes(v)) result.view = v;
    }

    if (params.has('layer')) {
        const l = parseInt(params.get('layer'), 10);
        if (!isNaN(l) && l >= 0 && l < 32) result.layer = l;
    }

    if (params.has('rank')) {
        const r = parseInt(params.get('rank'), 10);
        if (!isNaN(r) && r >= 1 && r <= 1024) result.rank = r;
    }

    if (params.has('inspector')) {
        result.inspector = params.get('inspector') === '1' || params.get('inspector') === 'true';
    }

    return result;
}

/**
 * Serializes application state into a canonical URL hash string.
 * @param {object} state - Application state
 * @returns {string} Serialized hash including leading '#'
 */
export function serializeStateToHash(state = {}) {
    const params = new URLSearchParams();

    if (state.model) params.set('model', state.model);
    if (state.module) params.set('module', state.module);
    if (state.view) params.set('view', state.view);
    if (state.layer !== undefined && state.layer !== null) params.set('layer', state.layer.toString());
    if (state.rank) params.set('rank', state.rank.toString());
    if (state.inspector) params.set('inspector', '1');

    const str = params.toString();
    return str ? `#${str}` : '';
}

/**
 * Synchronizes the current window hash without creating new history clutter.
 * @param {object} state - Current application state
 */
export function syncUrlHash(state = {}) {
    if (typeof window === 'undefined') return;
    const hash = serializeStateToHash(state);
    if (window.history && window.history.replaceState) {
        window.history.replaceState(null, '', `${window.location.pathname}${window.location.search}${hash}`);
    } else {
        window.location.hash = hash;
    }
}

/**
 * Copies the current shareable permalink to the user clipboard and triggers a toast.
 * @param {object} state - Current application state
 * @returns {Promise<string>} The copied URL
 */
export async function copyShareablePermalink(state = {}) {
    if (typeof window === 'undefined') return '';
    const hash = serializeStateToHash(state);
    const fullUrl = `${window.location.origin}${window.location.pathname}${window.location.search}${hash}`;

    if (navigator && navigator.clipboard && navigator.clipboard.writeText) {
        try {
            await navigator.clipboard.writeText(fullUrl);
        } catch (_) {
            // Fallback for older browsers
        }
    }

    // Show visual confirmation toast
    let toast = document.querySelector('.permalink-copied');
    if (!toast) {
        toast = document.createElement('div');
        toast.className = 'permalink-copied';
        document.body.appendChild(toast);
    }
    toast.textContent = '🔗 Permalink copied to clipboard!';
    toast.style.display = 'block';
    setTimeout(() => {
        if (toast) toast.remove();
    }, 2500);

    return fullUrl;
}

/**
 * Applies parsed hash state to the live visualizer environment.
 * @param {object} parsedState - State parsed from hash
 * @param {object} context - Engine hooks { store, catalogModules, rebuildSceneForModel, smoothFlyCamera, camera, controls }
 */
export function applyHashStateToVisualizer(parsedState, context) {
    if (!parsedState || !context) return;
    const { store, catalogModules, rebuildSceneForModel, smoothFlyCamera, camera, controls } = context;

    let targetModel = parsedState.model || store.get('currentModel') || 'distilbert';
    let targetModuleName = parsedState.module;
    let foundModule = null;

    if (targetModuleName && catalogModules && catalogModules.length) {
        foundModule = catalogModules.find(m =>
            (m.name && m.name.toLowerCase() === targetModuleName.toLowerCase()) ||
            (m.id && m.id.toLowerCase() === targetModuleName.toLowerCase()) ||
            (m.filename && m.filename.toLowerCase().includes(targetModuleName.toLowerCase()))
        );
    }

    if (foundModule) {
        store.set({ activeResearchModule: foundModule });
    }

    if (parsedState.model) {
        store.set({ currentModel: targetModel });
    }

    // Rebuild scene
    const paradigm = (foundModule && foundModule.paradigm) || 'svd';
    if (typeof rebuildSceneForModel === 'function') {
        rebuildSceneForModel(targetModel, paradigm);
    }

    // Handle 3D / 2D View
    if (parsedState.view) {
        const btn2d = document.getElementById('btn-view-2d');
        const btn3d = document.getElementById('btn-view-3d');
        if (parsedState.view === '2d' && btn2d) btn2d.click();
        else if (parsedState.view === '3d' && btn3d) btn3d.click();
    }

    // Handle target layer camera flight
    if (parsedState.layer !== undefined && typeof smoothFlyCamera === 'function') {
        // Approximate Y altitude per layer (layer height ~ 2.2 units)
        const targetY = 1.0 + parsedState.layer * 2.2;
        setTimeout(() => {
            smoothFlyCamera(camera, controls, targetY, 14.0, 0);
        }, 300);
    }

    // Handle Inspector toggle
    if (parsedState.inspector || foundModule) {
        const inspector = document.getElementById('inspector');
        if (inspector) inspector.classList.remove('hidden');
    }
}
