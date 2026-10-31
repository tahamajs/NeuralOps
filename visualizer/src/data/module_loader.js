/**
 * data/module_loader.js
 * =====================
 * Asynchronously loads modules-catalog.json and modules_components.json.
 */

/**
 * Loads the complete module catalog of 240+ research modules.
 * @param {string} url - Target catalog JSON path
 * @returns {Promise<Array<object>>} List of module catalog entries
 */
export async function loadModuleCatalog(url = 'modules-catalog.json') {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status} fetching module catalog`);
        }
        const data = await response.json();
        return Array.isArray(data) ? data : (data.modules || []);
    } catch (err) {
        console.warn('[module_loader] Could not load modules-catalog.json, trying fallback...', err);
        return [];
    }
}

/**
 * Loads the consolidated component definitions dictionary.
 * @param {string} url - Target components JSON path
 * @returns {Promise<object>} Module components index
 */
export async function loadModuleComponents(url = 'modules_components.json') {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status} fetching module components`);
        }
        const data = await response.json();
        return data;
    } catch (err) {
        console.warn('[module_loader] Could not load modules_components.json:', err);
        return { total: 0, modules: [] };
    }
}

/**
 * Loads a single module's component file from components/<id>.json on demand.
 * @param {string} moduleId - Module identifier
 * @returns {Promise<object|null>}
 */
export async function loadSingleComponent(moduleId) {
    try {
        const response = await fetch(`components/${moduleId}.json`);
        if (!response.ok) return null;
        return await response.json();
    } catch (err) {
        return null;
    }
}
