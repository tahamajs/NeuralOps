/**
 * data/spatial_loader.js
 * ======================
 * Asynchronously loads and validates spatial-data.json.
 */

/**
 * Loads spatial scene data from spatial-data.json.
 * @param {string} url - Target JSON path
 * @returns {Promise<object>} Parsed spatial data object
 */
export async function loadSpatialData(url = 'spatial-data.json') {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status} fetching spatial data`);
        }
        const data = await response.json();
        if (!data.models) {
            throw new Error("Invalid spatial data: missing 'models' root key");
        }
        return data;
    } catch (err) {
        console.error('[spatial_loader] Failed to load spatial data:', err);
        throw err;
    }
}
