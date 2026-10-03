/**
 * state/store.js
 * ==============
 * Observable application store replacing loose global variables with structured state management.
 */

class AppStore {
    constructor() {
        this._state = {
            currentModel: 'distilbert',
            currentCompression: 'svd',
            activeResearchModule: null,
            pinnedModule: null,
            simulationSpeed: 1.0,
            isPlaying: true,
            enablePulses: true,
            activeNodeId: null,
            isTourActive: false,
            tourStep: 0,
            lodLevel: 'mid',
            spatialData: null,
            modulesCatalog: [],
            modulesComponents: null,
        };
        this._listeners = new Set();
    }

    /**
     * Gets a snapshot value of a state key.
     * @param {string} key - State property name
     * @returns {any}
     */
    get(key) {
        return this._state[key];
    }

    /**
     * Returns the full readonly state object.
     * @returns {Readonly<object>}
     */
    getState() {
        return { ...this._state };
    }

    /**
     * Updates one or multiple state keys and notifies subscribers.
     * @param {object} updates - Key/value pairs to merge
     */
    set(updates) {
        let changed = false;
        for (const [key, val] of Object.entries(updates)) {
            if (this._state[key] !== val) {
                this._state[key] = val;
                changed = true;
            }
        }
        if (changed) {
            this._notify();
        }
    }

    /**
     * Subscribes a callback to state mutations.
     * @param {Function} listener - Callback taking (state)
     * @returns {Function} Unsubscribe function
     */
    subscribe(listener) {
        this._listeners.add(listener);
        return () => this._listeners.delete(listener);
    }

    _notify() {
        const snapshot = this.getState();
        for (const listener of this._listeners) {
            try {
                listener(snapshot);
            } catch (err) {
                console.error('[AppStore] Listener exception:', err);
            }
        }
    }
}

export const store = new AppStore();
