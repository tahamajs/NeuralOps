/**
 * react/store/useModuleStore.js
 * =============================
 * Lightweight Zustand-compatible reactive state store for 3D Visualizer.
 * Controls active research module, category/subcategory taxonomy, current animation state,
 * timeline scrubber position, and Three.js WebGL scene references.
 */

function createStore(initializer) {
  let state;
  const listeners = new Set();

  const getState = () => state;

  const setState = (partial, replace = false) => {
    const nextState = typeof partial === 'function' ? partial(state) : partial;
    if (!Object.is(nextState, state)) {
      state = replace ? nextState : Object.assign({}, state, nextState);
      listeners.forEach(listener => {
        try {
          listener(state);
        } catch (e) {
          console.error("Error in module store listener:", e);
        }
      });
    }
  };

  const subscribe = (listener) => {
    listeners.add(listener);
    return () => listeners.delete(listener);
  };

  // Callable hook / selector
  const useStore = (selector = (s) => s) => {
    return selector(getState());
  };

  useStore.getState = getState;
  useStore.setState = setState;
  useStore.subscribe = subscribe;

  state = initializer(setState, getState, useStore);
  return useStore;
}

export const useModuleStore = createStore((set, get) => ({
  // Active module identity
  activeModule: null,
  activeCategory: null,
  activeSubcategory: null,

  // Current state within the module's state machine
  activeState: 'idle',
  activeSubstate: null,

  // Timeline position (0.0 to 1.0)
  timelinePosition: 0,

  // Three.js scene & renderer references (imperative bridge)
  threeScene: null,
  threeRenderer: null,

  // Actions
  setActiveModule: (moduleId, taxonomy = {}) => {
    const entry = taxonomy[moduleId] || {};
    const cat = entry.category || null;
    const sub = entry.subcategory || null;

    set({
      activeModule: moduleId,
      activeCategory: cat,
      activeSubcategory: sub,
      activeState: 'idle',
      activeSubstate: null,
      timelinePosition: 0,
    });
  },

  setState: (state) => set({ activeState: state }),
  setSubstate: (sub) => set({ activeSubstate: sub }),
  setTimeline: (pos) => set({ timelinePosition: Math.max(0, Math.min(1, pos)) }),

  setThreeScene: (scene, renderer) =>
    set({ threeScene: scene, threeRenderer: renderer }),

  reset: () =>
    set({
      activeModule: null,
      activeCategory: null,
      activeSubcategory: null,
      activeState: 'idle',
      activeSubstate: null,
      timelinePosition: 0,
      threeScene: null,
      threeRenderer: null,
    }),
}));

export default useModuleStore;
