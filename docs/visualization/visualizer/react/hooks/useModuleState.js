/**
 * react/hooks/useModuleState.js
 * =============================
 * Ergonomic hook for accessing and mutating module state transitions and substate flags.
 */

import { useModuleStore } from '../store/useModuleStore.js';
import { ROUTING_MATRIX, validateModuleState } from '../components/visualization/routing_matrix.js';

export function useModuleState() {
  const activeModule = useModuleStore(s => s.activeModule);
  const activeCategory = useModuleStore(s => s.activeCategory);
  const activeSubcategory = useModuleStore(s => s.activeSubcategory);
  const activeState = useModuleStore(s => s.activeState);
  const activeSubstate = useModuleStore(s => s.activeSubstate);
  const timelinePosition = useModuleStore(s => s.timelinePosition);

  const setState = useModuleStore.getState().setState;
  const setSubstate = useModuleStore.getState().setSubstate;
  const setTimeline = useModuleStore.getState().setTimeline;
  const setActiveModule = useModuleStore.getState().setActiveModule;

  const transitionTo = (nextState) => {
    if (activeModule) {
      validateModuleState(activeModule, nextState, activeSubstate);
    }
    setState(nextState);
  };

  const toggleSubstate = (substateName) => {
    const nextSub = activeSubstate === substateName ? null : substateName;
    if (activeModule && nextSub) {
      validateModuleState(activeModule, activeState, nextSub);
    }
    setSubstate(nextSub);
  };

  const currentConfig = activeModule ? ROUTING_MATRIX[activeModule] : null;

  return {
    activeModule,
    activeCategory,
    activeSubcategory,
    activeState,
    activeSubstate,
    timelinePosition,
    currentConfig,
    transitionTo,
    toggleSubstate,
    setTimeline,
    setActiveModule,
  };
}

export default useModuleState;
