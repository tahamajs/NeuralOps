/**
 * react/components/visualization/VisualizationRouter.jsx
 * ======================================================
 * Central declarative category and state router.
 * Routes each of the 252 catalog modules to its dedicated category component tree,
 * validates against forbidden states, and manages substate rendering.
 */

import React from 'react';
import { useModuleStore } from '../../store/useModuleStore.js';
import { ROUTING_MATRIX, validateModuleState } from './routing_matrix.js';

import CategoryA_Dense from './CategoryA_Dense.jsx';
import CategoryB_Sparse from './CategoryB_Sparse.jsx';
import CategoryC_Window from './CategoryC_Window.jsx';
import CategoryD_Recurrent from './CategoryD_Recurrent.jsx';
import CategoryE_SVD from './CategoryE_SVD.jsx';
import CategoryF_MoE from './CategoryF_MoE.jsx';
import CategoryG_Pruning from './CategoryG_Pruning.jsx';
import CategoryH_Quant from './CategoryH_Quant.jsx';
import CategoryI_Hybrid from './CategoryI_Hybrid.jsx';

export const ROUTES = {
  A: CategoryA_Dense,
  B: CategoryB_Sparse,
  C: CategoryC_Window,
  D: CategoryD_Recurrent,
  E: CategoryE_SVD,
  F: CategoryF_MoE,
  G: CategoryG_Pruning,
  H: CategoryH_Quant,
  I: CategoryI_Hybrid,
};

export default function VisualizationRouter() {
  const { activeCategory, activeModule, activeState, activeSubstate } = useModuleStore();

  if (!activeModule) {
    return (
      <div className="empty-state-router" data-testid="empty-state">
        <p>Select a research module from the taxonomy catalog to inspect architecture states.</p>
      </div>
    );
  }

  // Resolve config from ROUTING_MATRIX
  const config = ROUTING_MATRIX[activeModule];
  const categoryToRender = activeCategory || config?.category || null;

  // Strict invariant validation: if activeState or substate is forbidden, this throws
  if (activeModule && activeState && activeState !== 'idle') {
    validateModuleState(activeModule, activeState, activeSubstate);
  }

  const Component = ROUTES[categoryToRender];
  if (!Component) {
    return (
      <div className="unknown-category-router" data-testid="unknown-category">
        <h3>Unrecognized Architecture Category: {categoryToRender}</h3>
      </div>
    );
  }

  return (
    <div className="visualization-router-wrapper" data-testid="visualization-router">
      <Component moduleId={activeModule} />
    </div>
  );
}
