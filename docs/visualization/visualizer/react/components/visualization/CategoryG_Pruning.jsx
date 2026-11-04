/**
 * react/components/visualization/CategoryG_Pruning.jsx
 * ===================================================
 * Weight and activation structured / unstructured pruning.
 */

import React from 'react';
import { useModuleStore } from '../../store/useModuleStore.js';
import { PRUNING_STATES } from '../../types/module.types.js';
import StateTimeline from '../states/StateTimeline.jsx';
import StateInfoPanel from '../states/StateInfoPanel.jsx';
import StateControls from '../states/StateControls.jsx';

export default function CategoryG_Pruning({ moduleId }) {
  const { activeState, activeSubstate, setState, setSubstate } = useModuleStore();

  const stateKey = PRUNING_STATES[activeState] ? activeState : 'importance_eval';
  const stateDef = PRUNING_STATES[stateKey];
  const statesList = Object.keys(PRUNING_STATES);

  const availableSubstates = [
    'threshold_cutoff_visible',
    'activation_weight_product_visible',
    'hessian_update_visible',
  ];

  return (
    <div className="category-view category-pruning" data-testid="category-G" data-module={moduleId}>
      <div className="category-header">
        <span className="category-badge">Category G · Weight Pruning & Structured Sparsity</span>
        <h3 className="module-title">{moduleId}</h3>
      </div>

      <StateTimeline
        states={statesList}
        current={stateKey}
        onChange={setState}
        labels={statesList.map(s => PRUNING_STATES[s]?.title || s)}
      />

      <StateInfoPanel
        title={stateDef.title}
        formula={stateDef.formula}
        description={stateDef.description}
        shapes={stateDef.shapes}
        flops={stateDef.flops}
      />

      <div className="pruning-stage" data-testid="pruning-stage">
        <div className="sparsity-mask-visual" data-testid="sparsity-mask">
          <div className="stage-tag">2:4 Semi-Structured Sparsity Pattern (50% Speedup)</div>
          <div className="mask-matrix-grid">
            {Array.from({ length: 32 }).map((_, i) => (
              <div
                key={i}
                className={`mask-dot ${i % 2 === 0 ? 'pruned-zero' : 'retained-weight'}`}
              />
            ))}
          </div>
        </div>

        {activeSubstate === 'activation_weight_product_visible' && (
          <div className="substate-overlay" data-testid="substate-wanda">
            <div className="substate-tag">⚡ Wanda Metric: S_{'{ij}'} = |W_{'{ij}'}| · ||X_{'{:j}'}||_2</div>
          </div>
        )}
      </div>

      <StateControls
        states={statesList}
        current={stateKey}
        onChange={setState}
        substates={availableSubstates}
        activeSubstate={activeSubstate}
        onToggleSubstate={sub => setSubstate(activeSubstate === sub ? null : sub)}
      />
    </div>
  );
}
