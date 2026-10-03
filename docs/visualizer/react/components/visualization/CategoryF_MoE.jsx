/**
 * react/components/visualization/CategoryF_MoE.jsx
 * ===============================================
 * Sparse Mixture of Experts (MoE) routing and parallel expert evaluation.
 */

import React from 'react';
import { useModuleStore } from '../../store/useModuleStore.js';
import { MOE_STATES } from '../../types/module.types.js';
import StateTimeline from '../states/StateTimeline.jsx';
import StateInfoPanel from '../states/StateInfoPanel.jsx';
import StateControls from '../states/StateControls.jsx';

export default function CategoryF_MoE({ moduleId }) {
  const { activeState, activeSubstate, setState, setSubstate } = useModuleStore();

  const stateKey = MOE_STATES[activeState] ? activeState : 'router_gate';
  const stateDef = MOE_STATES[stateKey];
  const statesList = Object.keys(MOE_STATES);

  const availableSubstates = [
    'capacity_limit_visible',
    'routing_probabilities_visible',
  ];

  return (
    <div className="category-view category-moe" data-testid="category-F" data-module={moduleId}>
      <div className="category-header">
        <span className="category-badge">Category F · Sparse Mixture of Experts (MoE)</span>
        <h3 className="module-title">{moduleId}</h3>
      </div>

      <StateTimeline
        states={statesList}
        current={stateKey}
        onChange={setState}
        labels={statesList.map(s => MOE_STATES[s]?.title || s)}
      />

      <StateInfoPanel
        title={stateDef.title}
        formula={stateDef.formula}
        description={stateDef.description}
        shapes={stateDef.shapes}
        flops={stateDef.flops}
      />

      <div className="moe-stage" data-testid="moe-stage">
        <div className="expert-dispatch-grid" data-testid="expert-grid">
          {['Expert 1 (GELU)', 'Expert 2 (SwiGLU)', 'Expert 3 (FFN)', 'Expert 4 (FFN)'].map((name, i) => (
            <div key={i} className={`expert-pod ${i < 2 ? 'activated' : 'idle'}`}>
              <div className="pod-header">{name}</div>
              <div className="pod-status">{i < 2 ? '⚡ Active (Top-2)' : '💤 Dormant'}</div>
            </div>
          ))}
        </div>

        {activeSubstate === 'capacity_limit_visible' && (
          <div className="substate-overlay" data-testid="substate-moe-capacity">
            <div className="substate-tag">⚖️ Expert Capacity Factor: 1.25 × (Tokens / NumExperts)</div>
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
