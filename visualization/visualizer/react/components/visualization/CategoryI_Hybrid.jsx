/**
 * react/components/visualization/CategoryI_Hybrid.jsx
 * ===================================================
 * Hybrid architectural routing interleaving Recurrent SSM (Mamba) and Global Multi-Head Attention.
 */

import React from 'react';
import { useModuleStore } from '../../store/useModuleStore.js';
import { HYBRID_STATES } from '../../types/module.types.js';
import StateTimeline from '../states/StateTimeline.jsx';
import StateInfoPanel from '../states/StateInfoPanel.jsx';
import StateControls from '../states/StateControls.jsx';

export default function CategoryI_Hybrid({ moduleId }) {
  const { activeState, activeSubstate, setState, setSubstate } = useModuleStore();

  const stateKey = HYBRID_STATES[activeState] ? activeState : 'hybrid_ssm_step';
  const stateDef = HYBRID_STATES[stateKey];
  const statesList = Object.keys(HYBRID_STATES);

  const availableSubstates = [
    'layer_index_indicator_visible',
    'cross_layer_fusion_visible',
  ];

  return (
    <div className="category-view category-hybrid" data-testid="category-I" data-module={moduleId}>
      <div className="category-header">
        <span className="category-badge">Category I · Hybrid Attention / SSM Composite</span>
        <h3 className="module-title">{moduleId}</h3>
      </div>

      <StateTimeline
        states={statesList}
        current={stateKey}
        onChange={setState}
        labels={statesList.map(s => HYBRID_STATES[s]?.title || s)}
      />

      <StateInfoPanel
        title={stateDef.title}
        formula={stateDef.formula}
        description={stateDef.description}
        shapes={stateDef.shapes}
        flops={stateDef.flops}
      />

      <div className="hybrid-stage" data-testid="hybrid-stage">
        <div className="interleave-ladder" data-testid="interleave-ladder">
          <div className="stage-tag">Interleaved Architecture Layer Schedule</div>
          <div className="ladder-steps">
            <div className="ladder-block block-ssm">Layer 1: Selective SSM (Mamba)</div>
            <div className="ladder-block block-attn">Layer 2: Sliding Window / Global Attn</div>
            <div className="ladder-block block-ssm">Layer 3: Selective SSM (Mamba)</div>
            <div className="ladder-block block-attn">Layer 4: Full Multi-Head Attention</div>
          </div>
        </div>

        {activeSubstate === 'cross_layer_fusion_visible' && (
          <div className="substate-overlay" data-testid="substate-fusion-gate">
            <div className="substate-tag">🔀 Gated Highway Blending: z = σ(W_g x) ⊙ y_{'{ssm}'} + (1 - σ(W_g x)) ⊙ y_{'{attn}'}</div>
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
