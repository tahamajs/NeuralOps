/**
 * react/components/visualization/CategoryH_Quant.jsx
 * ==================================================
 * Precision quantization (FP8, INT8, SmoothQuant) visualization.
 */

import React from 'react';
import { useModuleStore } from '../../store/useModuleStore.js';
import { QUANT_STATES } from '../../types/module.types.js';
import StateTimeline from '../states/StateTimeline.jsx';
import StateInfoPanel from '../states/StateInfoPanel.jsx';
import StateControls from '../states/StateControls.jsx';

export default function CategoryH_Quant({ moduleId }) {
  const { activeState, activeSubstate, setState, setSubstate } = useModuleStore();

  const stateKey = QUANT_STATES[activeState] ? activeState : 'quantize_int8_fp8';
  const stateDef = QUANT_STATES[stateKey];
  const statesList = Object.keys(QUANT_STATES);

  const availableSubstates = [
    'dynamic_scale_visible',
    'smoothing_factors_visible',
  ];

  return (
    <div className="category-view category-quant" data-testid="category-H" data-module={moduleId}>
      <div className="category-header">
        <span className="category-badge">Category H · Precision Quantization (FP8 / INT8)</span>
        <h3 className="module-title">{moduleId}</h3>
      </div>

      <StateTimeline
        states={statesList}
        current={stateKey}
        onChange={setState}
        labels={statesList.map(s => QUANT_STATES[s]?.title || s)}
      />

      <StateInfoPanel
        title={stateDef.title}
        formula={stateDef.formula}
        description={stateDef.description}
        shapes={stateDef.shapes}
        flops={stateDef.flops}
      />

      <div className="quant-stage" data-testid="quant-stage">
        <div className="quant-bit-representation" data-testid="quant-bits">
          <div className="stage-tag">FP8 E4M3 vs FP16 Bit Allocation</div>
          <div className="bit-blocks">
            <span className="bit-chunk bit-sign">1s</span>
            <span className="bit-chunk bit-exp">4e</span>
            <span className="bit-chunk bit-mantissa">3m</span>
            <span className="bit-tag">8-bit Total Footprint</span>
          </div>
        </div>

        {activeSubstate === 'dynamic_scale_visible' && (
          <div className="substate-overlay" data-testid="substate-quant-scale">
            <div className="substate-tag">⚖️ Per-Tensor Dynamic Scaling Factor: s = 448.0 / max(|X|)</div>
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
