/**
 * react/components/visualization/CategoryC_Window.jsx
 * ==================================================
 * Local and sliding-window banded neighborhood attention.
 */

import React from 'react';
import { useModuleStore } from '../../store/useModuleStore.js';
import { WINDOW_STATES } from '../../types/module.types.js';
import StateTimeline from '../states/StateTimeline.jsx';
import StateInfoPanel from '../states/StateInfoPanel.jsx';
import StateControls from '../states/StateControls.jsx';

export default function CategoryC_Window({ moduleId }) {
  const { activeState, activeSubstate, setState, setSubstate } = useModuleStore();

  const stateKey = WINDOW_STATES[activeState] ? activeState : 'banded_mask';
  const stateDef = WINDOW_STATES[stateKey];
  const statesList = Object.keys(WINDOW_STATES);

  const availableSubstates = [
    'local_window_mask_visible',
    'dynamic_span_visible',
    'dilation_stride_visible',
  ];

  return (
    <div className="category-view category-window" data-testid="category-C" data-module={moduleId}>
      <div className="category-header">
        <span className="category-badge">Category C · Sliding Window Local Attention</span>
        <h3 className="module-title">{moduleId}</h3>
      </div>

      <StateTimeline
        states={statesList}
        current={stateKey}
        onChange={setState}
        labels={statesList.map(s => WINDOW_STATES[s]?.title || s)}
      />

      <StateInfoPanel
        title={stateDef.title}
        formula={stateDef.formula}
        description={stateDef.description}
        shapes={stateDef.shapes}
        flops={stateDef.flops}
      />

      <div className="window-stage" data-testid="window-stage">
        <div className="banded-matrix-display" data-testid="banded-matrix">
          <div className="stage-tag">Banded Diagonal Attention Receptive Field (Window w = 4)</div>
          <div className="banded-grid">
            {Array.from({ length: 64 }).map((_, idx) => {
              const row = Math.floor(idx / 8);
              const col = idx % 8;
              const inBand = Math.abs(row - col) <= 2;
              return (
                <div
                  key={idx}
                  className={`band-cell ${inBand ? 'active-band' : 'masked-cell'}`}
                />
              );
            })}
          </div>
        </div>

        {activeSubstate === 'dynamic_span_visible' && (
          <div className="substate-overlay" data-testid="substate-dynamic-span">
            <div className="substate-tag">📏 Adaptive Span Gating: Head-Specific Dynamic Window Extension</div>
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
