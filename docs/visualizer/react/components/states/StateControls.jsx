/**
 * react/components/states/StateControls.jsx
 * =========================================
 * Scrubbing controls and substate overlays toggle bar.
 */

import React from 'react';
import { useModuleStore } from '../../store/useModuleStore.js';

export default function StateControls({ states = [], current, onChange, substates = [], activeSubstate, onToggleSubstate }) {
  const currentIndex = states.indexOf(current);
  const canPrev = currentIndex > 0;
  const canNext = currentIndex < states.length - 1;

  const handlePrev = () => {
    if (canPrev) onChange(states[currentIndex - 1]);
  };

  const handleNext = () => {
    if (canNext) onChange(states[currentIndex + 1]);
  };

  return (
    <div className="state-controls-dock" data-testid="state-controls">
      <div className="state-stepper-buttons">
        <button
          type="button"
          className="btn-step-prev"
          data-testid="btn-step-prev"
          disabled={!canPrev}
          onClick={handlePrev}
          title="Previous State"
        >
          ◀ Prev
        </button>

        <span className="state-counter" data-testid="state-counter">
          {currentIndex + 1} / {states.length}
        </span>

        <button
          type="button"
          className="btn-step-next"
          data-testid="btn-step-next"
          disabled={!canNext}
          onClick={handleNext}
          title="Next State"
        >
          Next ▶
        </button>
      </div>

      {substates && substates.length > 0 && (
        <div className="substate-toggles" data-testid="substate-toggles">
          <span className="substate-label">Overlays:</span>
          {substates.map(sub => {
            const isActive = activeSubstate === sub;
            return (
              <button
                key={sub}
                type="button"
                className={`btn-substate-toggle ${isActive ? 'active' : ''}`}
                data-testid={`substate-${sub}`}
                onClick={() => onToggleSubstate && onToggleSubstate(sub)}
              >
                {sub.replace(/_/g, ' ')}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
