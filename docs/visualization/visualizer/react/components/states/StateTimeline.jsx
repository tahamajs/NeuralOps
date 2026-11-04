/**
 * react/components/states/StateTimeline.jsx
 * ========================================
 * State timeline scrubber allowing users to jump directly to any valid discrete state.
 */

import React from 'react';

export default function StateTimeline({ states = [], current, onChange, labels = [] }) {
  return (
    <div className="state-timeline" data-testid="state-timeline" role="tablist" aria-label="Module State Progression">
      {states.map((s, i) => {
        const isActive = s === current;
        const label = labels[i] || s;
        return (
          <button
            key={s}
            type="button"
            className={`state-pill ${isActive ? 'active' : ''}`}
            data-testid={`state-pill-${s}`}
            data-active={isActive ? 'true' : 'false'}
            onClick={() => onChange && onChange(s)}
            title={`Step ${i + 1}: ${label}`}
            aria-selected={isActive}
            role="tab"
          >
            <span className="state-step-num">{i + 1}</span>
            <span className="state-step-name">{label}</span>
          </button>
        );
      })}
    </div>
  );
}
