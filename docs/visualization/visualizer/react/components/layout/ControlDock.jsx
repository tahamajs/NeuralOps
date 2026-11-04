/**
 * react/components/layout/ControlDock.jsx
 * =======================================
 * Bottom dock housing camera perspectives, trajectory speed, and rank sweeps.
 */

import React from 'react';
import { useModuleStore } from '../../store/useModuleStore.js';

export default function ControlDock() {
  const { timelinePosition, setTimeline } = useModuleStore();

  return (
    <div className="control-dock react-control-dock" data-testid="control-dock">
      <div className="timeline-scrubber-group">
        <label htmlFor="global-timeline-slider" className="scrubber-label">
          Continuous Animation Scrub: {(timelinePosition * 100).toFixed(0)}%
        </label>
        <input
          id="global-timeline-slider"
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={timelinePosition}
          onChange={(e) => setTimeline(parseFloat(e.target.value))}
          className="timeline-slider"
          aria-label="Continuous timeline slider"
        />
      </div>
    </div>
  );
}
