/**
 * react/components/visualization/CategoryE_SVD.jsx
 * ===============================================
 * SVD / Low-Rank Compression Visualization Component.
 * Implements SVD matrix factorization, singular spectrum decay, rank truncation,
 * low-rank reconstruction, and error heatmap.
 *
 * HARD RULE: Strictly FORBIDS Softmax, QKV, or Dense Attention Matrix.
 */

import React from 'react';
import { useModuleStore } from '../../store/useModuleStore.js';
import { SVD_STATES } from '../../types/module.types.js';
import StateTimeline from '../states/StateTimeline.jsx';
import StateInfoPanel from '../states/StateInfoPanel.jsx';
import StateControls from '../states/StateControls.jsx';

export default function CategoryE_SVD({ moduleId }) {
  const { activeState, activeSubstate, setState, setSubstate } = useModuleStore();

  const stateKey = SVD_STATES[activeState] ? activeState : 'decompose';
  const stateDef = SVD_STATES[stateKey];
  const statesList = Object.keys(SVD_STATES);

  const availableSubstates = [
    'fisher_weights_visible',
    'ln_recalibration_visible',
    'energy_threshold_visible',
  ];

  return (
    <div className="category-view category-svd" data-testid="category-E" data-module={moduleId}>
      <div className="category-header">
        <span className="category-badge">Category E · Low-Rank SVD Compression</span>
        <h3 className="module-title">{moduleId}</h3>
      </div>

      <StateTimeline
        states={statesList}
        current={stateKey}
        onChange={setState}
        labels={statesList.map(s => SVD_STATES[s]?.title || s)}
      />

      <StateInfoPanel
        title={stateDef.title}
        formula={stateDef.formula}
        description={stateDef.description}
        shapes={stateDef.shapes}
        flops={stateDef.flops}
      />

      <div className="svd-visualization-stage" data-testid="svd-stage">
        {stateKey === 'singular_values' && (
          <div className="svd-spectrum-container" data-testid="svd-spectrum-view">
            <div className="spectrum-label">Singular Value Spectrum (Log Scale)</div>
            <div className="spectrum-bars">
              {[100, 72, 45, 28, 16, 9, 5, 2.5, 1.2, 0.6].map((h, i) => (
                <div
                  key={i}
                  className={`spectrum-bar ${i < 4 ? 'retained' : 'truncated'}`}
                  style={{ height: `${h}%` }}
                  title={`σ_${i + 1} = ${(h * 0.05).toFixed(3)}`}
                />
              ))}
            </div>
          </div>
        )}

        {stateKey === 'error_heatmap' && (
          <div className="svd-error-container" data-testid="error-heatmap-view">
            <div className="error-title">Frobenius Residual Error: ||W - W_r||_F</div>
            <div className="error-matrix-grid">
              {Array.from({ length: 16 }).map((_, idx) => (
                <div
                  key={idx}
                  className="error-cell"
                  style={{ opacity: 0.15 + ((idx * 17) % 70) / 100 }}
                />
              ))}
            </div>
          </div>
        )}

        {stateKey === 'truncate' && (
          <div className="svd-factor-triplet" data-testid="svd-factor-triplet">
            <div className="matrix-slab slab-u">U_r (768×r)</div>
            <div className="matrix-slab slab-sigma">Σ_r (r×r)</div>
            <div className="matrix-slab slab-v">V_r^T (r×768)</div>
          </div>
        )}

        {/* Substate Overlays */}
        {activeSubstate === 'fisher_weights_visible' && (
          <div className="substate-overlay fisher-overlay" data-testid="substate-fisher-weights">
            <div className="substate-tag">⚡ Fisher Information Matrix Weighting: diag(F)^(1/2)</div>
          </div>
        )}

        {activeSubstate === 'ln_recalibration_visible' && (
          <div className="substate-overlay ln-overlay" data-testid="substate-ln-recal">
            <div className="substate-tag">🔄 LayerNorm Variance Recalibration Active</div>
          </div>
        )}

        {activeSubstate === 'energy_threshold_visible' && (
          <div className="substate-overlay energy-overlay" data-testid="substate-energy-threshold">
            <div className="substate-tag">📊 Adaptive Energy Retention Target: 95.0%</div>
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
