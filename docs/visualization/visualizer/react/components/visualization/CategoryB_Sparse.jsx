/**
 * react/components/visualization/CategoryB_Sparse.jsx
 * ==================================================
 * Sparse and Approximate Linearized Attention (Performer, Linformer, Nyström, LSH).
 *
 * HARD RULE: Strictly FORBIDS dense S×S attention matrix.
 */

import React from 'react';
import { useModuleStore } from '../../store/useModuleStore.js';
import { SPARSE_STATES } from '../../types/module.types.js';
import StateTimeline from '../states/StateTimeline.jsx';
import StateInfoPanel from '../states/StateInfoPanel.jsx';
import StateControls from '../states/StateControls.jsx';

export default function CategoryB_Sparse({ moduleId }) {
  const { activeState, activeSubstate, setState, setSubstate } = useModuleStore();

  const stateKey = SPARSE_STATES[activeState] ? activeState : 'feature_map';
  const stateDef = SPARSE_STATES[stateKey];
  const statesList = Object.keys(SPARSE_STATES);

  const availableSubstates = [
    'kernel_features_visible',
    'low_rank_projection_visible',
    'landmark_selection_visible',
    'hash_buckets_visible',
  ];

  return (
    <div className="category-view category-sparse" data-testid="category-B" data-module={moduleId}>
      <div className="category-header">
        <span className="category-badge">Category B · Sparse / Approximate Linear Attention</span>
        <h3 className="module-title">{moduleId}</h3>
      </div>

      <StateTimeline
        states={statesList}
        current={stateKey}
        onChange={setState}
        labels={statesList.map(s => SPARSE_STATES[s]?.title || s)}
      />

      <StateInfoPanel
        title={stateDef.title}
        formula={stateDef.formula}
        description={stateDef.description}
        shapes={stateDef.shapes}
        flops={stateDef.flops}
      />

      <div className="sparse-stage" data-testid="sparse-stage">
        {stateKey === 'feature_map' && (
          <div className="feature-map-container" data-testid="kernel-feature-map">
            <div className="stage-tag">Positive Orthogonal Random Features: ϕ(x) ∈ ℝ^{'{S × m}'}</div>
            <div className="feature-bars">
              {Array.from({ length: 8 }).map((_, idx) => (
                <div key={idx} className="feature-indicator" style={{ height: `${25 + idx * 8}%` }} />
              ))}
            </div>
          </div>
        )}

        {stateKey === 'associative_reorder' && (
          <div className="reorder-formula-block" data-testid="associative-reorder-block">
            <div className="stage-tag">Associative Parenthesization: ϕ(Q) · (ϕ(K)^T · V)</div>
            <div className="reorder-boxes">
              <span className="box-left">ϕ(Q) [S×m]</span>
              <span className="op-times">×</span>
              <span className="box-right">KV_summary [m×d]</span>
            </div>
          </div>
        )}

        {/* Substate Overlays */}
        {activeSubstate === 'kernel_features_visible' && (
          <div className="substate-overlay" data-testid="substate-kernel-features">
            <div className="substate-tag">🎲 Performer FAVOR+ Random Orthogonal Features</div>
          </div>
        )}

        {activeSubstate === 'low_rank_projection_visible' && (
          <div className="substate-overlay" data-testid="substate-linformer-proj">
            <div className="substate-tag">📉 Linformer Linear E, F Sequence Dimension Projections (k ≪ S)</div>
          </div>
        )}

        {activeSubstate === 'landmark_selection_visible' && (
          <div className="substate-overlay" data-testid="substate-nystrom-landmarks">
            <div className="substate-tag">📍 Nyström Landmark Token Anchor Sampling</div>
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
