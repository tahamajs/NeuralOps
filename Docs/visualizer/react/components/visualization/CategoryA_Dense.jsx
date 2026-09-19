/**
 * react/components/visualization/CategoryA_Dense.jsx
 * =================================================
 * Full quadratic dense attention component (Standard MHA, GQA, MQA, FlashAttention).
 */

import React from 'react';
import { useModuleStore } from '../../store/useModuleStore.js';
import { DENSE_STATES } from '../../types/module.types.js';
import StateTimeline from '../states/StateTimeline.jsx';
import StateInfoPanel from '../states/StateInfoPanel.jsx';
import StateControls from '../states/StateControls.jsx';

export default function CategoryA_Dense({ moduleId }) {
  const { activeState, activeSubstate, setState, setSubstate } = useModuleStore();

  const stateKey = DENSE_STATES[activeState] ? activeState : 'qkv_projection';
  const stateDef = DENSE_STATES[stateKey];
  const statesList = Object.keys(DENSE_STATES);

  const availableSubstates = [
    'kv_sharing_visible',
    'kv_sharing_collapsed',
    'tiling_blocks_visible',
    'head_split_visible',
  ];

  return (
    <div className="category-view category-dense" data-testid="category-A" data-module={moduleId}>
      <div className="category-header">
        <span className="category-badge">Category A · Quadratic Dense Attention</span>
        <h3 className="module-title">{moduleId}</h3>
      </div>

      <StateTimeline
        states={statesList}
        current={stateKey}
        onChange={setState}
        labels={statesList.map(s => DENSE_STATES[s]?.title || s)}
      />

      <StateInfoPanel
        title={stateDef.title}
        formula={stateDef.formula}
        description={stateDef.description}
        shapes={stateDef.shapes}
        flops={stateDef.flops}
      />

      <div className="dense-attention-stage" data-testid="dense-stage">
        {/* QKV Projection Blocks */}
        {stateKey === 'qkv_projection' && (
          <div className="qkv-projections-bundle" data-testid="qkv-bundle">
            <div className="proj-box proj-q">Query Proj (W_Q)</div>
            <div className="proj-box proj-k">Key Proj (W_K)</div>
            <div className="proj-box proj-v">Value Proj (W_V)</div>
          </div>
        )}

        {/* Attention Matrix Display */}
        {(stateKey === 'score_matrix' || stateKey === 'softmax') && (
          <div className="attention-matrix-container" data-testid="attention-matrix">
            <div className="matrix-title">
              {stateKey === 'score_matrix' ? 'Dot-Product Attention Scores (S × S)' : 'Normalized Softmax Probabilities'}
            </div>
            <div className="attention-heatmap-grid" data-testid="heatmap-grid">
              {Array.from({ length: 16 }).map((_, idx) => (
                <div
                  key={idx}
                  className="heatmap-cell"
                  style={{ opacity: 0.2 + ((idx * 23) % 80) / 100 }}
                />
              ))}
            </div>
          </div>
        )}

        {/* GQA Substate Overlay */}
        {activeSubstate === 'kv_sharing_visible' && (
          <div className="substate-overlay gqa-overlay" data-testid="substate-gqa">
            <div className="substate-tag">👥 Grouped-Query Attention: 8 Query Heads sharing 2 Key-Value Heads</div>
          </div>
        )}

        {/* FlashAttention Tiling Blocks Overlay */}
        {activeSubstate === 'tiling_blocks_visible' && (
          <div className="substate-overlay tiling-overlay" data-testid="substate-flash-tiling">
            <div className="substate-tag">🧱 FlashAttention SRAM Tiling: B_r × B_c Block Online Softmax</div>
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
