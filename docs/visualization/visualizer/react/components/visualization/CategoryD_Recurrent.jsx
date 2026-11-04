/**
 * react/components/visualization/CategoryD_Recurrent.jsx
 * =====================================================
 * Recurrent / Linear Attention Visualization Component.
 * Visualizes hidden state memory evolution, elementwise gating, and selective state-space scan.
 *
 * HARD RULE: Strictly FORBIDS Q/K/V projections and S×S attention square.
 */

import React from 'react';
import { useModuleStore } from '../../store/useModuleStore.js';
import { RECURRENT_STATES } from '../../types/module.types.js';
import StateTimeline from '../states/StateTimeline.jsx';
import StateInfoPanel from '../states/StateInfoPanel.jsx';
import StateControls from '../states/StateControls.jsx';

export default function CategoryD_Recurrent({ moduleId }) {
  const { activeState, activeSubstate, setState, setSubstate, timelinePosition } = useModuleStore();

  const stateKey = RECURRENT_STATES[activeState] ? activeState : 't1';
  const stateDef = RECURRENT_STATES[stateKey];
  const statesList = Object.keys(RECURRENT_STATES);

  const availableSubstates = [
    'forget_gate_visible',
    'input_gate_visible',
    'state_space_visible',
    'selective_scan_visible',
    'delta_rule_visible',
  ];

  return (
    <div className="category-view category-recurrent" data-testid="category-D" data-module={moduleId}>
      <div className="category-header">
        <span className="category-badge">Category D · Recurrent State-Space / Linear Sequence</span>
        <h3 className="module-title">{moduleId}</h3>
      </div>

      <StateTimeline
        states={statesList}
        current={stateKey}
        onChange={setState}
        labels={statesList.map(s => `Step ${s}`)}
      />

      <StateInfoPanel
        title={stateDef.title}
        formula={stateDef.formula}
        description={stateDef.description}
        shapes={stateDef.shapes}
        flops={stateDef.flops}
      />

      <div className="recurrent-stage" data-testid="recurrent-stage">
        {/* Recurrent State Evolution Ribbon */}
        <div className="recurrent-ribbon" data-testid="recurrent-ribbon">
          <div className="ribbon-label">Recurrent Memory Trunk: h_t ∈ ℝ^d (O(1) Spatial Footprint)</div>
          <div className="state-vector-strip">
            {['t0', 't1', 't2', 't3', 't4'].map((t, idx) => {
              const isCurrent = stateKey === t;
              return (
                <div
                  key={t}
                  className={`state-cell ${isCurrent ? 'active' : ''}`}
                  data-testid={`recurrent-cell-${t}`}
                  style={{ transform: isCurrent ? 'scale(1.08)' : 'scale(1.0)' }}
                >
                  <span className="cell-step">t={idx}</span>
                  <span className="cell-state">h_{idx}</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Dynamic Gates Chart */}
        <div className="gates-chart" data-testid="gates-chart">
          <div className="gate-metric">
            <span className="gate-name">Forget Gate (f_t):</span>
            <div className="gate-bar-track">
              <div
                className="gate-bar-fill forget-fill"
                style={{ width: `${((stateDef.forgetGate || 0.85) * 100).toFixed(0)}%` }}
              />
            </div>
            <span className="gate-val">{(stateDef.forgetGate || 0.85).toFixed(2)}</span>
          </div>

          <div className="gate-metric">
            <span className="gate-name">Input Modulation Gate (i_t):</span>
            <div className="gate-bar-track">
              <div
                className="gate-bar-fill input-fill"
                style={{ width: `${((stateDef.inputGate || 0.45) * 100).toFixed(0)}%` }}
              />
            </div>
            <span className="gate-val">{(stateDef.inputGate || 0.45).toFixed(2)}</span>
          </div>
        </div>

        {/* Substate Overlays */}
        {activeSubstate === 'forget_gate_visible' && (
          <div className="substate-overlay" data-testid="substate-forget-gate">
            <div className="substate-tag">🛡️ Forget Gate Elementwise Modulation: f_t ⊙ h_{'{t-1}'}</div>
          </div>
        )}

        {activeSubstate === 'selective_scan_visible' && (
          <div className="substate-overlay" data-testid="substate-selective-scan">
            <div className="substate-tag">⚡ Mamba / S6 Selective Scan: Δ_t = Softplus(Parameter + Linear(x_t))</div>
          </div>
        )}

        {activeSubstate === 'delta_rule_visible' && (
          <div className="substate-overlay" data-testid="substate-delta-rule">
            <div className="substate-tag">📐 Delta-Net Associative Retrieval Correction</div>
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
