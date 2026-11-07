/**
 * react/components/cards/CompareCard.jsx
 * ======================================
 * Direct baseline comparison card vs. Standard Dense Attention (O(S^2)).
 */

import React from 'react';
import { ROUTING_MATRIX } from '../visualization/routing_matrix.js';

export default function CompareCard({ moduleId }) {
  if (!moduleId) return null;
  const config = ROUTING_MATRIX[moduleId] || {};

  return (
    <div className="inspector-card compare-card" data-testid="compare-card">
      <div className="card-header">
        <span className="card-tag">VS. DENSE BASELINE</span>
        <h4 className="card-title">Complexity Profile</h4>
      </div>
      <div className="compare-table">
        <div className="compare-row header-row">
          <span>Dimension</span>
          <span>Module</span>
          <span>Baseline</span>
        </div>
        <div className="compare-row">
          <span>Time Complexity:</span>
          <strong>{config.category === 'D' || config.category === 'B' ? 'O(S · d)' : 'O(S² · d)'}</strong>
          <span className="baseline-val">O(S² · d)</span>
        </div>
        <div className="compare-row">
          <span>Space (KV Cache):</span>
          <strong>{config.category === 'D' ? 'O(1)' : 'O(S · d)'}</strong>
          <span className="baseline-val">O(S · d)</span>
        </div>
        <div className="compare-row">
          <span>Parameters:</span>
          <strong>{config.category === 'E' ? 'r · (d_in + d_out)' : 'd_in · d_out'}</strong>
          <span className="baseline-val">Full Rank</span>
        </div>
      </div>
    </div>
  );
}
