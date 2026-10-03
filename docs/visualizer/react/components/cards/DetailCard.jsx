/**
 * react/components/cards/DetailCard.jsx
 * =====================================
 * Technical specification card for the active module.
 */

import React from 'react';
import { ROUTING_MATRIX } from '../visualization/routing_matrix.js';

export default function DetailCard({ moduleId }) {
  if (!moduleId) {
    return (
      <div className="inspector-card detail-card empty" data-testid="detail-card">
        <p className="empty-text">No active module selected</p>
      </div>
    );
  }

  const config = ROUTING_MATRIX[moduleId] || {};

  return (
    <div className="inspector-card detail-card" data-testid="detail-card">
      <div className="card-header">
        <span className="card-tag">SPECIFICATION</span>
        <h4 className="card-title">{moduleId}</h4>
      </div>
      <div className="detail-meta-list">
        <div className="meta-row">
          <span className="meta-label">Category:</span>
          <span className="meta-val badge-category">Category {config.category}</span>
        </div>
        <div className="meta-row">
          <span className="meta-label">Subcategory:</span>
          <span className="meta-val">{config.subcategory || 'Standard'}</span>
        </div>
        <div className="meta-row">
          <span className="meta-label">Discrete States:</span>
          <span className="meta-val">{config.states?.length || 0} stages</span>
        </div>
        <div className="meta-row">
          <span className="meta-label">Substates:</span>
          <span className="meta-val">{config.substates?.length ? config.substates.join(', ') : 'None'}</span>
        </div>
      </div>
    </div>
  );
}
