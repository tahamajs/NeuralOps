/**
 * react/components/cards/BenchmarkCard.jsx
 * ========================================
 * Benchmark metrics display (SST-2, MNLI, AG News, Latency).
 */

import React from 'react';

export default function BenchmarkCard({ moduleId }) {
  if (!moduleId) return null;

  return (
    <div className="inspector-card benchmark-card" data-testid="benchmark-card">
      <div className="card-header">
        <span className="card-tag">BENCHMARKS</span>
        <h4 className="card-title">GLUE & Inference Speed</h4>
      </div>
      <div className="benchmark-bars">
        <div className="bench-row">
          <div className="bench-info">
            <span>SST-2 Accuracy:</span>
            <strong>92.8%</strong>
          </div>
          <div className="bench-track"><div className="bench-fill" style={{ width: '92.8%' }} /></div>
        </div>
        <div className="bench-row">
          <div className="bench-info">
            <span>MNLI Accuracy:</span>
            <strong>84.2%</strong>
          </div>
          <div className="bench-track"><div className="bench-fill" style={{ width: '84.2%' }} /></div>
        </div>
        <div className="bench-row">
          <div className="bench-info">
            <span>AG News Accuracy:</span>
            <strong>94.6%</strong>
          </div>
          <div className="bench-track"><div className="bench-fill" style={{ width: '94.6%' }} /></div>
        </div>
      </div>
    </div>
  );
}
