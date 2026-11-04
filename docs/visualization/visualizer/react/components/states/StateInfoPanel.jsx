/**
 * react/components/states/StateInfoPanel.jsx
 * ==========================================
 * Card displaying mathematical formulations, LaTeX, descriptions, and tensor shapes.
 */

import React from 'react';

export default function StateInfoPanel({ title = '', formula = '', description = '', shapes = [], flops = null }) {
  const renderFormula = (latex) => {
    if (typeof window !== 'undefined' && window.katex && typeof window.katex.renderToString === 'function') {
      try {
        return window.katex.renderToString(latex, { throwOnError: false, displayMode: true });
      } catch (e) {
        // Fallback to raw text
      }
    }
    return `<code class="raw-latex">\${latex}</code>`;
  };

  return (
    <aside className="state-info-panel" data-testid="state-info-panel" aria-label="State Mathematical Analysis">
      <div className="state-info-header">
        <h4 className="state-info-title">{title}</h4>
        {flops ? <span className="state-flops-tag">{(flops / 1e6).toFixed(2)} MFLOPs</span> : null}
      </div>

      <div
        className="state-formula"
        data-testid="state-formula"
        dangerouslySetInnerHTML={{ __html: renderFormula(formula) }}
      />

      <p className="state-description">{description}</p>

      {shapes && shapes.length > 0 && (
        <div className="state-shapes" data-testid="state-shapes">
          <span className="state-shapes-label">Tensors:</span>
          {shapes.map((s, idx) => (
            <span key={idx} className="shape-chip" data-testid="shape-chip">
              {s}
            </span>
          ))}
        </div>
      )}
    </aside>
  );
}
