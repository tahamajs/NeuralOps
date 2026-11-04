/**
 * react/components/layout/TopBar.jsx
 * ==================================
 * Header bar showing Atlas title, module counter, view switcher, and quick picker.
 */

import React from 'react';
import { useModuleStore } from '../../store/useModuleStore.js';
import { ROUTING_MATRIX } from '../visualization/routing_matrix.js';

export default function TopBar({ onOpenCatalog, onOpenLearn }) {
  const { activeModule, setActiveModule } = useModuleStore();
  const moduleKeys = Object.keys(ROUTING_MATRIX);

  return (
    <header className="topbar react-topbar" data-testid="topbar" role="banner">
      <div className="brand-group">
        <div className="eyebrow">NEURAL ARCHITECTURE ATLAS · 252 RESEARCH MODULES</div>
        <h1 className="brand-title">Attention Lab <span className="brand-sub">/ Declarative React Router</span></h1>
      </div>

      <div className="top-actions">
        <div className="quick-select-box">
          <select
            className="quick-module-select"
            value={activeModule || ''}
            onChange={(e) => setActiveModule(e.target.value, ROUTING_MATRIX)}
            aria-label="Quick pick research module"
          >
            <option value="" disabled>⚡ Quick Pick (252 Modules)...</option>
            {moduleKeys.map(k => (
              <option key={k} value={k}>
                [{ROUTING_MATRIX[k].category}] {k}
              </option>
            ))}
          </select>
        </div>

        <button
          type="button"
          className="round-button btn-catalog"
          onClick={onOpenCatalog}
          title="Browse Complete Catalog"
        >
          📚 Catalog
        </button>

        <button
          type="button"
          className="round-button btn-learn"
          onClick={onOpenLearn}
          title="Attention & SVD Primer"
        >
          📖 Learn
        </button>
      </div>
    </header>
  );
}
