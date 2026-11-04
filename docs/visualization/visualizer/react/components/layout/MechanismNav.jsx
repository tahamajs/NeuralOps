/**
 * react/components/layout/MechanismNav.jsx
 * ========================================
 * Mechanism Category Navigator tabs (A through I).
 */

import React from 'react';
import { useModuleStore } from '../../store/useModuleStore.js';
import { ROUTING_MATRIX } from '../visualization/routing_matrix.js';

const CATEGORIES = [
  { id: 'A', label: 'Dense (A)' },
  { id: 'B', label: 'Sparse (B)' },
  { id: 'C', label: 'Window (C)' },
  { id: 'D', label: 'Recurrent (D)' },
  { id: 'E', label: 'SVD / Low-Rank (E)' },
  { id: 'F', label: 'MoE (F)' },
  { id: 'G', label: 'Pruning (G)' },
  { id: 'H', label: 'Quant (H)' },
  { id: 'I', label: 'Hybrid (I)' },
];

export default function MechanismNav() {
  const { activeCategory, setActiveModule } = useModuleStore();

  const handleSelectCategory = (catId) => {
    // Pick first module matching category
    const firstMatch = Object.keys(ROUTING_MATRIX).find(
      k => ROUTING_MATRIX[k].category === catId
    );
    if (firstMatch) {
      setActiveModule(firstMatch, ROUTING_MATRIX);
    }
  };

  return (
    <nav className="mechanism-nav" data-testid="mechanism-nav" role="tablist" aria-label="Architecture Category Tabs">
      {CATEGORIES.map(cat => (
        <button
          key={cat.id}
          type="button"
          role="tab"
          aria-selected={activeCategory === cat.id}
          className={`nav-tab ${activeCategory === cat.id ? 'active' : ''}`}
          data-testid={`cat-tab-${cat.id}`}
          onClick={() => handleSelectCategory(cat.id)}
        >
          {cat.label}
        </button>
      ))}
    </nav>
  );
}
