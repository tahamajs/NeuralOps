/**
 * react/components/modals/CatalogModal.jsx
 * ========================================
 * Searchable modal listing all 252 architectural research modules with category filters.
 */

import React, { useState } from 'react';
import { useModuleStore } from '../../store/useModuleStore.js';
import { ROUTING_MATRIX } from '../visualization/routing_matrix.js';

export default function CatalogModal({ isOpen, onClose }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('ALL');
  const { setActiveModule } = useModuleStore();

  if (!isOpen) return null;

  const entries = Object.entries(ROUTING_MATRIX).filter(([id, config]) => {
    const matchesSearch = id.toLowerCase().includes(searchTerm.toLowerCase()) ||
                          (config.name && config.name.toLowerCase().includes(searchTerm.toLowerCase()));
    const matchesCat = categoryFilter === 'ALL' || config.category === categoryFilter;
    return matchesSearch && matchesCat;
  });

  const handleSelect = (modId) => {
    setActiveModule(modId, ROUTING_MATRIX);
    onClose();
  };

  return (
    <div className="modal-backdrop react-modal" data-testid="catalog-modal" role="dialog" aria-modal="true">
      <div className="modal-container">
        <div className="modal-header">
          <h3>Architecture Catalog (252 Modules)</h3>
          <button type="button" className="btn-close-modal" onClick={onClose} aria-label="Close modal">✕</button>
        </div>

        <div className="catalog-filters">
          <input
            type="text"
            className="input-search-catalog"
            placeholder="Search module name, author, or keyword..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />

          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            className="filter-category-select"
          >
            <option value="ALL">All Categories (A–I)</option>
            {['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'].map(c => (
              <option key={c} value={c}>Category {c}</option>
            ))}
          </select>
        </div>

        <div className="catalog-grid-scroll">
          {entries.map(([id, config]) => (
            <div
              key={id}
              className="catalog-card-item"
              onClick={() => handleSelect(id)}
              tabIndex={0}
              role="button"
            >
              <div className="item-header">
                <span className="item-badge">Cat {config.category}</span>
                <span className="item-id">{id}</span>
              </div>
              <p className="item-desc">{config.name || id}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
