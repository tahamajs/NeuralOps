/**
 * react/components/layout/Inspector.jsx
 * =====================================
 * Right-hand inspector panel holding architectural metadata, benchmarks, and comparison.
 */

import React from 'react';
import { useModuleStore } from '../../store/useModuleStore.js';
import DetailCard from '../cards/DetailCard.jsx';
import BenchmarkCard from '../cards/BenchmarkCard.jsx';
import CompareCard from '../cards/CompareCard.jsx';

export default function Inspector() {
  const { activeModule } = useModuleStore();

  return (
    <aside className="inspector-panel react-inspector" data-testid="inspector-panel" aria-label="Module Technical Inspector">
      <div className="inspector-scroll-wrap">
        <DetailCard moduleId={activeModule} />
        <BenchmarkCard moduleId={activeModule} />
        <CompareCard moduleId={activeModule} />
      </div>
    </aside>
  );
}
