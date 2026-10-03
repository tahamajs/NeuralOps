/**
 * react/App.jsx
 * =============
 * Root React application component orchestrating the declarative visualizer UI,
 * routing active research modules, and bridging to the imperative Three.js canvas.
 */

import React, { useState, useEffect, useRef } from 'react';
import { useModuleStore } from './store/useModuleStore.js';
import { useThreeScene } from './hooks/useThreeScene.js';
import { ROUTING_MATRIX } from './components/visualization/routing_matrix.js';

import TopBar from './components/layout/TopBar.jsx';
import MechanismNav from './components/layout/MechanismNav.jsx';
import VisualizationRouter from './components/visualization/VisualizationRouter.jsx';
import ControlDock from './components/layout/ControlDock.jsx';
import Inspector from './components/layout/Inspector.jsx';
import CatalogModal from './components/modals/CatalogModal.jsx';
import LearnModal from './components/modals/LearnModal.jsx';

export default function App() {
  const canvasRef = useRef(null);
  const [catalogOpen, setCatalogOpen] = useState(false);
  const [learnOpen, setLearnOpen] = useState(false);

  const { activeModule, setActiveModule } = useModuleStore();

  // Connect bridge hook to WebGL canvas without touching Three.js rendering code
  useThreeScene(canvasRef);

  // Initialize default research module on mount if unselected
  useEffect(() => {
    if (!activeModule) {
      const defaultMod = 'a3_compressor' in ROUTING_MATRIX ? 'a3_compressor' : Object.keys(ROUTING_MATRIX)[0];
      setActiveModule(defaultMod, ROUTING_MATRIX);
    }
  }, [activeModule, setActiveModule]);

  return (
    <div className="react-visualizer-app" data-testid="react-app">
      <TopBar
        onOpenCatalog={() => setCatalogOpen(true)}
        onOpenLearn={() => setLearnOpen(true)}
      />

      <MechanismNav />

      <main className="main-content-layout">
        <section className="visualization-center-pane">
          {/* Three.js Canvas Container (Mount point if rendered by React, or attached to DOM) */}
          <div className="three-canvas-wrap" data-testid="three-canvas-wrap">
            <canvas ref={canvasRef} id="react-three-canvas" className="webgl-canvas" />
          </div>

          {/* Declarative Module & State Routing Layer */}
          <div className="declarative-module-overlay">
            <VisualizationRouter />
          </div>

          <ControlDock />
        </section>

        <Inspector />
      </main>

      <CatalogModal isOpen={catalogOpen} onClose={() => setCatalogOpen(false)} />
      <LearnModal isOpen={learnOpen} onClose={() => setLearnOpen(false)} />
    </div>
  );
}
