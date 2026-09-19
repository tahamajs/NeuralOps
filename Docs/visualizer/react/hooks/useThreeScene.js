/**
 * react/hooks/useThreeScene.js
 * ============================
 * Bridge hook connecting declarative React state management to the imperative Three.js WebGL canvas.
 * Preserves Three.js engine and shaders 100% untouched while passing module state updates.
 */

import { useEffect, useRef } from 'react';
import { useModuleStore } from '../store/useModuleStore.js';

export function useThreeScene(canvasRef) {
  const activeModule = useModuleStore(s => s.activeModule);
  const activeState = useModuleStore(s => s.activeState);
  const timelinePosition = useModuleStore(s => s.timelinePosition);
  const sceneRef = useRef(null);

  // Bind to existing WebGL context and Three.js scene instance
  useEffect(() => {
    if (typeof window === 'undefined') return;

    // Attach to existing Three.js scene if instantiated by visualizer core
    const existingScene = window.__THREE_SCENE__ || window.scene || null;
    const existingRenderer = window.__THREE_RENDERER__ || window.renderer || null;

    if (existingScene && existingRenderer) {
      sceneRef.current = { scene: existingScene, renderer: existingRenderer };
      useModuleStore.getState().setThreeScene(existingScene, existingRenderer);
    }
  }, [canvasRef]);

  // Synchronize state transitions to 3D rendering pipeline
  useEffect(() => {
    if (!activeModule) return;

    if (typeof window !== 'undefined' && typeof window.dispatchEvent === 'function') {
      const event = new CustomEvent('visualizer:module-state-change', {
        detail: {
          moduleId: activeModule,
          state: activeState,
          timelinePosition,
        }
      });
      window.dispatchEvent(event);
    }
  }, [activeModule, activeState, timelinePosition]);

  return sceneRef;
}

export default useThreeScene;
