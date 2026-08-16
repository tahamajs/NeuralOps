// src/features/attentionDashboard/AttentionCanvasSuite.tsx
import React, { useEffect } from 'react';
import { Zone1_ContextBar } from './components/Zone1_ContextBar';
import { Zone2_AttentionCanvas } from './components/Zone2_AttentionCanvas';
import { Zone3_LiveMathPanel } from './components/Zone3_LiveMathPanel';
import { Zone4_ControlPanel } from './components/Zone4_ControlPanel';
import { useAttentionDashboardStore } from './stores/useAttentionDashboardStore';
import { ErrorBoundary } from '../../components/ui/ErrorBoundary';

export const AttentionCanvasSuite: React.FC = () => {
  const {
    context,
    setSelectedLayer,
    setSelectedHead,
  } = useAttentionDashboardStore();

  // Keyboard navigation shortcuts: Arrow keys (Left/Right for Layer, Up/Down for Head)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Avoid intercepting input when user is typing in textarea or inputs
      if (['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) {
        return;
      }

      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        setSelectedLayer(Math.max(0, context.selectedLayer - 1));
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        setSelectedLayer(Math.min(context.numLayers - 1, context.selectedLayer + 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedHead(Math.max(0, context.selectedHead - 1));
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedHead(Math.min(context.numHeads - 1, context.selectedHead + 1));
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [context.selectedLayer, context.selectedHead, context.numLayers, context.numHeads, setSelectedLayer, setSelectedHead]);

  return (
    <div className="min-h-screen w-full bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-white">
      {/* ZONE 1: Global Navigation & Context Bar */}
      <ErrorBoundary title="Failed to render Context Bar">
        <Zone1_ContextBar />
      </ErrorBoundary>

      {/* Main Workspace Dashboard (Zones 2, 3, 4) */}
      <main className="flex-1 max-w-[1920px] w-full mx-auto p-3 sm:p-4 lg:p-5 flex flex-col lg:flex-row gap-4">
        {/* ZONE 4: Control Panel (Left Orchestration Surface) */}
        <ErrorBoundary title="Failed to render Attention Controls">
          <Zone4_ControlPanel />
        </ErrorBoundary>

        {/* Right Section: Canvas (Zone 2) + Math Panel (Zone 3) */}
        <div className="flex-1 flex flex-col gap-4 min-w-0">
          {/* ZONE 2: Attention Canvas (The Central Visual Microscope) */}
          <ErrorBoundary title="Failed to render Attention Canvas">
            <Zone2_AttentionCanvas />
          </ErrorBoundary>

          {/* ZONE 3: Live Math Panel (3Blue1Brown-Style Mathematical Breakdown) */}
          <ErrorBoundary title="Failed to render Live Math Panel">
            <Zone3_LiveMathPanel />
          </ErrorBoundary>
        </div>
      </main>
    </div>
  );
};

export default AttentionCanvasSuite;
