// src/features/attentionDashboard/AttentionCanvasSuite.tsx
import React, { useState, useEffect } from 'react';
import { Zone1_ContextBar } from './components/Zone1_ContextBar';
import { Zone2_AttentionCanvas } from './components/Zone2_AttentionCanvas';
import { Zone3_LiveMathPanel } from './components/Zone3_LiveMathPanel';
import { Zone4_ControlPanel } from './components/Zone4_ControlPanel';
import { LogitLensPanel } from './components/LogitLensPanel';
import { CircuitTracerPanel } from './components/CircuitTracerPanel';
import { HeadAblationLab } from './components/HeadAblationLab';
import { SemanticPathwayPanel } from './components/SemanticPathwayPanel';
import { CrossPlatformIntegrationPanel } from './components/CrossPlatformIntegrationPanel';
import { useAttentionDashboardStore } from './stores/useAttentionDashboardStore';
import { ErrorBoundary } from '../../components/ui/ErrorBoundary';
import {
  Sparkles,
  Layers,
  Network,
  Scissors,
  Route,
  Terminal,
  Activity,
  Maximize2,
} from 'lucide-react';

export const AttentionCanvasSuite: React.FC = () => {
  const {
    context,
    setSelectedLayer,
    setSelectedHead,
  } = useAttentionDashboardStore();

  const [activeZone, setActiveZone] = useState<'canvas' | 'logit' | 'circuits' | 'ablation' | 'pathways' | 'triton'>('canvas');

  // Keyboard navigation shortcuts: Arrow keys (Left/Right for Layer, Up/Down for Head)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
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
    <div className="min-h-screen w-full bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-white pb-12">
      {/* ZONE 1: Global Navigation & Context Bar */}
      <ErrorBoundary title="Failed to render Context Bar">
        <Zone1_ContextBar />
      </ErrorBoundary>

      {/* Mode Navigation Tabs */}
      <div className="max-w-[1920px] w-full mx-auto px-4 pt-3 flex items-center justify-between gap-3 border-b border-white/5 pb-2 overflow-x-auto scrollbar-thin">
        <div className="flex items-center gap-1.5 p-1 bg-white/5 rounded-xl border border-white/10">
          {[
            { id: 'canvas', label: 'Attention Canvas & 3B1B Math', icon: Sparkles, color: 'text-cyan-400' },
            { id: 'logit', label: 'Logit Lens & Projections', icon: Layers, color: 'text-purple-400' },
            { id: 'circuits', label: 'Circuit Tracer & Induction', icon: Network, color: 'text-emerald-400' },
            { id: 'ablation', label: 'Head Ablation Lab', icon: Scissors, color: 'text-rose-400' },
            { id: 'pathways', label: 'Semantic Pathways', icon: Route, color: 'text-amber-400' },
            { id: 'triton', label: 'SVD Kernel Code Export', icon: Terminal, color: 'text-blue-400' },
          ].map((tab) => {
            const Icon = tab.icon;
            const isSelected = activeZone === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveZone(tab.id as any)}
                className={`px-3 py-1.5 rounded-lg text-xs font-bold flex items-center gap-2 transition-all whitespace-nowrap ${
                  isSelected
                    ? 'bg-gradient-to-r from-cyan-500/20 to-blue-500/20 border border-cyan-500/40 text-white shadow-lg shadow-cyan-500/10'
                    : 'text-slate-400 hover:text-white hover:bg-white/5'
                }`}
              >
                <Icon size={14} className={tab.color} />
                {tab.label}
              </button>
            );
          })}
        </div>

        <div className="text-[11px] font-mono text-slate-400 flex items-center gap-2">
          <span className="px-2 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/30 text-cyan-300">
            L{context.selectedLayer} : H{context.selectedHead}
          </span>
          <span className="hidden md:inline">Use Arrow Keys (← → ↑ ↓) to step</span>
        </div>
      </div>

      {/* Main Workspace Dynamic Render */}
      <main className="flex-1 max-w-[1920px] w-full mx-auto p-3 sm:p-4 lg:p-5">
        {activeZone === 'canvas' && (
          <div className="flex flex-col lg:flex-row gap-4">
            {/* ZONE 4: Control Panel */}
            <ErrorBoundary title="Failed to render Attention Controls">
              <Zone4_ControlPanel />
            </ErrorBoundary>

            {/* Right Section: Canvas (Zone 2) + Math Panel (Zone 3) */}
            <div className="flex-1 flex flex-col gap-4 min-w-0">
              <ErrorBoundary title="Failed to render Attention Canvas">
                <Zone2_AttentionCanvas />
              </ErrorBoundary>

              <ErrorBoundary title="Failed to render Live Math Panel">
                <Zone3_LiveMathPanel />
              </ErrorBoundary>
            </div>
          </div>
        )}

        {activeZone === 'logit' && (
          <ErrorBoundary title="Failed to render Logit Lens">
            <div className="glass p-4 rounded-2xl border border-white/10">
              <LogitLensPanel />
            </div>
          </ErrorBoundary>
        )}

        {activeZone === 'circuits' && (
          <ErrorBoundary title="Failed to render Circuit Tracer">
            <div className="glass p-4 rounded-2xl border border-white/10">
              <CircuitTracerPanel />
            </div>
          </ErrorBoundary>
        )}

        {activeZone === 'ablation' && (
          <ErrorBoundary title="Failed to render Head Ablation Lab">
            <div className="glass p-4 rounded-2xl border border-white/10">
              <HeadAblationLab />
            </div>
          </ErrorBoundary>
        )}

        {activeZone === 'pathways' && (
          <ErrorBoundary title="Failed to render Semantic Pathways">
            <div className="glass p-4 rounded-2xl border border-white/10">
              <SemanticPathwayPanel />
            </div>
          </ErrorBoundary>
        )}

        {activeZone === 'triton' && (
          <ErrorBoundary title="Failed to render Triton Export">
            <div className="glass p-4 rounded-2xl border border-white/10">
              <CrossPlatformIntegrationPanel />
            </div>
          </ErrorBoundary>
        )}
      </main>
    </div>
  );
};

export default AttentionCanvasSuite;
