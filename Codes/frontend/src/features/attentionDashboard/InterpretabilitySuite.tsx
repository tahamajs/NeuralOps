import React, { useState } from 'react';
import { useAttentionStore } from './useAttentionStore';
import { HeadScatter3DPanel } from './components/HeadScatter3DPanel';
import { HeadAblationLab } from './components/HeadAblationLab';
import { CircuitTracerPanel } from './components/CircuitTracerPanel';
import { SemanticPathwayPanel } from './components/SemanticPathwayPanel';
import { LogitLensPanel } from './components/LogitLensPanel';
import { SocialHubPanel } from './components/SocialHubPanel';
import { GamificationSidebar } from './components/GamificationSidebar';
import { CrossPlatformIntegrationPanel } from './components/CrossPlatformIntegrationPanel';
import { Button } from '../../components/ui/Button';
import { Badge } from '../../components/ui/Badge';
import { ErrorBoundary } from '../../components/ui/ErrorBoundary';
import {
  Layers,
  Scissors,
  Network,
  Route,
  Eye,
  Users,
  Trophy,
  Cpu,
  Sparkles,
  RefreshCw,
  Zap,
  Flame,
  ChevronRight,
  Sidebar,
} from 'lucide-react';

export const InterpretabilitySuite: React.FC = () => {
  const {
    selectedModel,
    setSelectedModel,
    selectedAttentionType,
    setSelectedAttentionType,
    inputText,
    setInputText,
    selectedLayer,
    selectedHead,
    activeZoneTab,
    setActiveZoneTab,
    recomputeAnalytics,
    ablatedHeads,
    gamification,
  } = useAttentionStore();

  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 sm:p-6 space-y-6">
      {/* Top Context Bar: Model, Attention Variant & Input Text */}
      <div className="p-4 sm:p-5 rounded-2xl bg-gradient-to-r from-slate-900 via-slate-900/90 to-slate-950 border border-slate-800 shadow-2xl backdrop-blur-xl space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white shadow-lg shadow-blue-500/20">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold text-white tracking-tight">Attention Analytics & Interpretability Suite</h1>
                <Badge variant="outline" className="text-[10px] font-mono border-blue-500/40 text-blue-400">
                  Zones 5-9 Master Lab
                </Badge>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Quantitative Diagnostics · Mechanistic Circuits · Ablation Sensitivity · Social Arena · Gamified Progress
              </p>
            </div>
          </div>

          {/* Quick Stats Toolbar */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono">
              <span className="text-slate-500">Active:</span>
              <span className="text-white font-bold">L{selectedLayer}H{selectedHead}</span>
            </div>

            {ablatedHeads.size > 0 && (
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-950/40 border border-rose-500/40 text-xs font-mono text-rose-300">
                <Scissors className="w-3.5 h-3.5" />
                <span>{ablatedHeads.size} Masked</span>
              </div>
            )}

            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className={`p-2 rounded-lg border text-xs font-mono flex items-center gap-1.5 transition-all ${
                sidebarOpen
                  ? 'bg-amber-500/20 border-amber-500/50 text-amber-300'
                  : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-white'
              }`}
              title="Toggle Gamification & Quest Sidebar"
            >
              <Flame className="w-4 h-4 text-amber-500" />
              <span className="hidden sm:inline">{gamification.currentXP} XP</span>
            </button>
          </div>
        </div>

        {/* Input Parameters Row */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2 border-t border-slate-800/80">
          {/* Model Selector */}
          <div>
            <label className="text-[11px] font-mono uppercase text-slate-400 block mb-1">Architecture / Model</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full py-2 px-3 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-white focus:outline-none focus:border-blue-500"
            >
              <option value="bert-base-uncased">bert-base-uncased (12L × 12H, 110M params)</option>
              <option value="gpt2-medium">gpt2-medium (24L × 16H, 355M params)</option>
              <option value="llama-3.2-1b">llama-3.2-1b (16L × 32H, GQA Transcoder)</option>
              <option value="pythia-160m">pythia-160m (12L × 12H, Pretrained CLT)</option>
            </select>
          </div>

          {/* Attention Variant */}
          <div>
            <label className="text-[11px] font-mono uppercase text-slate-400 block mb-1">Attention Mechanism</label>
            <select
              value={selectedAttentionType}
              onChange={(e) => setSelectedAttentionType(e.target.value)}
              className="w-full py-2 px-3 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-white focus:outline-none focus:border-blue-500"
            >
              <option value="Galerkin Attention (Rank 64)">Galerkin Attention (SVD Rank 64)</option>
              <option value="TLinFormer (Linear Spectral)">TLinFormer (Linear Spectral Projection)</option>
              <option value="PADRe (Rank Adaptive Dynamic)">PADRe (Rank Adaptive Dynamic)</option>
              <option value="Spectraformer SVD">Spectraformer (Truncated SVD)</option>
              <option value="Standard Softmax MHA">Standard Scaled Dot-Product MHA</option>
            </select>
          </div>

          {/* Input Text Prompt */}
          <div>
            <label className="text-[11px] font-mono uppercase text-slate-400 block mb-1">Input Text / Context</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="flex-1 py-2 px-3 bg-slate-950 border border-slate-800 rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:border-blue-500"
                placeholder="Type prompt to re-evaluate attention..."
              />
              <Button size="sm" variant="outline" onClick={recomputeAnalytics} className="text-xs h-9 px-3">
                <RefreshCw className="w-3.5 h-3.5" />
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs Header */}
      <div className="flex flex-wrap items-center gap-2 p-1.5 rounded-xl bg-slate-900/80 border border-slate-800 overflow-x-auto">
        <button
          onClick={() => setActiveZoneTab('metrics')}
          className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
            activeZoneTab === 'metrics'
              ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
          }`}
        >
          <Layers className="w-4 h-4" /> 3D Diagnostics & Scatter
        </button>

        <button
          onClick={() => setActiveZoneTab('ablation')}
          className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
            activeZoneTab === 'ablation'
              ? 'bg-rose-600 text-white shadow-lg shadow-rose-500/20'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
          }`}
        >
          <Scissors className="w-4 h-4" /> Ablation Laboratory
        </button>

        <button
          onClick={() => setActiveZoneTab('circuits')}
          className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
            activeZoneTab === 'circuits'
              ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/20'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
          }`}
        >
          <Network className="w-4 h-4" /> QK / OV Circuit Tracer
        </button>

        <button
          onClick={() => setActiveZoneTab('pathways')}
          className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
            activeZoneTab === 'pathways'
              ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-500/20'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
          }`}
        >
          <Route className="w-4 h-4" /> Semantic Pathways
        </button>

        <button
          onClick={() => setActiveZoneTab('logitlens')}
          className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
            activeZoneTab === 'logitlens'
              ? 'bg-amber-600 text-white shadow-lg shadow-amber-500/20'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
          }`}
        >
          <Eye className="w-4 h-4" /> Logit Lens Timeline
        </button>

        <button
          onClick={() => setActiveZoneTab('social')}
          className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
            activeZoneTab === 'social'
              ? 'bg-pink-600 text-white shadow-lg shadow-pink-500/20'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
          }`}
        >
          <Users className="w-4 h-4" /> Social & Arena Hub
        </button>

        <button
          onClick={() => setActiveZoneTab('integrations')}
          className={`px-4 py-2 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all ${
            activeZoneTab === 'integrations'
              ? 'bg-cyan-600 text-white shadow-lg shadow-cyan-500/20'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
          }`}
        >
          <Cpu className="w-4 h-4" /> Pipelines & Export
        </button>
      </div>

      {/* Main Content Layout with Responsive Sidebar */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 items-start">
        {/* Main Panel Content (3 or 4 columns based on sidebar toggle) */}
        <div className={sidebarOpen ? 'lg:col-span-3 space-y-6' : 'lg:col-span-4 space-y-6'}>
          {activeZoneTab === 'metrics' && (
            <ErrorBoundary title="Failed to render 3D Diagnostics & Scatter">
              <HeadScatter3DPanel />
            </ErrorBoundary>
          )}
          {activeZoneTab === 'ablation' && (
            <ErrorBoundary title="Failed to render Ablation Laboratory">
              <HeadAblationLab />
            </ErrorBoundary>
          )}
          {activeZoneTab === 'circuits' && (
            <ErrorBoundary title="Failed to render Circuit Tracer Panel">
              <CircuitTracerPanel />
            </ErrorBoundary>
          )}
          {activeZoneTab === 'pathways' && (
            <ErrorBoundary title="Failed to render Semantic Pathways Panel">
              <SemanticPathwayPanel />
            </ErrorBoundary>
          )}
          {activeZoneTab === 'logitlens' && (
            <ErrorBoundary title="Failed to render Logit Lens Timeline">
              <LogitLensPanel />
            </ErrorBoundary>
          )}
          {activeZoneTab === 'social' && (
            <ErrorBoundary title="Failed to render Social Hub Panel">
              <SocialHubPanel />
            </ErrorBoundary>
          )}
          {activeZoneTab === 'integrations' && (
            <ErrorBoundary title="Failed to render Integrations & Export Panel">
              <CrossPlatformIntegrationPanel />
            </ErrorBoundary>
          )}
        </div>

        {/* Zone 8 Gamification & Quest Sidebar */}
        {sidebarOpen && (
          <div className="lg:col-span-1">
            <ErrorBoundary title="Failed to render Quest Sidebar">
              <GamificationSidebar />
            </ErrorBoundary>
          </div>
        )}
      </div>
    </div>
  );
};
export default InterpretabilitySuite;
