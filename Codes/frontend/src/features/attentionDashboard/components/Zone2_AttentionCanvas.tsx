// src/features/attentionDashboard/components/Zone2_AttentionCanvas.tsx
import React, { useState, useMemo } from 'react';
import {
  Grid,
  Layers,
  GitCommit,
  ScatterChart,
  Eye,
  Sliders,
  Download,
  Maximize2,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Sparkles,
  Zap,
  Filter,
  Flame,
  Palette,
  Play,
  Pause,
  SkipForward,
  Info,
} from 'lucide-react';
import { useAttentionDashboardStore } from '../stores/useAttentionDashboardStore';
import { ViewMode, ColorPalette, SpecializationType } from '../types/attention';

// Helper for color mapping
function getColorForWeight(val: number, palette: ColorPalette): string {
  const clamped = Math.max(0, Math.min(1, val));
  if (palette === 'viridis') {
    const r = Math.floor(68 + clamped * 185);
    const g = Math.floor(1 + clamped * 230);
    const b = Math.floor(84 + (1 - clamped) * 100);
    return `rgb(${r}, ${g}, ${b})`;
  } else if (palette === 'blues') {
    const r = Math.floor(15 + (1 - clamped) * 30);
    const g = Math.floor(80 + clamped * 140);
    const b = Math.floor(180 + clamped * 75);
    return `rgb(${r}, ${g}, ${b})`;
  } else if (palette === 'turbo') {
    const r = Math.floor(30 + clamped * 225);
    const g = Math.floor(clamped < 0.5 ? clamped * 2 * 200 : (1 - clamped) * 2 * 200);
    const b = Math.floor((1 - clamped) * 240);
    return `rgb(${r}, ${g}, ${b})`;
  }
  // Default 'heat'
  const r = Math.floor(15 + clamped * 240);
  const g = Math.floor(23 + clamped * 140);
  const b = Math.floor(42 + (1 - clamped) * 160);
  return `rgb(${r}, ${g}, ${b})`;
}

const SPEC_COLORS: Record<SpecializationType, { bg: string; text: string; border: string }> = {
  'Positional Head': { bg: 'bg-blue-500/20', text: 'text-blue-400', border: 'border-blue-500/50' },
  'Induction Head': { bg: 'bg-purple-500/20', text: 'text-purple-400', border: 'border-purple-500/50' },
  'Semantic Head': { bg: 'bg-rose-500/20', text: 'text-rose-400', border: 'border-rose-500/50' },
  'Delimiter Head': { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/50' },
  'Broad Context Head': { bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500/50' },
};

export const Zone2_AttentionCanvas: React.FC = () => {
  const {
    context,
    headViewData,
    modelViewData,
    sankeyData,
    qkEmbeddingData,
    selection,
    setActiveView,
    setColorPalette,
    setMaskDiagonal,
    setMultiHeadOverlay,
    setMultiHeadMode,
    setDiffMode,
    setSelectedLayer,
    setSelectedHead,
    setHoveredCell,
    setSelectedQueryToken,
    setSelectedKeyToken,
    togglePinPair,
  } = useAttentionDashboardStore();

  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [generationStep, setGenerationStep] = useState<number>(context.tokens.length);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [hoveredHeadCard, setHoveredHeadCard] = useState<{ layer: number; head: number } | null>(null);

  const tokens = context.tokens;
  const N = tokens.length;
  const matrix = headViewData.matrix;

  // Autoregressive token generation stream timer
  useEffect(() => {
    let timer: any = null;
    if (isGenerating) {
      timer = setInterval(() => {
        setGenerationStep((prev) => (prev >= tokens.length ? 1 : prev + 1));
      }, 750);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isGenerating, tokens.length]);

  // Multi-view tabs
  const VIEW_TABS: Array<{ id: ViewMode; label: string; icon: React.ReactNode }> = [
    { id: 'head', label: 'Head View', icon: <Grid className="w-3.5 h-3.5" /> },
    { id: 'model', label: 'Model View', icon: <Layers className="w-3.5 h-3.5" /> },
    { id: 'sankey', label: 'Sankey Flow', icon: <GitCommit className="w-3.5 h-3.5" /> },
    { id: 'qk', label: 'Q/K Joint', icon: <ScatterChart className="w-3.5 h-3.5" /> },
    { id: '3d', label: '3D Topology', icon: <Sparkles className="w-3.5 h-3.5" /> },
    { id: 'generation', label: 'Stream Gen', icon: <Play className="w-3.5 h-3.5" /> },
  ];

  const exportCSV = () => {
    const header = ['Query_Token', ...tokens].join(',');
    const rows = matrix.map((row, i) => [tokens[i], ...row].join(',')).join('\n');
    const csvContent = 'data:text/csv;charset=utf-8,' + header + '\n' + rows;
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `attention_L${context.selectedLayer}_H${context.selectedHead}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="flex-1 flex flex-col bg-slate-900/60 rounded-xl border border-slate-800 shadow-xl overflow-hidden min-w-0">
      {/* Top Toolbar */}
      <div className="flex flex-wrap items-center justify-between px-4 py-2.5 bg-slate-950/70 border-b border-slate-800/80 gap-3">
        {/* View Mode Switcher */}
        <div className="flex items-center space-x-1 p-1 bg-slate-900 rounded-lg border border-slate-800">
          {VIEW_TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveView(tab.id)}
              className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                context.activeView === tab.id
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              {tab.icon}
              <span>{tab.label}</span>
            </button>
          ))}
        </div>

        {/* View Settings & Modifiers */}
        <div className="flex items-center space-x-2 text-xs">
          {/* Palette Selector */}
          <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded bg-slate-900 border border-slate-800">
            <Palette className="w-3.5 h-3.5 text-slate-400" />
            <select
              value={context.colorPalette}
              onChange={(e) => setColorPalette(e.target.value as ColorPalette)}
              className="bg-transparent text-slate-300 focus:outline-none cursor-pointer"
            >
              <option value="heat" className="bg-slate-900">Heat Palette</option>
              <option value="viridis" className="bg-slate-900">Viridis</option>
              <option value="blues" className="bg-slate-900">Blues</option>
              <option value="turbo" className="bg-slate-900">Turbo</option>
            </select>
          </div>

          {/* Mask Diagonal Toggle */}
          {context.activeView === 'head' && (
            <button
              onClick={() => setMaskDiagonal(!context.maskDiagonal)}
              className={`flex items-center space-x-1 px-2.5 py-1 rounded border transition-colors ${
                context.maskDiagonal
                  ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                  : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'
              }`}
              title="Mask self-attention diagonal to highlight cross-token relations"
            >
              <Filter className="w-3 h-3" />
              <span>Mask Diag</span>
            </button>
          )}

          {/* Multi-head Overlay */}
          {context.activeView === 'head' && (
            <button
              onClick={() => setMultiHeadOverlay(!context.multiHeadOverlay)}
              className={`flex items-center space-x-1 px-2.5 py-1 rounded border transition-colors ${
                context.multiHeadOverlay
                  ? 'bg-purple-500/20 text-purple-300 border-purple-500/40'
                  : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'
              }`}
            >
              <Layers className="w-3 h-3" />
              <span>Overlay (3H)</span>
            </button>
          )}

          {/* Diff Mode Toggle */}
          <button
            onClick={() => setDiffMode(!context.diffMode)}
            className={`flex items-center space-x-1 px-2.5 py-1 rounded border transition-colors ${
              context.diffMode
                ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'
            }`}
            title="Diff against baseline Softmax MHA"
          >
            <Sliders className="w-3 h-3" />
            <span>Diff Mode</span>
          </button>

          {/* Zoom controls */}
          <div className="flex items-center space-x-0.5 rounded bg-slate-900 border border-slate-800 p-0.5">
            <button
              onClick={() => setZoomLevel((z) => Math.max(0.7, z - 0.15))}
              className="p-1 text-slate-400 hover:text-slate-200"
            >
              <ZoomOut className="w-3 h-3" />
            </button>
            <span className="text-[10px] font-mono px-1 text-slate-400">{Math.round(zoomLevel * 100)}%</span>
            <button
              onClick={() => setZoomLevel((z) => Math.min(2.0, z + 0.15))}
              className="p-1 text-slate-400 hover:text-slate-200"
            >
              <ZoomIn className="w-3 h-3" />
            </button>
          </div>

          {/* Export CSV */}
          <button
            onClick={exportCSV}
            className="flex items-center space-x-1 px-2.5 py-1 rounded bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-300 transition-colors"
            title="Export Matrix as CSV"
          >
            <Download className="w-3 h-3 text-cyan-400" />
            <span>CSV</span>
          </button>
        </div>
      </div>

      {/* Main Canvas Viewport */}
      <div className="flex-1 p-4 overflow-auto flex items-center justify-center relative min-h-[460px]">
        {/* VIEW 1: HEAD VIEW (INTERACTIVE HEATMAP) */}
        {context.activeView === 'head' && (
          <div
            className="flex flex-col items-center justify-center transition-transform duration-200"
            style={{ transform: `scale(${zoomLevel})` }}
          >
            <div className="flex flex-col items-center">
              {/* Key Token Labels (Top Headers) */}
              <div className="flex ml-24 mb-2">
                {tokens.map((tok, j) => (
                  <div
                    key={j}
                    className={`w-10 text-[10px] font-mono font-medium truncate transform -rotate-45 origin-bottom-left transition-colors ${
                      selection.hoveredCell?.key === j || selection.selectedKeyToken === j
                        ? 'text-cyan-400 font-bold'
                        : 'text-slate-400'
                    }`}
                    title={`Key ${j}: ${tok}`}
                  >
                    {tok}
                  </div>
                ))}
              </div>

              {/* Heatmap Grid Rows */}
              <div className="space-y-1">
                {matrix.map((row, i) => (
                  <div key={i} className="flex items-center space-x-1">
                    {/* Query Token Label (Left Header) */}
                    <div
                      onClick={() =>
                        setSelectedQueryToken(selection.selectedQueryToken === i ? null : i)
                      }
                      className={`w-24 text-right pr-2 text-[11px] font-mono truncate cursor-pointer transition-all ${
                        selection.selectedQueryToken === i
                          ? 'text-cyan-300 font-bold bg-cyan-950/40 py-0.5 rounded'
                          : selection.hoveredCell?.query === i
                          ? 'text-cyan-400 font-semibold'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                      title={`Query ${i}: ${tokens[i]} (Click to isolate row)`}
                    >
                      {tokens[i]}
                    </div>

                    {/* Matrix Cells */}
                    {row.map((val, j) => {
                      const isDiag = i === j;
                      const isHovered = selection.hoveredCell?.query === i && selection.hoveredCell?.key === j;
                      const isRowSelected = selection.selectedQueryToken === i;
                      const isKeySelected = selection.selectedKeyToken === j;
                      const isMasked = context.maskDiagonal && isDiag;

                      const displayVal = isMasked ? 0 : val;
                      const bg = getColorForWeight(displayVal, context.colorPalette);

                      return (
                        <div
                          key={j}
                          onMouseEnter={() => setHoveredCell({ query: i, key: j })}
                          onMouseLeave={() => setHoveredCell(null)}
                          onClick={() => togglePinPair(i, j)}
                          className={`w-10 h-10 rounded-sm flex items-center justify-center cursor-pointer transition-all duration-150 relative ${
                            isHovered
                              ? 'ring-2 ring-white scale-110 z-20 shadow-lg'
                              : isRowSelected || isKeySelected
                              ? 'ring-1 ring-cyan-400 opacity-100'
                              : selection.selectedQueryToken !== null && !isRowSelected
                              ? 'opacity-25'
                              : 'opacity-90 hover:opacity-100'
                          }`}
                          style={{ backgroundColor: bg }}
                        >
                          <span
                            className={`text-[9px] font-mono select-none ${
                              val > 0.45 ? 'text-white font-bold' : 'text-slate-300'
                            }`}
                          >
                            {isMasked ? '—' : val.toFixed(2)}
                          </span>

                          {isDiag && !context.maskDiagonal && (
                            <div className="absolute inset-0 border border-white/20 rounded-sm pointer-events-none" />
                          )}
                        </div>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>

            {/* Quick Metrics Bar below heatmap */}
            <div className="mt-6 flex flex-wrap items-center justify-center gap-4 text-xs text-slate-400">
              <div className="flex items-center gap-1.5 bg-slate-950/80 px-3 py-1.5 rounded-lg border border-slate-800">
                <span className="text-slate-500">Entropy:</span>
                <span className="font-mono font-bold text-cyan-400">{headViewData.mean_entropy} bits</span>
              </div>
              <div className="flex items-center gap-1.5 bg-slate-950/80 px-3 py-1.5 rounded-lg border border-slate-800">
                <span className="text-slate-500">Top-3 Concentration:</span>
                <span className="font-mono font-bold text-emerald-400">
                  {(headViewData.concentration_top3 * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center gap-1.5 bg-slate-950/80 px-3 py-1.5 rounded-lg border border-slate-800">
                <span className="text-slate-500">Self-Attn Ratio:</span>
                <span className="font-mono font-bold text-amber-400">
                  {(headViewData.diagonal_ratio * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        )}

        {/* VIEW 2: MODEL VIEW (ALL LAYERS × HEADS BIRD'S EYE) */}
        {context.activeView === 'model' && (
          <div className="w-full max-w-4xl flex flex-col items-center">
            <div className="flex items-center justify-between w-full mb-3 text-xs text-slate-400">
              <span className="font-semibold text-slate-200">All Layers × All Heads Overview</span>
              <div className="flex items-center gap-2">
                {Object.entries(SPEC_COLORS).map(([spec, styles]) => (
                  <span key={spec} className="flex items-center gap-1 text-[11px]">
                    <span className={`w-2.5 h-2.5 rounded-sm border ${styles.bg} ${styles.border}`} />
                    <span className="text-slate-400">{spec.replace(' Head', '')}</span>
                  </span>
                ))}
              </div>
            </div>

            {/* Model Grid */}
            <div className="grid grid-cols-12 gap-2 p-4 bg-slate-950/90 rounded-xl border border-slate-800 w-full">
              {modelViewData.heads.map((hm) => {
                const isCurrent =
                  context.selectedLayer === hm.layer_idx && context.selectedHead === hm.head_idx;
                const specStyle = SPEC_COLORS[hm.specialization] || SPEC_COLORS['Semantic Head'];

                return (
                  <div
                    key={`${hm.layer_idx}_${hm.head_idx}`}
                    onClick={() => {
                      setSelectedLayer(hm.layer_idx);
                      setSelectedHead(hm.head_idx);
                      setActiveView('head');
                    }}
                    onMouseEnter={() => setHoveredHeadCard({ layer: hm.layer_idx, head: hm.head_idx })}
                    onMouseLeave={() => setHoveredHeadCard(null)}
                    className={`aspect-square rounded-lg p-1.5 flex flex-col justify-between cursor-pointer border transition-all relative ${
                      isCurrent
                        ? 'ring-2 ring-cyan-400 border-white bg-cyan-950/60 shadow-lg scale-105 z-10'
                        : `${specStyle.bg} ${specStyle.border} hover:scale-105 hover:border-slate-300`
                    }`}
                  >
                    <div className="flex items-center justify-between text-[9px] font-mono">
                      <span className="text-slate-400">L{hm.layer_idx}</span>
                      <span className="font-bold text-white">H{hm.head_idx}</span>
                    </div>

                    <div className="flex items-center justify-center my-0.5">
                      <div
                        className="rounded-full bg-white/80"
                        style={{
                          width: `${Math.max(6, hm.importance * 18)}px`,
                          height: `${Math.max(6, hm.importance * 18)}px`,
                        }}
                      />
                    </div>

                    <div className="text-[8px] font-mono text-center text-slate-400 truncate">
                      {hm.specialization.split(' ')[0]}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Top Important Heads Summary */}
            <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3 w-full">
              {modelViewData.most_important_heads.map((top, idx) => (
                <div
                  key={idx}
                  onClick={() => {
                    setSelectedLayer(top.layer);
                    setSelectedHead(top.head);
                    setActiveView('head');
                  }}
                  className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800 hover:border-cyan-500/40 cursor-pointer transition-colors text-xs"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono font-bold text-cyan-300">
                      Layer {top.layer} · Head {top.head}
                    </span>
                    <span className="px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-400 text-[10px] font-mono">
                      Imp: {top.importance}
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-400 mt-1">{top.specialization}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* VIEW 3: SANKEY FLOW DIAGRAM */}
        {context.activeView === 'sankey' && (
          <div className="w-full max-w-4xl flex flex-col items-center">
            <div className="text-xs text-slate-400 mb-2 font-medium">
              Multi-Layer Information Flow (Tokens → Intermediate Heads → Outputs)
            </div>
            <svg viewBox="0 0 800 380" className="w-full h-auto max-h-[420px] bg-slate-950/90 rounded-xl border border-slate-800 p-4">
              {/* Render Links with dynamic source and target coordinates */}
              {sankeyData.links.map((link, idx) => {
                const getCoords = (nodeId: string, isSource: boolean) => {
                  if (nodeId.startsWith('input_')) {
                    const i = parseInt(nodeId.replace('input_', ''), 10) || 0;
                    return { x: 120, y: 37 + (i % N) * 32 };
                  }
                  if (nodeId.startsWith('output_')) {
                    const i = parseInt(nodeId.replace('output_', ''), 10) || 0;
                    return { x: 680, y: 37 + (i % N) * 32 };
                  }
                  const match = nodeId.match(/L\d+_H(\d+)/);
                  const h = match ? parseInt(match[1], 10) : 0;
                  const y = 50 + (h % 4) * 70;
                  return isSource ? { x: 510, y } : { x: 420, y };
                };

                const p1 = getCoords(link.source, true);
                const p2 = getCoords(link.target, false);
                const cx1 = p1.x + (p2.x - p1.x) * 0.5;
                const cx2 = p1.x + (p2.x - p1.x) * 0.5;

                return (
                  <path
                    key={idx}
                    d={`M ${p1.x},${p1.y} C ${cx1},${p1.y} ${cx2},${p2.y} ${p2.x},${p2.y}`}
                    fill="none"
                    stroke="url(#flowGradient)"
                    strokeWidth={Math.max(1.5, link.weight * 5)}
                    strokeOpacity={0.65}
                    className="hover:stroke-opacity-100 hover:stroke-cyan-300 transition-all cursor-pointer"
                  >
                    <title>{`${link.source} → ${link.target} (Flow: ${link.weight.toFixed(3)})`}</title>
                  </path>
                );
              })}

              <defs>
                <linearGradient id="flowGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#3b82f6" />
                  <stop offset="50%" stopColor="#8b5cf6" />
                  <stop offset="100%" stopColor="#10b981" />
                </linearGradient>
              </defs>

              {/* Input Nodes Column */}
              <g transform="translate(40, 20)">
                <text x="0" y="0" fill="#94a3b8" fontSize="11" fontWeight="bold">Input Tokens</text>
                {tokens.map((tok, i) => (
                  <g key={i} transform={`translate(0, ${25 + i * 32})`}>
                    <rect width="80" height="24" rx="4" fill="#1e293b" stroke="#3b82f6" strokeWidth="1.5" />
                    <text x="40" y="16" fill="#f8fafc" fontSize="10" fontFamily="monospace" textAnchor="middle">
                      {tok}
                    </text>
                  </g>
                ))}
              </g>

              {/* Intermediate Heads Column */}
              <g transform="translate(420, 20)">
                <text x="0" y="0" fill="#94a3b8" fontSize="11" fontWeight="bold">Attention Heads</text>
                {[0, 1, 2, 3].map((h, i) => (
                  <g key={i} transform={`translate(0, ${30 + i * 70})`}>
                    <rect width="90" height="40" rx="6" fill="#1e293b" stroke="#8b5cf6" strokeWidth="1.5" />
                    <text x="45" y="20" fill="#f8fafc" fontSize="10" fontWeight="bold" fontFamily="monospace" textAnchor="middle">
                      L5 · Head {h}
                    </text>
                    <text x="45" y="32" fill="#c084fc" fontSize="8" textAnchor="middle">
                      SVD Petrov-Galerkin
                    </text>
                  </g>
                ))}
              </g>

              {/* Output Column */}
              <g transform="translate(680, 20)">
                <text x="0" y="0" fill="#94a3b8" fontSize="11" fontWeight="bold">Output States</text>
                {tokens.map((tok, i) => (
                  <g key={i} transform={`translate(0, ${25 + i * 32})`}>
                    <rect width="80" height="24" rx="4" fill="#1e293b" stroke="#10b981" strokeWidth="1.5" />
                    <text x="40" y="16" fill="#f8fafc" fontSize="10" fontFamily="monospace" textAnchor="middle">
                      {tok}
                    </text>
                  </g>
                ))}
              </g>
            </svg>
          </div>
        )}

        {/* VIEW 4: Q/K JOINT EMBEDDING (ATTENTIONVIZ 2D SCATTER) */}
        {context.activeView === 'qk' && (
          <div className="w-full max-w-4xl flex flex-col items-center">
            <div className="flex items-center justify-between w-full mb-2 text-xs text-slate-400">
              <span className="font-semibold text-slate-200">
                Query (●) vs Key (▲) Joint 2D Projections ({qkEmbeddingData.method})
              </span>
              <div className="flex items-center gap-3">
                <span className="flex items-center gap-1">
                  <span className="w-2.5 h-2.5 rounded-full bg-blue-500" />
                  <span>Queries (Q)</span>
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-2.5 h-2.5 rotate-45 bg-rose-500" />
                  <span>Keys (K)</span>
                </span>
              </div>
            </div>

            <svg viewBox="-120 -120 240 240" className="w-full max-w-lg aspect-square bg-slate-950/90 rounded-xl border border-slate-800 p-2">
              {/* Grid Lines */}
              <line x1="-100" y1="0" x2="100" y2="0" stroke="#334155" strokeDasharray="2 2" />
              <line x1="0" y1="-100" x2="0" y2="100" stroke="#334155" strokeDasharray="2 2" />

              {/* Connections between Q and K */}
              {qkEmbeddingData.connections.map((conn, idx) => {
                const qPt = qkEmbeddingData.points.find((p) => p.id === conn.source);
                const kPt = qkEmbeddingData.points.find((p) => p.id === conn.target);
                if (!qPt || !kPt) return null;

                return (
                  <line
                    key={idx}
                    x1={qPt.x}
                    y1={qPt.y}
                    x2={kPt.x}
                    y2={kPt.y}
                    stroke="#06b6d4"
                    strokeWidth={Math.max(1, conn.weight * 4)}
                    strokeOpacity={0.5}
                  />
                );
              })}

              {/* Points */}
              {qkEmbeddingData.points.map((pt) => {
                const isQuery = pt.point_type === 'query';
                return (
                  <g key={pt.id} transform={`translate(${pt.x}, ${pt.y})`} className="cursor-pointer group">
                    {isQuery ? (
                      <circle r="5" fill="#3b82f6" stroke="#ffffff" strokeWidth="1" className="group-hover:scale-150 transition-transform" />
                    ) : (
                      <polygon points="0,-6 6,5 -6,5" fill="#ef4444" stroke="#ffffff" strokeWidth="1" className="group-hover:scale-150 transition-transform" />
                    )}
                    <text x="7" y="3" fill="#cbd5e1" fontSize="6" fontFamily="monospace">
                      {pt.token}
                    </text>
                    <title>{`${pt.id}: ${pt.token} (magnitude: ${pt.magnitude})`}</title>
                  </g>
                );
              })}
            </svg>
          </div>
        )}

        {/* VIEW 5: 3D TOPOLOGY VIEW & PARTICLES */}
        {context.activeView === '3d' && (
          <div className="w-full max-w-4xl flex flex-col items-center justify-center p-6 bg-slate-950/90 rounded-xl border border-slate-800">
            <div className="flex items-center space-x-2 text-cyan-400 mb-4 font-semibold">
              <Sparkles className="w-5 h-5 animate-spin" />
              <span>3D Attention Topology Map (WebGL / Three.js Layer Canvas)</span>
            </div>
            <div className="w-full h-80 relative flex items-center justify-center border border-slate-800 rounded-lg overflow-hidden bg-radial from-slate-900 to-slate-950">
              {/* 3D Stack Cylindrical Simulation */}
              <div className="flex flex-col items-center space-y-6 transform perspective-1000 rotate-x-25">
                {[0, 1, 2].map((layerIdx) => (
                  <div
                    key={layerIdx}
                    className="w-72 h-16 rounded-full border-2 border-cyan-500/40 bg-cyan-950/20 flex items-center justify-around px-4 shadow-[0_0_20px_rgba(6,182,212,0.15)] relative"
                  >
                    <span className="absolute -left-16 text-[11px] font-mono text-cyan-400 font-bold">
                      Layer {layerIdx * 4}
                    </span>
                    {tokens.slice(0, 6).map((t, ti) => (
                      <div
                        key={ti}
                        className="w-3 h-3 rounded-full bg-cyan-400 animate-pulse shadow-[0_0_8px_#06b6d4]"
                        title={`Token: ${t}`}
                      />
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* VIEW 6: STREAM GENERATION TIMELINE */}
        {context.activeView === 'generation' && (
          <div className="w-full max-w-4xl flex flex-col items-center p-6 bg-slate-950/90 rounded-xl border border-slate-800">
            <div className="flex items-center justify-between w-full mb-4">
              <div className="flex items-center space-x-2 text-emerald-400 font-semibold text-xs">
                <Play className="w-4 h-4" />
                <span>Autoregressive Token Generation Step-by-Step Stream</span>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setIsGenerating(!isGenerating)}
                  className="px-3 py-1 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold flex items-center gap-1"
                >
                  {isGenerating ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                  <span>{isGenerating ? 'Pause' : 'Play Stream'}</span>
                </button>
              </div>
            </div>

            {/* Generated Sequence Badges */}
            <div className="flex flex-wrap gap-1.5 p-3 rounded-lg bg-slate-900 border border-slate-800 w-full mb-4">
              {tokens.map((tok, idx) => (
                <span
                  key={idx}
                  className={`px-2.5 py-1 rounded text-xs font-mono transition-all ${
                    idx < generationStep
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                      : 'bg-slate-950 text-slate-600 border border-slate-800'
                  }`}
                >
                  {tok}
                </span>
              ))}
            </div>

            {/* Scrubber slider */}
            <div className="w-full flex items-center space-x-3 text-xs text-slate-400">
              <span>Step 1</span>
              <input
                type="range"
                min={1}
                max={tokens.length}
                value={generationStep}
                onChange={(e) => setGenerationStep(Number(e.target.value))}
                className="flex-1 accent-cyan-400 cursor-pointer"
              />
              <span>Step {tokens.length}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
