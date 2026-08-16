// src/features/attentionDashboard/components/CircuitTracerPanel.tsx
import React, { useState } from 'react';
import { useAttentionStore } from '../useAttentionStore';
import { CircuitSVDResult } from '../types';
import { Card } from '../../../components/ui/Card';
import { Button } from '../../../components/ui/Button';
import { Badge } from '../../../components/ui/Badge';
import { Slider } from '../../../components/ui/Slider';
import {
  Network,
  Activity,
  GitCommit,
  Sliders,
  Maximize2,
  Minimize2,
  Sparkles,
  Zap,
  TrendingDown,
  Layers,
  Search,
} from 'lucide-react';

export const CircuitTracerPanel: React.FC = () => {
  const {
    circuitSVDs,
    attributionGraph,
    selectedCircuitHead,
    setSelectedCircuitHead,
    circuitEdgeThreshold,
    setCircuitEdgeThreshold,
    headMetrics,
  } = useAttentionStore();

  const [activeCircuitType, setActiveCircuitType] = useState<'QK' | 'OV'>('QK');
  const activeSVD = circuitSVDs.find((c) => c.circuit_type === activeCircuitType) || circuitSVDs[0];

  const maxSingular = activeSVD?.singular_values[0] || 1.0;

  return (
    <div className="space-y-6">
      {/* Top Header & Settings */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl bg-slate-900/70 border border-slate-800 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400 border border-purple-500/20">
            <Network className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-white">Anthropic-Style Circuit Tracer & SVD Eigendecomposition</h3>
            <p className="text-xs text-slate-400">
              Tracing QK (query-key matching) and OV (output-value write) low-rank circuit attributions across layers
            </p>
          </div>
        </div>

        {/* Head Selector & Threshold */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-xs text-slate-300">
            <span className="text-slate-500 font-mono">Target Head:</span>
            <select
              value={`${selectedCircuitHead.layer}-${selectedCircuitHead.head}`}
              onChange={(e) => {
                const [l, h] = e.target.value.split('-').map(Number);
                setSelectedCircuitHead({ layer: l, head: h });
              }}
              className="py-1 px-2.5 bg-slate-800 border border-slate-700 rounded text-xs font-mono text-white focus:outline-none focus:border-purple-500"
            >
              {headMetrics?.points.map((pt) => (
                <option key={`L${pt.layer}H${pt.head}`} value={`${pt.layer}-${pt.head}`}>
                  L{pt.layer}H{pt.head} ({pt.head_type})
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500 font-mono">Edge Cutoff:</span>
            <input
              type="range"
              min="0.1"
              max="0.8"
              step="0.05"
              value={circuitEdgeThreshold}
              onChange={(e) => setCircuitEdgeThreshold(parseFloat(e.target.value))}
              className="w-24 accent-purple-500"
            />
            <span className="text-xs font-mono text-purple-400">{circuitEdgeThreshold}</span>
          </div>
        </div>
      </div>

      {/* Main Grid: Attribution Graph (Left) vs SVD Spectrum & Eigenspace (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Attribution Network Graph */}
        <div className="lg:col-span-2 rounded-xl bg-slate-950 border border-slate-800 p-5 space-y-4 shadow-2xl flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-white flex items-center gap-2">
              <Zap className="w-4 h-4 text-purple-400" />
              Cross-Layer Circuit Attribution Graph (Supernodes & Pathways)
            </h4>
            <div className="flex items-center gap-3 text-xs text-slate-400">
              <div className="flex items-center gap-1">
                <span className="w-3 h-0.5 bg-purple-400" /> QK Edge
              </div>
              <div className="flex items-center gap-1">
                <span className="w-3 h-0.5 bg-cyan-400" /> OV Edge
              </div>
            </div>
          </div>

          {/* Graph Simulation Canvas */}
          <div className="relative w-full h-[380px] bg-slate-900/60 rounded-lg border border-slate-800 overflow-hidden p-6 flex flex-col justify-between">
            {/* Layer Rails */}
            <div className="absolute inset-0 grid grid-rows-4 divide-y divide-slate-800/40 pointer-events-none" />

            {/* SVG Visualizing Cross-Layer Circuit Attribution */}
            <svg className="w-full h-full">
              {/* Render Edges */}
              {attributionGraph?.edges.slice(0, 45).map((edge, idx) => {
                const srcNode = attributionGraph.nodes.find((n) => n.id === edge.source);
                const tgtNode = attributionGraph.nodes.find((n) => n.id === edge.target);
                if (!srcNode || !tgtNode) return null;

                const x1 = ((srcNode.head + 0.5) / 12) * 100;
                const y1 = ((srcNode.layer + 0.5) / 4) * 100;
                const x2 = ((tgtNode.head + 0.5) / 12) * 100;
                const y2 = ((tgtNode.layer + 0.5) / 4) * 100;

                const strokeColor = edge.type === 'qk' ? '#a855f7' : '#06b6d4';

                return (
                  <line
                    key={idx}
                    x1={`${x1}%`}
                    y1={`${y1}%`}
                    x2={`${x2}%`}
                    y2={`${y2}%`}
                    stroke={strokeColor}
                    strokeWidth={edge.weight * 3}
                    strokeOpacity={Math.max(0.2, edge.weight)}
                  />
                );
              })}

              {/* Render Nodes */}
              {attributionGraph?.nodes.slice(0, 36).map((node) => {
                const cx = ((node.head + 0.5) / 12) * 100;
                const cy = ((node.layer + 0.5) / 4) * 100;
                const isSelected = selectedCircuitHead.layer === node.layer && selectedCircuitHead.head === node.head;

                return (
                  <g
                    key={node.id}
                    onClick={() => setSelectedCircuitHead({ layer: node.layer, head: node.head })}
                    className="cursor-pointer transition-transform hover:scale-125"
                  >
                    <circle
                      cx={`${cx}%`}
                      cy={`${cy}%`}
                      r={isSelected ? 10 : node.size * 0.7}
                      fill={node.color}
                      stroke={isSelected ? '#ffffff' : 'transparent'}
                      strokeWidth={isSelected ? 3 : 0}
                      className="filter drop-shadow"
                    />
                    <text
                      x={`${cx}%`}
                      y={`${cy + 4}%`}
                      textAnchor="middle"
                      fill="#94a3b8"
                      fontSize="9"
                      fontFamily="monospace"
                    >
                      {node.id}
                    </text>
                  </g>
                );
              })}
            </svg>

            <div className="flex justify-between items-center text-xs text-slate-500 border-t border-slate-800/80 pt-2 font-mono">
              <span>{attributionGraph?.nodes.length || 0} Supernodes · {attributionGraph?.edges.length || 0} Attributed Edges</span>
              <span>Anthropic circuit-tracer schema compatible</span>
            </div>
          </div>
        </div>

        {/* Right 1 Col: SVD Spectrum & Eigendecomposition */}
        <div className="rounded-xl bg-slate-900/90 border border-slate-800 p-5 flex flex-col justify-between space-y-4 shadow-xl">
          <div>
            {/* Circuit Mode Selector */}
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center gap-1.5 p-1 bg-slate-950 rounded-lg border border-slate-800">
                <button
                  onClick={() => setActiveCircuitType('QK')}
                  className={`px-3 py-1 text-xs font-mono font-bold rounded ${
                    activeCircuitType === 'QK' ? 'bg-purple-600 text-white' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  $W_{`{QK}`}$ Matrix
                </button>
                <button
                  onClick={() => setActiveCircuitType('OV')}
                  className={`px-3 py-1 text-xs font-mono font-bold rounded ${
                    activeCircuitType === 'OV' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  $W_{`{OV}`}$ Matrix
                </button>
              </div>
              <Badge variant="outline">
                L{selectedCircuitHead.layer}H{selectedCircuitHead.head}
              </Badge>
            </div>

            {activeSVD && (
              <div className="space-y-4 mt-4">
                {/* Spectral Metric Stats Grid */}
                <div className="grid grid-cols-2 gap-2">
                  <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase font-mono">Effective Rank $r_{`{eff}`}$</span>
                    <p className="text-base font-mono font-bold text-emerald-400">{activeSVD.effective_rank}</p>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase font-mono">Condition Number $\kappa$</span>
                    <p className="text-base font-mono font-bold text-cyan-400">{activeSVD.condition_number}</p>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase font-mono">Spectral Radius $\rho$</span>
                    <p className="text-base font-mono font-bold text-purple-400">{activeSVD.spectral_radius}</p>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase font-mono">Matrix Trace $\operatorname{Tr}$</span>
                    <p className="text-base font-mono font-bold text-amber-400">{activeSVD.trace}</p>
                  </div>
                </div>

                {/* Singular Value Decay Bars */}
                <div className="space-y-1.5 pt-2">
                  <div className="flex justify-between text-xs text-slate-400 font-mono">
                    <span>Singular Value Spectrum $\sigma_i$ (Top 12)</span>
                    <span>Energy 95% Cutoff</span>
                  </div>
                  <div className="space-y-1 bg-slate-950 p-3 rounded-lg border border-slate-800">
                    {activeSVD.singular_values.slice(0, 8).map((sv, idx) => {
                      const widthPercent = (sv / maxSingular) * 100;
                      const energy = (activeSVD.cumulative_energy[idx] * 100).toFixed(0);
                      return (
                        <div key={idx} className="space-y-0.5">
                          <div className="flex justify-between text-[10px] font-mono text-slate-400">
                            <span>σ_{idx + 1} = {sv}</span>
                            <span className="text-slate-500">{energy}% Cumul.</span>
                          </div>
                          <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all"
                              style={{
                                width: `${widthPercent}%`,
                                backgroundColor: idx < 4 ? '#a855f7' : '#64748b',
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 text-[11px] text-slate-400">
            <strong className="text-white block mb-1">Mechanistic Insight:</strong>
            {activeCircuitType === 'QK'
              ? 'QK circuit singular value spectrum determines how selectively this head routes attention to prior keys.'
              : 'OV circuit eigenvalues quantify the subspace directions written into the residual stream for future layers.'}
          </div>
        </div>
      </div>
    </div>
  );
};
