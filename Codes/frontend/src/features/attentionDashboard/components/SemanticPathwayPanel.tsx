// src/features/attentionDashboard/components/SemanticPathwayPanel.tsx
import React, { useState } from 'react';
import { useAttentionStore } from '../useAttentionStore';
import { TokenTrajectory } from '../types';
import { Card } from '../../../components/ui/Card';
import { Button } from '../../../components/ui/Button';
import { Badge } from '../../../components/ui/Badge';
import {
  Compass,
  Route,
  Activity,
  Sparkles,
  ArrowRight,
  TrendingUp,
  Share2,
  Info,
} from 'lucide-react';

export const SemanticPathwayPanel: React.FC = () => {
  const { semanticPathways, inputText, setInputText } = useAttentionStore();
  const [selectedTokenIndex, setSelectedTokenIndex] = useState<number | null>(1);
  const [showArcs, setShowArcs] = useState(true);

  if (!semanticPathways) {
    return (
      <div className="p-8 text-center text-slate-400">
        <Sparkles className="w-8 h-8 mx-auto mb-2 animate-spin text-blue-500" />
        <p>Generating Cross-Layer Semantic Pathway Trajectories...</p>
      </div>
    );
  }

  const selectedTrajectory: TokenTrajectory | null =
    selectedTokenIndex !== null && semanticPathways.trajectories[selectedTokenIndex]
      ? semanticPathways.trajectories[selectedTokenIndex]
      : null;

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl bg-slate-900/70 border border-slate-800 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <Route className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-white">Semantic Pathway Cross-Layer Trajectories (IEEE VIS 2025)</h3>
            <p className="text-xs text-slate-400">
              Dimensionality-reduced representation evolution of tokens through layers with attention influence arcs
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowArcs(!showArcs)}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition-colors ${
              showArcs
                ? 'bg-emerald-600/20 border-emerald-500 text-emerald-300'
                : 'bg-slate-800 border-slate-700 text-slate-400'
            }`}
          >
            {showArcs ? '✓ Influence Arcs On' : 'Influence Arcs Off'}
          </button>
          <Badge variant="outline" className="font-mono text-xs">
            PCA Variance: {((semanticPathways.explained_variance_ratio[0] + semanticPathways.explained_variance_ratio[1]) * 100).toFixed(1)}%
          </Badge>
        </div>
      </div>

      {/* Main Grid: Trajectory Canvas (Left) vs Token Details & Velocity (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: 2D Embedding Canvas */}
        <div className="lg:col-span-2 rounded-xl bg-slate-950 border border-slate-800 p-5 space-y-4 shadow-2xl flex flex-col justify-between">
          <div className="flex items-center justify-between">
            {/* Token Selector Pills */}
            <div className="flex flex-wrap gap-1.5 items-center">
              <span className="text-xs text-slate-500 mr-1">Tokens:</span>
              {semanticPathways.tokens.map((tok, idx) => {
                const isSelected = selectedTokenIndex === idx;
                return (
                  <button
                    key={idx}
                    onClick={() => setSelectedTokenIndex(isSelected ? null : idx)}
                    className={`px-2.5 py-1 rounded-md text-xs font-mono transition-all ${
                      isSelected
                        ? 'bg-emerald-500 text-slate-950 font-bold ring-2 ring-emerald-400/50 shadow'
                        : 'bg-slate-900 text-slate-300 hover:bg-slate-800 border border-slate-800'
                    }`}
                  >
                    #{idx} {tok}
                  </button>
                );
              })}
            </div>
          </div>

          {/* 2D Trajectory Simulation Canvas */}
          <div className="relative w-full h-[380px] bg-slate-900/50 rounded-lg border border-slate-800 p-6 flex flex-col justify-between overflow-hidden">
            {/* Background Grid */}
            <div className="absolute inset-0 bg-[radial-gradient(#334155_1px,transparent_1px)] [background-size:16px_16px] opacity-30 pointer-events-none" />

            <svg className="w-full h-full">
              {/* Render Inactive Trajectory Paths */}
              {semanticPathways.trajectories.map((traj, idx) => {
                if (selectedTokenIndex === idx) return null; // Render selected on top
                const pointsStr = traj.coordinates
                  .map(([x, y]) => {
                    const canvasX = ((x + 8) / 16) * 100;
                    const canvasY = ((y + 6) / 12) * 100;
                    return `${canvasX}%,${canvasY}%`;
                  })
                  .join(' ');

                return (
                  <polyline
                    key={traj.token_index}
                    points={pointsStr}
                    fill="none"
                    stroke="#475569"
                    strokeWidth="1.5"
                    strokeOpacity="0.4"
                    strokeDasharray="2,2"
                  />
                );
              })}

              {/* Render Selected Trajectory Path (Glow & High Contrast) */}
              {selectedTrajectory && (
                <>
                  <polyline
                    points={selectedTrajectory.coordinates
                      .map(([x, y]) => `${((x + 8) / 16) * 100}%,${((y + 6) / 12) * 100}%`)
                      .join(' ')}
                    fill="none"
                    stroke="#10b981"
                    strokeWidth="4"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="filter drop-shadow-[0_0_8px_rgba(16,185,129,0.8)]"
                  />

                  {/* Nodes along the trajectory representing Layers */}
                  {selectedTrajectory.coordinates.map(([x, y], lIdx) => {
                    const cx = `${((x + 8) / 16) * 100}%`;
                    const cy = `${((y + 6) / 12) * 100}%`;
                    return (
                      <g key={lIdx}>
                        <circle
                          cx={cx}
                          cy={cy}
                          r={lIdx === 0 ? 6 : lIdx === selectedTrajectory.coordinates.length - 1 ? 7 : 4}
                          fill={lIdx === 0 ? '#3b82f6' : lIdx === selectedTrajectory.coordinates.length - 1 ? '#f59e0b' : '#10b981'}
                          stroke="#ffffff"
                          strokeWidth="1.5"
                        />
                        <text
                          x={cx}
                          y={cy}
                          dx="8"
                          dy="3"
                          fontSize="9"
                          fill="#94a3b8"
                          fontFamily="monospace"
                        >
                          L{lIdx}
                        </text>
                      </g>
                    );
                  })}
                </>
              )}

              {/* Render Influence Arcs */}
              {showArcs &&
                semanticPathways.influence_arcs.map((arc, idx) => {
                  const srcTraj = semanticPathways.trajectories[arc.source_index];
                  const tgtTraj = semanticPathways.trajectories[arc.target_index];
                  if (!srcTraj || !tgtTraj) return null;

                  const srcCoord = srcTraj.coordinates[arc.layer] || srcTraj.coordinates[0];
                  const tgtCoord = tgtTraj.coordinates[arc.layer] || tgtTraj.coordinates[0];

                  const x1 = ((srcCoord[0] + 8) / 16) * 100;
                  const y1 = ((srcCoord[1] + 6) / 12) * 100;
                  const x2 = ((tgtCoord[0] + 8) / 16) * 100;
                  const y2 = ((tgtCoord[1] + 6) / 12) * 100;

                  return (
                    <line
                      key={`arc-${idx}`}
                      x1={`${x1}%`}
                      y1={`${y1}%`}
                      x2={`${x2}%`}
                      y2={`${y2}%`}
                      stroke="#f59e0b"
                      strokeWidth={arc.attention_weight * 2.5}
                      strokeDasharray="4,3"
                      strokeOpacity="0.75"
                    />
                  );
                })}
            </svg>

            <div className="flex justify-between items-center text-xs text-slate-400 border-t border-slate-800/80 pt-2 font-mono">
              <div className="flex items-center gap-4">
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-500" /> Layer 0 (Embedding)</span>
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-500" /> Mid Transformations</span>
                <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-amber-500" /> Final Layer Output</span>
              </div>
              <span className="text-slate-500">Bézier Semantic Flow</span>
            </div>
          </div>
        </div>

        {/* Right 1 Col: Trajectory Metrics & Influence Ranks */}
        <div className="rounded-xl bg-slate-900/90 border border-slate-800 p-5 flex flex-col justify-between space-y-4 shadow-xl">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <h4 className="text-sm font-bold text-white flex items-center gap-2">
                <Compass className="w-4 h-4 text-emerald-400" />
                Selected Token Trajectory
              </h4>
              {selectedTrajectory && (
                <Badge variant="success">
                  "{selectedTrajectory.token}"
                </Badge>
              )}
            </div>

            {selectedTrajectory ? (
              <div className="space-y-4 mt-4">
                {/* Trajectory Stats */}
                <div className="grid grid-cols-2 gap-2">
                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase font-mono">Total Path Length</span>
                    <p className="text-base font-mono font-bold text-white">{selectedTrajectory.total_path_length}</p>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase font-mono">Path Curvature</span>
                    <p className="text-base font-mono font-bold text-emerald-400">{selectedTrajectory.curvature} rad</p>
                  </div>
                </div>

                {/* Layer Velocity Sparkline */}
                <div className="space-y-1.5">
                  <span className="text-xs text-slate-400 font-mono">Layer-to-Layer Displacement Velocity:</span>
                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1.5">
                    {selectedTrajectory.velocities.map((v, lIdx) => (
                      <div key={lIdx} className="flex items-center justify-between text-[11px] font-mono">
                        <span className="text-slate-500">L{lIdx} → L{lIdx + 1}</span>
                        <div className="w-32 bg-slate-800 h-1.5 rounded-full overflow-hidden mx-2">
                          <div
                            className="bg-emerald-400 h-full rounded-full"
                            style={{ width: `${Math.min(100, v * 70)}%` }}
                          />
                        </div>
                        <span className="text-slate-300">{v}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Cross-Token Influence Ranking */}
                <div className="space-y-1.5">
                  <span className="text-xs text-slate-400 font-mono">Attributed Cross-Token Pulls:</span>
                  <div className="space-y-1.5">
                    {semanticPathways.influence_arcs
                      .filter((a) => a.source_index === selectedTrajectory.token_index)
                      .map((arc, idx) => (
                        <div
                          key={idx}
                          className="p-2 rounded bg-slate-950 border border-slate-800 flex items-center justify-between text-xs font-mono"
                        >
                          <div className="flex items-center gap-1.5">
                            <span className="text-amber-400">⚡ Pull by</span>
                            <strong className="text-white">"{arc.target_token}"</strong>
                            <span className="text-slate-500">@L{arc.layer}</span>
                          </div>
                          <span className="text-emerald-400 font-bold">
                            {(arc.attention_weight * 100).toFixed(0)}% attn
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-8 text-center text-slate-500 text-xs">
                Click any token above to trace its cross-layer semantic trajectory.
              </div>
            )}
          </div>

          <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 text-[11px] text-slate-400">
            Tokens with sharp changes in trajectory indicate active contextual disambiguation through attention sublayers.
          </div>
        </div>
      </div>
    </div>
  );
};
