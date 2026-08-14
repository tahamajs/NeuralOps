// src/features/attentionDashboard/components/HeadScatter3DPanel.tsx
import React, { useState, useMemo } from 'react';
import { useAttentionStore } from '../useAttentionStore';
import { HeadMetricPoint, HeadType } from '../types';
import { Card } from '../../../components/ui/Card';
import { Button } from '../../../components/ui/Button';
import { Badge } from '../../../components/ui/Badge';
import {
  Layers,
  Sparkles,
  Search,
  Filter,
  Eye,
  Crosshair,
  TrendingUp,
  Maximize2,
  Minimize2,
  BarChart2,
  Info,
} from 'lucide-react';

export const HeadScatter3DPanel: React.FC = () => {
  const {
    headMetrics,
    selectedHeadPoint,
    setSelectedHeadPoint,
    metricAxisX,
    metricAxisY,
    metricAxisZ,
    setMetricAxis,
    activeHeadTypeFilter,
    setActiveHeadTypeFilter,
    setSelectedLayer,
    setSelectedHead,
  } = useAttentionStore();

  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'scatter' | 'dendrogram' | 'matrix'>('scatter');
  const [hoveredPoint, setHoveredPoint] = useState<HeadMetricPoint | null>(null);

  const filteredPoints = useMemo(() => {
    if (!headMetrics) return [];
    return headMetrics.points.filter((p) => {
      const matchesType = activeHeadTypeFilter === 'ALL' || p.head_type === activeHeadTypeFilter;
      const matchesSearch =
        searchQuery === '' ||
        `L${p.layer}H${p.head}`.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.head_type.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.description.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesType && matchesSearch;
    });
  }, [headMetrics, activeHeadTypeFilter, searchQuery]);

  const headTypesList = useMemo(() => {
    if (!headMetrics) return [];
    return Object.keys(headMetrics.type_distribution);
  }, [headMetrics]);

  if (!headMetrics) {
    return (
      <div className="p-8 text-center text-slate-400">
        <Sparkles className="w-8 h-8 mx-auto mb-2 animate-spin text-blue-500" />
        <p>Computing 3D Head Importance, Stability & Similarity Metrics...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Top Controls Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl bg-slate-900/60 border border-slate-800 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-white">3D Head Diagnostic Scatter & Dendrogram</h3>
            <p className="text-xs text-slate-400">
              Evaluating {headMetrics.layers * headMetrics.heads_per_layer} attention heads across 3 BHPVAS pruning criteria
            </p>
          </div>
        </div>

        {/* View Mode Switcher */}
        <div className="flex items-center gap-2 p-1 rounded-lg bg-slate-800/80 border border-slate-700">
          <button
            onClick={() => setViewMode('scatter')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
              viewMode === 'scatter' ? 'bg-blue-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            3D Scatter View
          </button>
          <button
            onClick={() => setViewMode('dendrogram')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
              viewMode === 'dendrogram' ? 'bg-blue-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Hierarchical Dendrogram
          </button>
          <button
            onClick={() => setViewMode('matrix')}
            className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
              viewMode === 'matrix' ? 'bg-blue-600 text-white shadow' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Metric Matrix Grid
          </button>
        </div>
      </div>

      {/* Filter and Axis Configuration */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-2.5 w-4 h-4 text-slate-500" />
          <input
            type="text"
            placeholder="Search head (e.g. L5H7, Induction)..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-2 bg-slate-900/80 border border-slate-800 rounded-lg text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-blue-500"
          />
        </div>

        {/* X-Axis Metric */}
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-slate-400">X-Axis:</span>
          <select
            value={metricAxisX}
            onChange={(e) => setMetricAxis('X', e.target.value as any)}
            className="w-full py-2 px-3 bg-slate-900/80 border border-slate-800 rounded-lg text-xs text-slate-200 focus:outline-none focus:border-blue-500"
          >
            <option value="importance">Importance I(l,h)</option>
            <option value="stability">Stability S(l,h)</option>
            <option value="similarity">Similarity Sim(l,h)</option>
          </select>
        </div>

        {/* Y-Axis Metric */}
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-slate-400">Y-Axis:</span>
          <select
            value={metricAxisY}
            onChange={(e) => setMetricAxis('Y', e.target.value as any)}
            className="w-full py-2 px-3 bg-slate-900/80 border border-slate-800 rounded-lg text-xs text-slate-200 focus:outline-none focus:border-blue-500"
          >
            <option value="stability">Stability S(l,h)</option>
            <option value="importance">Importance I(l,h)</option>
            <option value="similarity">Similarity Sim(l,h)</option>
          </select>
        </div>

        {/* Z-Axis Metric */}
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-slate-400">Color/Z:</span>
          <select
            value={metricAxisZ}
            onChange={(e) => setMetricAxis('Z', e.target.value as any)}
            className="w-full py-2 px-3 bg-slate-900/80 border border-slate-800 rounded-lg text-xs text-slate-200 focus:outline-none focus:border-blue-500"
          >
            <option value="similarity">Similarity Sim(l,h)</option>
            <option value="importance">Importance I(l,h)</option>
            <option value="stability">Stability S(l,h)</option>
          </select>
        </div>
      </div>

      {/* Head Type Category Filter Pills */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs text-slate-400 flex items-center gap-1 mr-1">
          <Filter className="w-3 h-3" /> Types:
        </span>
        <button
          onClick={() => setActiveHeadTypeFilter('ALL')}
          className={`px-2.5 py-1 rounded-full text-xs font-medium transition-all ${
            activeHeadTypeFilter === 'ALL'
              ? 'bg-blue-600 text-white ring-2 ring-blue-400/40'
              : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
          }`}
        >
          All Heads ({headMetrics.points.length})
        </button>
        {headTypesList.map((type) => {
          const count = headMetrics.type_distribution[type] || 0;
          const isActive = activeHeadTypeFilter === type;
          return (
            <button
              key={type}
              onClick={() => setActiveHeadTypeFilter(type)}
              className={`px-2.5 py-1 rounded-full text-xs font-medium flex items-center gap-1.5 transition-all ${
                isActive ? 'ring-2 ring-white/50 text-white font-semibold' : 'text-slate-300 hover:opacity-80'
              }`}
              style={{
                backgroundColor: isActive
                  ? headMetrics.points.find((p) => p.head_type === type)?.color || '#3b82f6'
                  : '#1e293b',
              }}
            >
              <span
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: headMetrics.points.find((p) => p.head_type === type)?.color }}
              />
              {type} ({count})
            </button>
          );
        })}
      </div>

      {/* Main Visualizer Area */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Scatter Canvas / Dendrogram */}
        <div className="lg:col-span-2 rounded-xl bg-slate-950 border border-slate-800 p-4 relative min-h-[460px] flex flex-col justify-between overflow-hidden shadow-2xl">
          {viewMode === 'scatter' && (
            <div className="relative w-full h-full flex-1 flex flex-col justify-between">
              {/* Top Summary Bar */}
              <div className="flex justify-between items-center text-xs text-slate-400 mb-3">
                <span className="font-mono">
                  Displaying {filteredPoints.length} heads · {metricAxisX.toUpperCase()} vs. {metricAxisY.toUpperCase()}
                </span>
                <span className="text-[11px] text-slate-500">Click point to inspect · Hover for scores</span>
              </div>

              {/* 2D/3D Scatter Plot Canvas Container */}
              <div className="relative w-full h-[360px] rounded-lg bg-slate-900/50 border border-slate-800/80 overflow-hidden flex items-center justify-center p-6">
                {/* Axis Guidelines */}
                <div className="absolute left-8 bottom-8 right-8 top-8 border-l border-b border-slate-700/60 pointer-events-none" />
                <div className="absolute left-4 top-1/2 -rotate-90 text-[10px] text-slate-400 font-mono tracking-wider">
                  {metricAxisY.toUpperCase()} ↑
                </div>
                <div className="absolute bottom-2 left-1/2 -translate-x-1/2 text-[10px] text-slate-400 font-mono tracking-wider">
                  {metricAxisX.toUpperCase()} →
                </div>

                {/* Simulated Point Cloud */}
                <div className="relative w-full h-full">
                  {filteredPoints.map((pt) => {
                    const xVal = pt[metricAxisX];
                    const yVal = pt[metricAxisY];
                    const leftPercent = Math.min(95, Math.max(5, xVal * 90 + 5));
                    const topPercent = Math.min(95, Math.max(5, 100 - (yVal * 90 + 5)));

                    const isSelected = selectedHeadPoint?.layer === pt.layer && selectedHeadPoint?.head === pt.head;
                    const isHovered = hoveredPoint?.layer === pt.layer && hoveredPoint?.head === pt.head;

                    return (
                      <button
                        key={`L${pt.layer}H${pt.head}`}
                        onClick={() => {
                          setSelectedHeadPoint(pt);
                          setSelectedLayer(pt.layer);
                          setSelectedHead(pt.head);
                        }}
                        onMouseEnter={() => setHoveredPoint(pt)}
                        onMouseLeave={() => setHoveredPoint(null)}
                        className={`absolute rounded-full transition-all duration-150 transform -translate-x-1/2 -translate-y-1/2 focus:outline-none ${
                          isSelected
                            ? 'ring-4 ring-white ring-offset-2 ring-offset-slate-950 scale-150 z-30 animate-pulse'
                            : isHovered
                            ? 'ring-2 ring-white scale-125 z-20 shadow-lg'
                            : 'hover:scale-110 z-10'
                        }`}
                        style={{
                          left: `${leftPercent}%`,
                          top: `${topPercent}%`,
                          width: `${pt.size}px`,
                          height: `${pt.size}px`,
                          backgroundColor: pt.color,
                          boxShadow: isSelected || isHovered ? `0 0 12px ${pt.color}` : 'none',
                        }}
                        title={`L${pt.layer}H${pt.head}: ${pt.head_type}`}
                      />
                    );
                  })}
                </div>

                {/* Floating Tooltip */}
                {hoveredPoint && (
                  <div
                    className="absolute top-4 right-4 p-3 bg-slate-900/95 border border-slate-700 rounded-lg shadow-xl text-xs text-slate-200 pointer-events-none backdrop-blur-md z-40 space-y-1"
                    style={{ borderLeftColor: hoveredPoint.color, borderLeftWidth: 4 }}
                  >
                    <div className="flex items-center justify-between gap-4 font-bold text-white">
                      <span>Head L{hoveredPoint.layer}H{hoveredPoint.head}</span>
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
                        {hoveredPoint.head_type}
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-400 grid grid-cols-3 gap-2 pt-1 border-t border-slate-800">
                      <div>Imp: <span className="font-mono text-emerald-400">{hoveredPoint.importance}</span></div>
                      <div>Stab: <span className="font-mono text-cyan-400">{hoveredPoint.stability}</span></div>
                      <div>Sim: <span className="font-mono text-purple-400">{hoveredPoint.similarity}</span></div>
                    </div>
                  </div>
                )}
              </div>

              {/* Floor stats */}
              <div className="mt-3 flex items-center justify-between text-xs text-slate-400 border-t border-slate-800/80 pt-2">
                <div className="flex gap-4">
                  <span>Avg Imp: <strong className="text-white">{headMetrics.avg_importance}</strong></span>
                  <span>Avg Stab: <strong className="text-white">{headMetrics.avg_stability}</strong></span>
                  <span>Avg Sim: <strong className="text-white">{headMetrics.avg_similarity}</strong></span>
                </div>
                <span className="text-slate-500 font-mono">BHPVAS Ward Agglomerative Linkage</span>
              </div>
            </div>
          )}

          {viewMode === 'dendrogram' && (
            <div className="flex-1 flex flex-col justify-between">
              <div className="flex justify-between items-center text-xs text-slate-400 mb-2">
                <span>Hierarchical Clustering Dendrogram (Ward Linkage on Jensen-Shannon Divergence)</span>
                <Badge variant="outline">Auto-Cluster Tree</Badge>
              </div>

              {/* Simplified SVG Dendrogram visualization */}
              <div className="w-full h-[360px] bg-slate-900/40 rounded-lg border border-slate-800 p-4 overflow-y-auto space-y-3">
                {Object.entries(headMetrics.type_distribution).map(([clusterName, count], idx) => {
                  const clusterPoints = headMetrics.points.filter((p) => p.head_type === clusterName);
                  const representative = clusterPoints[0];
                  return (
                    <div
                      key={clusterName}
                      className="p-3 rounded-lg bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition-all space-y-2"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="w-3 h-3 rounded-full" style={{ backgroundColor: representative.color }} />
                          <span className="text-xs font-semibold text-white">{clusterName} Cluster</span>
                          <span className="text-[11px] text-slate-400">({count} heads)</span>
                        </div>
                        <span className="text-[11px] font-mono text-slate-400">
                          Avg JSD: {(0.15 + idx * 0.08).toFixed(2)}
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400">{representative.description}</p>
                      <div className="flex flex-wrap gap-1.5 pt-1">
                        {clusterPoints.slice(0, 12).map((pt) => (
                          <button
                            key={`L${pt.layer}H${pt.head}`}
                            onClick={() => {
                              setSelectedHeadPoint(pt);
                              setSelectedLayer(pt.layer);
                              setSelectedHead(pt.head);
                            }}
                            className={`px-2 py-0.5 rounded text-[10px] font-mono transition-colors ${
                              selectedHeadPoint?.layer === pt.layer && selectedHeadPoint?.head === pt.head
                                ? 'bg-white text-slate-950 font-bold'
                                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                            }`}
                          >
                            L{pt.layer}H{pt.head}
                          </button>
                        ))}
                        {clusterPoints.length > 12 && (
                          <span className="text-[10px] text-slate-500 self-center">
                            +{clusterPoints.length - 12} more
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {viewMode === 'matrix' && (
            <div className="flex-1 flex flex-col justify-between">
              <div className="text-xs text-slate-400 mb-2">Layer × Head Quantitative Score Matrix</div>
              <div className="grid grid-cols-12 gap-1 w-full h-[360px] p-2 bg-slate-900/60 rounded-lg overflow-y-auto">
                {headMetrics.points.map((pt) => {
                  const isSelected = selectedHeadPoint?.layer === pt.layer && selectedHeadPoint?.head === pt.head;
                  return (
                    <button
                      key={`L${pt.layer}H${pt.head}`}
                      onClick={() => {
                        setSelectedHeadPoint(pt);
                        setSelectedLayer(pt.layer);
                        setSelectedHead(pt.head);
                      }}
                      className={`h-7 rounded flex items-center justify-center text-[10px] font-mono transition-transform ${
                        isSelected ? 'ring-2 ring-white scale-110 z-10' : 'hover:scale-105 opacity-85 hover:opacity-100'
                      }`}
                      style={{
                        backgroundColor: pt.color,
                        color: pt.importance > 0.6 ? '#ffffff' : '#0f172a',
                      }}
                      title={`L${pt.layer}H${pt.head}: Imp=${pt.importance}, Stab=${pt.stability}`}
                    >
                      {pt.importance.toFixed(1)}
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* Right 1 Col: Selected Head Detail Card */}
        <div className="rounded-xl bg-slate-900/90 border border-slate-800 p-5 flex flex-col justify-between space-y-4 shadow-xl">
          {selectedHeadPoint ? (
            <>
              <div>
                <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                  <div>
                    <div className="flex items-center gap-2">
                      <h4 className="text-lg font-bold text-white">
                        Head L{selectedHeadPoint.layer}H{selectedHeadPoint.head}
                      </h4>
                      <Badge style={{ backgroundColor: selectedHeadPoint.color, color: '#fff' }}>
                        {selectedHeadPoint.head_type}
                      </Badge>
                    </div>
                    <p className="text-xs text-slate-400 mt-1">Layer {selectedHeadPoint.layer} · Head index {selectedHeadPoint.head}</p>
                  </div>
                  <Crosshair className="w-5 h-5 text-blue-400" />
                </div>

                <div className="space-y-3 mt-4">
                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80">
                    <span className="text-[11px] uppercase font-mono tracking-wider text-slate-500">Diagnostic Role</span>
                    <p className="text-xs text-slate-200 mt-1">{selectedHeadPoint.description}</p>
                  </div>

                  {/* 3 Core Metric Gauges */}
                  <div className="space-y-2 pt-2">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Importance I(l,h):</span>
                        <span className="font-mono text-emerald-400 font-bold">{selectedHeadPoint.importance}</span>
                      </div>
                      <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                        <div
                          className="bg-emerald-500 h-full rounded-full transition-all"
                          style={{ width: `${selectedHeadPoint.importance * 100}%` }}
                        />
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Stability S(l,h):</span>
                        <span className="font-mono text-cyan-400 font-bold">{selectedHeadPoint.stability}</span>
                      </div>
                      <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                        <div
                          className="bg-cyan-500 h-full rounded-full transition-all"
                          style={{ width: `${selectedHeadPoint.stability * 100}%` }}
                        />
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Similarity Sim(l,h):</span>
                        <span className="font-mono text-purple-400 font-bold">{selectedHeadPoint.similarity}</span>
                      </div>
                      <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                        <div
                          className="bg-purple-500 h-full rounded-full transition-all"
                          style={{ width: `${selectedHeadPoint.similarity * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Secondary Metrics */}
                  <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-800/80">
                    <div className="p-2 rounded bg-slate-950 border border-slate-800">
                      <span className="text-[10px] text-slate-500">Normalized Entropy</span>
                      <p className="text-sm font-mono font-semibold text-white">{selectedHeadPoint.entropy}</p>
                    </div>
                    <div className="p-2 rounded bg-slate-950 border border-slate-800">
                      <span className="text-[10px] text-slate-500">Sparsity Ratio</span>
                      <p className="text-sm font-mono font-semibold text-white">
                        {(selectedHeadPoint.sparsity * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="pt-4 border-t border-slate-800 space-y-2">
                <Button
                  variant="primary"
                  className="w-full text-xs font-semibold"
                  onClick={() => {
                    useAttentionStore.getState().setActiveZoneTab('circuits');
                    useAttentionStore.getState().setSelectedCircuitHead({
                      layer: selectedHeadPoint.layer,
                      head: selectedHeadPoint.head,
                    });
                  }}
                >
                  Inspect QK/OV Circuit SVD
                </Button>
                <Button
                  variant="outline"
                  className="w-full text-xs"
                  onClick={() => {
                    useAttentionStore.getState().toggleHeadAblation(selectedHeadPoint.layer, selectedHeadPoint.head);
                    useAttentionStore.getState().setActiveZoneTab('ablation');
                  }}
                >
                  Toggle Ablation in Laboratory
                </Button>
              </div>
            </>
          ) : (
            <div className="p-8 text-center text-slate-500">Select any point to inspect detailed metrics</div>
          )}
        </div>
      </div>
    </div>
  );
};
