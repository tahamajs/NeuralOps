// src/features/attentionDashboard/components/HeadAblationLab.tsx
import React, { useState } from 'react';
import { useAttentionStore } from '../useAttentionStore';
import { Button } from '../../../components/ui/Button';
import { Badge } from '../../../components/ui/Badge';
import {
  Scissors,
  Download,
  Copy,
  RotateCcw,
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  TrendingDown,
  BarChart,
  Layers,
  FileJson,
} from 'lucide-react';

export const HeadAblationLab: React.FC = () => {
  const {
    headMetrics,
    ablatedHeads,
    ablationResult,
    pruningRecipe,
    toggleHeadAblation,
    ablateLayer,
    ablateRedundantHeads,
    resetAblations,
  } = useAttentionStore();

  const [copied, setCopied] = useState(false);
  const numLayers = headMetrics?.layers || 12;
  const numHeads = headMetrics?.heads_per_layer || 12;

  const handleCopyRecipe = () => {
    if (!pruningRecipe) return;
    navigator.clipboard.writeText(JSON.stringify(pruningRecipe, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadRecipe = () => {
    if (!pruningRecipe) return;
    const blob = new Blob([JSON.stringify(pruningRecipe, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pruning_recipe_${pruningRecipe.model_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Top Banner & Action Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl bg-slate-900/70 border border-slate-800 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-rose-500/10 text-rose-400 border border-rose-500/20">
            <Scissors className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-white">Head Ablation & Pruning Sensitivity Laboratory</h3>
            <p className="text-xs text-slate-400">
              Selectively zero out attention heads to quantify loss, perplexity, KL divergence, and token prediction flips
            </p>
          </div>
        </div>

        {/* Global Ablation Actions */}
        <div className="flex flex-wrap items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={ablateRedundantHeads}
            className="text-xs text-amber-400 border-amber-500/30 hover:bg-amber-500/10"
          >
            <Sparkles className="w-3.5 h-3.5 mr-1" />
            Ablate Redundant Heads
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={resetAblations}
            className="text-xs text-slate-300 hover:bg-slate-800"
          >
            <RotateCcw className="w-3.5 h-3.5 mr-1" />
            Reset All ({ablatedHeads.size})
          </Button>
          <Button
            variant="primary"
            size="sm"
            onClick={handleDownloadRecipe}
            className="text-xs bg-emerald-600 hover:bg-emerald-500"
          >
            <Download className="w-3.5 h-3.5 mr-1" />
            Export Pruning Recipe
          </Button>
        </div>
      </div>

      {/* Main Ablation Grid & Real-time Impact */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Layer x Head Grid */}
        <div className="lg:col-span-2 rounded-xl bg-slate-950 border border-slate-800 p-5 space-y-4 shadow-2xl">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-blue-400" />
              Interactive Head Grid (Click cell to toggle ablation)
            </h4>
            <div className="flex items-center gap-3 text-xs text-slate-400">
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded bg-blue-500" /> Active
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded bg-rose-500/40 border border-rose-500" /> Ablated (Zeroed)
              </div>
            </div>
          </div>

          {/* Grid Container */}
          <div className="overflow-x-auto">
            <div className="min-w-[540px] space-y-1.5">
              {/* Header row: Head indexes */}
              <div className="grid grid-cols-13 gap-1 text-center font-mono text-[10px] text-slate-500 pb-1 border-b border-slate-800">
                <div>Layer</div>
                {Array.from({ length: numHeads }, (_, h) => (
                  <div key={h}>H{h}</div>
                ))}
              </div>

              {/* Layer rows */}
              {Array.from({ length: numLayers }, (_, l) => (
                <div key={l} className="grid grid-cols-13 gap-1 items-center">
                  <button
                    onClick={() => ablateLayer(l)}
                    className="text-[10px] font-mono text-slate-400 hover:text-white hover:underline text-left"
                    title={`Click to ablate all heads in Layer ${l}`}
                  >
                    L{l}
                  </button>
                  {Array.from({ length: numHeads }, (_, h) => {
                    const key = `L${l}H${h}`;
                    const isAblated = ablatedHeads.has(key);
                    const pt = headMetrics?.points.find((p) => p.layer === l && p.head === h);
                    const baseColor = pt?.color || '#3b82f6';

                    return (
                      <button
                        key={key}
                        onClick={() => toggleHeadAblation(l, h)}
                        className={`h-7 rounded text-[10px] font-mono font-bold transition-all relative flex items-center justify-center ${
                          isAblated
                            ? 'bg-rose-950/80 border border-rose-500/80 text-rose-300 ring-1 ring-rose-500/40 shadow-inner'
                            : 'hover:brightness-125 opacity-90 hover:opacity-100 text-white shadow-sm'
                        }`}
                        style={{
                          backgroundColor: isAblated ? undefined : baseColor,
                        }}
                        title={`Head L${l}H${h} (${pt?.head_type || 'Head'}) - Click to ${isAblated ? 'Restore' : 'Ablate'}`}
                      >
                        {isAblated ? '✕' : h}
                      </button>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>

          <div className="pt-2 flex justify-between items-center text-xs text-slate-500 border-t border-slate-900 font-mono">
            <span>Total Ablated: <strong className="text-rose-400">{ablatedHeads.size}</strong> heads ({(ablatedHeads.size / (numLayers * numHeads) * 100).toFixed(1)}%)</span>
            <span>nnsight `model.edit()` deferred zero-masking</span>
          </div>
        </div>

        {/* Right 1 Col: Real-time Sensitivity & Metrics */}
        <div className="rounded-xl bg-slate-900/90 border border-slate-800 p-5 flex flex-col justify-between space-y-4 shadow-xl">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <h4 className="text-sm font-bold text-white flex items-center gap-2">
                <BarChart className="w-4 h-4 text-emerald-400" />
                Intervention Sensitivity Deltas
              </h4>
              <Badge variant={ablatedHeads.size > 0 ? 'warning' : 'outline'}>
                {ablatedHeads.size} Heads Masked
              </Badge>
            </div>

            {ablationResult && (
              <div className="space-y-4 mt-4">
                {/* 4 Core Quantitative Delta Cards */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase font-mono">Perplexity (PPL)</span>
                    <div className="flex items-baseline gap-1 mt-1">
                      <span className="text-lg font-bold text-white">{ablationResult.ablated_ppl}</span>
                      <span className="text-xs text-rose-400 font-mono font-semibold">
                        (+{ablationResult.ppl_delta})
                      </span>
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase font-mono">Cross-Entropy Loss</span>
                    <div className="flex items-baseline gap-1 mt-1">
                      <span className="text-lg font-bold text-white">{ablationResult.ablated_loss}</span>
                      <span className="text-xs text-rose-400 font-mono font-semibold">
                        (+{ablationResult.loss_delta})
                      </span>
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase font-mono">Output KL Divergence</span>
                    <div className="flex items-baseline gap-1 mt-1">
                      <span className="text-lg font-bold font-mono text-cyan-400">{ablationResult.kl_divergence}</span>
                      <span className="text-[10px] text-slate-400">nats</span>
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-950 border border-slate-800">
                    <span className="text-[10px] text-slate-500 uppercase font-mono">Prediction Flips</span>
                    <div className="flex items-baseline gap-1 mt-1">
                      <span className="text-lg font-bold text-amber-400">{ablationResult.top_k_flips}</span>
                      <span className="text-[10px] text-slate-400">tokens</span>
                    </div>
                  </div>
                </div>

                {/* Compound Redundancy Bar */}
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1.5">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-400">Sublinear Redundancy Score:</span>
                    <span className="font-mono text-emerald-400 font-bold">
                      {(ablationResult.compound_redundancy_score * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                    <div
                      className="bg-emerald-500 h-full rounded-full transition-all"
                      style={{ width: `${ablationResult.compound_redundancy_score * 100}%` }}
                    />
                  </div>
                  <p className="text-[10px] text-slate-500">
                    High score indicates ablating these heads together causes significantly less disruption than separate ablations.
                  </p>
                </div>

                {/* Affected Tokens Table */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-medium text-slate-300">Significant Token Probability Shifts</span>
                    <span className="text-[11px] text-slate-500">{ablationResult.affected_tokens.length} affected</span>
                  </div>
                  <div className="max-h-36 overflow-y-auto space-y-1.5 pr-1">
                    {ablationResult.affected_tokens.map((tok) => (
                      <div
                        key={tok.position}
                        className="p-2 rounded bg-slate-950/80 border border-slate-800/80 flex items-center justify-between text-xs font-mono"
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-slate-500">#{tok.position}</span>
                          <span className="text-white font-semibold">{tok.token}</span>
                          {tok.flipped && (
                            <span className="px-1 py-0.2 rounded bg-rose-500/20 text-rose-400 text-[9px] border border-rose-500/30">
                              FLIP
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="text-slate-400">{tok.original_prob} → {tok.ablated_prob}</span>
                          <span className={tok.prob_delta < 0 ? 'text-rose-400' : 'text-emerald-400'}>
                            ({tok.prob_delta > 0 ? '+' : ''}{tok.prob_delta})
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Pruning Recipe Summary & Copy */}
          {pruningRecipe && (
            <div className="p-3 rounded-lg bg-emerald-950/20 border border-emerald-500/30 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-emerald-400 flex items-center gap-1.5">
                  <FileJson className="w-3.5 h-3.5" /> Pruning Recipe Available
                </span>
                <Button size="sm" variant="outline" onClick={handleCopyRecipe} className="text-xs h-7">
                  {copied ? <CheckCircle2 className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                  {copied ? 'Copied' : 'Copy'}
                </Button>
              </div>
              <p className="text-[11px] text-slate-300">{pruningRecipe.rationale}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
