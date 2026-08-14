// src/features/attentionDashboard/components/LogitLensPanel.tsx
import React, { useEffect } from 'react';
import { useAttentionStore } from '../useAttentionStore';
import { Button } from '../../../components/ui/Button';
import { Badge } from '../../../components/ui/Badge';
import {
  Eye,
  Play,
  Pause,
  SkipForward,
  SkipBack,
  Sparkles,
  Zap,
  TrendingUp,
  Target,
  Layers,
} from 'lucide-react';

export const LogitLensPanel: React.FC = () => {
  const {
    logitLens,
    currentScrubberLayer,
    setCurrentScrubberLayer,
    isLogitPlaying,
    setIsLogitPlaying,
    inputText,
  } = useAttentionStore();

  const numLayers = logitLens?.layer_predictions.length || 12;

  // Auto-play interval
  useEffect(() => {
    let interval: any = null;
    if (isLogitPlaying) {
      interval = setInterval(() => {
        setCurrentScrubberLayer((currentScrubberLayer + 1) % numLayers);
      }, 700);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isLogitPlaying, currentScrubberLayer, numLayers, setCurrentScrubberLayer]);

  if (!logitLens) {
    return (
      <div className="p-8 text-center text-slate-400">
        <Sparkles className="w-8 h-8 mx-auto mb-2 animate-spin text-blue-500" />
        <p>Computing Logit Lens Intermediate Vocabulary Projections...</p>
      </div>
    );
  }

  const currentPrediction = logitLens.layer_predictions[currentScrubberLayer] || logitLens.layer_predictions[0];

  return (
    <div className="space-y-6">
      {/* Top Header & Timeline Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl bg-slate-900/70 border border-slate-800 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/20">
            <Eye className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-white">Logit Lens Prediction Timeline (nostalgebraist, 2020)</h3>
            <p className="text-xs text-slate-400">
              Unembedding intermediate activations $p_l = \\operatorname{softmax}(W_U \\cdot h_l)$ layer by layer
            </p>
          </div>
        </div>

        {/* Timeline Playback Bar */}
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setCurrentScrubberLayer(Math.max(0, currentScrubberLayer - 1))}
            disabled={currentScrubberLayer === 0}
            className="h-8 w-8 p-0"
          >
            <SkipBack className="w-3.5 h-3.5" />
          </Button>

          <Button
            variant="primary"
            size="sm"
            onClick={() => setIsLogitPlaying(!isLogitPlaying)}
            className="h-8 px-3 text-xs bg-amber-600 hover:bg-amber-500 font-semibold"
          >
            {isLogitPlaying ? <Pause className="w-3.5 h-3.5 mr-1" /> : <Play className="w-3.5 h-3.5 mr-1" />}
            {isLogitPlaying ? 'Pause' : 'Play Flow'}
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={() => setCurrentScrubberLayer(Math.min(numLayers - 1, currentScrubberLayer + 1))}
            disabled={currentScrubberLayer === numLayers - 1}
            className="h-8 w-8 p-0"
          >
            <SkipForward className="w-3.5 h-3.5" />
          </Button>

          <Badge variant="outline" className="ml-2 font-mono text-xs text-amber-400 border-amber-500/30">
            Layer {currentScrubberLayer} / {numLayers - 1}
          </Badge>
        </div>
      </div>

      {/* Layer Slider Track */}
      <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2 shadow-inner">
        <div className="flex justify-between items-center text-xs font-mono text-slate-400">
          <span>Early Layers (Diffuse)</span>
          <span className="text-amber-400 font-bold">
            Convergence @ Layer {logitLens.convergence_layer}
          </span>
          <span>Final Layer (Converged)</span>
        </div>

        <input
          type="range"
          min="0"
          max={numLayers - 1}
          value={currentScrubberLayer}
          onChange={(e) => setCurrentScrubberLayer(parseInt(e.target.value, 10))}
          className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
        />

        <div className="grid grid-cols-12 gap-1 pt-1">
          {logitLens.layer_predictions.map((lp, idx) => (
            <button
              key={idx}
              onClick={() => setCurrentScrubberLayer(idx)}
              className={`py-1 rounded text-[10px] font-mono transition-all text-center ${
                currentScrubberLayer === idx
                  ? 'bg-amber-500 text-slate-950 font-bold ring-2 ring-amber-300 shadow-md'
                  : idx >= logitLens.convergence_layer
                  ? 'bg-emerald-950/60 text-emerald-400 border border-emerald-800/40 hover:bg-emerald-900/50'
                  : 'bg-slate-900 text-slate-500 hover:bg-slate-800'
              }`}
            >
              L{idx}
            </button>
          ))}
        </div>
      </div>

      {/* Main Grid: Prediction Word Cloud & Top-K Probs */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Animated Word Cloud & Distribution */}
        <div className="lg:col-span-2 rounded-xl bg-slate-950 border border-slate-800 p-5 space-y-4 shadow-2xl flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-white flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-amber-400" />
              Layer {currentScrubberLayer} Vocabulary Projection Space
            </h4>
            <span className="text-xs font-mono text-slate-400">
              Entropy: <strong className="text-amber-400">{currentPrediction.entropy} nats</strong>
            </span>
          </div>

          {/* Interactive Word Cloud Visualizer */}
          <div className="relative w-full h-[320px] bg-slate-900/50 rounded-lg border border-slate-800 p-6 flex flex-wrap items-center justify-center gap-4 overflow-hidden">
            {currentPrediction.top_k_tokens.map((tok, idx) => {
              const prob = currentPrediction.top_k_probs[idx];
              const fontSize = Math.max(13, Math.min(46, prob * 56));
              const isTop1 = idx === 0;

              return (
                <div
                  key={tok}
                  className={`px-4 py-2 rounded-xl font-mono transition-all duration-300 transform flex items-center gap-2 ${
                    isTop1
                      ? 'bg-amber-500/20 border-2 border-amber-400 text-amber-300 font-bold shadow-[0_0_20px_rgba(245,158,11,0.3)] scale-110'
                      : 'bg-slate-900/80 border border-slate-800 text-slate-300 hover:border-slate-700'
                  }`}
                  style={{ fontSize: `${fontSize}px` }}
                >
                  <span>"{tok}"</span>
                  <span className="text-xs font-normal text-slate-400 opacity-80">
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </div>

          {/* Probability Bar Breakdown */}
          <div className="space-y-2 pt-2 border-t border-slate-800/80">
            <div className="text-xs text-slate-400 font-mono">Top Candidate Probabilities at Layer {currentScrubberLayer}:</div>
            <div className="space-y-1.5">
              {currentPrediction.top_k_tokens.map((tok, idx) => {
                const prob = currentPrediction.top_k_probs[idx];
                return (
                  <div key={tok} className="flex items-center justify-between text-xs font-mono">
                    <span className="w-28 truncate text-slate-300 font-semibold">{tok}</span>
                    <div className="flex-1 mx-3 bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-800">
                      <div
                        className={`h-full rounded-full transition-all duration-300 ${
                          idx === 0 ? 'bg-amber-500' : 'bg-blue-500/70'
                        }`}
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                    <span className="w-12 text-right text-slate-400">{(prob * 100).toFixed(1)}%</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right 1 Col: Convergence & Layer Trajectory Insights */}
        <div className="rounded-xl bg-slate-900/90 border border-slate-800 p-5 flex flex-col justify-between space-y-4 shadow-xl">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <h4 className="text-sm font-bold text-white flex items-center gap-2">
                <Target className="w-4 h-4 text-amber-400" />
                Convergence Diagnostics
              </h4>
              <Badge variant="success">Final: "{logitLens.final_prediction}"</Badge>
            </div>

            <div className="space-y-4 mt-4">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
                <span className="text-[10px] text-slate-500 uppercase font-mono">Early Convergence Layer</span>
                <div className="flex items-baseline gap-2">
                  <span className="text-2xl font-bold text-emerald-400 font-mono">
                    Layer {logitLens.convergence_layer}
                  </span>
                  <span className="text-xs text-slate-400">
                    ({((logitLens.convergence_layer / (numLayers - 1)) * 100).toFixed(0)}% depth)
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 pt-1">
                  The model reaches its final top-1 token "{logitLens.final_prediction}" at Layer {logitLens.convergence_layer} with confidence climbing to {(logitLens.final_confidence * 100).toFixed(1)}%.
                </p>
              </div>

              {/* Entropy Across Layers Sparkline List */}
              <div className="space-y-1.5">
                <span className="text-xs text-slate-400 font-mono">Layerwise Prediction Entropy:</span>
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1 max-h-48 overflow-y-auto">
                  {logitLens.layer_predictions.map((lp) => (
                    <div
                      key={lp.layer}
                      onClick={() => setCurrentScrubberLayer(lp.layer)}
                      className={`flex items-center justify-between text-[11px] font-mono p-1 rounded cursor-pointer transition-colors ${
                        currentScrubberLayer === lp.layer ? 'bg-slate-800 text-amber-400 font-bold' : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      <span>L{lp.layer}: "{lp.top_1_token}"</span>
                      <span>{lp.entropy} nats</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 text-[11px] text-slate-400">
            Logit Lens reveals how semantic hypothesis refining occurs: early layers maintain broad diffuse super-categories before locking into exact lexical choices.
          </div>
        </div>
      </div>
    </div>
  );
};
