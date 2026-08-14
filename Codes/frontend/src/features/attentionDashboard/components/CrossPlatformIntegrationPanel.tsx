// src/features/attentionDashboard/components/CrossPlatformIntegrationPanel.tsx
import React, { useState } from 'react';
import { useAttentionStore } from '../useAttentionStore';
import { Button } from '../../../components/ui/Button';
import { Badge } from '../../../components/ui/Badge';
import {
  Cpu,
  FileText,
  Volume2,
  VolumeX,
  Sparkles,
  Zap,
  Play,
  Download,
  Copy,
  CheckCircle2,
  ExternalLink,
  Target,
  Layers,
} from 'lucide-react';

export const CrossPlatformIntegrationPanel: React.FC = () => {
  const {
    selectedModel,
    selectedAttentionType,
    inputText,
    selectedLayer,
    selectedHead,
    universalScore,
    generatedPaper,
    generatePaperDraft,
    isSonificationPlaying,
    toggleSonification,
    particleFlowSpeed,
    setParticleFlowSpeed,
  } = useAttentionStore();

  const [copiedLatex, setCopiedLatex] = useState(false);

  const handleCopyLatex = () => {
    if (!generatedPaper) return;
    navigator.clipboard.writeText(generatedPaper.latexCode);
    setCopiedLatex(true);
    setTimeout(() => setCopiedLatex(false), 2000);
  };

  const handleLaunchExperiment = () => {
    const launcherUrl = `/launcher?model=${encodeURIComponent(selectedModel)}&attn=${encodeURIComponent(selectedAttentionType)}&layer=${selectedLayer}&head=${selectedHead}&text=${encodeURIComponent(inputText)}`;
    window.open(launcherUrl, '_blank');
  };

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl bg-slate-900/70 border border-slate-800 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-white">Cross-Platform Pipeline & Automated Workflows (Zone 9)</h3>
            <p className="text-xs text-slate-400">
              One-click experiment training launcher, 4D universal attention scoring, LaTeX paper authoring & sonification
            </p>
          </div>
        </div>

        <Button
          variant="primary"
          onClick={handleLaunchExperiment}
          className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 font-semibold text-xs shadow-lg"
        >
          <Zap className="w-3.5 h-3.5 mr-1" /> "Train with This" in Launcher (+100 XP)
        </Button>
      </div>

      {/* Grid: 4D Universal Scoring & Audio / Particle Flow */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Col 1: 4D Universal Score Radar */}
        <div className="rounded-xl bg-slate-950 border border-slate-800 p-5 space-y-4 shadow-2xl">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <h4 className="text-sm font-bold text-white flex items-center gap-2">
              <Target className="w-4 h-4 text-cyan-400" />
              Universal Attention 4D Score
            </h4>
            <Badge variant="success">
              {universalScore.percentile}th Percentile
            </Badge>
          </div>

          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-xs mb-1 font-mono">
                <span className="text-slate-400">Interpretability</span>
                <span className="text-emerald-400 font-bold">{universalScore.interpretability} / 100</span>
              </div>
              <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-800">
                <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${universalScore.interpretability}%` }} />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1 font-mono">
                <span className="text-slate-400">Specialization</span>
                <span className="text-cyan-400 font-bold">{universalScore.specialization} / 100</span>
              </div>
              <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-800">
                <div className="bg-cyan-500 h-full rounded-full" style={{ width: `${universalScore.specialization}%` }} />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1 font-mono">
                <span className="text-slate-400">Efficiency</span>
                <span className="text-purple-400 font-bold">{universalScore.efficiency} / 100</span>
              </div>
              <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-800">
                <div className="bg-purple-500 h-full rounded-full" style={{ width: `${universalScore.efficiency}%` }} />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1 font-mono">
                <span className="text-slate-400">Stability</span>
                <span className="text-amber-400 font-bold">{universalScore.stability} / 100</span>
              </div>
              <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-800">
                <div className="bg-amber-500 h-full rounded-full" style={{ width: `${universalScore.stability}%` }} />
              </div>
            </div>

            <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800 text-center space-y-1">
              <span className="text-[10px] text-slate-500 font-mono uppercase">Composite Quality Index</span>
              <p className="text-2xl font-bold font-mono text-white">{universalScore.compositeScore}</p>
            </div>
          </div>
        </div>

        {/* Col 2 & 3: Auto-Generated Paper Section & LaTeX Export */}
        <div className="lg:col-span-2 rounded-xl bg-slate-950 border border-slate-800 p-5 space-y-4 shadow-2xl flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <h4 className="text-sm font-bold text-white flex items-center gap-2">
                <FileText className="w-4 h-4 text-blue-400" />
                Auto-Generated Research Paper Section (Methods, Results & LaTeX)
              </h4>
              <Button size="sm" variant="primary" onClick={generatePaperDraft} className="text-xs bg-blue-600 hover:bg-blue-500">
                <Sparkles className="w-3 h-3 mr-1" /> Generate Section Draft
              </Button>
            </div>

            {generatedPaper ? (
              <div className="space-y-3 mt-4">
                <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800 space-y-1">
                  <span className="text-[10px] font-mono uppercase text-blue-400">Draft Title</span>
                  <h5 className="text-xs font-bold text-white">{generatedPaper.title}</h5>
                </div>

                <div className="space-y-2 text-xs text-slate-300 max-h-48 overflow-y-auto pr-1">
                  <div>
                    <strong className="text-slate-200 block text-[11px] font-mono text-blue-400">Methods:</strong>
                    <p>{generatedPaper.methods}</p>
                  </div>
                  <div>
                    <strong className="text-slate-200 block text-[11px] font-mono text-emerald-400">Results:</strong>
                    <p>{generatedPaper.results}</p>
                  </div>
                </div>

                {/* LaTeX Code Box */}
                <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 font-mono text-xs text-slate-300 relative">
                  <pre className="overflow-x-auto text-[11px]">{generatedPaper.latexCode}</pre>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={handleCopyLatex}
                    className="absolute top-2 right-2 text-xs h-6 px-2"
                  >
                    {copiedLatex ? <CheckCircle2 className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                  </Button>
                </div>
              </div>
            ) : (
              <div className="p-8 text-center text-slate-500 text-xs">
                Click "Generate Section Draft" to create publication-ready Methods, Results, and LaTeX formulas based on this attention pattern.
              </div>
            )}
          </div>

          {/* Interactive Media Extras: Sonification & Particle Flow */}
          <div className="pt-3 border-t border-slate-800 flex flex-wrap items-center justify-between gap-4">
            {/* Audio Sonification */}
            <div className="flex items-center gap-3">
              <button
                onClick={toggleSonification}
                className={`p-2 rounded-lg border transition-all flex items-center gap-1.5 text-xs font-mono ${
                  isSonificationPlaying
                    ? 'bg-purple-600/20 border-purple-500 text-purple-300 ring-2 ring-purple-500/40'
                    : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-white'
                }`}
              >
                {isSonificationPlaying ? <Volume2 className="w-4 h-4 text-purple-400" /> : <VolumeX className="w-4 h-4" />}
                {isSonificationPlaying ? 'Sonification: Synthesizing QK Sine Waves' : 'Play Attention Audio'}
              </button>
            </div>

            {/* Particle Flow Speed */}
            <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
              <span>Particle Flow:</span>
              <input
                type="range"
                min="0.2"
                max="3.0"
                step="0.2"
                value={particleFlowSpeed}
                onChange={(e) => setParticleFlowSpeed(parseFloat(e.target.value))}
                className="w-20 accent-cyan-500"
              />
              <span className="text-cyan-400">{particleFlowSpeed}x</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
