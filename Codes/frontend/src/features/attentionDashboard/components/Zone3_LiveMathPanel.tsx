// src/features/attentionDashboard/components/Zone3_LiveMathPanel.tsx
import React, { useState, useEffect } from 'react';
import {
  Calculator,
  Play,
  Pause,
  SkipForward,
  SkipBack,
  Copy,
  Check,
  Download,
  Sliders,
  Layers,
  Sparkles,
  Zap,
  Info,
  ChevronRight,
} from 'lucide-react';
import { useAttentionDashboardStore } from '../stores/useAttentionDashboardStore';

export const Zone3_LiveMathPanel: React.FC = () => {
  const {
    liveMath,
    currentMathStep,
    isMathPlaying,
    mathPlaySpeed,
    selection,
    setCurrentMathStep,
    setIsMathPlaying,
    setMathPlaySpeed,
  } = useAttentionDashboardStore();

  const [activeTab, setActiveTab] = useState<'stepper' | 'inspector' | 'diff' | 'complexity'>('stepper');
  const [copied, setCopied] = useState(false);

  const steps = liveMath.steps;
  const activeStep = steps.find((s) => s.step_number === currentMathStep) || steps[0];
  const numInspection = liveMath.numerical_inspection;

  // Auto-play timer for step-by-step animation
  useEffect(() => {
    let interval: any = null;
    if (isMathPlaying) {
      interval = setInterval(() => {
        setCurrentMathStep((currentMathStep % steps.length) + 1);
      }, 2500 / mathPlaySpeed);
    }
    return () => clearInterval(interval);
  }, [isMathPlaying, currentMathStep, mathPlaySpeed, steps.length, setCurrentMathStep]);

  const copyInspectionJSON = () => {
    navigator.clipboard.writeText(JSON.stringify(numInspection, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="w-full bg-slate-950/80 rounded-xl border border-slate-800 shadow-xl overflow-hidden flex flex-col">
      {/* Header with Variable Legend & Tabs */}
      <div className="px-4 py-3 bg-slate-900/90 border-b border-slate-800 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-purple-500/20 text-purple-400 border border-purple-500/30">
            <Calculator className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-slate-100 flex items-center gap-1.5">
              <span>Zone 3: Live Math Panel</span>
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-purple-950 text-purple-300 font-mono">
                {liveMath.attention_type}
              </span>
            </h3>
            <div className="flex items-center gap-2 mt-0.5 text-[10px]">
              <span className="text-blue-400 font-semibold">Q (Query)</span>
              <span className="text-slate-600">·</span>
              <span className="text-rose-400 font-semibold">K (Key)</span>
              <span className="text-slate-600">·</span>
              <span className="text-emerald-400 font-semibold">V (Value)</span>
              <span className="text-slate-600">·</span>
              <span className="text-purple-400 font-semibold">softmax</span>
              <span className="text-slate-600">·</span>
              <span className="text-cyan-400 font-semibold">SVD/Sketch</span>
            </div>
          </div>
        </div>

        {/* Tab Switcher */}
        <div className="flex items-center space-x-1 p-0.5 bg-slate-950 rounded-lg border border-slate-800 text-xs">
          <button
            onClick={() => setActiveTab('stepper')}
            className={`px-2.5 py-1 rounded font-medium transition-colors ${
              activeTab === 'stepper'
                ? 'bg-purple-600 text-white shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Step Stepper
          </button>
          <button
            onClick={() => setActiveTab('inspector')}
            className={`px-2.5 py-1 rounded font-medium transition-colors ${
              activeTab === 'inspector'
                ? 'bg-purple-600 text-white shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Value Inspector
          </button>
          <button
            onClick={() => setActiveTab('diff')}
            className={`px-2.5 py-1 rounded font-medium transition-colors ${
              activeTab === 'diff'
                ? 'bg-purple-600 text-white shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Math Diff
          </button>
          <button
            onClick={() => setActiveTab('complexity')}
            className={`px-2.5 py-1 rounded font-medium transition-colors ${
              activeTab === 'complexity'
                ? 'bg-purple-600 text-white shadow'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Complexity
          </button>
        </div>
      </div>

      {/* Main Content Body */}
      <div className="p-4 flex-1">
        {/* TAB 1: STEP-BY-STEP STEPPER */}
        {activeTab === 'stepper' && (
          <div className="space-y-4">
            {/* Step Navigation Dots & Scrubber */}
            <div className="flex items-center justify-between bg-slate-900/60 p-3 rounded-lg border border-slate-800/80">
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setIsMathPlaying(!isMathPlaying)}
                  className="p-1.5 rounded-md bg-purple-600 hover:bg-purple-500 text-white transition-colors"
                  title={isMathPlaying ? 'Pause Animation' : 'Play Step Animation'}
                >
                  {isMathPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                </button>
                <button
                  onClick={() => setCurrentMathStep(Math.max(1, currentMathStep - 1))}
                  disabled={currentMathStep === 1}
                  className="p-1 rounded text-slate-400 hover:text-white disabled:opacity-30"
                >
                  <SkipBack className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => setCurrentMathStep(Math.min(steps.length, currentMathStep + 1))}
                  disabled={currentMathStep === steps.length}
                  className="p-1 rounded text-slate-400 hover:text-white disabled:opacity-30"
                >
                  <SkipForward className="w-3.5 h-3.5" />
                </button>
                <span className="text-xs font-mono text-slate-300 ml-2">
                  Step <span className="text-purple-400 font-bold">{currentMathStep}</span> / {steps.length}
                </span>
              </div>

              {/* Step indicator pills */}
              <div className="flex items-center space-x-1.5">
                {steps.map((s) => (
                  <button
                    key={s.step_number}
                    onClick={() => setCurrentMathStep(s.step_number)}
                    className={`h-2 rounded-full transition-all ${
                      currentMathStep === s.step_number
                        ? 'w-8 bg-purple-400 shadow-[0_0_8px_#c084fc]'
                        : 'w-2 bg-slate-700 hover:bg-slate-500'
                    }`}
                    title={s.title}
                  />
                ))}
              </div>

              {/* Speed selector */}
              <div className="flex items-center space-x-1 text-[10px] font-mono">
                {[0.5, 1, 2].map((spd) => (
                  <button
                    key={spd}
                    onClick={() => setMathPlaySpeed(spd)}
                    className={`px-1.5 py-0.5 rounded ${
                      mathPlaySpeed === spd ? 'bg-purple-900 text-purple-200 font-bold' : 'text-slate-500 hover:text-slate-300'
                    }`}
                  >
                    {spd}x
                  </button>
                ))}
              </div>
            </div>

            {/* Active Step Card */}
            <div className="p-4 rounded-xl bg-gradient-to-br from-slate-900 to-slate-950 border border-purple-500/30 shadow-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-purple-400 uppercase tracking-wider">
                  Step {activeStep.step_number}: {activeStep.title}
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-purple-950/80 border border-purple-700/50 text-purple-300 font-mono">
                  Live KaTeX
                </span>
              </div>

              {/* LaTeX Equation Display Box */}
              <div className="p-3 my-2 rounded-lg bg-slate-950 border border-slate-800 text-center font-mono text-sm text-cyan-300 tracking-wide overflow-x-auto">
                {activeStep.latex_equation}
              </div>

              <p className="text-xs text-slate-300 mt-2 leading-relaxed">{activeStep.description}</p>

              {/* Variable Explanations */}
              <div className="mt-3 pt-3 border-t border-slate-800/80 grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
                {Object.entries(activeStep.variables).map(([v, desc]) => (
                  <div key={v} className="p-1.5 rounded bg-slate-900/80 border border-slate-800 flex flex-col">
                    <span className="font-mono font-bold text-purple-300">{v}</span>
                    <span className="text-[10px] text-slate-400 truncate">{desc}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: NUMERICAL VALUE INSPECTOR */}
        {activeTab === 'inspector' && (
          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400">
                Live Floating-Point Breakdown for{' '}
                <strong className="text-cyan-300">
                  '{numInspection.query_token}' (q={numInspection.query_index}) → '{numInspection.key_token}' (k={numInspection.key_index})
                </strong>
              </span>
              <button
                onClick={copyInspectionJSON}
                className="flex items-center gap-1 text-[11px] text-slate-400 hover:text-cyan-300 px-2 py-0.5 rounded bg-slate-900 border border-slate-800 transition-colors"
              >
                {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                <span>{copied ? 'Copied' : 'Copy JSON'}</span>
              </button>
            </div>

            <div className="overflow-x-auto rounded-lg border border-slate-800">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-slate-900 text-slate-400 text-[10px] uppercase border-b border-slate-800">
                  <tr>
                    <th className="p-2.5">Computation Phase</th>
                    <th className="p-2.5">Formula Expression</th>
                    <th className="p-2.5">Evaluated Value</th>
                    <th className="p-2.5">Description</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-200">
                  <tr className="hover:bg-slate-900/40">
                    <td className="p-2 text-cyan-400 font-bold">① Raw Dot Product</td>
                    <td className="p-2 text-slate-400">Q[i] · K[j]</td>
                    <td className="p-2 font-bold text-white">{numInspection.raw_dot_product}</td>
                    <td className="p-2 text-[11px] text-slate-400">Vector similarity score</td>
                  </tr>
                  <tr className="hover:bg-slate-900/40">
                    <td className="p-2 text-amber-400 font-bold">② Variance Scaling</td>
                    <td className="p-2 text-slate-400">Score / √{numInspection.head_dimension_d_k}</td>
                    <td className="p-2 font-bold text-amber-300">{numInspection.scaled_score}</td>
                    <td className="p-2 text-[11px] text-slate-400">Divided by √d_k = {numInspection.sqrt_d_k}</td>
                  </tr>
                  <tr className="hover:bg-slate-900/40">
                    <td className="p-2 text-purple-400 font-bold">③ Softmax Weight</td>
                    <td className="p-2 text-slate-400">exp(S_ij) / Σ exp</td>
                    <td className="p-2 font-bold text-purple-300">
                      {numInspection.attention_probability.toFixed(4)}
                    </td>
                    <td className="p-2 text-[11px] text-slate-400">Normalized probability mass</td>
                  </tr>
                  <tr className="hover:bg-slate-900/40">
                    <td className="p-2 text-emerald-400 font-bold">④ Value Synthesis</td>
                    <td className="p-2 text-slate-400">Weight × V[j] (head)</td>
                    <td className="p-2 font-bold text-emerald-300 truncate max-w-[150px]">
                      [{numInspection.weighted_value_contribution?.slice(0, 3).map((n) => n.toFixed(3)).join(', ')}...]
                    </td>
                    <td className="p-2 text-[11px] text-slate-400">Contextual representation contribution</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* TAB 3: ARCHITECTURE COMPARISON MATH DIFF */}
        {activeTab === 'diff' && (
          <div className="space-y-3">
            <div className="text-xs text-slate-400">
              Mathematical Formulation Diff:{' '}
              <span className="text-slate-200 font-semibold">{liveMath.comparison_diff?.baseline_name}</span> vs{' '}
              <span className="text-cyan-300 font-semibold">{liveMath.comparison_diff?.current_name}</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {/* Baseline Box */}
              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-xs">
                <div className="text-[10px] font-bold uppercase text-slate-400 mb-1">
                  Baseline (Softmax MHA)
                </div>
                <div className="font-mono text-slate-300 p-2 bg-slate-950 rounded border border-slate-800 mb-2">
                  {liveMath.comparison_diff?.baseline_formula}
                </div>
                <div className="text-[11px] text-slate-400">Standard quadratic scaled dot-product attention.</div>
              </div>

              {/* Current Box */}
              <div className="p-3 rounded-lg bg-slate-900 border border-cyan-500/40 text-xs">
                <div className="text-[10px] font-bold uppercase text-cyan-400 mb-1">
                  Active ({liveMath.attention_type})
                </div>
                <div className="font-mono text-cyan-300 p-2 bg-slate-950 rounded border border-cyan-900/50 mb-2">
                  {liveMath.comparison_diff?.current_formula}
                </div>
                <div className="text-[11px] text-cyan-200">
                  Subspace projection eliminates quadratic token bottlenecks.
                </div>
              </div>
            </div>

            {/* Operations Diff Table */}
            <div className="rounded-lg border border-slate-800 overflow-hidden text-xs font-mono">
              <table className="w-full text-left">
                <thead className="bg-slate-900 text-slate-400 text-[10px] uppercase">
                  <tr>
                    <th className="p-2">Operation</th>
                    <th className="p-2">Baseline</th>
                    <th className="p-2">Current Architecture</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 text-slate-300">
                  {liveMath.comparison_diff?.differing_operations.map((op, i) => (
                    <tr key={i} className="hover:bg-slate-900/40">
                      <td className="p-2 text-purple-400 font-semibold">{op.op}</td>
                      <td className="p-2 text-slate-400">{op.baseline}</td>
                      <td className="p-2 text-emerald-400 font-bold">{op.current}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* TAB 4: COMPLEXITY & FLOPS */}
        {activeTab === 'complexity' && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 flex flex-col">
              <span className="text-[10px] font-bold uppercase text-slate-400">Time Complexity</span>
              <span className="font-mono text-cyan-300 font-bold mt-1">
                {liveMath.complexity_table.time_complexity}
              </span>
              <span className="text-[10px] text-slate-500 mt-1">Linear sequence scaling</span>
            </div>

            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 flex flex-col">
              <span className="text-[10px] font-bold uppercase text-slate-400">Memory Footprint</span>
              <span className="font-mono text-emerald-300 font-bold mt-1">
                {liveMath.complexity_table.memory_complexity}
              </span>
              <span className="text-[10px] text-slate-500 mt-1">KV cache reduction</span>
            </div>

            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 flex flex-col">
              <span className="text-[10px] font-bold uppercase text-slate-400">Speedup Factor</span>
              <span className="font-mono text-amber-300 font-bold mt-1">
                {liveMath.complexity_table.speedup_vs_dense}
              </span>
              <span className="text-[10px] text-slate-500 mt-1">Measured on GPU</span>
            </div>

            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 flex flex-col">
              <span className="text-[10px] font-bold uppercase text-slate-400">FLOPs (N=1024)</span>
              <span className="font-mono text-purple-300 font-bold mt-1">
                {liveMath.complexity_table.flops_n1024}
              </span>
              <span className="text-[10px] text-slate-500 mt-1">Per forward pass</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
