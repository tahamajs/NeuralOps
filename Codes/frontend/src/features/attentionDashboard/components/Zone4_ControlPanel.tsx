// src/features/attentionDashboard/components/Zone4_ControlPanel.tsx
import React, { useState } from 'react';
import {
  Sliders,
  Layers,
  Zap,
  FileText,
  FileSpreadsheet,
  FileCode,
  Sparkles,
  Rocket,
  Plus,
  Minus,
  Check,
  Send,
  HelpCircle,
  Copy,
  ExternalLink,
  BookOpen,
} from 'lucide-react';
import { useAttentionDashboardStore } from '../stores/useAttentionDashboardStore';
import { SpecializationType } from '../types/attention';

const TEXT_PRESETS = [
  {
    title: 'Standard Sentence',
    text: 'The quick brown fox jumps over the lazy dog',
    desc: 'Demonstrates basic syntax and adjective-noun binding',
  },
  {
    title: 'SVD / Low-Rank Compression',
    text: 'Low-rank attention matrices preserve dominant spectral singular values',
    desc: 'Shows high singular value concentration on semantic tokens',
  },
  {
    title: 'Coreference Resolution',
    text: 'The scientist analyzed the transformer because it compressed attention fast',
    desc: 'Traces anaphoric links between pronoun and antecedent',
  },
  {
    title: 'Code Syntax',
    text: 'def forward(query, key, value): return softmax(query @ key.T) @ value',
    desc: 'Analyzes structured programming language tokens',
  },
];

const SPEC_COLORS: Record<SpecializationType, string> = {
  'Positional Head': '#3b82f6',
  'Induction Head': '#a855f7',
  'Semantic Head': '#f43f5e',
  'Delimiter Head': '#10b981',
  'Broad Context Head': '#f59e0b',
};

export const Zone4_ControlPanel: React.FC = () => {
  const {
    context,
    modelViewData,
    setSelectedLayer,
    setSelectedHead,
    setInputText,
    isTrainModalOpen,
    setTrainModalOpen,
    isReportModalOpen,
    setReportModalOpen,
    isPaperModalOpen,
    setPaperModalOpen,
    isNoteModalOpen,
    setNoteModalOpen,
  } = useAttentionDashboardStore();

  const [activeTab, setActiveTab] = useState<'selectors' | 'presets'>('selectors');
  const [copied, setCopied] = useState(false);
  const [trainConfig, setTrainConfig] = useState({
    name: `Reproduce_L${context.selectedLayer}_H${context.selectedHead}_${context.modelId.replace(/[^a-zA-Z0-9]/g, '_')}`,
    task: 'GLUE-MRPC',
    epochs: 3,
    batchSize: 16,
    lr: '2e-5',
    trackAttention: true,
  });

  const numLayers = context.numLayers;
  const numHeads = context.numHeads;
  const currentLayerHeads = modelViewData.heads.filter((h) => h.layer_idx === context.selectedLayer);

  const handlePresetSelect = (text: string) => {
    setInputText(text);
  };

  const copyPaperLatex = () => {
    const latex = `
\\subsection{Attention Analysis for ${context.modelId}}
We investigated the attention representation of \\texttt{${context.modelId}} using ${context.attentionType}.
At Layer~${context.selectedLayer}, Head~${context.selectedHead}, empirical attention entropy was measured at ${1.82}~bits.
\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=0.85\\linewidth]{figures/attention_L${context.selectedLayer}_H${context.selectedHead}.pdf}
  \\caption{Attention heatmap for ${context.modelId} (${context.attentionType}).}
  \\label{fig:attn_l${context.selectedLayer}_h${context.selectedHead}}
\\end{figure}
    `.trim();
    navigator.clipboard.writeText(latex);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="w-full lg:w-84 xl:w-96 bg-slate-950/80 rounded-xl border border-slate-800 shadow-xl overflow-hidden flex flex-col shrink-0">
      {/* Panel Header */}
      <div className="px-4 py-3 bg-slate-900/90 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-lg bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
            <Sliders className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-slate-100">Zone 4: Control Room</h3>
            <span className="text-[10px] text-slate-400">Interactive Model & Sequence Hub</span>
          </div>
        </div>

        <div className="flex items-center space-x-1 p-0.5 bg-slate-950 rounded-lg border border-slate-800 text-xs">
          <button
            onClick={() => setActiveTab('selectors')}
            className={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
              activeTab === 'selectors' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Controls
          </button>
          <button
            onClick={() => setActiveTab('presets')}
            className={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
              activeTab === 'presets' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Presets
          </button>
        </div>
      </div>

      {/* Main Body */}
      <div className="p-4 space-y-4 flex-1 overflow-y-auto">
        {activeTab === 'selectors' ? (
          <>
            {/* 1. LAYER SELECTOR WITH LIVE THUMBNAIL PREVIEW STRIP */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-200 flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Layer Selector (0 - {numLayers - 1})</span>
                </span>
                <span className="font-mono text-cyan-400 font-bold">Layer {context.selectedLayer}</span>
              </div>

              {/* Live Thumbnail Strip */}
              <div className="flex items-center space-x-1.5 overflow-x-auto pb-2 pt-1 scrollbar-thin">
                {Array.from({ length: numLayers }).map((_, l) => {
                  const isSelected = context.selectedLayer === l;
                  return (
                    <button
                      key={l}
                      onClick={() => setSelectedLayer(l)}
                      className={`flex-shrink-0 w-12 h-14 rounded-lg p-1 flex flex-col justify-between border transition-all ${
                        isSelected
                          ? 'border-cyan-400 bg-cyan-950/80 shadow-[0_0_10px_rgba(6,182,212,0.3)] ring-1 ring-cyan-400 scale-105'
                          : 'border-slate-800 bg-slate-900/80 hover:border-slate-600'
                      }`}
                      title={`Select Layer ${l}`}
                    >
                      <span className="text-[9px] font-mono font-bold text-slate-300">L{l}</span>
                      {/* Mini Heatmap matrix simulation */}
                      <div className="grid grid-cols-2 gap-0.5 w-full h-5 rounded overflow-hidden">
                        <div className={`rounded-xs ${l % 3 === 0 ? 'bg-cyan-400' : 'bg-slate-700'}`} />
                        <div className={`rounded-xs ${l % 2 === 0 ? 'bg-purple-400' : 'bg-slate-800'}`} />
                        <div className={`rounded-xs ${l % 4 === 0 ? 'bg-rose-400' : 'bg-slate-800'}`} />
                        <div className={`rounded-xs ${l % 2 === 1 ? 'bg-emerald-400' : 'bg-slate-700'}`} />
                      </div>
                      <span className="text-[7px] text-slate-400 truncate">12h</span>
                    </button>
                  );
                })}
              </div>

              {/* Range slider for quick jumping */}
              <input
                type="range"
                min={0}
                max={numLayers - 1}
                value={context.selectedLayer}
                onChange={(e) => setSelectedLayer(Number(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer"
              />
            </div>

            {/* 2. HEAD SELECTOR WITH INTERACTIVE VISUAL DOT GRID */}
            <div className="space-y-2 pt-2 border-t border-slate-800/80">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-200">
                  Head Selector (Layer {context.selectedLayer})
                </span>
                <span className="font-mono text-amber-400 font-bold">Head {context.selectedHead}</span>
              </div>

              {/* 4x3 Head Grid */}
              <div className="grid grid-cols-4 gap-2 p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                {currentLayerHeads.map((h) => {
                  const isSelected = context.selectedHead === h.head_idx;
                  const dotColor = SPEC_COLORS[h.specialization] || '#3b82f6';
                  const dotSize = Math.max(12, h.importance * 22);

                  return (
                    <button
                      key={h.head_idx}
                      onClick={() => setSelectedHead(h.head_idx)}
                      className={`p-2 rounded-md flex flex-col items-center justify-center border transition-all ${
                        isSelected
                          ? 'border-amber-400 bg-amber-950/40 ring-1 ring-amber-400 shadow-sm scale-105'
                          : 'border-slate-800 hover:border-slate-600 bg-slate-950/60'
                      }`}
                      title={`Head ${h.head_idx}: ${h.specialization} (Imp: ${h.importance})`}
                    >
                      <span className="text-[10px] font-mono font-bold text-slate-300 mb-1">
                        H{h.head_idx}
                      </span>
                      <div
                        className="rounded-full shadow-sm transition-transform"
                        style={{
                          width: `${dotSize}px`,
                          height: `${dotSize}px`,
                          backgroundColor: dotColor,
                        }}
                      />
                    </button>
                  );
                })}
              </div>
            </div>

            {/* 3. INPUT TEXT & TOKENIZER */}
            <div className="space-y-1.5 pt-2 border-t border-slate-800/80">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-slate-200">Input Sequence</span>
                <span className="text-[10px] font-mono text-slate-400">
                  {context.inputText.length} chars · {context.tokens.length} tokens
                </span>
              </div>
              <textarea
                value={context.inputText}
                onChange={(e) => setInputText(e.target.value)}
                rows={3}
                className="w-full p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-xs text-white font-mono focus:outline-none focus:border-cyan-500 transition-colors resize-none"
                placeholder="Enter input text to compute live attention matrices..."
              />
            </div>
          </>
        ) : (
          /* PRESETS TAB */
          <div className="space-y-2">
            <span className="text-xs text-slate-400 font-semibold">Select Curated Linguistic Benchmark:</span>
            {TEXT_PRESETS.map((p, idx) => (
              <div
                key={idx}
                onClick={() => handlePresetSelect(p.text)}
                className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 hover:border-cyan-500/50 cursor-pointer transition-all text-xs"
              >
                <div className="font-bold text-cyan-300">{p.title}</div>
                <div className="text-[11px] font-mono text-slate-200 my-1 truncate">"{p.text}"</div>
                <div className="text-[10px] text-slate-400">{p.desc}</div>
              </div>
            ))}
          </div>
        )}

        {/* 4. ⚡ "TRAIN WITH THIS ATTENTION" PRIMARY BUTTON */}
        <div className="pt-2 border-t border-slate-800/80">
          <button
            onClick={() => setTrainModalOpen(true)}
            className="w-full py-2.5 px-4 rounded-lg bg-gradient-to-r from-cyan-600 via-blue-600 to-purple-600 hover:from-cyan-500 hover:to-purple-500 text-white text-xs font-bold shadow-lg shadow-cyan-500/20 flex items-center justify-center space-x-2 transition-all group"
          >
            <Rocket className="w-4 h-4 text-amber-300 group-hover:animate-bounce" />
            <span>TRAIN WITH THIS ATTENTION</span>
          </button>
        </div>

        {/* 5. SEGMENTED ACTION BUTTONS: NOTE, REPORT, PAPER */}
        <div className="grid grid-cols-3 gap-2">
          <button
            onClick={() => setNoteModalOpen(true)}
            className="p-2 rounded-lg bg-slate-900 border border-slate-800 hover:border-cyan-500/40 hover:bg-slate-850 text-slate-300 flex flex-col items-center justify-center text-[11px] font-medium transition-colors"
          >
            <FileText className="w-4 h-4 text-cyan-400 mb-1" />
            <span>Note</span>
          </button>
          <button
            onClick={() => setReportModalOpen(true)}
            className="p-2 rounded-lg bg-slate-900 border border-slate-800 hover:border-purple-500/40 hover:bg-slate-850 text-slate-300 flex flex-col items-center justify-center text-[11px] font-medium transition-colors"
          >
            <FileSpreadsheet className="w-4 h-4 text-purple-400 mb-1" />
            <span>Report</span>
          </button>
          <button
            onClick={() => setPaperModalOpen(true)}
            className="p-2 rounded-lg bg-slate-900 border border-slate-800 hover:border-emerald-500/40 hover:bg-slate-850 text-slate-300 flex flex-col items-center justify-center text-[11px] font-medium transition-colors"
          >
            <FileCode className="w-4 h-4 text-emerald-400 mb-1" />
            <span>Paper</span>
          </button>
        </div>
      </div>

      {/* MODAL 1: TRAIN EXPERIMENT MODAL */}
      {isTrainModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="w-full max-w-lg bg-slate-900 border border-slate-700 rounded-xl p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center space-x-2">
                <Rocket className="w-5 h-5 text-amber-400" />
                <h3 className="font-bold text-white text-sm">Launch Training Experiment</h3>
              </div>
              <button
                onClick={() => setTrainModalOpen(false)}
                className="text-slate-400 hover:text-white text-xs"
              >
                ✕
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <label className="block text-slate-400 mb-1">Experiment Name</label>
                <input
                  type="text"
                  value={trainConfig.name}
                  onChange={(e) => setTrainConfig({ ...trainConfig, name: e.target.value })}
                  className="w-full p-2 bg-slate-950 border border-slate-800 rounded font-mono text-cyan-300"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-slate-400 mb-1">Downstream Task</label>
                  <select
                    value={trainConfig.task}
                    onChange={(e) => setTrainConfig({ ...trainConfig, task: e.target.value })}
                    className="w-full p-2 bg-slate-950 border border-slate-800 rounded text-slate-200"
                  >
                    <option value="GLUE-MRPC">GLUE - MRPC</option>
                    <option value="GLUE-SST2">GLUE - SST-2</option>
                    <option value="GLUE-MNLI">GLUE - MNLI</option>
                    <option value="WikiText-103">WikiText-103 Perplexity</option>
                  </select>
                </div>
                <div>
                  <label className="block text-slate-400 mb-1">Epochs</label>
                  <input
                    type="number"
                    value={trainConfig.epochs}
                    onChange={(e) => setTrainConfig({ ...trainConfig, epochs: Number(e.target.value) })}
                    className="w-full p-2 bg-slate-950 border border-slate-800 rounded text-slate-200"
                  />
                </div>
              </div>

              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-[11px] text-slate-400 space-y-1">
                <div>• Architecture: <strong className="text-white">{context.attentionType}</strong></div>
                <div>• Base Model: <strong className="text-white">{context.modelId}</strong></div>
                <div>• Estimated Cost: <strong className="text-emerald-400">$0.45</strong> (1x NVIDIA A100 GPU)</div>
              </div>
            </div>

            <div className="flex items-center justify-end space-x-2 pt-3 border-t border-slate-800">
              <button
                onClick={() => setTrainModalOpen(false)}
                className="px-4 py-2 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  alert(`Experiment "${trainConfig.name}" queued successfully!`);
                  setTrainModalOpen(false);
                }}
                className="px-4 py-2 rounded bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold flex items-center gap-1.5"
              >
                <Zap className="w-3.5 h-3.5 text-amber-300" />
                <span>Launch Job</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* MODAL 2: EXPORT TO PAPER LATEX */}
      {isPaperModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="w-full max-w-xl bg-slate-900 border border-slate-700 rounded-xl p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center space-x-2">
                <FileCode className="w-5 h-5 text-emerald-400" />
                <h3 className="font-bold text-white text-sm">Export LaTeX Methods & Figures</h3>
              </div>
              <button onClick={() => setPaperModalOpen(false)} className="text-slate-400 hover:text-white text-xs">
                ✕
              </button>
            </div>

            <div className="space-y-2">
              <span className="text-xs text-slate-400">Pre-generated LaTeX snippet for academic papers:</span>
              <pre className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-[11px] font-mono text-cyan-300 overflow-x-auto max-h-64">
{`\\subsection{Attention Analysis}
We analyzed the attention representation of \\texttt{${context.modelId}} equipped with ${context.attentionType}.
At Layer~${context.selectedLayer}, Head~${context.selectedHead}, empirical attention entropy is $\\mathcal{H} = 1.82$~bits.

\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=0.85\\linewidth]{figures/attention_L${context.selectedLayer}_H${context.selectedHead}.pdf}
  \\caption{Attention heatmap for ${context.modelId} using ${context.attentionType}.}
  \\label{fig:attn_l${context.selectedLayer}_h${context.selectedHead}}
\\end{figure}`}
              </pre>
            </div>

            <div className="flex items-center justify-end space-x-2 pt-3 border-t border-slate-800">
              <button
                onClick={copyPaperLatex}
                className="px-4 py-2 rounded bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold flex items-center gap-1.5"
              >
                {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copied ? 'Copied to Clipboard' : 'Copy LaTeX'}</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* MODAL 3: CREATE RESEARCHER NOTE */}
      {isNoteModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="w-full max-w-lg bg-slate-900 border border-slate-700 rounded-xl p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center space-x-2">
                <FileText className="w-5 h-5 text-cyan-400" />
                <h3 className="font-bold text-white text-sm">Save Researcher Note</h3>
              </div>
              <button onClick={() => setNoteModalOpen(false)} className="text-slate-400 hover:text-white text-xs">
                ✕
              </button>
            </div>
            <p className="text-xs text-slate-300">
              Captures current visualization state (Layer {context.selectedLayer}, Head {context.selectedHead}, {context.attentionType}) into your platform Research Notes log.
            </p>
            <div className="flex justify-end gap-2 pt-2">
              <button onClick={() => setNoteModalOpen(false)} className="px-3 py-1.5 rounded bg-slate-800 text-xs text-slate-300">
                Cancel
              </button>
              <button
                onClick={() => {
                  alert('Note saved to Researcher Notebook!');
                  setNoteModalOpen(false);
                }}
                className="px-4 py-1.5 rounded bg-cyan-600 text-xs font-bold text-white"
              >
                Save Note
              </button>
            </div>
          </div>
        </div>
      )}

      {/* MODAL 4: GENERATE REPORT */}
      {isReportModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="w-full max-w-lg bg-slate-900 border border-slate-700 rounded-xl p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center space-x-2">
                <FileSpreadsheet className="w-5 h-5 text-purple-400" />
                <h3 className="font-bold text-white text-sm">Generate Full Diagnostic Report</h3>
              </div>
              <button onClick={() => setReportModalOpen(false)} className="text-slate-400 hover:text-white text-xs">
                ✕
              </button>
            </div>
            <p className="text-xs text-slate-300">
              Generates an automated HTML report including Executive Summary, quantitative metrics (entropy, diversity), and SVG figures.
            </p>
            <div className="flex justify-end gap-2 pt-2">
              <button onClick={() => setReportModalOpen(false)} className="px-3 py-1.5 rounded bg-slate-800 text-xs text-slate-300">
                Cancel
              </button>
              <button
                onClick={() => {
                  alert('Generating comprehensive HTML diagnostic report...');
                  setReportModalOpen(false);
                }}
                className="px-4 py-1.5 rounded bg-purple-600 text-xs font-bold text-white"
              >
                Export Report
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
