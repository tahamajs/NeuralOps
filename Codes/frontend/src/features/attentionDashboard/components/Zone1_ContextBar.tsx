// src/features/attentionDashboard/components/Zone1_ContextBar.tsx
import React, { useState } from 'react';
import {
  Brain,
  Layers,
  ChevronRight,
  History,
  Star,
  Sparkles,
  Search,
  ExternalLink,
  Plus,
  Minus,
  Check,
  RotateCcw,
  BookOpen,
  Info,
} from 'lucide-react';
import { useAttentionDashboardStore } from '../stores/useAttentionDashboardStore';

const MODEL_PRESETS = [
  { id: 'BERT-base-uncased', name: 'BERT-base (110M)', params: '110M', type: 'Encoder' },
  { id: 'GPT-2 (fine-tuned)', name: 'GPT-2 (124M)', params: '124M', type: 'Autoregressive' },
  { id: 'RoBERTa-large', name: 'RoBERTa-large (355M)', params: '355M', type: 'Encoder' },
  { id: 'LLaMA-2-7B-LoRA', name: 'LLaMA-2-7B (SVD rank 64)', params: '7B', type: 'Decoder' },
  { id: 'Mistral-7B-v0.1', name: 'Mistral-7B (GQA 8-KV)', params: '7B', type: 'Decoder' },
  { id: 'SVD-Linear-Transformer', name: 'SVD Linear Transformer', params: '85M', type: 'Hybrid' },
];

const ATTENTION_TYPES = [
  {
    id: 'SVD Linear Attention (rank=64)',
    name: 'SVD Linear Attention',
    rank: 64,
    complexity: 'O(N · r · d)',
    color: '#06b6d4',
    badge: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/40',
    desc: 'Truncated SVD low-rank decomposition with adaptive energy preservation',
  },
  {
    id: 'Galerkin Attention (rank=64)',
    name: 'Galerkin Attention',
    rank: 64,
    complexity: 'O(N · r · d)',
    color: '#3b82f6',
    badge: 'bg-blue-500/20 text-blue-400 border-blue-500/40',
    desc: 'Petrov-Galerkin projection with randomized sketching',
  },
  {
    id: 'Softmax MHA',
    name: 'Softmax MHA (Standard)',
    complexity: 'O(N² · d)',
    color: '#a855f7',
    badge: 'bg-purple-500/20 text-purple-400 border-purple-500/40',
    desc: 'Classic quadratic scaled dot-product multi-head attention',
  },
  {
    id: 'Grouped Query Attention (GQA)',
    name: 'Grouped Query Attention (GQA)',
    complexity: 'O(N² · d_kv)',
    color: '#f59e0b',
    badge: 'bg-amber-500/20 text-amber-400 border-amber-500/40',
    desc: 'Shared Key/Value heads across grouped query heads',
  },
  {
    id: 'Linear Kernel Attention',
    name: 'Linear Kernel Attention',
    complexity: 'O(N · d²)',
    color: '#10b981',
    badge: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40',
    desc: 'Kernel feature map with linear matrix associativity',
  },
];

export const Zone1_ContextBar: React.FC = () => {
  const {
    context,
    history,
    setModel,
    setAttentionType,
    setSelectedLayer,
    setSelectedHead,
    setInputText,
    isHistoryOpen,
    setHistoryOpen,
    restoreHistoryEntry,
    toggleStarHistory,
  } = useAttentionDashboardStore();

  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);
  const [isAttnDropdownOpen, setIsAttnDropdownOpen] = useState(false);
  const [isEditingText, setIsEditingText] = useState(false);
  const [tempText, setTempText] = useState(context.inputText);
  const [modelSearch, setModelSearch] = useState('');

  const currentAttn = ATTENTION_TYPES.find((a) => a.id === context.attentionType) || ATTENTION_TYPES[0];

  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setInputText(tempText);
    setIsEditingText(false);
  };

  const filteredModels = MODEL_PRESETS.filter((m) =>
    m.name.toLowerCase().includes(modelSearch.toLowerCase()) || m.type.toLowerCase().includes(modelSearch.toLowerCase())
  );

  return (
    <header className="sticky top-0 z-50 w-full bg-slate-950/90 backdrop-blur-md border-b border-slate-800/80 px-4 py-2.5 shadow-lg">
      <div className="max-w-[1920px] mx-auto flex flex-wrap items-center justify-between gap-3">
        {/* Left: Breadcrumbs & History Trigger */}
        <div className="flex items-center space-x-2 text-xs">
          <div className="flex items-center space-x-1.5 font-semibold text-slate-300">
            <Brain className="w-4 h-4 text-cyan-400" />
            <span className="hidden sm:inline text-slate-400">NeuralOps</span>
            <ChevronRight className="w-3.5 h-3.5 text-slate-600" />
            <span className="text-slate-200">Attention Lab</span>
            <ChevronRight className="w-3.5 h-3.5 text-slate-600" />
            <span className="text-cyan-400 font-mono font-medium">{context.modelId}</span>
          </div>

          {/* History Button */}
          <div className="relative">
            <button
              onClick={() => setHistoryOpen(!isHistoryOpen)}
              className="flex items-center space-x-1 px-2 py-1 rounded bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700/60 transition-colors"
              title="Exploration History"
            >
              <History className="w-3.5 h-3.5 text-amber-400" />
              <span className="text-[11px] font-mono">{history.length}</span>
            </button>

            {/* History Dropdown */}
            {isHistoryOpen && (
              <div className="absolute left-0 mt-2 w-80 bg-slate-900 border border-slate-700/80 rounded-lg shadow-2xl p-3 z-50 max-h-96 overflow-y-auto">
                <div className="flex items-center justify-between pb-2 border-b border-slate-800 mb-2">
                  <span className="font-semibold text-slate-200 text-xs flex items-center gap-1.5">
                    <History className="w-3.5 h-3.5 text-cyan-400" /> Exploration History
                  </span>
                  <span className="text-[10px] text-slate-400">{history.length} saved</span>
                </div>
                <div className="space-y-1.5">
                  {history.map((h) => (
                    <div
                      key={h.id}
                      className="p-2 rounded bg-slate-800/50 hover:bg-slate-800 border border-slate-700/40 text-xs flex items-center justify-between cursor-pointer group transition-all"
                      onClick={() => restoreHistoryEntry(h)}
                    >
                      <div className="flex-1 min-w-0 pr-2">
                        <div className="font-medium text-slate-200 truncate flex items-center gap-1.5">
                          <span className="w-2 h-2 rounded-full bg-cyan-400" />
                          {h.modelId} · L{h.layer}H{h.head}
                        </div>
                        <div className="text-[10px] text-slate-400 truncate">{h.textSnippet}</div>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleStarHistory(h.id);
                        }}
                        className="p-1 hover:text-amber-400 text-slate-500"
                      >
                        <Star className={`w-3.5 h-3.5 ${h.isStarred ? 'text-amber-400 fill-amber-400' : ''}`} />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Center: Context Chips Chain */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Model Chip */}
          <div className="relative">
            <button
              onClick={() => setIsModelDropdownOpen(!isModelDropdownOpen)}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-md bg-slate-900 border border-slate-700/80 hover:border-cyan-500/60 transition-all text-xs group"
            >
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Model:</span>
              <span className="font-semibold text-white group-hover:text-cyan-300 transition-colors">
                {context.modelId}
              </span>
            </button>

            {isModelDropdownOpen && (
              <div className="absolute left-0 mt-2 w-72 bg-slate-900 border border-slate-700 rounded-lg shadow-2xl p-2 z-50">
                <div className="relative mb-2">
                  <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-slate-400" />
                  <input
                    type="text"
                    value={modelSearch}
                    onChange={(e) => setModelSearch(e.target.value)}
                    placeholder="Search models..."
                    className="w-full pl-8 pr-2 py-1.5 bg-slate-950 border border-slate-700 rounded text-xs text-white focus:outline-none focus:border-cyan-500"
                  />
                </div>
                <div className="space-y-1 max-h-56 overflow-y-auto">
                  {filteredModels.map((m) => (
                    <button
                      key={m.id}
                      onClick={() => {
                        setModel(m.id);
                        setIsModelDropdownOpen(false);
                      }}
                      className={`w-full text-left p-2 rounded text-xs flex items-center justify-between transition-colors ${
                        context.modelId === m.id
                          ? 'bg-cyan-950/60 border border-cyan-500/40 text-cyan-300'
                          : 'hover:bg-slate-800 text-slate-300'
                      }`}
                    >
                      <div>
                        <div className="font-medium text-slate-100">{m.name}</div>
                        <div className="text-[10px] text-slate-400">{m.params} · {m.type}</div>
                      </div>
                      {context.modelId === m.id && <Check className="w-4 h-4 text-cyan-400" />}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Attention Type Chip */}
          <div className="relative">
            <button
              onClick={() => setIsAttnDropdownOpen(!isAttnDropdownOpen)}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-md bg-slate-900 border border-slate-700/80 hover:border-cyan-500/60 transition-all text-xs group"
            >
              <span
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: currentAttn.color }}
              />
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Attn:</span>
              <span className="font-semibold text-white group-hover:text-cyan-300 transition-colors">
                {currentAttn.name}
              </span>
            </button>

            {isAttnDropdownOpen && (
              <div className="absolute left-0 mt-2 w-84 bg-slate-900 border border-slate-700 rounded-lg shadow-2xl p-2 z-50 space-y-1">
                {ATTENTION_TYPES.map((a) => (
                  <button
                    key={a.id}
                    onClick={() => {
                      setAttentionType(a.id);
                      setIsAttnDropdownOpen(false);
                    }}
                    className={`w-full text-left p-2.5 rounded text-xs transition-all ${
                      context.attentionType === a.id
                        ? 'bg-slate-800 border border-cyan-500/50'
                        : 'hover:bg-slate-800/60 border border-transparent'
                    }`}
                  >
                    <div className="flex items-center justify-between font-semibold text-slate-200">
                      <span className="flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: a.color }} />
                        {a.name}
                      </span>
                      <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-slate-950 text-slate-400">
                        {a.complexity}
                      </span>
                    </div>
                    <p className="text-[10px] text-slate-400 mt-1">{a.desc}</p>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Text Input Chip */}
          {isEditingText ? (
            <form onSubmit={handleTextSubmit} className="flex items-center">
              <input
                type="text"
                value={tempText}
                onChange={(e) => setTempText(e.target.value)}
                autoFocus
                onBlur={() => setIsEditingText(false)}
                className="px-2.5 py-1 bg-slate-950 border border-cyan-500 rounded text-xs text-white focus:outline-none w-56 font-mono"
              />
            </form>
          ) : (
            <button
              onClick={() => {
                setTempText(context.inputText);
                setIsEditingText(true);
              }}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-md bg-slate-900 border border-slate-700/80 hover:border-cyan-500/60 transition-all text-xs max-w-[200px] truncate"
              title="Click to edit input sequence text"
            >
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Text:</span>
              <span className="text-slate-300 font-mono truncate">"{context.inputText}"</span>
            </button>
          )}

          {/* Layer Stepper Chip */}
          <div className="flex items-center rounded-md bg-slate-900 border border-slate-700/80 p-0.5 text-xs">
            <span className="px-2 text-[10px] font-bold uppercase tracking-wider text-slate-400">Layer</span>
            <button
              onClick={() => setSelectedLayer(Math.max(0, context.selectedLayer - 1))}
              disabled={context.selectedLayer === 0}
              className="p-1 hover:bg-slate-800 disabled:opacity-30 rounded text-slate-300"
            >
              <Minus className="w-3 h-3" />
            </button>
            <span className="px-1.5 font-mono font-bold text-cyan-400">{context.selectedLayer}</span>
            <button
              onClick={() => setSelectedLayer(Math.min(context.numLayers - 1, context.selectedLayer + 1))}
              disabled={context.selectedLayer === context.numLayers - 1}
              className="p-1 hover:bg-slate-800 disabled:opacity-30 rounded text-slate-300"
            >
              <Plus className="w-3 h-3" />
            </button>
          </div>

          {/* Head Stepper Chip */}
          <div className="flex items-center rounded-md bg-slate-900 border border-slate-700/80 p-0.5 text-xs">
            <span className="px-2 text-[10px] font-bold uppercase tracking-wider text-slate-400">Head</span>
            <button
              onClick={() => setSelectedHead(Math.max(0, context.selectedHead - 1))}
              disabled={context.selectedHead === 0}
              className="p-1 hover:bg-slate-800 disabled:opacity-30 rounded text-slate-300"
            >
              <Minus className="w-3 h-3" />
            </button>
            <span className="px-1.5 font-mono font-bold text-amber-400">{context.selectedHead}</span>
            <button
              onClick={() => setSelectedHead(Math.min(context.numHeads - 1, context.selectedHead + 1))}
              disabled={context.selectedHead === context.numHeads - 1}
              className="p-1 hover:bg-slate-800 disabled:opacity-30 rounded text-slate-300"
            >
              <Plus className="w-3 h-3" />
            </button>
          </div>
        </div>

        {/* Right Actions & Status */}
        <div className="flex items-center space-x-2 text-xs">
          <div className="hidden xl:flex items-center gap-2 px-2.5 py-1 rounded bg-slate-900/80 border border-slate-800 text-[11px] text-slate-400">
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              <span>Reactive GPU WebGL</span>
            </span>
          </div>
        </div>
      </div>
    </header>
  );
};
