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
  { id: 'SVD-Linear-Transformer-v2', name: 'SVD Linear Transformer v2 (1.3B)', params: '1.3B', type: 'Riemannian SVD' },
  { id: 'DeepSeek-V3-MLA', name: 'DeepSeek-V3 (671B MLA MoE)', params: '671B', type: 'Latent Attention' },
  { id: 'LLaMA-3-8B-Instruct', name: 'LLaMA-3 (8B GQA)', params: '8B', type: 'Decoder' },
  { id: 'Griffin-3B-SSD', name: 'Griffin-SSD (3B Recurrent)', params: '3B', type: 'Linear RNN' },
  { id: 'Hymba-TransX-1.5B', name: 'Hymba-TransX (1.5B SSM Hybrid)', params: '1.5B', type: 'Hybrid SSM' },
  { id: 'Mistral-7B-v0.1', name: 'Mistral-7B (GQA 8-KV)', params: '7B', type: 'Decoder' },
  { id: 'BERT-base-uncased', name: 'BERT-base (110M)', params: '110M', type: 'Encoder' },
  { id: 'RoBERTa-large', name: 'RoBERTa-large (355M)', params: '355M', type: 'Encoder' },
  { id: 'GPT-2-Large', name: 'GPT-2 (774M)', params: '774M', type: 'Autoregressive' },
];

export const ALL_ATTENTION_MODULES = [
  // SVD & Manifold Compression
  {
    id: 'SVD Linear Attention (rank=64)',
    name: 'SVD Linear Attention',
    category: 'svd',
    rank: 64,
    complexity: 'O(N · r · d)',
    speedup: '2.4×',
    vramReduction: '-42%',
    color: '#06b6d4',
    badge: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40',
    desc: 'Truncated SVD low-rank decomposition with Riemannian Stiefel manifold QR retraction',
  },
  {
    id: 'Dynamic Entropy SVD Compressor',
    name: 'Dynamic Entropy SVD (r_l ∝ H(Σ_l))',
    category: 'svd',
    rank: 32,
    complexity: 'O(N · r_l · d)',
    speedup: '2.8×',
    vramReduction: '-48%',
    color: '#38bdf8',
    badge: 'bg-sky-500/20 text-sky-300 border-sky-500/40',
    desc: 'Von Neumann singular entropy rank allocation allocating higher capacity to critical layers',
  },
  {
    id: 'ResSVD Residual Low-Rank Attention',
    name: 'ResSVD Residual Attention',
    category: 'svd',
    rank: 16,
    complexity: 'O(N · (r + k) · d)',
    speedup: '2.1×',
    vramReduction: '-38%',
    color: '#818cf8',
    badge: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40',
    desc: 'Dual-path rank decomposition: primary low-rank core + residual error recovery branch',
  },
  {
    id: 'ASVD Adaptive Singular Pruning',
    name: 'ASVD Adaptive Pruning',
    category: 'svd',
    rank: 48,
    complexity: 'O(N · r · d)',
    speedup: '2.2×',
    vramReduction: '-35%',
    color: '#a78bfa',
    badge: 'bg-violet-500/20 text-violet-300 border-violet-500/40',
    desc: 'Activation-aware singular value scaling mitigating outlier channel degradation',
  },

  // State-of-the-Art Linear & Recurrent
  {
    id: 'FlashAttention-3 (Hopper FP8)',
    name: 'FlashAttention-3 (Hopper FP8)',
    category: 'linear',
    complexity: 'O(N² · d) in-SRAM',
    speedup: '2.2×',
    vramReduction: '-50%',
    color: '#f43f5e',
    badge: 'bg-rose-500/20 text-rose-300 border-rose-500/40',
    desc: 'Asynchronous ping-pong TMA GEMMs with hardware-accelerated FP8 tensor cores',
  },
  {
    id: 'Griffin-SSD & Hawk Recurrent',
    name: 'Griffin-SSD (Gated Linear RNN)',
    category: 'linear',
    complexity: 'O(N · d)',
    speedup: '3.1×',
    vramReduction: '-60%',
    color: '#10b981',
    badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
    desc: 'Gated linear recurrence combining local 1D depthwise convolutions with sliding context',
  },
  {
    id: 'Gated DeltaNet & HGRN2',
    name: 'Gated DeltaNet (Delta Rule)',
    category: 'linear',
    complexity: 'O(N · d²)',
    speedup: '2.6×',
    vramReduction: '-45%',
    color: '#34d399',
    badge: 'bg-teal-500/20 text-teal-300 border-teal-500/40',
    desc: 'Associative memory updates with delta-rule error correction and decay gating',
  },
  {
    id: 'MLA Multi-Head Latent Attention',
    name: 'MLA Latent Attention (DeepSeek-V3)',
    category: 'linear',
    complexity: 'O(N · d_c)',
    speedup: '2.5×',
    vramReduction: '-75% KV',
    color: '#ec4899',
    badge: 'bg-pink-500/20 text-pink-300 border-pink-500/40',
    desc: 'Low-rank KV compression into shared latent vector c_t, eliminating KV cache bottlenecks',
  },
  {
    id: 'Hymba-TransX Hybrid Attention SSM',
    name: 'Hymba-TransX (Attention + SSM)',
    category: 'hybrid',
    complexity: 'O(N · d + N_local²)',
    speedup: '2.7×',
    vramReduction: '-52%',
    color: '#f59e0b',
    badge: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
    desc: 'Parallel fusion of linear state-space models and local cross-attention heads',
  },
  {
    id: 'Jamba2 & Samba-MoE',
    name: 'Jamba2 (Mamba-2 + MoE Hybrid)',
    category: 'hybrid',
    complexity: 'O(N · d)',
    speedup: '3.0×',
    vramReduction: '-55%',
    color: '#eab308',
    badge: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/40',
    desc: 'Interleaved Mamba-2 SSD blocks with Mixture-of-Experts transformer layers',
  },
  {
    id: 'NSA Native Sparse Attention',
    name: 'NSA Native Sparse Attention',
    category: 'sparse',
    complexity: 'O(N · k · d)',
    speedup: '2.9×',
    vramReduction: '-62%',
    color: '#8b5cf6',
    badge: 'bg-purple-500/20 text-purple-300 border-purple-500/40',
    desc: 'Hardware-aligned block-sparse token clustering with top-k dynamic routing',
  },
  {
    id: 'Infini-Attention Compressive Memory',
    name: 'Infini-Attention (1M+ Context)',
    category: 'sparse',
    complexity: 'O(N · d_mem)',
    speedup: '2.8×',
    vramReduction: '-70%',
    color: '#6366f1',
    badge: 'bg-indigo-500/20 text-indigo-300 border-indigo-500/40',
    desc: 'Compressive memory bounded recurrent state managing million-token long contexts',
  },
  {
    id: 'RetNet Retentive Network',
    name: 'RetNet Retentive Network',
    category: 'linear',
    complexity: 'O(N · d)',
    speedup: '3.2×',
    vramReduction: '-58%',
    color: '#14b8a6',
    badge: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40',
    desc: 'Tri-paradigm formulation: parallel training, recurrent inference, and chunkwise processing',
  },
  {
    id: 'RWKV-6 Eagle Time-Mixing',
    name: 'RWKV-6 Eagle (Linear RNN)',
    category: 'linear',
    complexity: 'O(N · d)',
    speedup: '3.4×',
    vramReduction: '-65%',
    color: '#0ea5e9',
    badge: 'bg-blue-500/20 text-blue-300 border-blue-500/40',
    desc: 'Data-dependent decay time-mixing linear attention with constant-memory O(1) generation',
  },

  // Randomized & Kernel Feature Maps
  {
    id: 'Galerkin Attention (rank=64)',
    name: 'Petrov-Galerkin Attention',
    category: 'svd',
    rank: 64,
    complexity: 'O(N · r · d)',
    speedup: '2.0×',
    vramReduction: '-40%',
    color: '#3b82f6',
    badge: 'bg-blue-500/20 text-blue-300 border-blue-500/40',
    desc: 'Petrov-Galerkin projection with randomized sketching and orthogonal test functions',
  },
  {
    id: 'Performer (FAVOR+ Positive Random)',
    name: 'Performer (FAVOR+ Random Features)',
    category: 'linear',
    complexity: 'O(N · m · d)',
    speedup: '2.3×',
    vramReduction: '-45%',
    color: '#059669',
    badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
    desc: 'Positive orthogonal random features approximating softmax kernel without variance blowup',
  },
  {
    id: 'Linformer Low-Rank Projections',
    name: 'Linformer Low-Rank Projection',
    category: 'linear',
    complexity: 'O(N · k)',
    speedup: '2.1×',
    vramReduction: '-40%',
    color: '#d97706',
    badge: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
    desc: 'Linear dimension reduction matrices projecting key-value sequences to length k',
  },

  // Standard Reference Baselines
  {
    id: 'Softmax MHA',
    name: 'Softmax Multi-Head Attention (Standard)',
    category: 'standard',
    complexity: 'O(N² · d)',
    speedup: '1.0×',
    vramReduction: '0%',
    color: '#a855f7',
    badge: 'bg-purple-500/20 text-purple-300 border-purple-500/40',
    desc: 'Classic quadratic scaled dot-product multi-head attention baseline (Vaswani et al.)',
  },
  {
    id: 'Grouped Query Attention (GQA)',
    name: 'Grouped Query Attention (GQA)',
    category: 'standard',
    complexity: 'O(N² · d_kv)',
    speedup: '1.4×',
    vramReduction: '-50% KV',
    color: '#f59e0b',
    badge: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
    desc: 'Grouped Key-Value heads shared across multiple query heads for balanced throughput',
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
  const [attnSearch, setAttnSearch] = useState('');
  const [attnCategory, setAttnCategory] = useState<'all' | 'svd' | 'linear' | 'hybrid' | 'sparse' | 'standard'>('all');

  const currentAttn = ALL_ATTENTION_MODULES.find((a) => a.id === context.attentionType) || ALL_ATTENTION_MODULES[0];

  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setInputText(tempText);
    setIsEditingText(false);
  };

  const filteredModels = MODEL_PRESETS.filter((m) =>
    m.name.toLowerCase().includes(modelSearch.toLowerCase()) || m.type.toLowerCase().includes(modelSearch.toLowerCase())
  );

  const filteredAttentions = ALL_ATTENTION_MODULES.filter((a) => {
    const matchesCat = attnCategory === 'all' || a.category === attnCategory;
    const matchesSearch =
      a.name.toLowerCase().includes(attnSearch.toLowerCase()) ||
      a.desc.toLowerCase().includes(attnSearch.toLowerCase()) ||
      a.complexity.toLowerCase().includes(attnSearch.toLowerCase());
    return matchesCat && matchesSearch;
  });

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
              <div className="absolute left-0 mt-2 w-96 bg-slate-900/95 backdrop-blur-xl border border-slate-700 rounded-2xl shadow-2xl p-3 z-50 space-y-2">
                {/* Search Bar */}
                <div className="relative">
                  <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-slate-400" />
                  <input
                    type="text"
                    value={attnSearch}
                    onChange={(e) => setAttnSearch(e.target.value)}
                    placeholder="Search 20+ Attention & SVD modules..."
                    className="w-full pl-8 pr-2 py-1.5 bg-slate-950 border border-slate-700 rounded-xl text-xs text-white focus:outline-none focus:border-cyan-500"
                  />
                </div>

                {/* Category Filtering Pills */}
                <div className="flex items-center gap-1 overflow-x-auto pb-1 scrollbar-none text-[10px] font-mono">
                  {[
                    { id: 'all', label: 'All' },
                    { id: 'svd', label: 'SVD & Manifold' },
                    { id: 'linear', label: 'Linear & RNN' },
                    { id: 'hybrid', label: 'SSM Hybrid' },
                    { id: 'sparse', label: 'Sparse' },
                    { id: 'standard', label: 'Baselines' },
                  ].map((cat) => (
                    <button
                      key={cat.id}
                      onClick={() => setAttnCategory(cat.id as any)}
                      className={`px-2 py-0.5 rounded-lg whitespace-nowrap transition-colors cursor-pointer ${
                        attnCategory === cat.id
                          ? 'bg-cyan-500 text-black font-bold'
                          : 'bg-white/5 text-slate-400 hover:text-white hover:bg-white/10'
                      }`}
                    >
                      {cat.label}
                    </button>
                  ))}
                </div>

                {/* Module List */}
                <div className="space-y-1 max-h-72 overflow-y-auto pr-1 scrollbar-thin">
                  {filteredAttentions.map((a) => (
                    <button
                      key={a.id}
                      onClick={() => {
                        setAttentionType(a.id);
                        setIsAttnDropdownOpen(false);
                      }}
                      className={`w-full text-left p-2.5 rounded-xl text-xs transition-all cursor-pointer ${
                        context.attentionType === a.id
                          ? 'bg-cyan-950/40 border border-cyan-500/50 shadow-md'
                          : 'hover:bg-slate-800/60 border border-transparent'
                      }`}
                    >
                      <div className="flex items-center justify-between font-semibold text-slate-200 mb-1">
                        <span className="flex items-center gap-1.5 text-xs text-white font-bold">
                          <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: a.color }} />
                          {a.name}
                        </span>
                        <div className="flex items-center gap-1">
                          {a.speedup && (
                            <span className="text-[9px] font-mono font-bold px-1.5 py-0.2 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                              {a.speedup}
                            </span>
                          )}
                          <span className="font-mono text-[9px] px-1.5 py-0.2 rounded bg-slate-950 text-cyan-300 border border-white/5">
                            {a.complexity}
                          </span>
                        </div>
                      </div>
                      <p className="text-[10px] text-slate-400 leading-tight font-light">{a.desc}</p>
                    </button>
                  ))}
                </div>
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
