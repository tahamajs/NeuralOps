// src/features/transformerStudios/ComputationStudios.tsx
import React, { useState, useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip as RechartsTooltip,
  CartesianGrid,
  Legend,
  BarChart,
  Bar,
  AreaChart,
  Area,
} from 'recharts';
import {
  Cpu,
  Layers,
  Activity,
  Maximize2,
  Sliders,
  Play,
  RotateCcw,
  Sparkles,
  Zap,
  Target,
  ArrowRight,
  TrendingUp,
  Flame,
  CheckCircle2,
  AlertTriangle,
  Info,
  Shield,
  HelpCircle,
  Database,
  BarChart3,
  Split,
  Search,
  Hash,
  Binary,
} from 'lucide-react';
import { Badge } from '../../components/ui/Badge';
import { Button } from '../../components/ui/Button';
import { ErrorBoundary } from '../../components/ui/ErrorBoundary';
import type {
  ComputationStudioTab,
  FFNNeuronRecord,
  SAEFeatureRecord,
  KVCacheMemoryPoint,
  DeltaKVSinkRecord,
  SingularValuePoint,
  ModelPreset,
} from './types';

// =====================================================================
// Model Presets
// =====================================================================
const MODEL_PRESETS: ModelPreset[] = [
  {
    id: 'llama-3-8b',
    name: 'LLaMA-3 8B',
    family: 'LLaMA',
    dModel: 4096,
    numLayers: 32,
    numHeads: 32,
    vocabSize: 128256,
    normType: 'RMSNorm',
    posType: 'RoPE',
    baseFrequency: 500000,
  },
  {
    id: 'mistral-7b',
    name: 'Mistral 7B v0.3',
    family: 'Mistral',
    dModel: 4096,
    numLayers: 32,
    numHeads: 32,
    vocabSize: 32768,
    normType: 'RMSNorm',
    posType: 'RoPE',
    baseFrequency: 1000000,
  },
  {
    id: 'gpt2-medium',
    name: 'GPT-2 Medium',
    family: 'GPT',
    dModel: 1024,
    numLayers: 24,
    numHeads: 16,
    vocabSize: 50257,
    normType: 'LayerNorm',
    posType: 'Sinusoidal',
    baseFrequency: 10000,
  },
  {
    id: 'bert-base',
    name: 'BERT Base (Uncased)',
    family: 'BERT',
    dModel: 768,
    numLayers: 12,
    numHeads: 12,
    vocabSize: 30522,
    normType: 'LayerNorm',
    posType: 'Sinusoidal',
    baseFrequency: 10000,
  },
];

const SAMPLE_TOKENS = [
  'The', 'transformer', 'architecture', 'processes', 'tokens', 'in', 'parallel',
  'using', 'multi-head', 'attention', 'and', 'feed-forward', 'networks', 'with',
  'residual', 'connections', 'and', 'layer', 'normalization', 'layers'
];

export const ComputationStudios: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ComputationStudioTab>('ffn');
  const [selectedPreset, setSelectedPreset] = useState<ModelPreset>(MODEL_PRESETS[0]);
  const [selectedLayer, setSelectedLayer] = useState<number>(4);

  // -------------------------------------------------------------------
  // 1. FFN / MLP State
  // -------------------------------------------------------------------
  const [ffnThreshold, setFfnThreshold] = useState<number>(0.001);
  const [neuronFilter, setNeuronFilter] = useState<'all' | 'active' | 'selective' | 'dead'>('all');
  const [selectedNeuronIdx, setSelectedNeuronIdx] = useState<number>(12);
  const [hoveredCell, setHoveredCell] = useState<{ token: string; neuron: number; val: number } | null>(null);

  // Generate synthetic FFN neuron activations
  const ffnData = useMemo(() => {
    const numNeurons = 32;
    const records: FFNNeuronRecord[] = [];
    const grid: number[][] = [];

    for (let t = 0; t < SAMPLE_TOKENS.length; t++) {
      const row: number[] = [];
      for (let n = 0; n < numNeurons; n++) {
        // Create realistic sparsity: most are 0, some fire strongly
        const isDeadNeuron = n === 3 || n === 17 || n === 28;
        if (isDeadNeuron) {
          row.push(0);
        } else {
          const raw = Math.sin(t * 0.4 + n * 0.8) * Math.cos(t * 0.2 + n * 0.3);
          const val = raw > 0.3 ? (raw - 0.3) * 3.5 : 0;
          row.push(Math.round(val * 100) / 100);
        }
      }
      grid.push(row);
    }

    for (let n = 0; n < numNeurons; n++) {
      const activations = grid.map(row => row[n]);
      const maxAct = Math.max(...activations);
      const meanAct = activations.reduce((a, b) => a + b, 0) / activations.length;
      const variance = activations.reduce((a, b) => a + Math.pow(b - meanAct, 2), 0) / activations.length;
      const isDead = maxAct <= ffnThreshold;

      const topTokens = activations
        .map((act, pos) => ({ token: SAMPLE_TOKENS[pos], activation: act, position: pos }))
        .sort((a, b) => b.activation - a.activation)
        .slice(0, 4);

      const selectivity = isDead ? 0 : Math.min(1, Math.max(0, 1 - (meanAct / (maxAct + 0.001))));

      records.push({
        neuronIndex: n,
        layerIndex: selectedLayer,
        meanActivation: Math.round(meanAct * 100) / 100,
        maxActivation: Math.round(maxAct * 100) / 100,
        variance: Math.round(variance * 100) / 100,
        selectivity: Math.round(selectivity * 100) / 100,
        isDead,
        topTokens,
      });
    }

    return { records, grid, numNeurons };
  }, [selectedLayer, ffnThreshold]);

  const deadNeuronCount = useMemo(() => ffnData.records.filter(r => r.isDead).length, [ffnData]);
  const deadNeuronRatio = useMemo(() => (deadNeuronCount / ffnData.numNeurons) * 100, [deadNeuronCount, ffnData]);

  const saeFeatures: SAEFeatureRecord[] = useMemo(() => [
    { featureIndex: 104, firingRate: 0.15, maxActivation: 4.82, meanActivation: 0.72, label: 'Syntax: Verb Phrase Connectors' },
    { featureIndex: 219, firingRate: 0.08, maxActivation: 6.14, meanActivation: 0.49, label: 'Entity: AI & Neural Systems' },
    { featureIndex: 342, firingRate: 0.22, maxActivation: 3.91, meanActivation: 0.86, label: 'Math: Parallelism & Tensor Ops' },
    { featureIndex: 512, firingRate: 0.04, maxActivation: 7.30, meanActivation: 0.31, label: 'Punctuation: Hyphenated Modifiers' },
    { featureIndex: 881, firingRate: 0.19, maxActivation: 4.25, meanActivation: 0.65, label: 'Hierarchy: Layer-wise Residuals' },
  ], []);

  // -------------------------------------------------------------------
  // 2. KV Cache Monitor State
  // -------------------------------------------------------------------
  const [seqLength, setSeqLength] = useState<number>(4096);
  const [precisionBytes, setPrecisionBytes] = useState<number>(2.0); // FP16
  const [kvHeads, setKvHeads] = useState<number>(selectedPreset.numHeads);
  const [headDim, setHeadDim] = useState<number>(128);

  const kvMemoryScaling: KVCacheMemoryPoint[] = useMemo(() => {
    const lens = [512, 1024, 2048, 4096, 8192, 16384, 32768];
    return lens.map(n => {
      const baseBytes = 2 * selectedPreset.numLayers * kvHeads * n * headDim * precisionBytes;
      const denseMB = baseBytes / (1024 * 1024);
      return {
        seqLen: n,
        denseMB: Math.round(denseMB),
        deltaKVMB: Math.round(denseMB * 0.29),
        h2oMB: Math.round(denseMB * 0.35),
        streamingLLMMB: Math.round(denseMB * 0.22),
        snapKVMB: Math.round(denseMB * 0.30),
      };
    });
  }, [selectedPreset, kvHeads, headDim, precisionBytes]);

  const currentKVMemoryMB = useMemo(() => {
    const bytes = 2 * selectedPreset.numLayers * kvHeads * seqLength * headDim * precisionBytes;
    return Math.round((bytes / (1024 * 1024)) * 10) / 10;
  }, [selectedPreset, kvHeads, seqLength, headDim, precisionBytes]);

  const deltaKVSinks: DeltaKVSinkRecord[] = useMemo(() => {
    return SAMPLE_TOKENS.map((token, pos) => {
      const isSink = pos < 3; // first 3 tokens are attention sinks
      const sinkScore = isSink ? 45.0 - pos * 8.0 : Math.max(1.2, 5.0 - Math.random() * 3.5);
      const temporalDelta = isSink ? 0.05 : 0.45 + Math.sin(pos) * 0.25;
      return {
        position: pos,
        token,
        sinkScore: Math.round(sinkScore * 10) / 10,
        isSink,
        temporalDelta: Math.round(temporalDelta * 100) / 100,
      };
    });
  }, []);

  // -------------------------------------------------------------------
  // 3. Weight Matrix Decomposer State
  // -------------------------------------------------------------------
  const [matrixType, setMatrixType] = useState<'W_Q' | 'W_K' | 'W_V' | 'W_O' | 'W_gate' | 'W_up' | 'W_down'>('W_Q');
  const [svdRank, setSvdRank] = useState<number>(32);
  const [energyThreshold, setEnergyThreshold] = useState<number>(95);
  const [decompMethod, setDecompMethod] = useState<'svd' | 'cur' | 'rpca'>('svd');

  const singularValuesData: SingularValuePoint[] = useMemo(() => {
    const maxRank = 64;
    const points: SingularValuePoint[] = [];
    let cumEnergy = 0;
    const rawS = Array.from({ length: maxRank }, (_, i) => Math.exp(-i * 0.08) * 10.0);
    const totalSq = rawS.reduce((a, b) => a + b * b, 0);

    for (let i = 0; i < maxRank; i++) {
      const s = rawS[i];
      cumEnergy += (s * s) / totalSq;
      const rankNum = i + 1;
      points.push({
        rank: rankNum,
        singularValue: Math.round(s * 100) / 100,
        logValue: Math.round(Math.log10(s + 0.001) * 100) / 100,
        cumulativeEnergy: Math.round(cumEnergy * 1000) / 10,
        isKept: rankNum <= svdRank,
      });
    }
    return points;
  }, [svdRank]);

  const energyRetained = useMemo(() => {
    if (svdRank <= 0 || svdRank > singularValuesData.length) return 0;
    return singularValuesData[svdRank - 1]?.cumulativeEnergy || 0;
  }, [svdRank, singularValuesData]);

  const frobeniusError = useMemo(() => {
    return Math.round(Math.max(0.01, (100 - energyRetained) * 0.035) * 1000) / 1000;
  }, [energyRetained]);

  const paramReduction = useMemo(() => {
    const d = selectedPreset.dModel;
    const origParams = d * d;
    const factorParams = d * svdRank + svdRank * d;
    const ratio = (1 - factorParams / origParams) * 100;
    return Math.round(ratio * 10) / 10;
  }, [selectedPreset, svdRank]);

  // -------------------------------------------------------------------
  // 4. LoRA Adapter Studio State
  // -------------------------------------------------------------------
  const [loraRank, setLoraRank] = useState<number>(16);
  const [loraAlpha, setLoraAlpha] = useState<number>(32);
  const [loraInitMethod, setLoraInitMethod] = useState<'standard' | 'svd'>('standard');
  const [learningRate, setLearningRate] = useState<number>(0.0002);

  const loraParams = useMemo(() => {
    const d = selectedPreset.dModel;
    const adapterParams = loraRank * (d + d);
    const fullParams = d * d;
    const paramRatio = (adapterParams / fullParams) * 100;
    return {
      adapterParams: adapterParams.toLocaleString(),
      fullParams: fullParams.toLocaleString(),
      ratio: Math.round(paramRatio * 100) / 100,
      scaling: Math.round((loraAlpha / loraRank) * 100) / 100,
    };
  }, [selectedPreset, loraRank, loraAlpha]);

  const loraSpectrum = useMemo(() => {
    return Array.from({ length: loraRank }, (_, i) => ({
      index: i + 1,
      standardMag: Math.round(Math.random() * 0.4 * 100) / 100,
      svdInitMag: Math.round((Math.exp(-i * 0.15) * 3.8) * 100) / 100,
      cumEnergy: Math.round((1 - Math.exp(-i * 0.25)) * 1000) / 10,
    }));
  }, [loraRank]);

  const loraGradientDynamics = useMemo(() => {
    return Array.from({ length: 10 }, (_, step) => ({
      step: (step + 1) * 50,
      gradANorm: Math.round((0.85 * Math.exp(-step * 0.18) + 0.08) * 1000) / 1000,
      gradBNorm: Math.round((0.62 * Math.exp(-step * 0.14) + 0.05) * 1000) / 1000,
      subspaceAlignment: Math.round((0.45 + 0.52 * (1 - Math.exp(-step * 0.22))) * 100) / 100,
    }));
  }, []);

  return (
    <div className="flex flex-col h-full bg-[#0B0F19] text-slate-100 font-sans">
      {/* ================================================================= */}
      {/* Top Navigation & Studio Control Bar */}
      {/* ================================================================= */}
      <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md px-6 py-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-xl bg-gradient-to-tr from-cyan-600 to-indigo-600 shadow-lg shadow-cyan-500/20">
            <Cpu className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight bg-gradient-to-r from-cyan-400 via-sky-300 to-indigo-300 bg-clip-text text-transparent">
              Transformer Computation, Weight & Memory Studios
            </h1>
            <p className="text-xs text-slate-400">
              Interactive Mechanistic Interpretability, KV Cache Profiler & SVD Decomposer
            </p>
          </div>
        </div>

        {/* Studio Tabs */}
        <div className="flex items-center bg-slate-950/70 p-1 rounded-xl border border-slate-800">
          <button
            onClick={() => setActiveTab('ffn')}
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'ffn'
                ? 'bg-cyan-600 text-white shadow-md shadow-cyan-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Activity className="w-3.5 h-3.5" />
            <span>FFN / MLP Explorer</span>
          </button>
          <button
            onClick={() => setActiveTab('kvcache')}
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'kvcache'
                ? 'bg-cyan-600 text-white shadow-md shadow-cyan-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Database className="w-3.5 h-3.5" />
            <span>KV Cache Monitor</span>
          </button>
          <button
            onClick={() => setActiveTab('weight')}
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'weight'
                ? 'bg-cyan-600 text-white shadow-md shadow-cyan-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Split className="w-3.5 h-3.5" />
            <span>Weight Decomposer</span>
          </button>
          <button
            onClick={() => setActiveTab('lora')}
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'lora'
                ? 'bg-cyan-600 text-white shadow-md shadow-cyan-600/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Zap className="w-3.5 h-3.5" />
            <span>LoRA Adapter Studio</span>
          </button>
        </div>

        {/* Model Presets & Layer Picker */}
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2 bg-slate-950/60 px-3 py-1.5 rounded-lg border border-slate-800">
            <span className="text-xs text-slate-400">Model:</span>
            <select
              value={selectedPreset.id}
              onChange={(e) => {
                const p = MODEL_PRESETS.find((m) => m.id === e.target.value);
                if (p) setSelectedPreset(p);
              }}
              className="bg-transparent text-xs font-semibold text-cyan-300 focus:outline-none cursor-pointer"
            >
              {MODEL_PRESETS.map((m) => (
                <option key={m.id} value={m.id} className="bg-slate-900 text-slate-200">
                  {m.name}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center space-x-2 bg-slate-950/60 px-3 py-1.5 rounded-lg border border-slate-800">
            <span className="text-xs text-slate-400">Layer:</span>
            <input
              type="number"
              min={0}
              max={selectedPreset.numLayers - 1}
              value={selectedLayer}
              onChange={(e) => setSelectedLayer(Math.max(0, Math.min(selectedPreset.numLayers - 1, Number(e.target.value))))}
              className="w-12 bg-transparent text-xs font-mono text-cyan-300 focus:outline-none text-center"
            />
          </div>
        </div>
      </header>

      {/* ================================================================= */}
      {/* Studio View Content Body */}
      {/* ================================================================= */}
      <main className="flex-1 overflow-y-auto p-6">
        {/* TAB 1: FFN / MLP NEURON EXPLORER */}
        {activeTab === 'ffn' && (
          <ErrorBoundary title="Failed to render FFN Universe">
            <div className="space-y-6">
              {/* Top Stat Cards & Health Bar */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="bg-slate-900/70 border border-slate-800/80 rounded-xl p-4 shadow-sm">
                  <div className="flex justify-between items-center text-xs text-slate-400 mb-1">
                    <span>Neuron Universe</span>
                    <Activity className="w-4 h-4 text-cyan-400" />
                  </div>
                  <div className="text-xl font-bold text-slate-100 font-mono">
                    {ffnData.activeCount} <span className="text-xs font-normal text-slate-500">/ {ffnData.records.length}</span>
                  </div>
                  <div className="text-[11px] text-emerald-400 mt-1 flex items-center space-x-1">
                    <span>{ffnData.sparsity}% Sparsity Rate</span>
                  </div>
                </div>

                <div className="bg-slate-900/70 border border-slate-800/80 rounded-xl p-4 shadow-sm">
                  <div className="flex justify-between items-center text-xs text-slate-400 mb-1">
                    <span>Dead Neurons</span>
                    <AlertTriangle className="w-4 h-4 text-rose-400" />
                  </div>
                  <div className="text-xl font-bold text-rose-400 font-mono">
                    {ffnData.deadCount}
                  </div>
                  <div className="text-[11px] text-slate-500 mt-1">
                    {((ffnData.deadCount / ffnData.records.length) * 100).toFixed(1)}% Capacity Lost
                  </div>
                </div>

                <div className="bg-slate-900/70 border border-slate-800/80 rounded-xl p-4 shadow-sm">
                  <div className="flex justify-between items-center text-xs text-slate-400 mb-1">
                    <span>Activation Mode</span>
                    <Zap className="w-4 h-4 text-amber-400" />
                  </div>
                  <div className="text-xl font-bold text-amber-300 font-mono">SwiGLU</div>
                  <div className="text-[11px] text-slate-500 mt-1">Gated Linear Unit</div>
                </div>

                <div className="bg-slate-900/70 border border-slate-800/80 rounded-xl p-4 shadow-sm">
                  <div className="flex justify-between items-center text-xs text-slate-400 mb-1">
                    <span>SAE Features Decoded</span>
                    <Sparkles className="w-4 h-4 text-purple-400" />
                  </div>
                  <div className="text-xl font-bold text-purple-300 font-mono">{saeFeatures.length}</div>
                  <div className="text-[11px] text-slate-500 mt-1">Dictionary Exp: 4x</div>
                </div>
              </div>

              {/* Main FFN Grid & Heatmap */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* 2 Cols: Interactive Neuron Activation Heatmap */}
                <div className="lg:col-span-2 bg-slate-900/60 border border-slate-800/80 rounded-xl p-5 shadow-md">
                  <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
                    <div>
                      <h2 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
                        <Layers className="w-4 h-4 text-cyan-400" />
                        <span>Token × Intermediate Neuron Activation Matrix</span>
                      </h2>
                      <p className="text-xs text-slate-400">
                        Inspecting Layer {selectedLayer} Intermediate Projection Activations
                      </p>
                    </div>

                    <div className="flex items-center space-x-2">
                      <div className="flex items-center space-x-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
                        {(['all', 'active', 'selective', 'dead'] as const).map((filter) => (
                          <button
                            key={filter}
                            onClick={() => setNeuronFilter(filter)}
                            className={`px-2.5 py-1 rounded capitalize ${
                              neuronFilter === filter
                                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                                : 'text-slate-400 hover:text-white'
                            }`}
                          >
                            {filter}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Heatmap Grid */}
                  <div className="overflow-x-auto pb-2">
                    <div className="min-w-[500px]">
                      {/* Column Header (Neurons) */}
                      <div className="flex items-center mb-1 pl-20">
                        {ffnData.filteredRecords.slice(0, 24).map((rec) => (
                          <div
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: KV CACHE MONITOR */}
        {activeTab === 'kvcache' && (
          <div className="space-y-6">
            {/* Top Parameter Sliders & Formula */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Sliders (2 cols) */}
              <div className="lg:col-span-2 bg-slate-900/60 border border-slate-800/80 rounded-xl p-5 shadow-md space-y-4">
                <div className="flex justify-between items-center">
                  <h2 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
                    <Database className="w-4 h-4 text-cyan-400" />
                    <span>KV Cache Parameters & Memory Scaling Engine</span>
                  </h2>
                  <Badge variant="outline">Auto-Compute</Badge>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                  <div>
                    <div className="flex justify-between text-slate-300 mb-1">
                      <span>Sequence Length (Tokens N):</span>
                      <span className="font-mono text-cyan-400 font-bold">{seqLength.toLocaleString()}</span>
                    </div>
                    <input
                      type="range"
                      min={512}
                      max={32768}
                      step={512}
                      value={seqLength}
                      onChange={(e) => setSeqLength(Number(e.target.value))}
                      className="w-full accent-cyan-500 cursor-pointer"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-slate-300 mb-1">
                      <span>KV Heads (H):</span>
                      <span className="font-mono text-cyan-400 font-bold">{kvHeads}</span>
                    </div>
                    <input
                      type="range"
                      min={1}
                      max={64}
                      value={kvHeads}
                      onChange={(e) => setKvHeads(Number(e.target.value))}
                      className="w-full accent-cyan-500 cursor-pointer"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-slate-300 mb-1">
                      <span>Head Dimension (d_head):</span>
                      <span className="font-mono text-cyan-400 font-bold">{headDim}</span>
                    </div>
                    <input
                      type="range"
                      min={32}
                      max={256}
                      step={32}
                      value={headDim}
                      onChange={(e) => setHeadDim(Number(e.target.value))}
                      className="w-full accent-cyan-500 cursor-pointer"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-slate-300 mb-1">
                      <span>Precision Format:</span>
                      <span className="font-mono text-cyan-400 font-bold">
                        {precisionBytes === 2 ? 'FP16 (2 Bytes)' : precisionBytes === 1 ? 'INT8 (1 Byte)' : 'INT4 (0.5 Bytes)'}
                      </span>
                    </div>
                    <div className="flex space-x-2 mt-1">
                      {[
                        { label: 'FP16', bytes: 2.0 },
                        { label: 'INT8', bytes: 1.0 },
                        { label: 'INT4', bytes: 0.5 },
                      ].map((p) => (
                        <button
                          key={p.label}
                          onClick={() => setPrecisionBytes(p.bytes)}
                          className={`flex-1 py-1 rounded text-xs font-semibold border ${
                            precisionBytes === p.bytes
                              ? 'bg-cyan-600 border-cyan-500 text-white'
                              : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          {p.label}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Real-time Math Formula */}
                <div className="bg-slate-950/70 p-3 rounded-lg border border-slate-800 font-mono text-xs text-slate-300">
                  <div className="text-slate-500 text-[10px] uppercase font-bold tracking-wider mb-1">
                    Live Memory Formula:
                  </div>
                  <div>
                    Memory = 2 × L × H × N × d_head × bytes
                  </div>
                  <div className="text-cyan-300 mt-1">
                    = 2 × {selectedPreset.numLayers} × {kvHeads} × {seqLength} × {headDim} × {precisionBytes} B ={' '}
                    <strong className="text-amber-300 text-sm">{currentKVMemoryMB} MB ({(currentKVMemoryMB / 1024).toFixed(2)} GB)</strong>
                  </div>
                </div>
              </div>

              {/* VRAM Gauge & Compression Summary */}
              <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-5 shadow-md space-y-4">
                <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
                  <Flame className="w-4 h-4 text-amber-400" />
                  <span>VRAM Consumption Gauge</span>
                </h3>

                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-400">NVIDIA RTX 4090 (24 GB)</span>
                      <span className="font-mono text-slate-200">{((currentKVMemoryMB / 1024 / 24) * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                      <div
                        className="bg-cyan-500 h-full"
                        style={{ width: `${Math.min(100, (currentKVMemoryMB / 1024 / 24) * 100)}%` }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-slate-400">NVIDIA A100 (80 GB)</span>
                      <span className="font-mono text-slate-200">{((currentKVMemoryMB / 1024 / 80) * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                      <div
                        className="bg-indigo-500 h-full"
                        style={{ width: `${Math.min(100, (currentKVMemoryMB / 1024 / 80) * 100)}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="p-3 bg-slate-950/60 rounded-lg border border-slate-800 space-y-2 text-xs">
                  <div className="text-slate-400 font-semibold">DeltaKV Compression Savings:</div>
                  <div className="flex justify-between items-center text-emerald-400 font-mono font-bold">
                    <span>Memory Saved:</span>
                    <span>-71.0% ({(currentKVMemoryMB * 0.71).toFixed(1)} MB)</span>
                  </div>
                  <div className="flex justify-between items-center text-slate-300 font-mono text-[11px]">
                    <span>Compressed Footprint:</span>
                    <span>{(currentKVMemoryMB * 0.29).toFixed(1)} MB</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Memory Scaling Multi-Curve Chart */}
            <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-5 shadow-md">
              <div className="flex justify-between items-center mb-4">
                <div>
                  <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
                    <TrendingUp className="w-4 h-4 text-emerald-400" />
                    <span>KV Cache Memory Scaling: Dense vs Compression Strategies</span>
                  </h3>
                  <p className="text-xs text-slate-400">Comparing Full Dense Cache vs DeltaKV, H2O, and StreamingLLM</p>
                </div>
                <Badge variant="outline">Log / Linear Scaling</Badge>
              </div>

              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={kvMemoryScaling}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                    <XAxis dataKey="seqLen" stroke="#64748B" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#64748B" tick={{ fontSize: 11 }} unit=" MB" />
                    <RechartsTooltip
                      contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', borderRadius: '8px' }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                    <Line type="monotone" dataKey="denseMB" name="Full Dense Baseline" stroke="#EF4444" strokeWidth={2.5} dot />
                    <Line type="monotone" dataKey="deltaKVMB" name="DeltaKV (Our SVD)" stroke="#06B6D4" strokeWidth={2.5} dot />
                    <Line type="monotone" dataKey="h2oMB" name="H2O Heavy-Hitter" stroke="#F59E0B" strokeWidth={2} dot />
                    <Line type="monotone" dataKey="streamingLLMMB" name="StreamingLLM" stroke="#10B981" strokeWidth={2} dot />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Delta-KV Temporal Difference & Attention Sinks */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-5 shadow-md">
                <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2 mb-3">
                  <Activity className="w-4 h-4 text-cyan-400" />
                  <span>Attention Sink Detection (KVSink Method)</span>
                </h3>
                <p className="text-xs text-slate-400 mb-3">
                  Tokens absorbing massive cross-layer attention mass (preserved with full rank in DeltaKV)
                </p>

                <div className="space-y-2">
                  {deltaKVSinks.slice(0, 8).map((s) => (
                    <div
                      key={s.position}
                      className={`flex items-center justify-between px-3 py-2 rounded-lg border text-xs ${
                        s.isSink
                          ? 'bg-cyan-950/40 border-cyan-800/60 text-cyan-200'
                          : 'bg-slate-950/40 border-slate-800/60 text-slate-300'
                      }`}
                    >
                      <div className="flex items-center space-x-2 font-mono">
                        <span className="text-slate-500 text-[10px]">#{s.position}</span>
                        <strong>"{s.token}"</strong>
                        {s.isSink && <Badge variant="info">ATTN SINK</Badge>}
                      </div>
                      <div className="flex items-center space-x-3 font-mono">
                        <span className="text-slate-400 text-[11px]">Mass: {s.sinkScore}%</span>
                        <span className="text-amber-400 font-semibold">Δ: {s.temporalDelta}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-5 shadow-md flex flex-col justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2 mb-3">
                    <Shield className="w-4 h-4 text-indigo-400" />
                    <span>Compression Policy Deployment</span>
                  </h3>
                  <p className="text-xs text-slate-400 mb-4">
                    Deploy KV Cache Compression Hook directly into the active inference engine
                  </p>

                  <div className="space-y-2.5 text-xs">
                    <div className="flex justify-between p-2.5 bg-slate-950 rounded border border-slate-800">
                      <span className="text-slate-400">Target Module:</span>
                      <span className="font-mono text-slate-200">Attention KV Store</span>
                    </div>
                    <div className="flex justify-between p-2.5 bg-slate-950 rounded border border-slate-800">
                      <span className="text-slate-400">Sink Token Protection:</span>
                      <span className="font-mono text-emerald-400">First 4 Tokens Preserved</span>
                    </div>
                    <div className="flex justify-between p-2.5 bg-slate-950 rounded border border-slate-800">
                      <span className="text-slate-400">Delta-KV Rank (r):</span>
                      <span className="font-mono text-cyan-300 font-bold">16 / 128</span>
                    </div>
                  </div>
                </div>

                <Button variant="primary" className="w-full mt-4 flex items-center justify-center space-x-2">
                  <Zap className="w-4 h-4" />
                  <span>Apply KV Cache Strategy to Model</span>
                </Button>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: WEIGHT MATRIX DECOMPOSER */}
        {activeTab === 'weight' && (
          <div className="space-y-6">
            {/* Controls Bar */}
            <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-5 shadow-md">
              <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
                <div className="flex items-center space-x-3">
                  <div className="flex items-center space-x-2 bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 text-xs">
                    <span className="text-slate-400">Matrix:</span>
                    <select
                      value={matrixType}
                      onChange={(e) => setMatrixType(e.target.value as any)}
                      className="bg-transparent text-xs font-semibold text-cyan-300 focus:outline-none cursor-pointer"
                    >
                      {['W_Q', 'W_K', 'W_V', 'W_O', 'W_gate', 'W_up', 'W_down'].map((m) => (
                        <option key={m} value={m} className="bg-slate-900 text-slate-200">
                          {m} Projection ({selectedPreset.dModel}×{selectedPreset.dModel})
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="flex items-center bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
                    {(['svd', 'cur', 'rpca'] as const).map((m) => (
                      <button
                        key={m}
                        onClick={() => setDecompMethod(m)}
                        className={`px-3 py-1 rounded text-xs font-semibold uppercase transition-all ${
                          decompMethod === m
                            ? 'bg-cyan-600 text-white shadow-sm'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {m}
                      </button>
                    ))}
                  </div>
                </div>

                {/* SVD Rank Slider */}
                <div className="flex items-center space-x-4 min-w-[280px]">
                  <div className="text-xs text-slate-300 shrink-0">
                    Rank (k): <strong className="text-cyan-400 font-mono text-sm">{svdRank}</strong>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={64}
                    value={svdRank}
                    onChange={(e) => setSvdRank(Number(e.target.value))}
                    className="w-full accent-cyan-500 cursor-pointer"
                  />
                </div>
              </div>

              {/* Metrics Summary Strip */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                  <div className="text-slate-400 text-[11px]">Energy Retained (Cumulative)</div>
                  <div className="text-lg font-mono font-bold text-emerald-400">{energyRetained}%</div>
                </div>
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                  <div className="text-slate-400 text-[11px]">Frobenius Reconstruction Error</div>
                  <div className="text-lg font-mono font-bold text-amber-400">{frobeniusError}</div>
                </div>
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                  <div className="text-slate-400 text-[11px]">Parameter Reduction</div>
                  <div className="text-lg font-mono font-bold text-cyan-300">-{paramReduction}%</div>
                </div>
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                  <div className="text-slate-400 text-[11px]">Estimated FLOPs Speedup</div>
                  <div className="text-lg font-mono font-bold text-indigo-300">
                    {(1 / (1 - paramReduction / 100)).toFixed(2)}x
                  </div>
                </div>
              </div>
            </div>

            {/* SVD Singular Spectrum Chart & Cumulative Energy */}
            <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-5 shadow-md">
              <div className="flex justify-between items-center mb-3">
                <div>
                  <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
                    <BarChart3 className="w-4 h-4 text-cyan-400" />
                    <span>Singular Value Spectrum & Cumulative Energy Retention</span>
                  </h3>
                  <p className="text-xs text-slate-400">
                    Components 1..{svdRank} kept in low-rank factor matrices U_k, Σ_k, V_k^T
                  </p>
                </div>
                <Badge variant={energyRetained >= 95 ? 'success' : 'warning'}>
                  {energyRetained >= 95 ? 'TARGET ≥95% MET' : 'BELOW 95% ENERGY'}
                </Badge>
              </div>

              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={singularValuesData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                    <XAxis dataKey="rank" stroke="#64748B" tick={{ fontSize: 10 }} />
                    <YAxis stroke="#64748B" tick={{ fontSize: 10 }} />
                    <RechartsTooltip
                      contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', borderRadius: '8px' }}
                    />
                    <Bar dataKey="singularValue" name="Singular Value (σ_i)" fill="#06B6D4" radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Triple Matrix Reconstruction View */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 shadow-md">
                <h4 className="text-xs font-semibold text-slate-200 mb-2">Original Matrix W (Dense)</h4>
                <div className="aspect-square bg-slate-950 rounded-lg p-2 flex items-center justify-center border border-slate-800">
                  <div className="grid grid-cols-8 gap-0.5 w-full h-full">
                    {Array.from({ length: 64 }).map((_, i) => (
                      <div
                        key={i}
                        className="bg-cyan-600 rounded-xs"
                        style={{ opacity: Math.sin(i * 0.3) * 0.4 + 0.5 }}
                      />
                    ))}
                  </div>
                </div>
              </div>

              <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 shadow-md">
                <h4 className="text-xs font-semibold text-slate-200 mb-2">Reconstructed W_k (Rank {svdRank})</h4>
                <div className="aspect-square bg-slate-950 rounded-lg p-2 flex items-center justify-center border border-slate-800">
                  <div className="grid grid-cols-8 gap-0.5 w-full h-full">
                    {Array.from({ length: 64 }).map((_, i) => (
                      <div
                        key={i}
                        className="bg-indigo-600 rounded-xs"
                        style={{ opacity: Math.sin(i * 0.3) * 0.38 + 0.48 }}
                      />
                    ))}
                  </div>
                </div>
              </div>

              <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 shadow-md">
                <h4 className="text-xs font-semibold text-slate-200 mb-2">Error Map |W - W_k|</h4>
                <div className="aspect-square bg-slate-950 rounded-lg p-2 flex items-center justify-center border border-slate-800">
                  <div className="grid grid-cols-8 gap-0.5 w-full h-full">
                    {Array.from({ length: 64 }).map((_, i) => (
                      <div
                        key={i}
                        className="bg-rose-500 rounded-xs"
                        style={{ opacity: Math.max(0.05, 0.5 - (svdRank / 64) * 0.45) }}
                      />
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: LoRA ADAPTER STUDIO */}
        {activeTab === 'lora' && (
          <div className="space-y-6">
            {/* Top LoRA Configuration Bar */}
            <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-5 shadow-md">
              <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
                <div>
                  <h2 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
                    <Zap className="w-4 h-4 text-cyan-400" />
                    <span>Low-Rank Adaptation (LoRA) Factorizer & Gradient Dynamics</span>
                  </h2>
                  <p className="text-xs text-slate-400">
                    W' = W_base + (α / r) · B · A where A ∈ ℝ^(r×d), B ∈ ℝ^(d×r)
                  </p>
                </div>

                <div className="flex items-center space-x-2 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
                  <button
                    onClick={() => setLoraInitMethod('standard')}
                    className={`px-3 py-1 rounded text-xs font-semibold transition-all ${
                      loraInitMethod === 'standard' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Standard Zero-Init
                  </button>
                  <button
                    onClick={() => setLoraInitMethod('svd')}
                    className={`px-3 py-1 rounded text-xs font-semibold transition-all ${
                      loraInitMethod === 'svd' ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    SVD Initialization
                  </button>
                </div>
              </div>

              {/* Sliders Grid */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs mb-4">
                <div>
                  <div className="flex justify-between text-slate-300 mb-1">
                    <span>Adapter Rank (r):</span>
                    <span className="font-mono text-cyan-400 font-bold">{loraRank}</span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={64}
                    value={loraRank}
                    onChange={(e) => setLoraRank(Number(e.target.value))}
                    className="w-full accent-cyan-500 cursor-pointer"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-slate-300 mb-1">
                    <span>LoRA Alpha (α):</span>
                    <span className="font-mono text-cyan-400 font-bold">{loraAlpha}</span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={64}
                    value={loraAlpha}
                    onChange={(e) => setLoraAlpha(Number(e.target.value))}
                    className="w-full accent-cyan-500 cursor-pointer"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-slate-300 mb-1">
                    <span>Learning Rate (η):</span>
                    <span className="font-mono text-cyan-400 font-bold">{learningRate}</span>
                  </div>
                  <input
                    type="range"
                    min={0.00005}
                    max={0.001}
                    step={0.00005}
                    value={learningRate}
                    onChange={(e) => setLearningRate(Number(e.target.value))}
                    className="w-full accent-cyan-500 cursor-pointer"
                  />
                </div>
              </div>

              {/* LoRA Parameter Impact Badge Strip */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                  <div className="text-slate-400 text-[11px]">Trainable Parameters</div>
                  <div className="text-lg font-mono font-bold text-cyan-300">{loraParams.adapterParams}</div>
                </div>
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                  <div className="text-slate-400 text-[11px]">Parameter Footprint Ratio</div>
                  <div className="text-lg font-mono font-bold text-emerald-400">{loraParams.ratio}%</div>
                </div>
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                  <div className="text-slate-400 text-[11px]">Scaling Multiplier (α/r)</div>
                  <div className="text-lg font-mono font-bold text-amber-300">{loraParams.scaling}x</div>
                </div>
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800">
                  <div className="text-slate-400 text-[11px]">Memory Reduction</div>
                  <div className="text-lg font-mono font-bold text-indigo-300">-99.2%</div>
                </div>
              </div>
            </div>

            {/* Gradient Dynamics & Subspace Alignment Chart */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-5 shadow-md">
                <div className="flex justify-between items-center mb-3">
                  <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
                    <TrendingUp className="w-4 h-4 text-emerald-400" />
                    <span>Low-Rank Gradient Dynamics & Subspace Alignment</span>
                  </h3>
                  <Badge variant="outline">Simulated SGD</Badge>
                </div>
                <p className="text-xs text-slate-400 mb-3">
                  Cosine alignment between low-rank gradient product and full-rank gradient
                </p>

                <div className="h-60 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={loraGradientDynamics}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                      <XAxis dataKey="step" stroke="#64748B" tick={{ fontSize: 10 }} />
                      <YAxis stroke="#64748B" tick={{ fontSize: 10 }} />
                      <RechartsTooltip
                        contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', borderRadius: '8px' }}
                      />
                      <Legend wrapperStyle={{ fontSize: '11px' }} />
                      <Line type="monotone" dataKey="gradANorm" name="||∇_A L||_F" stroke="#06B6D4" strokeWidth={2} dot />
                      <Line type="monotone" dataKey="gradBNorm" name="||∇_B L||_F" stroke="#8B5CF6" strokeWidth={2} dot />
                      <Line type="monotone" dataKey="subspaceAlignment" name="Subspace Alignment" stroke="#10B981" strokeWidth={2.5} dot />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* SVD vs Zero-Init Comparison */}
              <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-5 shadow-md flex flex-col justify-between">
                <div>
                  <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2 mb-3">
                    <Sparkles className="w-4 h-4 text-indigo-400" />
                    <span>SVD-Init vs Standard Zero-Init Quality</span>
                  </h3>
                  <p className="text-xs text-slate-400 mb-4">
                    SVD-initialized adapters capture dominant base singular components from step 0
                  </p>

                  <div className="space-y-3 text-xs">
                    <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 space-y-1.5">
                      <div className="font-bold text-slate-200 flex justify-between">
                        <span>Standard Initialization (PEFT Default):</span>
                        <Badge variant="outline">A ~ N(0, 1/d), B = 0</Badge>
                      </div>
                      <div className="text-slate-400 text-[11px]">
                        Initial update ΔW = 0. Requires multiple warmup epochs to align with primary task directions.
                      </div>
                    </div>

                    <div className="p-3 bg-indigo-950/40 rounded-lg border border-indigo-800/60 space-y-1.5">
                      <div className="font-bold text-indigo-200 flex justify-between">
                        <span>SVD Initialization (ICML 2026):</span>
                        <Badge variant="info">B = U_r √Σ_r, A = √Σ_r V_r^T</Badge>
                      </div>
                      <div className="text-indigo-300 text-[11px]">
                        Captures 84.6% of base weight energy immediately. 2.4x faster convergence in fine-tuning.
                      </div>
                    </div>
                  </div>
                </div>

                <Button variant="primary" className="w-full mt-4 flex items-center justify-center space-x-2">
                  <Play className="w-4 h-4" />
                  <span>Launch LoRA Fine-Tuning Run</span>
                </Button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default ComputationStudios;
