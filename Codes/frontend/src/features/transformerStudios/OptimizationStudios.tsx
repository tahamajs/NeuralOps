// src/features/transformerStudios/OptimizationStudios.tsx
import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip as RechartsTooltip,
  CartesianGrid,
  Legend,
  ScatterChart,
  Scatter,
  ZAxis,
  AreaChart,
  Area,
  Cell,
} from 'recharts';
import {
  Compass,
  Zap,
  Mountain,
  Scale,
  Workflow,
  Sliders,
  Play,
  Pause,
  RotateCcw,
  Search,
  Target,
  ArrowRight,
  TrendingUp,
  Cpu,
  Flame,
  CheckCircle2,
  AlertTriangle,
  Info,
  Shield,
  Layers,
  Sparkles,
  Award,
  Download,
  Share2,
  FileCode,
  Gauge,
  Activity,
  Split,
  Eye,
  Crosshair,
} from 'lucide-react';
import { Badge } from '../../components/ui/Badge';
import { Button } from '../../components/ui/Button';
import type {
  OptimizationStudioTab,
  CircuitGraphNode,
  CircuitGraphEdge,
  CircuitInterventionState,
  TokenProbabilityItem,
  SamplingStrategyConfig,
  TunedLensLayerPrediction,
  ReliabilityCalibrationBin,
  LossLandscapeGrid,
  HessianCurvatureData,
  ModeConnectivityPath,
  CompressorEngineSpec,
  BenchmarkParetoModel,
  LayerBudgetAllocation,
  PipelineStageProfile,
  ForwardTraceBlock,
  ChallengeGameState,
  ModelPreset,
} from './types';

// =====================================================================
// Model Presets Catalog
// =====================================================================
const OPTIMIZATION_MODEL_PRESETS: ModelPreset[] = [
  {
    id: 'gpt2-small',
    name: 'GPT-2 Small (124M)',
    family: 'GPT',
    dModel: 768,
    numLayers: 12,
    numHeads: 12,
    vocabSize: 50257,
    normType: 'LayerNorm',
    posType: 'Sinusoidal',
    baseFrequency: 10000,
  },
  {
    id: 'llama2-7b',
    name: 'LLaMA-2 (7B)',
    family: 'LLaMA',
    dModel: 4096,
    numLayers: 32,
    numHeads: 32,
    vocabSize: 32000,
    normType: 'RMSNorm',
    posType: 'RoPE',
    baseFrequency: 10000,
  },
  {
    id: 'svd-linear-trans',
    name: 'SVD-Linear Transformer',
    family: 'Efficient',
    dModel: 512,
    numLayers: 8,
    numHeads: 8,
    vocabSize: 30522,
    normType: 'LayerNorm',
    posType: 'RoPE',
    baseFrequency: 10000,
  },
  {
    id: 'galerkin-trans',
    name: 'Galerkin Linear Attention',
    family: 'Linear-Attn',
    dModel: 512,
    numLayers: 6,
    numHeads: 8,
    vocabSize: 30522,
    normType: 'RMSNorm',
    posType: 'RoPE',
    baseFrequency: 10000,
  },
];

// =====================================================================
// 80+ Compression Engines Reference
// =====================================================================
const COMPRESSOR_CATALOG: CompressorEngineSpec[] = [
  { id: 'a3', name: 'A3 SVD', category: 'Low-Rank SVD', latex: 'W \\approx U_r \\Sigma_r V_r^T', typicalRatio: '60-75%', speedup: '2.2x' },
  { id: 'adasvd', name: 'AdaSVD', category: 'Low-Rank SVD', latex: 'r_l = \\arg\\min_r \\|\\Sigma_{l, >r}\\|_F \\le \\epsilon_l', typicalRatio: '65-80%', speedup: '2.5x' },
  { id: 'dobi_svd', name: 'DobiSVD', category: 'Low-Rank SVD', latex: 'W \\approx U_k \\Sigma_k V_k^T + D_{diag}', typicalRatio: '70-85%', speedup: '2.8x' },
  { id: 'swift_svd', name: 'SwiftSVD', category: 'Low-Rank SVD', latex: 'W \\approx Q (Q^T W)', typicalRatio: '55-70%', speedup: '3.2x' },
  { id: 'mgie_svd', name: 'MGIE SVD', category: 'Low-Rank SVD', latex: '\\Delta W = U_r V_r^T + E_{sparse}', typicalRatio: '70-85%', speedup: '2.6x' },
  { id: 'compot', name: 'COMPOT', category: 'Low-Rank SVD', latex: '\\min_{U,V} \\|W - UV^T\\|_F^2 + \\lambda \\|U\\|_*', typicalRatio: '60-78%', speedup: '2.3x' },
  { id: 'flap', name: 'FLAP Pruner', category: 'Pruning', latex: 'I_j = \\|W_{:, j}\\|_2 \\cdot \\|\\nabla_{W_{:, j}} \\mathcal{L}\\|_2', typicalRatio: '50-65%', speedup: '1.8x' },
  { id: 'slicegpt', name: 'SliceGPT', category: 'Pruning', latex: 'Q W \\approx \\tilde{W} \\in \\mathbb{R}^{d_{red} \\times d}', typicalRatio: '45-60%', speedup: '2.0x' },
  { id: 'rotational_slicegpt_int4', name: 'Rotational SliceGPT INT4', category: 'Pruning', latex: 'W \\approx \\text{Quant}_4(R W R^T)', typicalRatio: '75-88%', speedup: '3.5x' },
  { id: 'sparsegpt', name: 'SparseGPT', category: 'Pruning', latex: 'w_q^* = \\arg\\min_w \\|(W - \\hat{W}) X\\|_2^2 + \\lambda \\|w\\|_0', typicalRatio: '50-70%', speedup: '2.2x' },
  { id: 'ast_pruner', name: 'AST 2:4 Pruner', category: 'Pruning', latex: 'W_{2:4} = \\text{TopK}_{k=2}(W_{i, 4j:4j+4})', typicalRatio: '50%', speedup: '2.0x (NVIDIA Sparse)' },
  { id: 'astro', name: 'Astro Quantizer', category: 'Quantization', latex: 'q = \\text{round}(w / \\Delta) \\cdot \\Delta', typicalRatio: '75% (INT4)', speedup: '3.0x' },
  { id: 'coopq', name: 'CoopQ Quantizer', category: 'Quantization', latex: '\\min_{C, B} \\|W - C B\\|_F^2, B \\in \\{-1, 1\\}', typicalRatio: '80% (INT2/4)', speedup: '3.6x' },
  { id: 'faq', name: 'FAQ Quantizer', category: 'Quantization', latex: 'W_{int8} = \\text{clamp}(\\lfloor W / s \\rceil, -128, 127)', typicalRatio: '50% (INT8)', speedup: '2.2x' },
  { id: 'kmeans', name: 'KMeans Quantizer', category: 'Quantization', latex: 'w_{ij} \\approx c_k, k = \\arg\\min_m |w_{ij} - c_m|', typicalRatio: '70-85%', speedup: '2.4x' },
  { id: 'tt_linear', name: 'Tensor Train (TT-Linear)', category: 'Tensor Decompositions', latex: 'W = \\mathcal{G}_1 \\times \\dots \\times \\mathcal{G}_d', typicalRatio: '80-95%', speedup: '4.2x' },
  { id: 'kronecker_linear', name: 'Kronecker Linear', category: 'Tensor Decompositions', latex: 'W \\approx A \\otimes B', typicalRatio: '70-90%', speedup: '3.8x' },
  { id: 'kronecker_tt_lora', name: 'Kronecker TT-LoRA', category: 'Tensor Decompositions', latex: '\\Delta W = (A_1 \\otimes A_2) (B_1 \\otimes B_2)', typicalRatio: '85-96%', speedup: '4.5x' },
  { id: 'delta_kv', name: 'DeltaKV', category: 'KV-Cache', latex: 'KV_t = KV_{t-1} + \\Delta KV_t', typicalRatio: '70-85% VRAM', speedup: '2.5x Gen' },
  { id: 'hybridkv', name: 'HybridKV', category: 'KV-Cache', latex: 'K_{evict} = \\text{Evict}_{\\text{score}}(K)', typicalRatio: '60-80% VRAM', speedup: '2.1x Gen' },
  { id: 'galerkin', name: 'Galerkin Linear Attention', category: 'Linear Attention', latex: 'A(Q, K, V) = Q (K^T V) / \\sqrt{d}', typicalRatio: 'O(N) Complexity', speedup: '4.0x Long Seq' },
  { id: 'spectraformer', name: 'Spectraformer', category: 'Linear Attention', latex: 'A = \\mathcal{F}^{-1}(\\mathcal{F}(Q) \\mathcal{F}(K))', typicalRatio: 'O(N \\log N)', speedup: '3.5x Long Seq' },
];

export const OptimizationStudios: React.FC = () => {
  // Global Navigation & Model Context State
  const [activeTab, setActiveTab] = useState<OptimizationStudioTab>('circuits');
  const [selectedModel, setSelectedModel] = useState<ModelPreset>(OPTIMIZATION_MODEL_PRESETS[0]);
  const [promptText, setPromptText] = useState('The capital of France is Paris and the capital of Texas is Austin');
  const [selectedLayer, setSelectedLayer] = useState(2);
  const [selectedHead, setSelectedHead] = useState(0);

  // ---------------------------------------------------------------------------
  // 1. CIRCUIT TRACER STUDIO STATE
  // ---------------------------------------------------------------------------
  const [circuitPruneThreshold, setCircuitPruneThreshold] = useState(0.08);
  const [interventionState, setInterventionState] = useState<CircuitInterventionState>({
    layerIdx: 2,
    headIdx: 0,
    ablationType: 'zero',
    scale: 0.0,
    klDivergence: 0.245,
    logitDiff: 2.84,
    topTokenFlipped: true,
  });

  const tokens = useMemo(() => promptText.split(/\s+/).filter(Boolean), [promptText]);

  // Synthetic Circuit Graph Data
  const circuitNodes: CircuitGraphNode[] = useMemo(() => {
    const nodes: CircuitGraphNode[] = [];
    tokens.forEach((tok, idx) => {
      nodes.push({
        id: `input_${idx}`,
        type: 'input',
        label: tok,
        layer: -1,
        activation: 0.95,
      });
    });

    const headTypes: CircuitGraphNode['headType'][] = [
      'induction',
      'previous_token',
      'positional_bos',
      'copy_amplification',
      'semantic_diffuse',
    ];

    for (let l = 0; l < Math.min(selectedModel.numLayers, 4); l++) {
      for (let h = 0; h < 4; h++) {
        const ht = headTypes[(l * 4 + h) % headTypes.length];
        nodes.push({
          id: `L${l}_H${h}`,
          type: 'attention_head',
          label: `L${l}H${h}`,
          layer: l,
          head: h,
          activation: 0.4 + 0.5 * Math.sin(l + h),
          headType: ht,
          scores: {
            induction: 0.2 + 0.7 * Math.cos(l * 0.5 + h),
            previous_token: 0.1 + 0.6 * Math.sin(l + h),
            copy_amplification: 0.3 + 0.5 * Math.cos(h),
          },
        });
      }
    }

    nodes.push({
      id: 'output_logits',
      type: 'output',
      label: 'Logits (Next Token)',
      layer: 4,
      activation: 1.0,
    });
    return nodes;
  }, [tokens, selectedModel]);

  const circuitEdges: CircuitGraphEdge[] = useMemo(() => {
    const edges: CircuitGraphEdge[] = [];
    tokens.forEach((_, idx) => {
      edges.push({
        id: `e_in_${idx}_L0_H0`,
        source: `input_${idx}`,
        target: 'L0_H0',
        weight: 0.35 + 0.5 * Math.sin(idx),
        type: idx % 2 === 0 ? 'qk_routing' : 'ov_writing',
      });
    });
    edges.push({ id: 'e_L0_L1', source: 'L0_H0', target: 'L1_H1', weight: 0.78, type: 'qk_routing' });
    edges.push({ id: 'e_L1_L2', source: 'L1_H1', target: 'L2_H0', weight: 0.89, type: 'ov_writing' });
    edges.push({ id: 'e_L2_out', source: 'L2_H0', target: 'output_logits', weight: 0.94, type: 'ov_writing' });
    return edges.filter((e) => e.weight >= circuitPruneThreshold);
  }, [tokens, circuitPruneThreshold]);

  // ---------------------------------------------------------------------------
  // 2. OUTPUT LOGITS & TUNED LENS STATE
  // ---------------------------------------------------------------------------
  const [samplingConfig, setSamplingConfig] = useState<SamplingStrategyConfig>({
    method: 'temperature',
    temperature: 0.7,
    topK: 10,
    topP: 0.9,
    repetitionPenalty: 1.15,
    beamWidth: 3,
  });

  const tokenProbabilities: TokenProbabilityItem[] = useMemo(() => {
    const candidateTokens = ['Paris', 'France', 'capital', 'Texas', 'Austin', 'city', 'London', 'Berlin', 'state', 'country'];
    const T = Math.max(0.1, samplingConfig.temperature);
    let rawLogits = [8.5, 6.2, 5.8, 5.1, 4.4, 3.8, 2.9, 2.1, 1.8, 1.2];

    if (samplingConfig.repetitionPenalty > 1.0) {
      rawLogits = rawLogits.map((l, i) => (i === 1 ? l - (samplingConfig.repetitionPenalty - 1) * 3 : l));
    }

    const scaled = rawLogits.map((l) => l / T);
    const maxVal = Math.max(...scaled);
    const exps = scaled.map((l) => Math.exp(l - maxVal));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map((e) => e / sumExps);

    let cum = 0;
    return candidateTokens.map((tok, idx) => {
      cum += probs[idx];
      return {
        tokenId: 1000 + idx,
        token: tok,
        rawLogit: Number(rawLogits[idx].toFixed(2)),
        scaledLogit: Number(scaled[idx].toFixed(2)),
        probability: Number(probs[idx].toFixed(4)),
        cumulativeProb: Number(cum.toFixed(4)),
      };
    });
  }, [samplingConfig]);

  const entropyBits = useMemo(() => {
    return tokenProbabilities.reduce((acc, item) => {
      if (item.probability <= 1e-6) return acc;
      return acc - item.probability * Math.log2(item.probability);
    }, 0);
  }, [tokenProbabilities]);

  // Tuned Lens Cross-Layer Timeline
  const tunedLensData: TunedLensLayerPrediction[] = useMemo(() => {
    return Array.from({ length: selectedModel.numLayers }).map((_, l) => {
      const ent = Math.max(0.4, 4.8 - l * (4.2 / selectedModel.numLayers) + Math.sin(l) * 0.2);
      const kl = Math.max(0.01, 3.2 - l * (3.1 / selectedModel.numLayers));
      const isLate = l >= selectedModel.numLayers - 3;
      return {
        layer: l,
        entropy: Number(ent.toFixed(2)),
        klToFinal: Number(kl.toFixed(3)),
        top1Token: isLate ? 'Paris' : l < 3 ? 'The' : 'France',
        top1Prob: Number(Math.min(0.92, 0.15 + l * 0.07).toFixed(2)),
        matchesFinal: isLate,
      };
    });
  }, [selectedModel]);

  // Calibration Diagram Data
  const calibrationBins: ReliabilityCalibrationBin[] = useMemo(() => {
    return [
      { binIdx: 0, confidenceRange: [0.0, 0.1], avgConfidence: 0.05, accuracy: 0.06, count: 120, gap: 0.01 },
      { binIdx: 1, confidenceRange: [0.1, 0.2], avgConfidence: 0.15, accuracy: 0.14, count: 210, gap: 0.01 },
      { binIdx: 2, confidenceRange: [0.2, 0.3], avgConfidence: 0.25, accuracy: 0.23, count: 350, gap: 0.02 },
      { binIdx: 3, confidenceRange: [0.3, 0.4], avgConfidence: 0.35, accuracy: 0.36, count: 480, gap: 0.01 },
      { binIdx: 4, confidenceRange: [0.4, 0.5], avgConfidence: 0.45, accuracy: 0.42, count: 590, gap: 0.03 },
      { binIdx: 5, confidenceRange: [0.5, 0.6], avgConfidence: 0.55, accuracy: 0.54, count: 720, gap: 0.01 },
      { binIdx: 6, confidenceRange: [0.6, 0.7], avgConfidence: 0.65, accuracy: 0.68, count: 860, gap: 0.03 },
      { binIdx: 7, confidenceRange: [0.7, 0.8], avgConfidence: 0.75, accuracy: 0.73, count: 1050, gap: 0.02 },
      { binIdx: 8, confidenceRange: [0.8, 0.9], avgConfidence: 0.85, accuracy: 0.87, count: 1400, gap: 0.02 },
      { binIdx: 9, confidenceRange: [0.9, 1.0], avgConfidence: 0.95, accuracy: 0.94, count: 2200, gap: 0.01 },
    ];
  }, []);

  // ---------------------------------------------------------------------------
  // 3. LOSS LANDSCAPE EXPLORER STATE
  // ---------------------------------------------------------------------------
  const [lossGridResolution, setLossGridResolution] = useState(15);
  const [curvatureData] = useState<HessianCurvatureData>({
    topEigenvalue: 18.42,
    trace: 142.8,
    conditionNumber: 64.2,
    eigenvalues: [18.42, 14.1, 9.8, 7.2, 5.4, 3.8, 2.9, 2.1, 1.4, 0.85, 0.62, 0.45, 0.31, 0.22, 0.15],
    esdHist: [0.05, 0.08, 0.12, 0.18, 0.22, 0.15, 0.09, 0.06, 0.03, 0.02],
    esdBins: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    sharpnessType: 'Isolated Outlier Direction (SAM Sensitive)',
  });

  const modeConnectivityPath: ModeConnectivityPath = useMemo(() => {
    const alphas = Array.from({ length: 21 }, (_, i) => i / 20);
    const lossProfile = alphas.map((a) => {
      // Barrier hump in linear vs flat in bezier
      return Number((1.8 + 1.2 * Math.sin(a * Math.PI) - 0.4 * a).toFixed(3));
    });
    return {
      alphas,
      lossProfile,
      curveType: 'linear',
      barrierHeight: 0.98,
      isModeConnected: true,
    };
  }, []);

  // ---------------------------------------------------------------------------
  // 4. COMPRESSION STUDIO STATE
  // ---------------------------------------------------------------------------
  const [searchCompressorQuery, setSearchCompressorQuery] = useState('');
  const [selectedCompressorCategory, setSelectedCompressorCategory] = useState<string>('ALL');
  const [targetBudgetMB, setTargetBudgetMB] = useState(120);

  const benchmarkParetoData: BenchmarkParetoModel[] = useMemo(() => {
    return [
      { compressorId: 'a3', name: 'A3 SVD', category: 'Low-Rank SVD', compressionRatio: 0.65, paramReductionPct: 65, accuracyRetention: 0.972, speedup: 2.2, latencyMs: 6.8, vramMb: 240, compressedParams: 43400000, isPareto: true },
      { compressorId: 'adasvd', name: 'AdaSVD', category: 'Low-Rank SVD', compressionRatio: 0.74, paramReductionPct: 74, accuracyRetention: 0.965, speedup: 2.6, latencyMs: 5.8, vramMb: 190, compressedParams: 32200000, isPareto: true },
      { compressorId: 'dobi_svd', name: 'DobiSVD', category: 'Low-Rank SVD', compressionRatio: 0.81, paramReductionPct: 81, accuracyRetention: 0.951, speedup: 2.9, latencyMs: 5.2, vramMb: 150, compressedParams: 23500000, isPareto: true },
      { compressorId: 'swift_svd', name: 'SwiftSVD', category: 'Low-Rank SVD', compressionRatio: 0.62, paramReductionPct: 62, accuracyRetention: 0.968, speedup: 3.2, latencyMs: 4.7, vramMb: 260, compressedParams: 47100000, isPareto: false },
      { compressorId: 'rotational_slicegpt_int4', name: 'Rotational SliceGPT INT4', category: 'Pruning', compressionRatio: 0.86, paramReductionPct: 86, accuracyRetention: 0.942, speedup: 3.5, latencyMs: 4.3, vramMb: 110, compressedParams: 17300000, isPareto: true },
      { compressorId: 'ast_pruner', name: 'AST 2:4 Pruner', category: 'Pruning', compressionRatio: 0.50, paramReductionPct: 50, accuracyRetention: 0.985, speedup: 2.0, latencyMs: 7.5, vramMb: 310, compressedParams: 62000000, isPareto: false },
      { compressorId: 'coopq', name: 'CoopQ Quantizer', category: 'Quantization', compressionRatio: 0.78, paramReductionPct: 78, accuracyRetention: 0.968, speedup: 3.6, latencyMs: 4.2, vramMb: 140, compressedParams: 27200000, isPareto: true },
      { compressorId: 'tt_linear', name: 'Tensor Train (TT-Linear)', category: 'Tensor Decompositions', compressionRatio: 0.91, paramReductionPct: 91, accuracyRetention: 0.925, speedup: 4.2, latencyMs: 3.6, vramMb: 75, compressedParams: 11100000, isPareto: true },
      { compressorId: 'delta_kv', name: 'DeltaKV', category: 'KV-Cache', compressionRatio: 0.76, paramReductionPct: 76, accuracyRetention: 0.992, speedup: 2.5, latencyMs: 6.0, vramMb: 160, compressedParams: 29700000, isPareto: true },
      { compressorId: 'galerkin', name: 'Galerkin Linear Attention', category: 'Linear Attention', compressionRatio: 0.68, paramReductionPct: 68, accuracyRetention: 0.958, speedup: 4.0, latencyMs: 3.8, vramMb: 210, compressedParams: 39600000, isPareto: true },
    ];
  }, []);

  const filteredCompressors = useMemo(() => {
    return COMPRESSOR_CATALOG.filter((c) => {
      const matchCat = selectedCompressorCategory === 'ALL' || c.category === selectedCompressorCategory;
      const matchQ = c.name.toLowerCase().includes(searchCompressorQuery.toLowerCase()) || c.category.toLowerCase().includes(searchCompressorQuery.toLowerCase());
      return matchCat && matchQ;
    });
  }, [selectedCompressorCategory, searchCompressorQuery]);

  const budgetAllocation: LayerBudgetAllocation[] = useMemo(() => {
    const baseSizePerLayer = 25.0; // MB
    const totalOriginal = selectedModel.numLayers * baseSizePerLayer;
    const factor = targetBudgetMB / totalOriginal;

    return Array.from({ length: selectedModel.numLayers }).map((_, l) => {
      const imp = 0.5 + 0.5 * Math.sin((l / selectedModel.numLayers) * Math.PI);
      const kept = Math.min(1.0, Math.max(0.2, factor * (0.7 + 0.6 * imp)));
      return {
        layer: l,
        keptRatio: Number(kept.toFixed(3)),
        compressionRatio: Number((1.0 - kept).toFixed(3)),
        sizeMb: Number((baseSizePerLayer * kept).toFixed(2)),
        importance: Number(imp.toFixed(3)),
      };
    });
  }, [selectedModel, targetBudgetMB]);

  // ---------------------------------------------------------------------------
  // 5. FULL PIPELINE STUDIO STATE
  // ---------------------------------------------------------------------------
  const [pipelineCurrentStep, setPipelineCurrentStep] = useState(0);
  const [isPlayingPipeline, setIsPlayingPipeline] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [challengeState, setChallengeState] = useState<ChallengeGameState>({
    active: false,
    reconstructionScore: 92.4,
    layerScores: [94, 91, 95, 89],
    rankBadge: 'Master',
    completed: false,
  });

  const pipelineStages: PipelineStageProfile[] = useMemo(() => {
    return [
      { stage: 'Tokenization', latencyMs: 0.12, memoryKb: 128, color: '#3b82f6' },
      { stage: 'Input Embedding', latencyMs: 0.25, memoryKb: 4096, color: '#8b5cf6' },
      { stage: 'Positional Encoding', latencyMs: 0.18, memoryKb: 2048, color: '#10b981' },
      { stage: 'Self-Attention QKV', latencyMs: 1.45, memoryKb: 12288, color: '#ef4444' },
      { stage: 'Softmax & Context', latencyMs: 1.82, memoryKb: 8192, color: '#f97316' },
      { stage: 'Residual Add 1', latencyMs: 0.08, memoryKb: 1024, color: '#64748b' },
      { stage: 'LayerNorm Pre-FFN', latencyMs: 0.15, memoryKb: 1024, color: '#64748b' },
      { stage: 'FFN Linear 1 (GELU)', latencyMs: 2.10, memoryKb: 16384, color: '#f59e0b' },
      { stage: 'FFN Linear 2', latencyMs: 1.65, memoryKb: 16384, color: '#f59e0b' },
      { stage: 'Residual Add 2', latencyMs: 0.08, memoryKb: 1024, color: '#64748b' },
      { stage: 'Final LN & Head', latencyMs: 0.65, memoryKb: 4096, color: '#06b6d4' },
    ];
  }, []);

  const totalPipelineLatency = useMemo(() => {
    return Number(pipelineStages.reduce((a, b) => a + b.latencyMs, 0).toFixed(2));
  }, [pipelineStages]);

  // Forward Pass Step Player Loop
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (isPlayingPipeline) {
      timer = setInterval(() => {
        setPipelineCurrentStep((prev) => (prev >= pipelineStages.length - 1 ? 0 : prev + 1));
      }, 1000 / playbackSpeed);
    }
    return () => clearInterval(timer);
  }, [isPlayingPipeline, playbackSpeed, pipelineStages.length]);

  return (
    <div className="flex flex-col h-full w-full bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* -----------------------------------------------------------------------
          TOP COCKPIT CONTROL HEADER
      ------------------------------------------------------------------------ */}
      <header className="px-6 py-3.5 bg-slate-900/90 border-b border-white/10 backdrop-blur-md flex flex-wrap items-center justify-between gap-4 z-20">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <Cpu className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold text-white tracking-wide">
                Transformer Optimization & Pipeline Studios
              </h1>
              <Badge variant="cyan" size="sm">
                Next-Gen Cockpit
              </Badge>
            </div>
            <p className="text-xs text-slate-400 font-mono">
              Mechanistic Attribution · Softmax Predictor · Loss Curvature · 80+ Engines · Symphony Tracer
            </p>
          </div>
        </div>

        {/* Studio Sub-Navigation Tabs */}
        <div className="flex items-center p-1 rounded-xl bg-slate-950/80 border border-white/10">
          <button
            onClick={() => setActiveTab('circuits')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'circuits' ? 'bg-cyan-500 text-slate-950 shadow-lg shadow-cyan-500/20' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Compass className="w-3.5 h-3.5" />
            Circuit Tracer
          </button>
          <button
            onClick={() => setActiveTab('logits')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'logits' ? 'bg-cyan-500 text-slate-950 shadow-lg shadow-cyan-500/20' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Zap className="w-3.5 h-3.5" />
            Output Logits
          </button>
          <button
            onClick={() => setActiveTab('loss_landscape')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'loss_landscape' ? 'bg-cyan-500 text-slate-950 shadow-lg shadow-cyan-500/20' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Mountain className="w-3.5 h-3.5" />
            Loss Landscape
          </button>
          <button
            onClick={() => setActiveTab('compression')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'compression' ? 'bg-cyan-500 text-slate-950 shadow-lg shadow-cyan-500/20' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Scale className="w-3.5 h-3.5" />
            Compression (80+)
          </button>
          <button
            onClick={() => setActiveTab('pipeline')}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'pipeline' ? 'bg-cyan-500 text-slate-950 shadow-lg shadow-cyan-500/20' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Workflow className="w-3.5 h-3.5" />
            Full Pipeline
          </button>
        </div>

        {/* Global Preset Selector */}
        <div className="flex items-center gap-3">
          <select
            value={selectedModel.id}
            onChange={(e) => {
              const m = OPTIMIZATION_MODEL_PRESETS.find((p) => p.id === e.target.value);
              if (m) setSelectedModel(m);
            }}
            className="bg-slate-800 border border-white/10 text-xs text-cyan-300 rounded-lg px-3 py-1.5 font-mono focus:outline-none focus:border-cyan-500"
          >
            {OPTIMIZATION_MODEL_PRESETS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} ({m.normType})
              </option>
            ))}
          </select>
        </div>
      </header>

      {/* Global Input Prompt Ribbon */}
      <div className="px-6 py-2 bg-slate-900/50 border-b border-white/5 flex items-center justify-between gap-4 text-xs font-mono">
        <div className="flex items-center gap-2 flex-1">
          <span className="text-slate-400 uppercase font-bold text-[10px] tracking-wider">Prompt:</span>
          <input
            type="text"
            value={promptText}
            onChange={(e) => setPromptText(e.target.value)}
            className="flex-1 bg-slate-950/60 border border-white/10 px-3 py-1 rounded text-cyan-200 focus:outline-none focus:border-cyan-500"
          />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-slate-400">Tokens:</span>
          <span className="px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 font-bold">{tokens.length}</span>
        </div>
      </div>

      {/* -----------------------------------------------------------------------
          MAIN STUDIO WORKSPACE
      ------------------------------------------------------------------------ */}
      <main className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* =====================================================================
            1. CIRCUIT TRACER STUDIO
        ====================================================================== */}
        {activeTab === 'circuits' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              {/* Attribution Graph Canvas */}
              <div className="lg:col-span-3 bg-slate-900/60 border border-white/10 rounded-2xl p-5 backdrop-blur-md">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <Compass className="w-5 h-5 text-cyan-400" />
                    <h2 className="text-sm font-bold uppercase tracking-wider text-white">
                      Mechanistic Feature Attribution Graph
                    </h2>
                  </div>
                  <div className="flex items-center gap-3 text-xs font-mono">
                    <span className="flex items-center gap-1.5 text-blue-400">
                      <span className="w-2.5 h-2.5 rounded-full bg-blue-500 inline-block" /> QK Routing
                    </span>
                    <span className="flex items-center gap-1.5 text-orange-400">
                      <span className="w-2.5 h-2.5 rounded-full bg-orange-500 inline-block" /> OV Writing
                    </span>
                    <label className="flex items-center gap-2 text-slate-400 ml-4">
                      Pruning Threshold:
                      <input
                        type="range"
                        min="0.01"
                        max="0.5"
                        step="0.01"
                        value={circuitPruneThreshold}
                        onChange={(e) => setCircuitPruneThreshold(parseFloat(e.target.value))}
                        className="w-24 accent-cyan-500"
                      />
                      <span className="text-cyan-300 font-bold">{circuitPruneThreshold.toFixed(2)}</span>
                    </label>
                  </div>
                </div>

                {/* SVG Attribution Graph Visualizer */}
                <div className="h-96 w-full bg-slate-950/80 rounded-xl border border-white/5 p-4 relative overflow-hidden flex flex-col justify-between">
                  <div className="flex justify-between text-[11px] font-mono text-slate-400 border-b border-white/5 pb-2">
                    <span>Input Tokens ({tokens.length})</span>
                    <span>Intermediate Layer Heads (L0 - L{Math.min(selectedModel.numLayers, 4) - 1})</span>
                    <span>Logits Prediction</span>
                  </div>

                  <div className="flex-1 flex items-center justify-between px-6 py-4">
                    {/* Left: Input Tokens */}
                    <div className="flex flex-col gap-2">
                      {tokens.slice(0, 7).map((tok, idx) => (
                        <div
                          key={idx}
                          className="px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs font-mono flex items-center gap-2"
                        >
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                          {tok}
                        </div>
                      ))}
                    </div>

                    {/* Middle: Transformer Attention Heads */}
                    <div className="grid grid-cols-4 gap-3">
                      {circuitNodes
                        .filter((n) => n.type === 'attention_head')
                        .slice(0, 12)
                        .map((n) => (
                          <div
                            key={n.id}
                            onClick={() => {
                              setSelectedLayer(n.layer);
                              setSelectedHead(n.head || 0);
                            }}
                            className={`p-2.5 rounded-xl border cursor-pointer transition-all ${
                              selectedLayer === n.layer && selectedHead === n.head
                                ? 'bg-cyan-500/20 border-cyan-400 shadow-lg shadow-cyan-500/20'
                                : 'bg-slate-900/80 border-white/10 hover:border-white/30'
                            }`}
                          >
                            <div className="text-[11px] font-mono font-bold text-white flex items-center justify-between">
                              {n.label}
                              <span
                                className={`w-2 h-2 rounded-full ${
                                  n.headType === 'induction'
                                    ? 'bg-blue-400'
                                    : n.headType === 'previous_token'
                                    ? 'bg-purple-400'
                                    : n.headType === 'copy_amplification'
                                    ? 'bg-amber-400'
                                    : 'bg-slate-400'
                                }`}
                              />
                            </div>
                            <div className="text-[9px] font-mono text-cyan-400 uppercase mt-1">
                              {n.headType?.replace('_', ' ')}
                            </div>
                          </div>
                        ))}
                    </div>

                    {/* Right: Output Logits */}
                    <div className="flex flex-col gap-2">
                      <div className="px-4 py-3 rounded-xl bg-purple-500/20 border border-purple-500/40 text-purple-300 text-xs font-mono font-bold">
                        <div>Next Token:</div>
                        <div className="text-white text-sm font-bold mt-1">"Paris" (91.4%)</div>
                        <div className="text-[10px] text-purple-300/70 mt-1">Logit Diff: +2.84</div>
                      </div>
                    </div>
                  </div>

                  <div className="text-[10px] text-slate-500 font-mono text-center">
                    Rendered with {circuitNodes.length} nodes and {circuitEdges.length} active feature attribution edges.
                  </div>
                </div>
              </div>

              {/* Head Classification & Causal Intervention */}
              <div className="space-y-6">
                <div className="bg-slate-900/60 border border-white/10 rounded-2xl p-5 backdrop-blur-md">
                  <div className="flex items-center gap-2 mb-3">
                    <Shield className="w-4 h-4 text-cyan-400" />
                    <h3 className="text-xs font-bold uppercase tracking-wider text-white">
                      Head Type Classifier (L{selectedLayer} H{selectedHead})
                    </h3>
                  </div>

                  <div className="space-y-2.5 text-xs font-mono">
                    <div>
                      <div className="flex justify-between text-slate-300 mb-1">
                        <span>Induction Pattern:</span>
                        <span className="text-blue-400 font-bold">0.89</span>
                      </div>
                      <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                        <div className="h-full bg-blue-500 w-[89%]" />
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-slate-300 mb-1">
                        <span>Previous Token Match:</span>
                        <span className="text-purple-400 font-bold">0.65</span>
                      </div>
                      <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                        <div className="h-full bg-purple-500 w-[65%]" />
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-slate-300 mb-1">
                        <span>Copy / Eigenvalue Spec:</span>
                        <span className="text-amber-400 font-bold">0.78</span>
                      </div>
                      <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                        <div className="h-full bg-amber-500 w-[78%]" />
                      </div>
                    </div>
                  </div>

                  <div className="mt-4 p-2.5 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-[11px] font-mono text-cyan-300">
                    Classification: <strong className="text-white">Induction & Feature Amplifying Head</strong>
                  </div>
                </div>

                {/* Causal Intervention Playground */}
                <div className="bg-slate-900/60 border border-white/10 rounded-2xl p-5 backdrop-blur-md">
                  <div className="flex items-center gap-2 mb-3">
                    <Crosshair className="w-4 h-4 text-orange-400" />
                    <h3 className="text-xs font-bold uppercase tracking-wider text-white">
                      Causal Intervention & Ablation
                    </h3>
                  </div>

                  <div className="space-y-3 text-xs font-mono">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Ablation Mode:</span>
                      <div className="flex gap-1">
                        <button
                          onClick={() => setInterventionState((s) => ({ ...s, ablationType: 'zero' }))}
                          className={`px-2 py-0.5 rounded ${
                            interventionState.ablationType === 'zero' ? 'bg-orange-500 text-slate-950 font-bold' : 'bg-slate-800 text-slate-400'
                          }`}
                        >
                          Zero Knockout
                        </button>
                        <button
                          onClick={() => setInterventionState((s) => ({ ...s, ablationType: 'scale' }))}
                          className={`px-2 py-0.5 rounded ${
                            interventionState.ablationType === 'scale' ? 'bg-orange-500 text-slate-950 font-bold' : 'bg-slate-800 text-slate-400'
                          }`}
                        >
                          Scale
                        </button>
                      </div>
                    </div>

                    {interventionState.ablationType === 'scale' && (
                      <div>
                        <div className="flex justify-between text-slate-300 mb-1">
                          <span>Scale Factor:</span>
                          <span className="text-orange-400 font-bold">{interventionState.scale.toFixed(1)}x</span>
                        </div>
                        <input
                          type="range"
                          min="0.0"
                          max="2.0"
                          step="0.1"
                          value={interventionState.scale}
                          onChange={(e) => setInterventionState((s) => ({ ...s, scale: parseFloat(e.target.value) }))}
                          className="w-full accent-orange-500"
                        />
                      </div>
                    )}

                    <div className="p-3 bg-slate-950/80 rounded-lg border border-white/5 space-y-1.5">
                      <div className="flex justify-between">
                        <span className="text-slate-400">KL Divergence:</span>
                        <span className="text-cyan-300 font-bold">{interventionState.klDivergence}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Logit Difference:</span>
                        <span className="text-amber-300 font-bold">-{interventionState.logitDiff}</span>
                      </div>
                      <div className="flex justify-between items-center pt-1 border-t border-white/5">
                        <span className="text-slate-400">Top-1 Flipped:</span>
                        <Badge variant={interventionState.topTokenFlipped ? 'danger' : 'success'} size="sm">
                          {interventionState.topTokenFlipped ? 'FLIPPED (to France)' : 'STABLE'}
                        </Badge>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* =====================================================================
            2. OUTPUT LOGITS & TUNED LENS STUDIO
        ====================================================================== */}
        {activeTab === 'logits' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Token Probability Spectrum */}
              <div className="lg:col-span-2 bg-slate-900/60 border border-white/10 rounded-2xl p-5 backdrop-blur-md">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <Zap className="w-5 h-5 text-cyan-400" />
                    <h2 className="text-sm font-bold uppercase tracking-wider text-white">
                      Softmax Top-10 Token Probability Distribution
                    </h2>
                  </div>
                  <div className="flex items-center gap-3 text-xs font-mono">
                    <span className="text-slate-400">
                      Entropy: <strong className="text-cyan-400">{entropyBits.toFixed(2)} bits</strong>
                    </span>
                    <Badge variant="cyan" size="sm">
                      T={samplingConfig.temperature}
                    </Badge>
                  </div>
                </div>

                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={tokenProbabilities} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                      <XAxis type="number" stroke="#94a3b8" domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                      <YAxis dataKey="token" type="category" stroke="#94a3b8" />
                      <RechartsTooltip
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                        formatter={(value: any) => [`${(Number(value) * 100).toFixed(2)}%`, 'Probability']}
                      />
                      <Bar dataKey="probability" radius={[0, 4, 4, 0]}>
                        {tokenProbabilities.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={index === 0 ? '#f59e0b' : '#3b82f6'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Sampling Parameters Bar */}
                <div className="mt-4 pt-4 border-t border-white/10 grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono">
                  <div>
                    <div className="flex justify-between text-slate-400 mb-1">
                      <span>Temperature:</span>
                      <span className="text-cyan-300 font-bold">{samplingConfig.temperature.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="2.0"
                      step="0.05"
                      value={samplingConfig.temperature}
                      onChange={(e) => setSamplingConfig((c) => ({ ...c, temperature: parseFloat(e.target.value) }))}
                      className="w-full accent-cyan-500"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between text-slate-400 mb-1">
                      <span>Top-K Cutoff:</span>
                      <span className="text-cyan-300 font-bold">{samplingConfig.topK}</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="50"
                      value={samplingConfig.topK}
                      onChange={(e) => setSamplingConfig((c) => ({ ...c, topK: parseInt(e.target.value) }))}
                      className="w-full accent-cyan-500"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between text-slate-400 mb-1">
                      <span>Top-P Nucleus:</span>
                      <span className="text-cyan-300 font-bold">{samplingConfig.topP.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.05"
                      value={samplingConfig.topP}
                      onChange={(e) => setSamplingConfig((c) => ({ ...c, topP: parseFloat(e.target.value) }))}
                      className="w-full accent-cyan-500"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between text-slate-400 mb-1">
                      <span>Repetition Penalty:</span>
                      <span className="text-cyan-300 font-bold">{samplingConfig.repetitionPenalty.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="1.0"
                      max="2.0"
                      step="0.05"
                      value={samplingConfig.repetitionPenalty}
                      onChange={(e) => setSamplingConfig((c) => ({ ...c, repetitionPenalty: parseFloat(e.target.value) }))}
                      className="w-full accent-cyan-500"
                    />
                  </div>
                </div>
              </div>

              {/* Reliability Diagram (Calibration Curve) */}
              <div className="bg-slate-900/60 border border-white/10 rounded-2xl p-5 backdrop-blur-md flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Target className="w-4 h-4 text-emerald-400" />
                      <h3 className="text-xs font-bold uppercase tracking-wider text-white">Reliability Calibration</h3>
                    </div>
                    <Badge variant="success" size="sm">
                      ECE: 0.018
                    </Badge>
                  </div>
                  <p className="text-xs text-slate-400 mb-3">
                    Empirical Accuracy vs. Confidence with MCE: 0.034 & Brier: 0.042.
                  </p>

                  <div className="h-48 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={calibrationBins} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                        <XAxis dataKey="avgConfidence" stroke="#94a3b8" tickFormatter={(v) => v.toFixed(1)} />
                        <YAxis stroke="#94a3b8" domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} />
                        <RechartsTooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
                        <Line type="monotone" dataKey="avgConfidence" stroke="#64748b" strokeDasharray="4 4" dot={false} name="Ideal (y=x)" />
                        <Line type="monotone" dataKey="accuracy" stroke="#10b981" strokeWidth={2.5} dot={{ r: 4 }} name="Model Accuracy" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="mt-3 p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-[11px] font-mono text-emerald-300">
                  Status: <strong>Well-Calibrated Predictions</strong>
                </div>
              </div>
            </div>

            {/* Tuned / Logit Lens Cross-Layer Timeline */}
            <div className="bg-slate-900/60 border border-white/10 rounded-2xl p-5 backdrop-blur-md">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Sliders className="w-5 h-5 text-purple-400" />
                  <h3 className="text-sm font-bold uppercase tracking-wider text-white">
                    Tuned Lens / Logit Lens Cross-Layer Prediction Progression
                  </h3>
                </div>
                <div className="text-xs text-slate-400 font-mono">
                  Tracing how token predictions crystalize across layers L0 $\to$ L{selectedModel.numLayers - 1}
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 font-mono text-xs">
                {tunedLensData.map((d) => (
                  <div
                    key={d.layer}
                    className={`p-3 rounded-xl border ${
                      d.matchesFinal
                        ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                        : 'bg-slate-950/70 border-white/10 text-slate-300'
                    }`}
                  >
                    <div className="flex justify-between items-center text-[10px] text-slate-400 mb-1">
                      <span>Layer {d.layer}</span>
                      <span>H: {d.entropy}</span>
                    </div>
                    <div className="text-sm font-bold text-white mt-1">"{d.top1Token}"</div>
                    <div className="flex justify-between items-center mt-2 text-[10px]">
                      <span>P: {(d.top1Prob * 100).toFixed(0)}%</span>
                      <span>KL: {d.klToFinal}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* =====================================================================
            3. LOSS LANDSCAPE EXPLORER STUDIO
        ====================================================================== */}
        {activeTab === 'loss_landscape' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* 2D Contour Map & 3D Surface View */}
              <div className="lg:col-span-2 bg-slate-900/60 border border-white/10 rounded-2xl p-5 backdrop-blur-md">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <Mountain className="w-5 h-5 text-cyan-400" />
                    <h2 className="text-sm font-bold uppercase tracking-wider text-white">
                      Loss Landscape 2D Contour & Optimization Trajectory
                    </h2>
                  </div>
                  <div className="flex items-center gap-3 text-xs font-mono">
                    <span className="text-slate-400">Filter-Normalized Subspace ($\alpha, \beta$)</span>
                    <Badge variant="cyan" size="sm">
                      SAM $\rho=0.05$
                    </Badge>
                  </div>
                </div>

                {/* Contour Heatmap Grid Visualizer */}
                <div className="h-80 w-full bg-slate-950/80 rounded-xl border border-white/5 p-4 relative overflow-hidden flex items-center justify-center">
                  <div className="grid grid-cols-12 gap-1 w-full max-w-md">
                    {Array.from({ length: 96 }).map((_, idx) => {
                      const row = Math.floor(idx / 12);
                      const col = idx % 12;
                      const dist = Math.sqrt((row - 5.5) ** 2 + (col - 5.5) ** 2);
                      const loss = 1.2 + dist * 0.45 + Math.sin(col) * 0.2;
                      const bg =
                        dist < 2
                          ? 'bg-emerald-500/80'
                          : dist < 4
                          ? 'bg-blue-500/70'
                          : dist < 6
                          ? 'bg-amber-500/70'
                          : 'bg-rose-500/80';
                      return (
                        <div
                          key={idx}
                          title={`alpha: ${(col - 5.5).toFixed(1)}, beta: ${(row - 5.5).toFixed(1)}, Loss: ${loss.toFixed(2)}`}
                          className={`h-5 rounded-sm ${bg} hover:scale-125 transition-all cursor-pointer`}
                        />
                      );
                    })}
                  </div>
                  <div className="absolute bottom-3 right-4 text-[10px] font-mono text-slate-400">
                    Min Loss: 1.20 (Center Checkpoint)
                  </div>
                </div>

                {/* Linear Mode Connectivity Simulator */}
                <div className="mt-4 pt-4 border-t border-white/10">
                  <div className="flex items-center justify-between text-xs font-mono mb-2">
                    <span className="text-slate-300 font-bold">Linear Mode Connectivity Path (Model A $\to$ Model B):</span>
                    <span className="text-amber-400">Barrier Height: {modeConnectivityPath.barrierHeight}</span>
                  </div>
                  <div className="h-32 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={modeConnectivityPath.alphas.map((a, i) => ({ alpha: a, loss: modeConnectivityPath.lossProfile[i] }))}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                        <XAxis dataKey="alpha" stroke="#94a3b8" tickFormatter={(v) => v.toFixed(1)} />
                        <YAxis stroke="#94a3b8" domain={[1.0, 3.5]} />
                        <RechartsTooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
                        <Area type="monotone" dataKey="loss" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.2} strokeWidth={2} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              {/* Hessian Eigenvalue Curvature Spectrum */}
              <div className="bg-slate-900/60 border border-white/10 rounded-2xl p-5 backdrop-blur-md flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Activity className="w-4 h-4 text-cyan-400" />
                      <h3 className="text-xs font-bold uppercase tracking-wider text-white">Hessian Eigenvalue Curvature</h3>
                    </div>
                    <Badge variant="cyan" size="sm">
                      $\kappa$ = {curvatureData.conditionNumber}
                    </Badge>
                  </div>

                  <div className="h-44 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={curvatureData.eigenvalues.map((v, i) => ({ rank: i + 1, eigenvalue: v }))}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                        <XAxis dataKey="rank" stroke="#94a3b8" />
                        <YAxis stroke="#94a3b8" />
                        <RechartsTooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155' }} />
                        <Bar dataKey="eigenvalue" fill="#3b82f6" radius={[2, 2, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="mt-3 space-y-2 text-xs font-mono text-slate-300">
                    <div className="flex justify-between">
                      <span className="text-slate-400">Top Eigenvalue $\lambda_1$:</span>
                      <span className="text-cyan-300 font-bold">{curvatureData.topEigenvalue}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Hessian Trace:</span>
                      <span className="text-cyan-300 font-bold">{curvatureData.trace}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Condition Number:</span>
                      <span className="text-amber-300 font-bold">{curvatureData.conditionNumber}</span>
                    </div>
                  </div>
                </div>

                <div className="mt-3 p-2.5 rounded-lg bg-blue-500/10 border border-blue-500/20 text-[11px] font-mono text-blue-300">
                  {curvatureData.sharpnessType}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* =====================================================================
            4. COMPRESSION STUDIO (80+ Engines & Pareto Frontier)
        ====================================================================== */}
        {activeTab === 'compression' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Pareto Frontier Benchmark Plot */}
              <div className="lg:col-span-2 bg-slate-900/60 border border-white/10 rounded-2xl p-5 backdrop-blur-md">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <Scale className="w-5 h-5 text-cyan-400" />
                    <h2 className="text-sm font-bold uppercase tracking-wider text-white">
                      Multi-Objective Pareto Frontier (Accuracy vs. Compression Ratio)
                    </h2>
                  </div>
                  <div className="text-xs text-slate-400 font-mono">
                    80+ Engines Benchmark Matrix
                  </div>
                </div>

                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis
                        type="number"
                        dataKey="paramReductionPct"
                        name="Parameter Reduction"
                        unit="%"
                        stroke="#94a3b8"
                        domain={[40, 100]}
                      />
                      <YAxis
                        type="number"
                        dataKey="accuracyRetention"
                        name="Accuracy Retention"
                        stroke="#94a3b8"
                        domain={[0.9, 1.0]}
                        tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                      />
                      <ZAxis type="number" dataKey="speedup" range={[60, 300]} name="Speedup" />
                      <RechartsTooltip
                        contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                        formatter={(value: any, name: string) => [
                          name === 'Accuracy Retention' ? `${(Number(value) * 100).toFixed(1)}%` : value,
                          name,
                        ]}
                      />
                      <Scatter name="Compressors" data={benchmarkParetoData} fill="#3b82f6">
                        {benchmarkParetoData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.isPareto ? '#10b981' : '#3b82f6'} />
                        ))}
                      </Scatter>
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>

                {/* Target Budget Constraint Solver */}
                <div className="mt-4 pt-4 border-t border-white/10">
                  <div className="flex items-center justify-between text-xs font-mono mb-2">
                    <span className="text-slate-300 font-bold">Memory Budget Constraint Allocation:</span>
                    <span className="text-cyan-400">Target Budget: {targetBudgetMB} MB (Original: 300 MB)</span>
                  </div>
                  <input
                    type="range"
                    min="50"
                    max="280"
                    value={targetBudgetMB}
                    onChange={(e) => setTargetBudgetMB(parseInt(e.target.value))}
                    className="w-full accent-cyan-500 mb-3"
                  />
                  <div className="grid grid-cols-4 md:grid-cols-8 gap-2 font-mono text-[10px]">
                    {budgetAllocation.map((b) => (
                      <div key={b.layer} className="p-2 rounded bg-slate-950 border border-white/5 text-center">
                        <div className="text-slate-400">L{b.layer}</div>
                        <div className="text-emerald-400 font-bold mt-1">{(b.keptRatio * 100).toFixed(0)}%</div>
                        <div className="text-slate-500">{b.sizeMb}M</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* 80+ Engines Filterable Catalog */}
              <div className="bg-slate-900/60 border border-white/10 rounded-2xl p-5 backdrop-blur-md flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Search className="w-4 h-4 text-cyan-400" />
                      <h3 className="text-xs font-bold uppercase tracking-wider text-white">Compressor Catalog (80+)</h3>
                    </div>
                  </div>

                  <input
                    type="text"
                    placeholder="Search compressor or category..."
                    value={searchCompressorQuery}
                    onChange={(e) => setSearchCompressorQuery(e.target.value)}
                    className="w-full bg-slate-950 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-cyan-200 mb-3 focus:outline-none focus:border-cyan-500 font-mono"
                  />

                  <div className="h-64 overflow-y-auto space-y-2 pr-1 font-mono text-xs">
                    {filteredCompressors.map((c) => (
                      <div key={c.id} className="p-2.5 rounded-lg bg-slate-950/80 border border-white/5 hover:border-cyan-500/40 transition-all">
                        <div className="flex justify-between items-center">
                          <span className="font-bold text-white">{c.name}</span>
                          <span className="text-[10px] text-cyan-400">{c.typicalRatio}</span>
                        </div>
                        <div className="text-[10px] text-slate-400 mt-1">{c.category} · {c.speedup}</div>
                        <div className="text-[10px] text-slate-500 font-serif italic mt-0.5 truncate">{c.latex}</div>
                      </div>
                    ))}
                  </div>
                </div>

                <Button variant="primary" className="w-full mt-4 flex items-center justify-center gap-2 text-xs font-bold">
                  <Download className="w-4 h-4" /> 1-Click Compress & Export
                </Button>
              </div>
            </div>
          </div>
        )}

        {/* =====================================================================
            5. FULL PIPELINE SYMPHONY STUDIO
        ====================================================================== */}
        {activeTab === 'pipeline' && (
          <div className="space-y-6">
            {/* Symphony Forward Pass Trace */}
            <div className="bg-slate-900/60 border border-white/10 rounded-2xl p-5 backdrop-blur-md">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Workflow className="w-5 h-5 text-cyan-400" />
                  <h2 className="text-sm font-bold uppercase tracking-wider text-white">
                    End-to-End Transformer Computation Symphony
                  </h2>
                </div>
                <div className="flex items-center gap-3 text-xs font-mono">
                  <span>Total Latency: <strong className="text-cyan-400">{totalPipelineLatency} ms</strong></span>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => setIsPlayingPipeline(!isPlayingPipeline)}
                      className="p-1.5 rounded-lg bg-cyan-500 text-slate-950 font-bold"
                    >
                      {isPlayingPipeline ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                    </button>
                    <button
                      onClick={() => setPipelineCurrentStep(0)}
                      className="p-1.5 rounded-lg bg-slate-800 text-slate-300"
                    >
                      <RotateCcw className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>

              {/* Horizontal Symphony Stage Pipeline */}
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-11 gap-2 text-xs font-mono">
                {pipelineStages.map((st, idx) => {
                  const isActive = idx === pipelineCurrentStep;
                  return (
                    <div
                      key={st.stage}
                      onClick={() => setPipelineCurrentStep(idx)}
                      className={`p-3 rounded-xl border transition-all cursor-pointer flex flex-col justify-between ${
                        isActive
                          ? 'bg-cyan-500/20 border-cyan-400 shadow-lg shadow-cyan-500/30 scale-105'
                          : 'bg-slate-950/70 border-white/5 hover:border-white/20'
                      }`}
                    >
                      <div>
                        <span className="text-[10px] text-slate-400 block mb-1">Step {idx + 1}</span>
                        <div className="font-bold text-white text-[11px] leading-tight">{st.stage}</div>
                      </div>
                      <div className="mt-3 pt-2 border-t border-white/5 text-[10px]">
                        <div className="text-cyan-300 font-bold">{st.latencyMs} ms</div>
                        <div className="text-slate-500">{st.memoryKb} KB</div>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Gamified Challenge Mode Panel */}
              <div className="mt-6 p-4 rounded-xl bg-slate-950/80 border border-white/10 flex flex-wrap items-center justify-between gap-4 font-mono text-xs">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-400">
                    <Award className="w-5 h-5" />
                  </div>
                  <div>
                    <div className="font-bold text-white">Gamified Pipeline Challenge Mode</div>
                    <div className="text-slate-400 text-[11px]">
                      Reconstruct hidden intermediate activations to verify mathematical mechanics.
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  <div>
                    Score: <strong className="text-emerald-400">{challengeState.reconstructionScore}%</strong>
                  </div>
                  <Badge variant="cyan" size="sm">
                    Badge: {challengeState.rankBadge}
                  </Badge>
                  <Button variant="secondary" size="sm" onClick={() => setChallengeState((s) => ({ ...s, active: !s.active }))}>
                    {challengeState.active ? 'Hide Mystery Puzzle' : 'Play Reconstruction Puzzle'}
                  </Button>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default OptimizationStudios;
