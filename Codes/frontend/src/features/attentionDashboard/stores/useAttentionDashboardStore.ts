// src/features/attentionDashboard/stores/useAttentionDashboardStore.ts
import { create } from 'zustand';
import {
  AttentionContext,
  HeadViewData,
  ModelViewData,
  SankeyData,
  QKEmbeddingData,
  LiveMathBreakdown,
  ExplorationHistoryEntry,
  SelectionState,
  ViewMode,
  ColorPalette,
  SpecializationType,
} from '../types/attention';

// Helper function to tokenize simple text
export function tokenizeText(text: string): string[] {
  if (!text || !text.trim()) {
    return ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'];
  }
  const clean = text.trim();
  const words = clean.split(/\s+/).slice(0, 32);
  return words;
}

// Generate realistic synthetic attention tensor for interactive client-side exploration
export function generateAttentionData(
  tokens: string[],
  layer: number,
  head: number,
  numLayers: number = 12,
  numHeads: number = 12,
  attentionType: string = 'SVD Linear Attention (rank=64)',
  modelName: string = 'BERT-base-uncased'
): {
  headView: HeadViewData;
  modelView: ModelViewData;
  sankey: SankeyData;
  qkEmbeddings: QKEmbeddingData;
  liveMath: LiveMathBreakdown;
} {
  const N = tokens.length;
  const d_k = 64;
  const rank = 16;

  // 1. Build N x N attention matrix for this specific head
  const matrix: number[][] = [];
  const entropies: number[] = [];

  for (let i = 0; i < N; i++) {
    const row: number[] = [];
    let sumExp = 0;
    const rawScores: number[] = [];

    for (let j = 0; j < N; j++) {
      let base = 0.5 + Math.sin(i * 1.5 + j * 0.8 + layer) * 0.3;

      // Inject recognizable attention motifs
      if (head % 4 === 0) {
        // Positional diagonal
        if (i === j) base += 3.5;
        if (i > 0 && j === i - 1) base += 1.8;
      } else if (head % 4 === 1) {
        // Induction / Previous token
        if (i > 0 && j === i - 1) base += 4.0;
        if (i > 1 && j === i - 2) base += 1.5;
      } else if (head % 4 === 2) {
        // Delimiter tokens (first or last token)
        if (j === 0) base += 3.0;
        if (j === N - 1) base += 1.5;
      } else {
        // Semantic / broad distribution
        base += ((i * 7 + j * 13) % 17) / 10.0;
      }

      // SVD low-rank filtering simulation
      if (attentionType.toLowerCase().includes('svd') || attentionType.toLowerCase().includes('galerkin')) {
        base = Math.max(0.1, base * 0.95);
      }

      const expVal = Math.exp(base);
      rawScores.push(base);
      sumExp += expVal;
    }

    let rowEntropy = 0;
    for (let j = 0; j < N; j++) {
      const prob = Math.exp(rawScores[j]) / sumExp;
      row.push(Number(prob.toFixed(4)));
      if (prob > 1e-12) {
        rowEntropy -= prob * Math.log2(prob);
      }
    }
    matrix.push(row);
    entropies.push(Number(rowEntropy.toFixed(3)));
  }

  const meanEntropy = Number((entropies.reduce((a, b) => a + b, 0) / N).toFixed(3));

  // Concentration: top 3 keys per row average
  let top3Total = 0;
  for (let i = 0; i < N; i++) {
    const sorted = [...matrix[i]].sort((a, b) => b - a);
    top3Total += (sorted[0] || 0) + (sorted[1] || 0) + (sorted[2] || 0);
  }
  const concentrationTop3 = Number((top3Total / N).toFixed(3));

  // Diagonal ratio
  let diagSum = 0;
  for (let i = 0; i < N; i++) {
    diagSum += matrix[i][i];
  }
  const diagonalRatio = Number((diagSum / N).toFixed(3));

  // Top pairs
  const topPairs: Array<{ query_idx: number; key_idx: number; query_token: string; key_token: string; weight: number }> = [];
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      topPairs.push({
        query_idx: i,
        key_idx: j,
        query_token: tokens[i],
        key_token: tokens[j],
        weight: matrix[i][j],
      });
    }
  }
  topPairs.sort((a, b) => b.weight - a.weight);

  const headView: HeadViewData = {
    layer_idx: layer,
    head_idx: head,
    model_name: modelName,
    attention_type: attentionType,
    tokens,
    matrix,
    entropy_per_token: entropies,
    mean_entropy: meanEntropy,
    concentration_top3: concentrationTop3,
    diagonal_ratio: diagonalRatio,
    top_attended_pairs: topPairs.slice(0, 10),
  };

  // 2. Build Model View across all layers x all heads
  const headMetrics: any[] = [];
  const layerEntropies: number[] = [];
  const layerImportances: number[] = [];
  const specCounts: Record<string, number> = {
    'Positional Head': 0,
    'Induction Head': 0,
    'Semantic Head': 0,
    'Delimiter Head': 0,
    'Broad Context Head': 0,
  };

  for (let l = 0; l < numLayers; l++) {
    let layerEntSum = 0;
    let layerImpSum = 0;

    for (let h = 0; h < numHeads; h++) {
      let spec: SpecializationType = 'Semantic Head';
      if (h % 4 === 0) spec = 'Positional Head';
      else if (h % 4 === 1) spec = 'Induction Head';
      else if (h % 4 === 2) spec = 'Delimiter Head';
      else if (l > 8) spec = 'Broad Context Head';

      specCounts[spec] = (specCounts[spec] || 0) + 1;

      const importance = Number((0.35 + ((l * 7 + h * 11) % 13) / 20.0).toFixed(2));
      const diversity = Number((0.4 + ((h * 17 + l * 5) % 11) / 20.0).toFixed(2));
      const ent = Number((1.5 + ((l + h) % 7) * 0.25).toFixed(2));

      layerEntSum += ent;
      layerImpSum += importance;

      // Small 4x4 thumbnail
      const thumb: number[][] = [];
      for (let tr = 0; tr < 4; tr++) {
        const trRow: number[] = [];
        for (let tc = 0; tc < 4; tc++) {
          trRow.push(tr === tc ? 0.8 : ((tr + tc) % 3 === 0 ? 0.4 : 0.1));
        }
        thumb.push(trRow);
      }

      headMetrics.push({
        layer_idx: l,
        head_idx: h,
        importance,
        diversity,
        entropy: ent,
        concentration: 0.65,
        specialization: spec,
        confidence: 0.88,
        thumbnail: thumb,
      });
    }

    layerEntropies.push(Number((layerEntSum / numHeads).toFixed(2)));
    layerImportances.push(Number((layerImpSum / numHeads).toFixed(2)));
  }

  const modelView: ModelViewData = {
    num_layers: numLayers,
    num_heads: numHeads,
    heads: headMetrics,
    layer_entropies: layerEntropies,
    layer_importances: layerImportances,
    most_important_heads: [
      { layer: 5, head: 7, importance: 0.95, specialization: 'Induction Head', entropy: 1.82 },
      { layer: 3, head: 2, importance: 0.89, specialization: 'Positional Head', entropy: 1.45 },
      { layer: 8, head: 4, importance: 0.86, specialization: 'Semantic Head', entropy: 2.31 },
      { layer: 1, head: 0, importance: 0.82, specialization: 'Delimiter Head', entropy: 1.12 },
    ],
    specialization_counts: specCounts,
  };

  // 3. Build Sankey Data
  const sankeyNodes: any[] = [];
  const sankeyLinks: any[] = [];

  tokens.forEach((t, i) => {
    sankeyNodes.push({
      id: `input_${i}`,
      name: t,
      layer: 0,
      node_type: 'input',
      value: 1.0,
      color: '#3b82f6',
    });
  });

  for (let l = 0; l < Math.min(3, numLayers); l++) {
    for (let h = 0; h < Math.min(4, numHeads); h++) {
      sankeyNodes.push({
        id: `L${l}_H${h}`,
        name: `L${l}·H${h}`,
        layer: l + 1,
        node_type: 'head',
        value: 1.5,
        color: l % 2 === 0 ? '#8b5cf6' : '#ec4899',
      });
    }
  }

  tokens.forEach((t, i) => {
    sankeyNodes.push({
      id: `output_${i}`,
      name: `Out:${t}`,
      layer: 4,
      node_type: 'output',
      value: 1.0,
      color: '#10b981',
    });
  });

  // Sample flow links
  for (let i = 0; i < N; i++) {
    sankeyLinks.push({
      source: `input_${i}`,
      target: `L0_H${i % 4}`,
      value: matrix[i][i] || 0.5,
      weight: matrix[i][i] || 0.5,
    });
  }
  for (let h = 0; h < 4; h++) {
    sankeyLinks.push({
      source: `L0_H${h}`,
      target: `L1_H${(h + 1) % 4}`,
      value: 0.75,
      weight: 0.75,
    });
    sankeyLinks.push({
      source: `L1_H${h}`,
      target: `L2_H${h}`,
      value: 0.65,
      weight: 0.65,
    });
    sankeyLinks.push({
      source: `L2_H${h}`,
      target: `output_${h % N}`,
      value: 0.8,
      weight: 0.8,
    });
  }

  const sankey: SankeyData = {
    nodes: sankeyNodes,
    links: sankeyLinks,
    layers_count: 5,
    tokens,
  };

  // 4. Build Q/K Joint Embeddings (PCA 2D simulation)
  const embeddingPoints: any[] = [];
  const connections: any[] = [];

  tokens.forEach((t, i) => {
    const angle = (i / N) * 2 * Math.PI;
    const radiusQ = 45 + ((i * 13) % 25);
    const radiusK = 50 + ((i * 17) % 30);

    embeddingPoints.push({
      id: `Q_${i}`,
      token_idx: i,
      token: t,
      point_type: 'query',
      x: Math.cos(angle) * radiusQ,
      y: Math.sin(angle) * radiusQ,
      magnitude: Number((2.5 + (i % 3) * 0.4).toFixed(2)),
      attention_sum: Number((0.8 + (i % 4) * 0.1).toFixed(2)),
    });

    embeddingPoints.push({
      id: `K_${i}`,
      token_idx: i,
      token: t,
      point_type: 'key',
      x: Math.cos(angle + 0.3) * radiusK,
      y: Math.sin(angle + 0.3) * radiusK,
      magnitude: Number((2.8 + (i % 2) * 0.5).toFixed(2)),
      attention_sum: Number((0.9 + (i % 3) * 0.15).toFixed(2)),
    });

    if (matrix[i] && matrix[i][(i + 1) % N] > 0.15) {
      connections.push({
        source: `Q_${i}`,
        target: `K_${(i + 1) % N}`,
        query_idx: i,
        key_idx: (i + 1) % N,
        weight: matrix[i][(i + 1) % N],
        raw_score: 1.25,
      });
    }
  });

  const qkEmbeddings: QKEmbeddingData = {
    method: 'PCA',
    points: embeddingPoints,
    connections,
    tokens,
    explained_variance_ratio: [0.62, 0.24],
  };

  // 5. Build Live Math Breakdown
  const liveMath: LiveMathBreakdown = {
    attention_type: attentionType,
    primary_formula:
      '\\text{Attention}_{\\text{SVD}}(Q, K, V) = \\text{softmax}\\left(\\frac{Q (K S_r)^T}{\\sqrt{d_k}}\\right) (V S_r)',
    variable_color_map: {
      Q: '#3b82f6',
      K: '#ef4444',
      V: '#10b981',
      d_k: '#94a3b8',
      softmax: '#a855f7',
      scale: '#f59e0b',
      SVD: '#ec4899',
    },
    steps: [
      {
        step_number: 1,
        title: 'Low-Rank SVD / Sketch Projection',
        latex_equation: 'K_r = K \\cdot S_r \\in \\mathbb{R}^{N \\times r}, \\quad V_r = V \\cdot S_r \\in \\mathbb{R}^{N \\times r}',
        description: `Projects Key & Value matrices into rank r=${rank} subspace via Petrov-Galerkin sketching.`,
        variables: { K: 'Key matrix', S_r: `Sketch matrix (N x ${rank})`, V_r: 'Compressed values' },
        sub_values: { rank, compression_ratio: '4.0x', dim_reduced: rank },
      },
      {
        step_number: 2,
        title: 'Low-Rank Scaled Dot-Product',
        latex_equation: 'S_{\\text{SVD}} = \\frac{Q K_r^T}{\\sqrt{d_k}} = \\frac{Q (K S_r)^T}{\\sqrt{64}}',
        description: 'Computes query-key compatibility in the compressed low-rank bottleneck.',
        variables: { Q: 'Query vector', K_r: 'Projected key', '\\sqrt{d_k}': 'Scaling factor = 8.0' },
        sub_values: { raw_dot: 0.847, scaled_score: 0.106 },
      },
      {
        step_number: 3,
        title: 'Row-wise Softmax Probability Distribution',
        latex_equation: 'A_{\\text{SVD}} = \\text{softmax}(S_{\\text{SVD}})_{ij} = \\frac{\\exp(S_{\\text{SVD}, ij})}{\\sum_k \\exp(S_{\\text{SVD}, ik})}',
        description: 'Normalizes compatibility scores into valid attention weights summing to 1.0.',
        variables: { A: 'Attention matrix', softmax: 'Normalized exponential' },
        sub_values: { exp_score: 1.112, attention_weight: matrix[0]?.[1] || 0.23 },
      },
      {
        step_number: 4,
        title: 'Compressed Value Synthesis',
        latex_equation: 'O_{\\text{SVD}} = A_{\\text{SVD}} \\cdot V_r = \\sum_{j=1}^N A_{\\text{SVD}, ij} V_{r, j}',
        description: 'Aggregates value representations weighted by attention probabilities.',
        variables: { O: 'Contextual representation', V_r: 'Low-rank value representations' },
        sub_values: { output_dim: `d_v = ${d_k}` },
      },
    ],
    complexity_table: {
      time_complexity: `O(N \\cdot r \\cdot d_k) = O(${rank} N d_k)`,
      memory_complexity: `O(N \\cdot r) = O(${rank} N)`,
      speedup_vs_dense: '4.0x quadratic bypass',
      flops_n1024: '1.05 MFLOPs (vs 8.39 MFLOPs dense)',
    },
    numerical_inspection: {
      query_token: tokens[0] || 'The',
      key_token: tokens[1] || 'quick',
      query_index: 0,
      key_index: 1,
      head_dimension_d_k: d_k,
      sqrt_d_k: 8.0,
      query_vector_norm: 3.42,
      key_vector_norm: 2.89,
      raw_dot_product: 0.847,
      scaled_score: 0.106,
      exp_numerator: 1.112,
      row_partition_sum: 5.432,
      attention_probability: matrix[0]?.[1] || 0.2047,
      value_vector_sample: [0.12, -0.34, 0.56, 0.08, -0.22],
      weighted_value_contribution: [0.024, -0.069, 0.114, 0.016, -0.045],
    },
    comparison_diff: {
      baseline_name: 'Standard Softmax MHA',
      baseline_formula: '\\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V',
      current_name: attentionType,
      current_formula:
        '\\text{softmax}\\left(\\frac{Q (K S_r)^T}{\\sqrt{d_k}}\\right) (V S_r)',
      differing_operations: [
        { op: 'Bottleneck Projection', baseline: 'None (Dense N x N)', current: `Rank r=${rank} Petrov-Galerkin SVD` },
        { op: 'Time Complexity', baseline: 'O(N² · d)', current: `O(N · ${rank} · d)` },
        { op: 'Memory Footprint', baseline: 'O(N²)', current: `O(N · ${rank})` },
      ],
    },
  };

  return { headView, modelView, sankey, qkEmbeddings, liveMath };
}

interface AttentionDashboardState {
  // Zone 1 Context
  context: AttentionContext;
  tokens: string[];
  history: ExplorationHistoryEntry[];

  // Zone 2 Canvas Data
  headViewData: HeadViewData;
  modelViewData: ModelViewData;
  sankeyData: SankeyData;
  qkEmbeddingData: QKEmbeddingData;

  // Cross-view selection
  selection: SelectionState;

  // Zone 3 Math Stepper
  liveMath: LiveMathBreakdown;
  currentMathStep: number;
  isMathPlaying: boolean;
  mathPlaySpeed: number;

  // Zone 4 Controls
  isTrainModalOpen: boolean;
  isReportModalOpen: boolean;
  isPaperModalOpen: boolean;
  isNoteModalOpen: boolean;
  isHistoryOpen: boolean;

  // Actions
  setModel: (modelId: string) => void;
  setAttentionType: (type: string) => void;
  setInputText: (text: string) => void;
  setSelectedLayer: (layer: number) => void;
  setSelectedHead: (head: number) => void;
  setActiveView: (view: ViewMode) => void;
  setColorPalette: (palette: ColorPalette) => void;
  setMaskDiagonal: (mask: boolean) => void;
  setMultiHeadOverlay: (enabled: boolean) => void;
  setMultiHeadMode: (mode: 'average' | 'max' | 'variance') => void;
  toggleHeadOverlay: (head: number) => void;
  setDiffMode: (diff: boolean) => void;

  // Selection Actions
  setHoveredCell: (cell: { query: number; key: number } | null) => void;
  setSelectedQueryToken: (idx: number | null) => void;
  setSelectedKeyToken: (idx: number | null) => void;
  togglePinPair: (query: number, key: number) => void;

  // Math Stepper Actions
  setCurrentMathStep: (step: number) => void;
  setIsMathPlaying: (playing: boolean) => void;
  setMathPlaySpeed: (speed: number) => void;

  // Modal Actions
  setTrainModalOpen: (open: boolean) => void;
  setReportModalOpen: (open: boolean) => void;
  setPaperModalOpen: (open: boolean) => void;
  setNoteModalOpen: (open: boolean) => void;
  setHistoryOpen: (open: boolean) => void;
  restoreHistoryEntry: (entry: ExplorationHistoryEntry) => void;
  toggleStarHistory: (id: string) => void;
}

const initialText = 'The quick brown fox jumps over the lazy dog';
const initialTokens = tokenizeText(initialText);
const initialData = generateAttentionData(initialTokens, 5, 7, 12, 12, 'SVD Linear Attention (rank=64)', 'BERT-base-uncased');

export const useAttentionDashboardStore = create<AttentionDashboardState>((set, get) => ({
  context: {
    modelId: 'BERT-base-uncased',
    modelSource: 'registry',
    modelVersion: 'v1.4.0',
    attentionType: 'SVD Linear Attention (rank=64)',
    attentionConfig: { rank: 64, temperature: 1.0, headDim: 64 },
    inputText: initialText,
    tokens: initialTokens,
    selectedLayer: 5,
    selectedHead: 7,
    numLayers: 12,
    numHeads: 12,
    activeView: 'head',
    colorPalette: 'heat',
    maskDiagonal: false,
    multiHeadOverlay: false,
    multiHeadMode: 'average',
    selectedHeadsOverlay: [7, 3, 5],
    diffMode: false,
  },
  tokens: initialTokens,
  history: [
    {
      id: 'hist-1',
      timestamp: Date.now() - 120000,
      modelId: 'BERT-base-uncased',
      attentionType: 'SVD Linear Attention (rank=64)',
      layer: 5,
      head: 7,
      textSnippet: 'The quick brown fox...',
      meanEntropy: 1.82,
      isStarred: true,
    },
    {
      id: 'hist-2',
      timestamp: Date.now() - 600000,
      modelId: 'GPT-2 (fine-tuned)',
      attentionType: 'Softmax MHA',
      layer: 3,
      head: 2,
      textSnippet: 'Attention mechanisms compress context...',
      meanEntropy: 2.14,
    },
  ],

  headViewData: initialData.headView,
  modelViewData: initialData.modelView,
  sankeyData: initialData.sankey,
  qkEmbeddingData: initialData.qkEmbeddings,
  liveMath: initialData.liveMath,

  selection: {
    hoveredCell: null,
    selectedQueryToken: null,
    selectedKeyToken: null,
    brushedRange: null,
    pinnedPairs: [],
  },

  currentMathStep: 1,
  isMathPlaying: false,
  mathPlaySpeed: 1,

  isTrainModalOpen: false,
  isReportModalOpen: false,
  isPaperModalOpen: false,
  isNoteModalOpen: false,
  isHistoryOpen: false,

  setModel: (modelId: string) => {
    const { context, tokens } = get();
    const newData = generateAttentionData(
      tokens,
      context.selectedLayer,
      context.selectedHead,
      context.numLayers,
      context.numHeads,
      context.attentionType,
      modelId
    );
    set({
      context: { ...context, modelId },
      headViewData: newData.headView,
      modelViewData: newData.modelView,
      sankeyData: newData.sankey,
      qkEmbeddingData: newData.qkEmbeddings,
      liveMath: newData.liveMath,
    });
  },

  setAttentionType: (attentionType: string) => {
    const { context, tokens } = get();
    const newData = generateAttentionData(
      tokens,
      context.selectedLayer,
      context.selectedHead,
      context.numLayers,
      context.numHeads,
      attentionType,
      context.modelId
    );
    set({
      context: { ...context, attentionType },
      headViewData: newData.headView,
      modelViewData: newData.modelView,
      sankeyData: newData.sankey,
      qkEmbeddingData: newData.qkEmbeddings,
      liveMath: newData.liveMath,
    });
  },

  setInputText: (inputText: string) => {
    const { context } = get();
    const newTokens = tokenizeText(inputText);
    const newData = generateAttentionData(
      newTokens,
      context.selectedLayer,
      context.selectedHead,
      context.numLayers,
      context.numHeads,
      context.attentionType,
      context.modelId
    );
    set({
      tokens: newTokens,
      context: { ...context, inputText, tokens: newTokens },
      headViewData: newData.headView,
      modelViewData: newData.modelView,
      sankeyData: newData.sankey,
      qkEmbeddingData: newData.qkEmbeddings,
      liveMath: newData.liveMath,
    });
  },

  setSelectedLayer: (selectedLayer: number) => {
    const { context, tokens } = get();
    const newData = generateAttentionData(
      tokens,
      selectedLayer,
      context.selectedHead,
      context.numLayers,
      context.numHeads,
      context.attentionType,
      context.modelId
    );
    set({
      context: { ...context, selectedLayer },
      headViewData: newData.headView,
      liveMath: newData.liveMath,
    });
  },

  setSelectedHead: (selectedHead: number) => {
    const { context, tokens } = get();
    const newData = generateAttentionData(
      tokens,
      context.selectedLayer,
      selectedHead,
      context.numLayers,
      context.numHeads,
      context.attentionType,
      context.modelId
    );
    set({
      context: { ...context, selectedHead },
      headViewData: newData.headView,
      liveMath: newData.liveMath,
    });
  },

  setActiveView: (activeView: ViewMode) =>
    set((state) => ({ context: { ...state.context, activeView } })),

  setColorPalette: (colorPalette: ColorPalette) =>
    set((state) => ({ context: { ...state.context, colorPalette } })),

  setMaskDiagonal: (maskDiagonal: boolean) =>
    set((state) => ({ context: { ...state.context, maskDiagonal } })),

  setMultiHeadOverlay: (multiHeadOverlay: boolean) =>
    set((state) => ({ context: { ...state.context, multiHeadOverlay } })),

  setMultiHeadMode: (multiHeadMode: 'average' | 'max' | 'variance') =>
    set((state) => ({ context: { ...state.context, multiHeadMode } })),

  toggleHeadOverlay: (head: number) =>
    set((state) => {
      const current = state.context.selectedHeadsOverlay;
      const next = current.includes(head) ? current.filter((h) => h !== head) : [...current, head];
      return { context: { ...state.context, selectedHeadsOverlay: next } };
    }),

  setDiffMode: (diffMode: boolean) =>
    set((state) => ({ context: { ...state.context, diffMode } })),

  setHoveredCell: (cell) =>
    set((state) => {
      if (!cell) {
        return { selection: { ...state.selection, hoveredCell: null } };
      }
      const qTok = state.tokens[cell.query] || `tok_${cell.query}`;
      const kTok = state.tokens[cell.key] || `tok_${cell.key}`;
      const weight = state.headViewData.matrix[cell.query]?.[cell.key] || 0.05;
      const rawDot = Number((weight * 8.0).toFixed(3));
      const scaled = Number((rawDot / 8.0).toFixed(3));

      const updatedLiveMath: LiveMathBreakdown = {
        ...state.liveMath,
        numerical_inspection: {
          ...state.liveMath.numerical_inspection,
          query_token: qTok,
          key_token: kTok,
          query_index: cell.query,
          key_index: cell.key,
          raw_dot_product: rawDot,
          scaled_score: scaled,
          attention_probability: weight,
        },
      };

      return {
        selection: { ...state.selection, hoveredCell: cell },
        liveMath: updatedLiveMath,
      };
    }),

  setSelectedQueryToken: (idx) =>
    set((state) => ({ selection: { ...state.selection, selectedQueryToken: idx } })),

  setSelectedKeyToken: (idx) =>
    set((state) => ({ selection: { ...state.selection, selectedKeyToken: idx } })),

  togglePinPair: (query, key) =>
    set((state) => {
      const current = state.selection.pinnedPairs;
      const exists = current.some((p) => p.query === query && p.key === key);
      const next = exists ? current.filter((p) => !(p.query === query && p.key === key)) : [...current, { query, key }];
      return { selection: { ...state.selection, pinnedPairs: next } };
    }),

  setCurrentMathStep: (step) => set({ currentMathStep: step }),
  setIsMathPlaying: (playing) => set({ isMathPlaying: playing }),
  setMathPlaySpeed: (speed) => set({ mathPlaySpeed: speed }),

  setTrainModalOpen: (open) => set({ isTrainModalOpen: open }),
  setReportModalOpen: (open) => set({ isReportModalOpen: open }),
  setPaperModalOpen: (open) => set({ isPaperModalOpen: open }),
  setNoteModalOpen: (open) => set({ isNoteModalOpen: open }),
  setHistoryOpen: (open) => set({ isHistoryOpen: open }),

  restoreHistoryEntry: (entry) => {
    const { context, tokens } = get();
    const newData = generateAttentionData(
      tokens,
      entry.layer,
      entry.head,
      context.numLayers,
      context.numHeads,
      entry.attentionType,
      entry.modelId
    );
    set({
      context: {
        ...context,
        modelId: entry.modelId,
        attentionType: entry.attentionType,
        selectedLayer: entry.layer,
        selectedHead: entry.head,
      },
      headViewData: newData.headView,
      modelViewData: newData.modelView,
      sankeyData: newData.sankey,
      qkEmbeddingData: newData.qkEmbeddings,
      liveMath: newData.liveMath,
      isHistoryOpen: false,
    });
  },

  toggleStarHistory: (id) =>
    set((state) => ({
      history: state.history.map((h) => (h.id === id ? { ...h, isStarred: !h.isStarred } : h)),
    })),
}));
