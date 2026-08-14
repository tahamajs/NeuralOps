// src/features/attentionDashboard/types/attention.ts

export type ViewMode = 'head' | 'model' | 'sankey' | 'qk' | '3d' | 'generation';
export type ColorPalette = 'heat' | 'viridis' | 'blues' | 'turbo' | 'monokai';
export type SpecializationType =
  | 'Positional Head'
  | 'Induction Head'
  | 'Semantic Head'
  | 'Delimiter Head'
  | 'Broad Context Head';

export interface AttentionContext {
  modelId: string;
  modelSource: 'registry' | 'huggingface' | 'upload';
  modelVersion?: string;
  attentionType: string;
  attentionConfig: {
    rank?: number;
    order?: number;
    temperature?: number;
    headDim?: number;
  };
  inputText: string;
  tokens: string[];
  selectedLayer: number;
  selectedHead: number;
  numLayers: number;
  numHeads: number;
  activeView: ViewMode;
  colorPalette: ColorPalette;
  maskDiagonal: boolean;
  multiHeadOverlay: boolean;
  multiHeadMode: 'average' | 'max' | 'variance';
  selectedHeadsOverlay: number[];
  diffMode: boolean;
  diffTargetAttentionType?: string;
}

export interface HeadCell {
  queryIdx: number;
  keyIdx: number;
  queryToken: string;
  keyToken: string;
  weight: number;
  rawScore?: number;
  isDiagonal: boolean;
}

export interface TopAttendedPair {
  query_idx: number;
  key_idx: number;
  query_token: string;
  key_token: string;
  weight: number;
}

export interface HeadViewData {
  layer_idx: number;
  head_idx: number;
  model_name: string;
  attention_type: string;
  tokens: string[];
  matrix: number[][];
  entropy_per_token: number[];
  mean_entropy: number;
  concentration_top3: number;
  diagonal_ratio: number;
  top_attended_pairs: TopAttendedPair[];
  svg_heatmap?: string;
}

export interface HeadMetric {
  layer_idx: number;
  head_idx: number;
  importance: number;
  diversity: number;
  entropy: number;
  concentration: number;
  specialization: SpecializationType;
  confidence: number;
  thumbnail?: number[][];
}

export interface ModelViewData {
  num_layers: number;
  num_heads: number;
  heads: HeadMetric[];
  layer_entropies: number[];
  layer_importances: number[];
  most_important_heads: Array<{
    layer: number;
    head: number;
    importance: number;
    specialization: string;
    entropy: number;
  }>;
  specialization_counts: Record<string, number>;
}

export interface SankeyNode {
  id: string;
  name: string;
  layer: number;
  node_type: 'input' | 'head' | 'output';
  value: number;
  color?: string;
}

export interface SankeyLink {
  source: string;
  target: string;
  value: number;
  weight: number;
  color?: string;
}

export interface SankeyData {
  nodes: SankeyNode[];
  links: SankeyLink[];
  layers_count: number;
  tokens: string[];
}

export interface EmbeddingPoint {
  id: string;
  token_idx: number;
  token: string;
  point_type: 'query' | 'key';
  x: number;
  y: number;
  magnitude: number;
  attention_sum: number;
}

export interface QKEmbeddingData {
  method: string;
  points: EmbeddingPoint[];
  connections: Array<{
    source: string;
    target: string;
    query_idx: number;
    key_idx: number;
    weight: number;
    raw_score: number;
  }>;
  tokens: string[];
  explained_variance_ratio?: number[];
}

export interface MathStep {
  step_number: number;
  title: string;
  latex_equation: string;
  description: string;
  variables: Record<string, string>;
  sub_values: Record<string, any>;
}

export interface LiveMathBreakdown {
  attention_type: string;
  primary_formula: string;
  variable_color_map: Record<string, string>;
  steps: MathStep[];
  complexity_table: {
    time_complexity: string;
    memory_complexity: string;
    speedup_vs_dense: string;
    flops_n1024: string;
  };
  numerical_inspection: {
    query_token: string;
    key_token: string;
    query_index: number;
    key_index: number;
    head_dimension_d_k: number;
    sqrt_d_k: number;
    query_vector_norm: number;
    key_vector_norm: number;
    raw_dot_product: number;
    scaled_score: number;
    exp_numerator: number;
    row_partition_sum: number;
    attention_probability: number;
    value_vector_sample: number[];
    weighted_value_contribution: number[];
  };
  comparison_diff?: {
    baseline_name: string;
    baseline_formula: string;
    current_name: string;
    current_formula: string;
    differing_operations: Array<{
      op: string;
      baseline: string;
      current: string;
    }>;
  };
}

export interface ExplorationHistoryEntry {
  id: string;
  timestamp: number;
  modelId: string;
  attentionType: string;
  layer: number;
  head: number;
  textSnippet: string;
  meanEntropy: number;
  isStarred?: boolean;
}

export interface SelectionState {
  hoveredCell: { query: number; key: number } | null;
  selectedQueryToken: number | null;
  selectedKeyToken: number | null;
  brushedRange: { queryStart: number; queryEnd: number; keyStart: number; keyEnd: number } | null;
  pinnedPairs: Array<{ query: number; key: number }>;
}
