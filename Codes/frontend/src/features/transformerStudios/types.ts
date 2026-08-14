export type StudioTab = 'embeddings' | 'positional' | 'residual' | 'layernorm';

export interface TokenNode3D {
  id: string;
  token: string;
  x: number;
  y: number;
  z: number;
  cluster: number;
  norm: number;
  color?: string;
}

export interface SemanticAxis3D {
  name: string;
  direction: [number, number, number];
  positiveLabel: string;
  negativeLabel: string;
  color: string;
}

export interface RoPEDimPair {
  pairIndex: number;
  dimRange: [number, number];
  frequency: number;
  angleRad: number;
  angleDeg: number;
  wavelength: number;
  cosVal: number;
  sinVal: number;
}

export interface ResidualLayerState {
  layer: number;
  hiddenNorm: number;
  attnNorm: number;
  ffnNorm: number;
  totalNorm: number;
  attnFraction: number;
  cosSimAttnFfn: number;
  topLogits: { token: string; probability: number }[];
}

export interface Vector2DSandboxToken {
  token: string;
  initial: [number, number];
  layerPositions: [number, number][];
  attnDeltas: [number, number][];
  ffnDeltas: [number, number][];
}

export interface LayerNormLayerStat {
  layer: number;
  preMean: number;
  preVar: number;
  preRms: number;
  postMean: number;
  postVar: number;
  postRms: number;
  cosSim: number;
  gammaDrift: number;
  betaDrift: number;
}

export interface ModelPreset {
  id: string;
  name: string;
  family: string;
  dModel: number;
  numLayers: number;
  numHeads: number;
  vocabSize: number;
  normType: 'LayerNorm' | 'RMSNorm';
  posType: 'RoPE' | 'Sinusoidal' | 'ALiBi';
  baseFrequency: number;
}

// =====================================================================
// Computation Studios Types
// =====================================================================

export type ComputationStudioTab = 'ffn' | 'kvcache' | 'weight' | 'lora';

export interface FFNNeuronRecord {
  neuronIndex: number;
  layerIndex: number;
  meanActivation: number;
  maxActivation: number;
  variance: number;
  selectivity: number;
  isDead: boolean;
  topTokens: { token: string; activation: number; position: number }[];
}

export interface SAEFeatureRecord {
  featureIndex: number;
  firingRate: number;
  maxActivation: number;
  meanActivation: number;
  label: string;
}

export interface KVCacheMemoryPoint {
  seqLen: number;
  denseMB: number;
  deltaKVMB: number;
  h2oMB: number;
  streamingLLMMB: number;
  snapKVMB: number;
}

export interface DeltaKVSinkRecord {
  position: number;
  token: string;
  sinkScore: number;
  isSink: boolean;
  temporalDelta: number;
}

export interface SingularValuePoint {
  rank: number;
  singularValue: number;
  logValue: number;
  cumulativeEnergy: number;
  isKept: boolean;
}

export interface MatrixDecompositionConfig {
  matrixType: 'W_Q' | 'W_K' | 'W_V' | 'W_O' | 'W_gate' | 'W_up' | 'W_down';
  layerIndex: number;
  targetRank: number;
  energyThreshold: number;
  method: 'svd' | 'cur' | 'rpca';
}

export interface LoRAStudioState {
  rank: number;
  alpha: number;
  dIn: number;
  dOut: number;
  initMethod: 'standard' | 'svd';
  learningRate: number;
}

// =====================================================================
// Optimization & Full Pipeline Studios Types
// =====================================================================

export type OptimizationStudioTab =
  | 'circuits'
  | 'logits'
  | 'loss_landscape'
  | 'compression'
  | 'pipeline';

export interface CircuitGraphNode {
  id: string;
  type: 'input' | 'attention_head' | 'output';
  label: string;
  layer: number;
  head?: number;
  activation: number;
  headType?: 'induction' | 'previous_token' | 'positional_bos' | 'copy_amplification' | 'semantic_diffuse';
  scores?: Record<string, number>;
}

export interface CircuitGraphEdge {
  id: string;
  source: string;
  target: string;
  weight: number;
  type: 'qk_routing' | 'ov_writing';
  queryPos?: number;
  keyPos?: number;
}

export interface CircuitInterventionState {
  layerIdx: number;
  headIdx: number;
  ablationType: 'zero' | 'scale';
  scale: number;
  klDivergence: number;
  logitDiff: number;
  topTokenFlipped: boolean;
}

export interface TokenProbabilityItem {
  tokenId: number;
  token: string;
  rawLogit: number;
  scaledLogit: number;
  probability: number;
  cumulativeProb: number;
}

export interface SamplingStrategyConfig {
  method: 'greedy' | 'temperature' | 'top_k' | 'top_p' | 'beam';
  temperature: number;
  topK: number;
  topP: number;
  repetitionPenalty: number;
  beamWidth: number;
}

export interface TunedLensLayerPrediction {
  layer: number;
  entropy: number;
  klToFinal: number;
  top1Token: string;
  top1Prob: number;
  matchesFinal: boolean;
}

export interface ReliabilityCalibrationBin {
  binIdx: number;
  confidenceRange: [number, number];
  avgConfidence: number;
  accuracy: number;
  count: number;
  gap: number;
}

export interface LossLandscapeGrid {
  alphas: number[];
  betas: number[];
  lossGrid: number[][];
  centerLoss: number;
  maxLoss: number;
  minLoss: number;
  sharpnessSam: number;
  resolution: number;
}

export interface HessianCurvatureData {
  topEigenvalue: number;
  trace: number;
  conditionNumber: number;
  eigenvalues: number[];
  esdHist: number[];
  esdBins: number[];
  sharpnessType: string;
}

export interface ModeConnectivityPath {
  alphas: number[];
  lossProfile: number[];
  curveType: 'linear' | 'bezier';
  barrierHeight: number;
  isModeConnected: boolean;
}

export interface CompressorEngineSpec {
  id: string;
  name: string;
  category: 'Low-Rank SVD' | 'Pruning' | 'Quantization' | 'Tensor Decompositions' | 'KV-Cache' | 'Linear Attention';
  latex: string;
  typicalRatio: string;
  speedup: string;
}

export interface BenchmarkParetoModel {
  compressorId: string;
  name: string;
  category: string;
  compressionRatio: number;
  paramReductionPct: number;
  accuracyRetention: number;
  speedup: number;
  latencyMs: number;
  vramMb: number;
  compressedParams: number;
  isPareto?: boolean;
  latex?: string;
}

export interface LayerBudgetAllocation {
  layer: number;
  keptRatio: number;
  compressionRatio: number;
  sizeMb: number;
  importance: number;
}

export interface PipelineStageProfile {
  stage: string;
  latencyMs: number;
  memoryKb: number;
  color: string;
}

export interface ForwardTraceBlock {
  layerIndex: number;
  hiddenInNorm: number;
  attnOutputNorm: number;
  ffnOutputNorm: number;
  hiddenOutNorm: number;
  residualRatio: number;
}

export interface ChallengeGameState {
  active: boolean;
  reconstructionScore: number;
  layerScores: number[];
  rankBadge: 'Grandmaster' | 'Master' | 'Apprentice' | 'Unranked';
  completed: boolean;
}

