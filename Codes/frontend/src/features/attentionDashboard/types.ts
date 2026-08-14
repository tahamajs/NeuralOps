// src/features/attentionDashboard/types.ts

export type HeadType =
  | 'Positional (Self)'
  | 'Positional (Local)'
  | 'Delimiter / Sink'
  | 'Semantic (Focused)'
  | 'Sparse / Selective'
  | 'Induction'
  | 'Redundant / Uniform'
  | 'Mixed / Diffuse';

export interface HeadMetricPoint {
  layer: number;
  head: number;
  importance: number;
  stability: number;
  similarity: number;
  head_type: HeadType;
  entropy: number;
  sparsity: number;
  color: string;
  size: number;
  description: string;
}

export interface Head3DMetricsResult {
  points: HeadMetricPoint[];
  layers: number;
  heads_per_layer: number;
  avg_importance: number;
  avg_stability: number;
  avg_similarity: number;
  type_distribution: Record<string, number>;
}

export interface AffectedToken {
  position: number;
  token: string;
  original_top1: number;
  ablated_top1: number;
  original_prob: number;
  ablated_prob: number;
  prob_delta: number;
  flipped: boolean;
}

export interface AblationResult {
  ablated_heads: { layer: number; head: number }[];
  original_loss: number;
  ablated_loss: number;
  loss_delta: number;
  original_ppl: number;
  ablated_ppl: number;
  ppl_delta: number;
  kl_divergence: number;
  top_k_flips: number;
  affected_tokens: AffectedToken[];
  compound_redundancy_score: number;
}

export interface PruningRecipe {
  model_id: string;
  ablated_heads: { layer: number; head: number }[];
  pruning_ratio: number;
  expected_ppl_delta: number;
  expected_kl_divergence: number;
  recommended: boolean;
  rationale: string;
}

export interface CircuitSVDResult {
  layer: number;
  head: number;
  circuit_type: 'QK' | 'OV';
  singular_values: number[];
  cumulative_energy: number[];
  effective_rank: number;
  condition_number: number;
  eigenvalues_real: number[];
  eigenvalues_imag: number[];
  spectral_radius: number;
  trace: number;
}

export interface AttributionNode {
  id: string;
  label: string;
  layer: number;
  head: number;
  head_type: string;
  activation: number;
  color: string;
  size: number;
}

export interface AttributionEdge {
  source: string;
  target: string;
  weight: number;
  type: 'qk' | 'ov';
}

export interface AttributionGraph {
  nodes: AttributionNode[];
  edges: AttributionEdge[];
}

export interface TokenTrajectory {
  token: string;
  token_index: number;
  coordinates: [number, number][];
  velocities: number[];
  total_path_length: number;
  curvature: number;
}

export interface InfluenceArc {
  source_token: string;
  source_index: number;
  target_token: string;
  target_index: number;
  layer: number;
  attention_weight: number;
  influence_score: number;
  head_idx?: number;
}

export interface SemanticPathwayResult {
  tokens: string[];
  trajectories: TokenTrajectory[];
  influence_arcs: InfluenceArc[];
  explained_variance_ratio: number[];
}

export interface LayerPrediction {
  layer: number;
  top_k_tokens: string[];
  top_k_probs: number[];
  entropy: number;
  top_1_token: string;
  top_1_prob: number;
}

export interface LogitLensResult {
  token_position: number;
  target_token: string | null;
  layer_predictions: LayerPrediction[];
  convergence_layer: number;
  final_prediction: string;
  final_confidence: number;
}

// ---------------------------------------------------------------------------
// Zone 7: Social Types
// ---------------------------------------------------------------------------

export type ReactionType = 'interesting' | 'efficient' | 'specialized' | 'redundant' | 'novel';

export interface ReactionCount {
  interesting: number;
  efficient: number;
  specialized: number;
  redundant: number;
  novel: number;
}

export interface ArenaMatch {
  id: string;
  patternA: {
    name: string;
    model: string;
    attentionType: string;
    score: number;
    layer: number;
    head: number;
  };
  patternB: {
    name: string;
    model: string;
    attentionType: string;
    score: number;
    layer: number;
    head: number;
  };
  revealed: boolean;
  eloA: number;
  eloB: number;
  criterion: string;
}

export interface ForkNode {
  id: string;
  name: string;
  author: string;
  date: string;
  attentionType: string;
  reactions: number;
  parentId: string | null;
}

export interface DiscussionComment {
  id: string;
  author: string;
  authorAvatar: string;
  timestamp: string;
  anchor: {
    type: 'head' | 'layer' | 'token_pair' | 'pattern';
    label: string;
  };
  content: string;
  latex?: string;
  likes: number;
  replies: DiscussionComment[];
}

// ---------------------------------------------------------------------------
// Zone 8: Gamification Types
// ---------------------------------------------------------------------------

export interface ProgressionTier {
  level: number;
  name: string;
  requiredXP: number;
  badge: string;
  color: string;
}

export interface UserGamificationState {
  currentXP: number;
  currentLevel: number;
  tierName: string;
  nextLevelXP: number;
  streakCount: number;
  longestStreak: number;
  lastActivityDate: string;
  weekStreakHistory: boolean[];
}

export interface DailyPuzzle {
  id: string;
  date: string;
  title: string;
  type: string;
  difficulty: 'Easy' | 'Medium' | 'Hard';
  description: string;
  options: string[];
  correctOptionIndex: number;
  explanation: string;
  xpReward: number;
  solved: boolean;
  userAnswer: number | null;
}

export interface BadgeItem {
  id: string;
  name: string;
  category: 'discovery' | 'community' | 'engagement' | 'expertise';
  tier: 'bronze' | 'silver' | 'gold' | 'platinum';
  icon: string;
  description: string;
  unlocked: boolean;
  unlockedAt?: string;
  progressPercent: number;
}

export interface LeaderboardEntry {
  rank: number;
  username: string;
  tier: string;
  xp: number;
  patternsDiscovered: number;
  arenaElo: number;
  change: number; // +3, -1, 0
  isCurrentUser?: boolean;
}

export interface BountyItem {
  id: string;
  title: string;
  description: string;
  rewardAmount: number;
  rewardCurrency: string;
  skills: string[];
  daysLeft: number;
  status: 'open' | 'in_progress' | 'completed';
  claimedBy?: string;
}

// ---------------------------------------------------------------------------
// Zone 9: Integration Types
// ---------------------------------------------------------------------------

export interface UniversalAttentionScore {
  interpretability: number;
  specialization: number;
  efficiency: number;
  stability: number;
  compositeScore: number;
  percentile: number;
}

export interface GeneratedPaperSection {
  title: string;
  methods: string;
  results: string;
  discussion: string;
  latexCode: string;
  bibtex: string;
}
