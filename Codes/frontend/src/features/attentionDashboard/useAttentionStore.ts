// src/features/attentionDashboard/useAttentionStore.ts
import { create } from 'zustand';
import {
  AblationResult,
  ArenaMatch,
  AttributionGraph,
  BadgeItem,
  BountyItem,
  CircuitSVDResult,
  DailyPuzzle,
  DiscussionComment,
  ForkNode,
  GeneratedPaperSection,
  Head3DMetricsResult,
  HeadMetricPoint,
  HeadType,
  LogitLensResult,
  PruningRecipe,
  ReactionCount,
  ReactionType,
  SemanticPathwayResult,
  UniversalAttentionScore,
  UserGamificationState,
} from './types';

interface AttentionState {
  // Model & Input Selection
  selectedModel: string;
  selectedAttentionType: string;
  inputText: string;
  selectedLayer: number;
  selectedHead: number;
  activeZoneTab: 'metrics' | 'ablation' | 'circuits' | 'pathways' | 'logitlens' | 'social' | 'gamification' | 'integrations';

  // Zone 5: Analytics
  headMetrics: Head3DMetricsResult | null;
  semanticPathways: SemanticPathwayResult | null;
  logitLens: LogitLensResult | null;
  selectedHeadPoint: HeadMetricPoint | null;
  metricAxisX: 'importance' | 'stability' | 'similarity';
  metricAxisY: 'importance' | 'stability' | 'similarity';
  metricAxisZ: 'importance' | 'stability' | 'similarity';
  activeHeadTypeFilter: string | 'ALL';
  currentScrubberLayer: number;
  isLogitPlaying: boolean;

  // Zone 6: Interpretability & Ablation
  ablatedHeads: Set<string>; // "L{layer}H{head}"
  ablationResult: AblationResult | null;
  pruningRecipe: PruningRecipe | null;
  circuitSVDs: CircuitSVDResult[];
  attributionGraph: AttributionGraph | null;
  selectedCircuitHead: { layer: number; head: number };
  circuitEdgeThreshold: number;

  // Zone 7: Social & Community
  reactions: ReactionCount;
  userReactions: Set<ReactionType>;
  arenaMatch: ArenaMatch | null;
  arenaVotesCount: number;
  forkLineage: ForkNode[];
  discussions: DiscussionComment[];

  // Zone 8: Gamification
  gamification: UserGamificationState;
  dailyPuzzle: DailyPuzzle;
  badges: BadgeItem[];
  bounties: BountyItem[];
  celebrationBadge: BadgeItem | null;

  // Zone 9: Integrations
  universalScore: UniversalAttentionScore;
  generatedPaper: GeneratedPaperSection | null;
  isSonificationPlaying: boolean;
  particleFlowSpeed: number;

  // Global Actions
  setSelectedModel: (model: string) => void;
  setSelectedAttentionType: (type: string) => void;
  setInputText: (text: string) => void;
  setSelectedLayer: (layer: number) => void;
  setSelectedHead: (head: number) => void;
  setActiveZoneTab: (tab: AttentionState['activeZoneTab']) => void;
  setSelectedHeadPoint: (point: HeadMetricPoint | null) => void;
  setMetricAxis: (axis: 'X' | 'Y' | 'Z', metric: 'importance' | 'stability' | 'similarity') => void;
  setActiveHeadTypeFilter: (filter: string) => void;
  setCurrentScrubberLayer: (layer: number) => void;
  setIsLogitPlaying: (playing: boolean) => void;

  // Ablation Actions
  toggleHeadAblation: (layer: number, head: number) => void;
  ablateLayer: (layer: number) => void;
  ablateRedundantHeads: () => void;
  resetAblations: () => void;
  setCircuitEdgeThreshold: (threshold: number) => void;
  setSelectedCircuitHead: (lh: { layer: number; head: number }) => void;

  // Social Actions
  toggleReaction: (type: ReactionType) => void;
  voteArena: (winner: 'A' | 'B') => void;
  revealArena: () => void;
  nextArenaMatch: () => void;
  addDiscussionComment: (content: string, latex?: string, anchorType?: string, anchorLabel?: string) => void;

  // Gamification Actions
  submitPuzzleAnswer: (optionIndex: number) => boolean;
  addXP: (amount: number, reason?: string) => void;
  dismissCelebration: () => void;
  claimBounty: (bountyId: string) => void;

  // Integration Actions
  generatePaperDraft: () => void;
  toggleSonification: () => void;
  setParticleFlowSpeed: (speed: number) => void;
  recomputeAnalytics: () => void;
}

// ---------------------------------------------------------------------------
// Mock Data Generators for Real-Time Responsive Exploration
// ---------------------------------------------------------------------------

const HEAD_TYPES: HeadType[] = [
  'Positional (Self)',
  'Positional (Local)',
  'Delimiter / Sink',
  'Semantic (Focused)',
  'Sparse / Selective',
  'Induction',
  'Redundant / Uniform',
  'Mixed / Diffuse',
];

const HEAD_COLORS: Record<HeadType, string> = {
  'Positional (Self)': '#3b82f6',
  'Positional (Local)': '#06b6d4',
  'Delimiter / Sink': '#f59e0b',
  'Semantic (Focused)': '#10b981',
  'Sparse / Selective': '#6366f1',
  'Induction': '#a855f7',
  'Redundant / Uniform': '#94a3b8',
  'Mixed / Diffuse': '#ec4899',
};

const DEFAULT_TOKENS = ['The', 'quantum', 'algorithm', 'accelerates', 'attention', 'computation', 'significantly', '.'];











const INITIAL_METRICS = null as any;
const INITIAL_CIRCUITS: any[] = [];
const INITIAL_GRAPH = null as any;
const INITIAL_PATHWAYS = null as any;
const INITIAL_LOGITLENS = null as any;

export const useAttentionStore = create<AttentionState>((set, get) => ({
  selectedModel: 'bert-base-uncased',
  selectedAttentionType: 'Galerkin Attention (Rank 64)',
  inputText: 'The quantum algorithm accelerates attention computation significantly.',
  selectedLayer: 5,
  selectedHead: 7,
  activeZoneTab: 'metrics',

  headMetrics: INITIAL_METRICS,
  semanticPathways: INITIAL_PATHWAYS,
  logitLens: INITIAL_LOGITLENS,
  selectedHeadPoint: null,
  metricAxisX: 'importance',
  metricAxisY: 'stability',
  metricAxisZ: 'similarity',
  activeHeadTypeFilter: 'ALL',
  currentScrubberLayer: 5,
  isLogitPlaying: false,

  ablatedHeads: new Set<string>(['L2H3', 'L7H11']),
  ablationResult: {
    ablated_heads: [
      { layer: 2, head: 3 },
      { layer: 7, head: 11 },
    ],
    original_loss: 1.84,
    ablated_loss: 1.89,
    loss_delta: 0.05,
    original_ppl: 6.3,
    ablated_ppl: 6.62,
    ppl_delta: 0.32,
    kl_divergence: 0.042,
    top_k_flips: 1,
    affected_tokens: [
      {
        position: 4,
        token: 'attention',
        original_top1: 1420,
        ablated_top1: 1420,
        original_prob: 0.88,
        ablated_prob: 0.82,
        prob_delta: -0.06,
        flipped: false,
      },
      {
        position: 6,
        token: 'significantly',
        original_top1: 3812,
        ablated_top1: 5904,
        original_prob: 0.45,
        ablated_prob: 0.39,
        prob_delta: -0.06,
        flipped: true,
      },
    ],
    compound_redundancy_score: 0.86,
  },
  pruningRecipe: {
    model_id: 'bert-base-uncased',
    ablated_heads: [
      { layer: 0, head: 11 },
      { layer: 1, head: 9 },
      { layer: 2, head: 3 },
      { layer: 4, head: 10 },
      { layer: 7, head: 11 },
      { layer: 11, head: 8 },
    ],
    pruning_ratio: 0.042,
    expected_ppl_delta: 0.08,
    expected_kl_divergence: 0.015,
    recommended: true,
    rationale: 'Pruning 6 redundant and diffuse heads preserves 99.2% output fidelity while reducing FLOPs by 4.2%.',
  },
  circuitSVDs: INITIAL_CIRCUITS,
  attributionGraph: INITIAL_GRAPH,
  selectedCircuitHead: { layer: 5, head: 7 },
  circuitEdgeThreshold: 0.25,

  reactions: {
    interesting: 42,
    efficient: 68,
    specialized: 35,
    redundant: 7,
    novel: 19,
  },
  userReactions: new Set<ReactionType>(['interesting', 'efficient']),
  arenaMatch: {
    id: 'match_9012',
    patternA: {
      name: 'Pattern A',
      model: 'BERT-Base (Galerkin SVD)',
      attentionType: 'Galerkin',
      score: 0.89,
      layer: 5,
      head: 7,
    },
    patternB: {
      name: 'Pattern B',
      model: 'RoBERTa-Base (Standard MHA)',
      attentionType: 'Standard Softmax',
      score: 0.84,
      layer: 8,
      head: 2,
    },
    revealed: false,
    eloA: 1642,
    eloB: 1598,
    criterion: 'Which attention pattern demonstrates cleaner semantic specialization?',
  },
  arenaVotesCount: 14,
  forkLineage: [
    { id: 'f1', name: 'Original BERT-Base Galerkin Exploration', author: 'Dr. Sarah Vance', date: '2026-04-12', attentionType: 'Galerkin', reactions: 128, parentId: null },
    { id: 'f2', name: 'Low-Rank Induction Head Probe', author: 'Alex Thorne', date: '2026-05-02', attentionType: 'Galerkin Rank 32', reactions: 64, parentId: 'f1' },
    { id: 'f3', name: 'Cross-Layer Trajectory Fork', author: 'You (Current)', date: '2026-05-26', attentionType: 'Galerkin Rank 64', reactions: 19, parentId: 'f2' },
  ],
  discussions: [
    {
      id: 'd1',
      author: 'Prof. Elena Rostova',
      authorAvatar: '👩‍🔬',
      timestamp: '2 hours ago',
      anchor: { type: 'head', label: 'Layer 5, Head 7' },
      content: 'Notice how this head maintains an effective rank of $r_{eff} = 6.42$, indicating strong low-rank compression while preserving the QK induction subspace:',
      latex: 'W_{QK} = W_Q W_K^T \\implies \\operatorname{rank}(W_{QK}) \\le 64',
      likes: 12,
      replies: [
        {
          id: 'd1-r1',
          author: 'Marc Dupont',
          authorAvatar: '👨‍💻',
          timestamp: '45 mins ago',
          anchor: { type: 'head', label: 'Layer 5, Head 7' },
          content: 'Agreed! When ablated, the KL divergence is only 0.042. Very safe candidate for selective rank reduction.',
          likes: 5,
          replies: [],
        },
      ],
    },
  ],

  gamification: {
    currentXP: 2845,
    currentLevel: 5,
    tierName: 'Head Hunter',
    nextLevelXP: 4860,
    streakCount: 12,
    longestStreak: 24,
    lastActivityDate: '2026-08-14',
    weekStreakHistory: [true, true, true, true, true, true, true],
  },
  dailyPuzzle: {
    id: 'puz_today',
    date: '2026-08-14',
    title: 'Guess the Head Specialization',
    type: 'Classification',
    difficulty: 'Medium',
    description: 'An attention matrix displays strong diagonal values (0.42) with significant off-diagonal copying on token transitions. What is the primary specialization of this head?',
    options: ['Induction Head', 'Positional Self-Attention', 'Delimiter Sink', 'Uniform Redundant'],
    correctOptionIndex: 0,
    explanation: 'Correct! The transition offset attention combined with diagonal tracking identifies an Induction Head ([A][B] ... [A] -> [B]).',
    xpReward: 35,
    solved: false,
    userAnswer: null,
  },
  badges: [
    { id: 'b1', name: 'Head Hunter (Gold)', category: 'discovery', tier: 'gold', icon: '🔬', description: 'Identified 50 distinct attention head types across architectures', unlocked: true, unlockedAt: '2026-07-20', progressPercent: 100 },
    { id: 'b2', name: 'Circuit Master', category: 'expertise', tier: 'silver', icon: '⚡', description: 'Published 25 QK/OV circuit SVD decomposition graphs', unlocked: true, unlockedAt: '2026-08-01', progressPercent: 100 },
    { id: 'b3', name: 'Ablation Virtuoso', category: 'expertise', tier: 'gold', icon: '✂️', description: 'Exported 10 validated pruning recipes to model compression', unlocked: false, progressPercent: 65 },
    { id: 'b4', name: 'Streak Master (Silver)', category: 'engagement', tier: 'silver', icon: '🔥', description: 'Maintained a 14-day daily puzzle exploration streak', unlocked: false, progressPercent: 85 },
  ],
  bounties: [
    { id: 'bnty_1', title: 'Trace Induction Circuits in Llama-3-8B', description: 'Deconstruct QK and OV eigenvalue spectra for layers 12-16 to identify induction heads under long-context prompts.', rewardAmount: 150, rewardCurrency: 'USDC', skills: ['circuit_tracer', 'llama3', 'induction_heads'], daysLeft: 14, status: 'open' },
    { id: 'bnty_2', title: 'Ablation Sensitivity on Fine-Tuned MRPC', description: 'Provide systematic pruning recipe identifying >= 15% redundant heads on GLUE benchmark without drop in F1.', rewardAmount: 85, rewardCurrency: 'USD', skills: ['ablation', 'compression', 'glue'], daysLeft: 6, status: 'open' },
  ],
  celebrationBadge: null,

  universalScore: {
    interpretability: 88,
    specialization: 82,
    efficiency: 91,
    stability: 85,
    compositeScore: 86.5,
    percentile: 94,
  },
  generatedPaper: null,
  isSonificationPlaying: false,
  particleFlowSpeed: 1.0,

  setSelectedModel: (model) => {
    set({ selectedModel: model });
    get().recomputeAnalytics();
  },

  setSelectedAttentionType: (type) => {
    set({ selectedAttentionType: type });
    get().recomputeAnalytics();
  },

  setInputText: (text) => {
    set({ inputText: text });
    get().recomputeAnalytics();
  },

  setSelectedLayer: (layer) => {
    const metrics = get().headMetrics;
    const pt = metrics?.points.find((p) => p.layer === layer && p.head === get().selectedHead) || null;
    set({
      selectedLayer: layer,
      selectedHeadPoint: pt,
          });
    get().addXP(1, 'layer_switch');
  },

  setSelectedHead: (head) => {
    const metrics = get().headMetrics;
    const pt = metrics?.points.find((p) => p.layer === get().selectedLayer && p.head === head) || null;
    set({
      selectedHead: head,
      selectedHeadPoint: pt,
          });
    get().addXP(1, 'head_switch');
  },

  setActiveZoneTab: (tab) => set({ activeZoneTab: tab }),

  setSelectedHeadPoint: (point) => {
    if (point) {
      set({
        selectedHeadPoint: point,
        selectedLayer: point.layer,
        selectedHead: point.head,
              });
    }
  },

  setMetricAxis: (axis, metric) => {
    if (axis === 'X') set({ metricAxisX: metric });
    else if (axis === 'Y') set({ metricAxisY: metric });
    else set({ metricAxisZ: metric });
  },

  setActiveHeadTypeFilter: (filter) => set({ activeHeadTypeFilter: filter }),

  setCurrentScrubberLayer: (layer) => set({ currentScrubberLayer: layer }),

  setIsLogitPlaying: (playing) => set({ isLogitPlaying: playing }),

  toggleHeadAblation: (layer, head) => {
    const key = `L${layer}H${head}`;
    const newAblated = new Set(get().ablatedHeads);
    if (newAblated.has(key)) {
      newAblated.delete(key);
    } else {
      newAblated.add(key);
    }

    const headsList = Array.from(newAblated).map((k) => {
      const match = k.match(/L(\d+)H(\d+)/);
      return match ? { layer: parseInt(match[1], 10), head: parseInt(match[2], 10) } : { layer: 0, head: 0 };
    });

    const count = newAblated.size;
    const lossDelta = Number((count * 0.024 + (Math.sin(count) * 0.01)).toFixed(3));
    const pplDelta = Number((count * 0.15 + (Math.cos(count) * 0.05)).toFixed(2));
    const kl = Number((count * 0.018).toFixed(4));
    const flips = Math.floor(count * 0.6);

    set({
      ablatedHeads: newAblated,
      ablationResult: {
        ablated_heads: headsList,
        original_loss: 1.84,
        ablated_loss: Number((1.84 + lossDelta).toFixed(3)),
        loss_delta: lossDelta,
        original_ppl: 6.3,
        ablated_ppl: Number((6.3 + pplDelta).toFixed(2)),
        ppl_delta: pplDelta,
        kl_divergence: kl,
        top_k_flips: flips,
        affected_tokens: get().ablationResult?.affected_tokens || [],
        compound_redundancy_score: Number(Math.max(0.2, 1 - count * 0.05).toFixed(2)),
      },
    });

    get().addXP(5, 'head_ablation_toggle');
  },

  ablateLayer: (layer) => {
    const newAblated = new Set(get().ablatedHeads);
    for (let h = 0; h < (get().headMetrics?.heads_per_layer || 12); h++) {
      newAblated.add(`L${layer}H${h}`);
    }
    set({ ablatedHeads: newAblated });
    get().addXP(15, 'layer_ablation');
  },

  ablateRedundantHeads: () => {
    const metrics = get().headMetrics;
    if (!metrics) return;
    const newAblated = new Set(get().ablatedHeads);
    metrics.points.forEach((p) => {
      if (p.head_type === 'Redundant / Uniform' || p.similarity > 0.75) {
        newAblated.add(`L${p.layer}H${p.head}`);
      }
    });
    set({ ablatedHeads: newAblated });
    get().addXP(25, 'ablate_redundant');
  },

  resetAblations: () => {
    set({ ablatedHeads: new Set() });
  },

  setCircuitEdgeThreshold: (threshold) => {
    set({ circuitEdgeThreshold: threshold });
    const metrics = get().headMetrics;
    if (metrics) {
      get().recomputeAnalytics();
    }
  },

  setSelectedCircuitHead: (lh) => {
    set({
      selectedCircuitHead: lh,
          });
  },

  toggleReaction: (type) => {
    const userReacts = new Set(get().userReactions);
    const reactions = { ...get().reactions };

    if (userReacts.has(type)) {
      userReacts.delete(type);
      reactions[type] = Math.max(0, reactions[type] - 1);
    } else {
      userReacts.add(type);
      reactions[type] = reactions[type] + 1;
      get().addXP(2, 'reaction_given');
    }

    set({ userReactions: userReacts, reactions });
  },

  voteArena: (winner) => {
    const match = get().arenaMatch;
    if (!match) return;

    const k = 32;
    const expectedA = 1 / (1 + Math.pow(10, (match.eloB - match.eloA) / 400));
    const expectedB = 1 - expectedA;
    const scoreA = winner === 'A' ? 1 : 0;
    const scoreB = winner === 'B' ? 1 : 0;

    const newEloA = Math.round(match.eloA + k * (scoreA - expectedA));
    const newEloB = Math.round(match.eloB + k * (scoreB - expectedB));

    set({
      arenaMatch: {
        ...match,
        revealed: true,
        eloA: newEloA,
        eloB: newEloB,
      },
      arenaVotesCount: get().arenaVotesCount + 1,
    });

    get().addXP(10, 'arena_vote');
  },

  revealArena: () => {
    if (get().arenaMatch) {
      set({ arenaMatch: { ...get().arenaMatch!, revealed: true } });
    }
  },

  nextArenaMatch: () => {
    const matchNum = get().arenaVotesCount + 1;
    set({
      arenaMatch: {
        id: `match_${matchNum}`,
        patternA: {
          name: 'Pattern A',
          model: 'TLinFormer (Linearized)',
          attentionType: 'Linear Spectral',
          score: 0.87,
          layer: 4,
          head: 3,
        },
        patternB: {
          name: 'Pattern B',
          model: 'GPT-2 Medium',
          attentionType: 'Standard Causal MHA',
          score: 0.83,
          layer: 6,
          head: 1,
        },
        revealed: false,
        eloA: 1610,
        eloB: 1580,
        criterion: 'Which attention distribution demonstrates higher information density?',
      },
    });
  },

  addDiscussionComment: (content, latex, anchorType = 'pattern', anchorLabel = 'General') => {
    const newComment: DiscussionComment = {
      id: `comment_${Date.now()}`,
      author: 'You (Researcher)',
      authorAvatar: '🧑‍🔬',
      timestamp: 'Just now',
      anchor: {
        type: anchorType as any,
        label: anchorLabel,
      },
      content,
      latex,
      likes: 1,
      replies: [],
    };

    set({ discussions: [newComment, ...get().discussions] });
    get().addXP(15, 'comment_posted');
  },

  submitPuzzleAnswer: (optionIndex) => {
    const puz = get().dailyPuzzle;
    if (puz.solved) return false;

    const isCorrect = optionIndex === puz.correctOptionIndex;
    set({
      dailyPuzzle: {
        ...puz,
        solved: true,
        userAnswer: optionIndex,
      },
    });

    if (isCorrect) {
      get().addXP(puz.xpReward, 'daily_puzzle_solved');
      const g = get().gamification;
      set({
        gamification: {
          ...g,
          streakCount: g.streakCount + 1,
          longestStreak: Math.max(g.longestStreak, g.streakCount + 1),
        },
      });
    }

    return isCorrect;
  },

  addXP: (amount, reason) => {
    const g = get().gamification;
    const newXP = g.currentXP + amount;
    let newLevel = g.currentLevel;
    let tierName = g.tierName;
    let nextLevelXP = g.nextLevelXP;

    if (newXP >= g.nextLevelXP) {
      newLevel += 1;
      nextLevelXP = Math.round(nextLevelXP * 1.6);
      if (newLevel >= 15) tierName = 'Grandmaster';
      else if (newLevel >= 10) tierName = 'Attention Sage';
      else if (newLevel >= 5) tierName = 'Head Hunter';
      else if (newLevel >= 2) tierName = 'Pattern Scout';
    }

    set({
      gamification: {
        ...g,
        currentXP: newXP,
        currentLevel: newLevel,
        tierName,
        nextLevelXP,
      },
    });
  },

  dismissCelebration: () => set({ celebrationBadge: null }),

  claimBounty: (bountyId) => {
    const bounties = get().bounties.map((b) =>
      b.id === bountyId ? { ...b, status: 'in_progress' as const, claimedBy: 'You (Current User)' } : b
    );
    set({ bounties });
    get().addXP(20, 'bounty_claimed');
  },

  generatePaperDraft: () => {
    const m = get().selectedModel;
    const attn = get().selectedAttentionType;
    const l = get().selectedLayer;
    const h = get().selectedHead;
    const pt = get().selectedHeadPoint;

    const draft: GeneratedPaperSection = {
      title: `Mechanistic Interpretability & Low-Rank Attention Dynamics in ${m}`,
      methods: `We examine the attention topology and spectral characteristics of ${m} employing ${attn}. Intermediate activations at Layer ${l}, Head ${h} are evaluated using singular value decomposition on QK/OV circuit matrices alongside empirical head masking sensitivity.`,
      results: `Analysis revealed Head L${l}H${h} functions primarily as a ${pt?.head_type || 'Positional'} operator (Importance: ${pt?.importance || 0.85}, Stability: ${pt?.stability || 0.91}). Ablation induces a KL divergence of $\\Delta D_{KL} = ${get().ablationResult?.kl_divergence || 0.042}$, verifying minimal degradation in target vocabulary perplexity.`,
      discussion: `The empirical divergence confirms that redundant heads in early-to-mid layers exhibit high Jensen-Shannon similarity with neighboring projection spaces, enabling aggressive compression without destabilizing downstream syntax tracking.`,
      latexCode: `\\section{Attention Diagnostics & Empirical Verification}\n\\label{sec:attention_viz}\n\nWe evaluated \\textbf{${m}} with \\emph{${attn}} at Layer ${l}, Head ${h}.\n\\begin{equation}\n  I(l,h) = \\| L(x) - L_{\\neg(l,h)}(x) \\|_2 \\approx ${pt?.importance || 0.85}\n\\end{equation}\nHead classification identified functional category \\texttt{${pt?.head_type || 'Positional'}} with singular value effective rank $r_{eff} = 6.42$.`,
      bibtex: `@article{neuralops2026attention,\n  title={Quantitative Attention Analytics and Low-Rank SVD Decomposition},\n  author={NeuralOps Research Team},\n  journal={Journal of Machine Learning Interpretability},\n  year={2026}\n}`,
    };

    set({ generatedPaper: draft });
    get().addXP(40, 'paper_section_generated');
  },

  toggleSonification: () => {
    set({ isSonificationPlaying: !get().isSonificationPlaying });
  },

  setParticleFlowSpeed: (speed) => set({ particleFlowSpeed: speed }),

  recomputeAnalytics: async () => {
    const { selectedModel, selectedLayer, selectedHead, inputText } = get();
    try {
        const res = await fetch(`/api/attention-visualization/interpretability/?model=${encodeURIComponent(selectedModel)}&layer=${selectedLayer}&head=${selectedHead}&text=${encodeURIComponent(inputText)}`);
        if (res.ok) {
            const data = await res.json();
            // Just typecast the backend data to avoid TS errors, as we assume backend returns correct shape
            set({
                headMetrics: data.heads ? { points: data.heads, layers: 12, heads_per_layer: 12, avg_importance: 0.5, avg_stability: 0.5, avg_similarity: 0.5, type_distribution: {} } as any : null,
                circuitSVDs: data.circuit_spectrum ? data.circuit_spectrum : [],
                attributionGraph: data.attributionGraph as any || null,
                semanticPathways: data.semanticPathways as any || null,
                logitLens: data.logit_lens as any || null,
            });
        }
    } catch (e) {
        console.error(e);
    }
  },
}));
