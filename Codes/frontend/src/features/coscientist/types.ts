// src/features/coscientist/types.ts
export type HypothesisStatus =
  | "generated"
  | "reviewed"
  | "ranked"
  | "evolved"
  | "selected"
  | "proposed";
export type AgentType =
  | "generation"
  | "reflection"
  | "ranking"
  | "evolution"
  | "proximity"
  | "meta_review";
export type TournamentRound = "quarterfinals" | "semifinals" | "finals";

export interface Agent {
  id: string;
  type: AgentType;
  name: string;
  status: "idle" | "working" | "completed" | "error";
  started_at?: string;
  completed_at?: string;
  metrics?: {
    tokens_used: number;
    duration_ms: number;
  };
}

export interface Hypothesis {
  id: string;
  title: string;
  description: string;
  scientific_merit: number;
  novelty_score: number;
  testability_score: number;
  impact_score: number;
  total_score: number;
  elo_rating: number;
  generation_round: number;
  parent_id?: string;
  status: HypothesisStatus;
  created_at: string;
  metrics?: {
    citation_count?: number;
    relevant_papers?: string[];
  };
}

export interface TournamentMatch {
  id: string;
  round: number;
  hypothesis_a_id: string;
  hypothesis_b_id: string;
  winner_id: string | null;
  vote_distribution: { a: number; b: number };
  completed: boolean;
  created_at: string;
}

export interface ResearchSession {
  id: string;
  session_id?: string;
  goal: string;
  status: "running" | "completed" | "failed";
  hypotheses: Hypothesis[];
  matches: TournamentMatch[];
  agents: Agent[];
  created_at: string;
  updated_at: string;
}

export interface CoscientistConfig {
  model_name: string;
  max_iterations: number;
  hypotheses_per_generation: number;
  tournament_size: number;
  evolution_top_k: number;
  proximity_threshold: number;
}
