export type RecommendationType =
  "hyperparameter" | "architecture" | "task" | "data_augmentation";
export type RecommendationStatus = "pending" | "accepted" | "dismissed";

export interface Recommendation {
  id: string;
  type: RecommendationType;
  title: string;
  description: string;
  expected_improvement: number; // percentage
  confidence: number;
  parameters?: Record<string, any>;
  created_at: string;
  status: RecommendationStatus;
}

export interface CreativeVariant {
  id: string;
  name: string;
  description: string;
  config_diff: Record<string, any>;
  expected_performance_lift: number;
  tags: string[];
}

export interface OpportunityIndex {
  total_score: number;
  missing_attributes: string[];
  high_impact_suggestions: string[];
  rank: number;
}
