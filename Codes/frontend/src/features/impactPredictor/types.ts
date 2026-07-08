export interface ImpactMetrics {
  h_index: number;
  i10_index: number;
  total_citations: number;
  avg_citations_per_paper: number;
  field_weighted_citation_impact: number;
  output_count: number;
  collaborations_count: number;
  international_collaborations: number;
}

export interface ImpactScore {
  overall_score: number;
  percentile_rank: number;
  confidence_interval: [number, number];
  breakdown: {
    novelty: number;
    technical_feasibility: number;
    community_interest: number;
    reproducibility: number;
    potential_citations: number;
  };
  predicted_h_index: number;
  predicted_citations_5y: number;
  risk_factors: string[];
  opportunities: string[];
}

export interface ImpactHistory {
  date: string;
  score: number;
  h_index: number;
  citations: number;
}

export interface ProjectComparison {
  id: string;
  name: string;
  owner: string;
  impact_score: number;
  metrics: ImpactMetrics;
  status: "active" | "completed" | "archived";
  last_updated: string;
  trend: "up" | "down" | "stable";
  change: number;
}

export interface PredictionRequest {
  project_id: string;
  title: string;
  abstract: string;
  keywords: string[];
  expected_venue?: string;
  estimated_pages?: number;
  collaboration_type?: "solo" | "small_team" | "large_team";
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  description: string;
}
