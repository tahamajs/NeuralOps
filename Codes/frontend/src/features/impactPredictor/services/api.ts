import { request as api } from "../../../api/client";
import {
  ImpactMetrics,
  ImpactScore,
  ImpactHistory,
  ProjectComparison,
  PredictionRequest,
  FeatureImportance,
} from "../types";

export const impactAPI = {
  // Get impact score for a project
  getImpactScore: (projectId: string) =>
    api.get<ImpactScore>(`/impact/projects/${projectId}/score/`),

  // Get metrics for a project
  getMetrics: (projectId: string) =>
    api.get<ImpactMetrics>(`/impact/projects/${projectId}/metrics/`),

  // Get historical impact data
  getHistory: (projectId: string, months: number = 12) =>
    api.get<ImpactHistory[]>(`/impact/projects/${projectId}/history/`, {
      params: { months },
    }),

  // Get feature importance for impact prediction
  getFeatureImportance: () =>
    api.get<FeatureImportance[]>("/impact/feature-importance/"),

  // Predict impact for a new project (before creation)
  predictImpact: (data: PredictionRequest) =>
    api.post<ImpactScore>("/impact/predict/", data),

  // Get projects comparison (leaderboard)
  getProjectsComparison: (params?: {
    sort_by?: string;
    order?: "asc" | "desc";
    limit?: number;
    offset?: number;
    status?: string;
  }) =>
    api.get<{ results: ProjectComparison[]; count: number }>(
      "/impact/projects/",
      { params },
    ),

  // Get a single project's comparison data
  getProjectComparison: (projectId: string) =>
    api.get<ProjectComparison>(`/impact/projects/${projectId}/comparison/`),
};
