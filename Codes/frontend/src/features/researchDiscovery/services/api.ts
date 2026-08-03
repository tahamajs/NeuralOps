import { request } from "../../../api/client";
import {
  Paper,
  LiteratureMap,
  SynthesisResult,
  Hypothesis,
  Grant,
} from "../types";

export const researchDiscoveryAPI = {
  // Literature synthesis
  synthesizeLiterature: (query: string, limit?: number) =>
    request.post<SynthesisResult>(
      "/research-discovery/literature/synthesize/",
      {
        query,
        limit,
      },
    ),
  getLiteratureMap: (paperId: string, depth: number = 2) =>
    request.get<LiteratureMap>(
      `/research-discovery/literature/map/${paperId}/`,
      {
        params: { depth },
      },
    ),

  // Hypothesis generation
  generateHypotheses: (goal: string, parameters?: any) =>
    request.post<Hypothesis[]>("/research-discovery/hypotheses/generate/", {
      goal,
      parameters,
    }),
  evolveHypothesis: (hypothesisId: string, feedback?: string) =>
    request.post<Hypothesis>(
      `/research-discovery/hypotheses/${hypothesisId}/evolve/`,
      { feedback },
    ),
  getHypothesisTimeline: (hypothesisId: string) =>
    request.get<Hypothesis[]>(
      `/research-discovery/hypotheses/${hypothesisId}/timeline/`,
    ),

  // Cross‑database search
  crossSearch: (query: string, databases: string[], limit?: number) =>
    request.post<Paper[]>("/research-discovery/databases/search/", {
      query,
      databases,
      limit,
    }),
  getDatabaseStatus: () =>
    request.get<Record<string, boolean>>(
      "/research-discovery/databases/status/",
    ),

  // Grant matching
  matchGrants: (profile: any) =>
    request.post<Grant[]>("/research-discovery/grants/match/", { profile }),
  getGrantDetail: (id: string) =>
    request.get<Grant>(`/research-discovery/grants/${id}/`),
};
