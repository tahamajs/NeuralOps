import api from "../../../api/client";
import {
  FailureAnalysis,
  FixAction,
  ErrorPattern,
  FailureStats,
} from "../types";

export const forensicsAPI = {
  // Failure analyses
  getFailures: (params?: {
    status?: string;
    severity?: string;
    run_id?: string;
  }) =>
    api.get<{ results: FailureAnalysis[]; count: number }>(
      "/forensics/analyses/",
      { params },
    ),
  getFailureDetail: (id: string) =>
    api.get<FailureAnalysis>(`/forensics/analyses/${id}/`),

  // Trigger analysis for a specific run
  analyzeRun: (runId: string) => api.post(`/forensics/analyze/${runId}/`),

  // Get analysis for a run (if exists)
  getRunAnalysis: (runId: string) =>
    api.get<FailureAnalysis>(`/forensics/runs/${runId}/analysis/`),

  // Resolve / ignore
  resolveFailure: (id: string) =>
    api.post(`/forensics/analyses/${id}/resolve/`),
  ignoreFailure: (id: string) => api.post(`/forensics/analyses/${id}/ignore/`),

  // Fixes
  getFixes: (analysisId: string) =>
    api.get<FixAction[]>(`/forensics/analyses/${analysisId}/fixes/`),
  applyFix: (fixId: string) => api.post(`/forensics/fixes/${fixId}/apply/`),

  // Error patterns
  getErrorPatterns: () => api.get<ErrorPattern[]>("/forensics/patterns/"),
  createErrorPattern: (data: any) =>
    api.post("/forensics/patterns/create/", data),

  // Stats
  getFailureStats: () => api.get<FailureStats>("/forensics/stats/"),
};
