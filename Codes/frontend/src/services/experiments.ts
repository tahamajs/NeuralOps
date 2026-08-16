import { request } from "../api/client";
import { ENDPOINTS } from "../api/endpoints";
import { Experiment } from "../types/experiment";

export interface RunMetrics {
  loss: { steps: number[]; values: number[] };
  accuracy?: { steps: number[]; values: number[] };
  [key: string]: any;
}

export interface RunAnalysis {
  overview: string;
  suggestions: string[];
  bottlenecks: string[];
}

export interface RunArtifact {
  name: string;
  size: string;
  type: string;
}

export interface DataLineage {
  nodes: Array<{ id: string; label: string; type: string }>;
  edges: Array<{ source: string; target: string }>;
}

export const experimentsService = {
  getExperiments: (params?: { search?: string; status?: string }) =>
    request.get<Experiment[]>(ENDPOINTS.EXPERIMENTS.LIST, { params }),

  getExperiment: (id: string) =>
    request.get<Experiment>(ENDPOINTS.EXPERIMENTS.DETAIL(id)),

  getRun: (runId: string) =>
    request.get<any>(ENDPOINTS.RUNS.DETAIL(runId)),

  stopRun: (runId: string) =>
    request.post<{ success: boolean }>(ENDPOINTS.RUNS.STOP(runId)),

  getRunResults: (runId: string) =>
    request.get<any>(ENDPOINTS.RUNS.RESULTS(runId)),

  getRunMetrics: (runId: string) =>
    request.get<RunMetrics>(`/runs/${runId}/metrics/`),

  getRunAnalysis: (runId: string) =>
    request.get<RunAnalysis>(ENDPOINTS.TRAINING_ANALYSIS.RUN_ANALYSIS(runId)),

  getRunArtifacts: (runId: string) =>
    request.get<RunArtifact[]>(`/runs/${runId}/artifacts/`),

  getDataLineage: (runId: string) =>
    request.get<DataLineage>(ENDPOINTS.RUNS.DATA_LINEAGE(runId)),

  getModelCard: (runId: string) =>
    request.get<any>(ENDPOINTS.RUNS.MODEL_CARD(runId)),

  getRunDrift: (runId: string) =>
    request.get<any>(ENDPOINTS.RUNS.DRIFT_REPORT(runId)),

  getRunFairness: (runId: string) =>
    request.get<any>(ENDPOINTS.RUNS.FAIRNESS(runId)),

  getRunRobustness: (runId: string) =>
    request.get<any>(ENDPOINTS.RUNS.ROBUSTNESS(runId)),
};
