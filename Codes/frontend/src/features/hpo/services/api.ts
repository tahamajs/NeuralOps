import { request as api } from "../../../api/client";
import { Sweep, Trial, SweepStats, ParameterImportance } from "../types";

export const hpoAPI = {
  // Sweeps
  getSweeps: () => api.get<{ results: Sweep[]; count: number }>("/sweeps/"),
  getSweep: (id: string) => api.get<Sweep>(`/sweeps/${id}/`),
  getSweepStats: (id: string) => api.get<SweepStats>(`/sweeps/${id}/stats/`),
  createSweep: (data: any) => api.post("/sweeps/", data),
  stopSweep: (id: string) => api.post(`/sweeps/${id}/stop/`),
  deleteSweep: (id: string) => api.delete(`/sweeps/${id}/`),

  // Trials
  getTrials: (sweepId: string) =>
    api.get<{ results: Trial[]; count: number }>(`/sweeps/${sweepId}/trials/`),
  getTrialDetail: (sweepId: string, trialNumber: number) =>
    api.get<Trial>(`/sweeps/${sweepId}/trials/${trialNumber}/`),

  // Visualizations
  getParameterImportance: (sweepId: string) =>
    api.get<ParameterImportance[]>(`/sweeps/${sweepId}/importance/`),
  getParallelCoordinatesData: (sweepId: string) =>
    api.get<any>(`/sweeps/${sweepId}/parallel/`),
  getContourData: (sweepId: string) =>
    api.get<any>(`/sweeps/${sweepId}/contour/`),
  getHistoryData: (sweepId: string) =>
    api.get<any[]>(`/sweeps/${sweepId}/history/`),
  getSliceData: (sweepId: string, param: string) =>
    api.get<any>(`/sweeps/${sweepId}/slice/?param=${param}`),
  getEdData: (sweepId: string) => api.get<any>(`/sweeps/${sweepId}/edf/`),
};
