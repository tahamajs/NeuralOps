import { request } from "../../../api/client";
import { Anomaly, FixAction } from "../types";

export const anomalyAPI = {
  // List anomalies with filters
  getAnomalies: (params?: {
    status?: string;
    severity?: string;
    run_id?: string;
  }) =>
    request.get<{ results: Anomaly[]; count: number }>("/anomaly-hpo/anomalies/", {
      params,
    }),

  // Get single anomaly
  getAnomaly: (id: string) => request.get<Anomaly>(`/anomaly-hpo/anomalies/${id}/`),

  // Trigger detection for a specific run
  detectAnomaly: (runId: string) =>
    request.post(`/anomaly-hpo/anomalies/detect/${runId}/`),

  // Get summary stats
  getAnomalySummary: () => request.get("/anomaly-hpo/anomalies/summary/"),

  // Resolve anomaly
  resolveAnomaly: (id: string, resolved: boolean) =>
    request.patch(`/anomaly-hpo/anomalies/${id}/`, {
      status: resolved ? "resolved" : "open",
    }),

  // Get suggested fixes for an anomaly
  getFixes: (anomalyId: string) =>
    request.get<FixAction[]>(`/anomaly-hpo/anomalies/${anomalyId}/fixes/`),

  // Apply a fix
  applyFix: (fixId: string) => request.post(`/anomaly-hpo/fixes/${fixId}/apply/`),
};
