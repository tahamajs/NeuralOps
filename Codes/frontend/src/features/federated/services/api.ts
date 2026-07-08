import api from "../../../api/client";
import type { GlobalMetrics, Client, FLRun } from "../types";

export const federatedAPI = {
  getRun: (runId: string) => api.get<FLRun>(`/federated/${runId}/`),
  getGlobalMetrics: (runId: string) =>
    api.get<GlobalMetrics>(`/federated/${runId}/global/`),
  getClients: (runId: string) =>
    api.get<{ clients: Client[] }>(`/federated/${runId}/clients/`),
  startNewRun: (config: any) => api.post("/federated/start/", config),
};
