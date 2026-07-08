import api from "../../../api/client";
import {
  ExperimentRun,
  DashboardLayout,
  SQLConnection,
  SQLQueryResult,
} from "../types";

export const experimentExplorerAPI = {
  // Hyperparameter explorer
  getExperiments: (params?: { limit?: number; sort?: string }) =>
    api.get<ExperimentRun[]>("/explorer/experiments/", { params }),
  getParameterCorrelations: (metric: string) =>
    api.get<ParameterMetricCorrelation[]>(`/explorer/correlations/`, {
      params: { metric },
    }),

  // Dashboard layouts
  getDashboardLayouts: () =>
    api.get<DashboardLayout[]>("/explorer/dashboards/"),
  getDashboardLayout: (id: string) =>
    api.get<DashboardLayout>(`/explorer/dashboards/${id}/`),
  saveDashboardLayout: (layout: Partial<DashboardLayout>) =>
    api.post<DashboardLayout>("/explorer/dashboards/", layout),
  updateDashboardLayout: (id: string, layout: Partial<DashboardLayout>) =>
    api.patch(`/explorer/dashboards/${id}/`, layout),
  deleteDashboardLayout: (id: string) =>
    api.delete(`/explorer/dashboards/${id}/`),

  // SQL Editor
  getSQLConnections: () =>
    api.get<SQLConnection[]>("/explorer/sql/connections/"),
  executeSQL: (connectionId: string, query: string) =>
    api.post<SQLQueryResult>(`/explorer/sql/execute/`, {
      connection_id: connectionId,
      query,
    }),
  getSchemaInfo: (connectionId: string) =>
    api.get(`/explorer/sql/schema/${connectionId}/`),
};
