import { request as api } from "../../../api/client";
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
    api.get<any>(`/explorer/correlations/`, {
      params: { metric },
    }),

  // Dashboard layouts (Mocked)
  getDashboardLayouts: async () => ({ data: [] }),
  getDashboardLayout: async (id: string) => ({ data: {} as DashboardLayout }),
  saveDashboardLayout: async (layout: Partial<DashboardLayout>) => ({
    data: layout as DashboardLayout,
  }),
  updateDashboardLayout: async (
    id: string,
    layout: Partial<DashboardLayout>,
  ) => ({ data: layout as DashboardLayout }),
  deleteDashboardLayout: async (id: string) => ({ data: null }),

  // SQL Editor (Mocked)
  getSQLConnections: async () => ({ data: [] }),
  executeSQL: async (connectionId: string, query: string) => ({
    data: {} as SQLQueryResult,
  }),
  getSchemaInfo: async (connectionId: string) => ({ data: {} }),
};
