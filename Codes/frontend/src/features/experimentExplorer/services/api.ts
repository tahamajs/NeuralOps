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
  getDashboardLayouts: async (): Promise<DashboardLayout[]> => [],
  getDashboardLayout: async (id: string): Promise<DashboardLayout> =>
    ({} as DashboardLayout),
  saveDashboardLayout: async (
    layout: Partial<DashboardLayout>,
  ): Promise<DashboardLayout> => layout as DashboardLayout,
  updateDashboardLayout: async (
    id: string,
    layout: Partial<DashboardLayout>,
  ): Promise<DashboardLayout> => layout as DashboardLayout,
  deleteDashboardLayout: async (id: string): Promise<null> => null,

  // SQL Editor (Mocked)
  getSQLConnections: async (): Promise<SQLConnection[]> => [],
  executeSQL: async (
    connectionId: string,
    query: string,
  ): Promise<SQLQueryResult> => ({} as SQLQueryResult),
  getSchemaInfo: async (connectionId: string): Promise<any> => ({}),
};
