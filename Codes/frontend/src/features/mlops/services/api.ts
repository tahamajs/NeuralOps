import { request } from "../../../api/client";
import {
  FeatureGroup,
  DriftMonitor,
  DriftCheck,
  Alert,
  AlertChannel,
  PerformanceSnapshot,
} from "../types";

export const mlopsAPI = {
  // Feature Store (Mocked structural endpoints)
  getFeatureGroups: async () => ({ data: [] }),
  getFeatureGroup: async (id: string) => ({ data: {} as FeatureGroup }),
  createFeatureGroup: async (data: Partial<FeatureGroup>) => ({
    data: data as FeatureGroup,
  }),
  updateFeatureGroup: async (id: string, data: Partial<FeatureGroup>) => ({
    data,
  }),
  deleteFeatureGroup: async (id: string) => ({ data: null }),
  ingestFeatures: (groupId: string, records: any[]) =>
    request.post(`/mlops/features/ingest/`, { records }), // the backend actually has /features/ingest/
  serveFeatures: (groupId: string, entityKeys: string[]) =>
    request.post(`/mlops/features/serve/`, {
      entity_keys: entityKeys,
    }),

  // Drift Monitoring (Mocked)
  getDriftMonitors: async () => ({ data: [] }),
  createDriftMonitor: async (data: Partial<DriftMonitor>) => ({
    data: data as DriftMonitor,
  }),
  deleteDriftMonitor: async (id: string) => ({ data: null }),
  getDriftChecks: async (monitorId: string, limit?: number) => ({ data: [] }),
  runDriftCheck: async (monitorId: string) => ({ data: {} as DriftCheck }),

  // Alerts (Mocked)
  getAlerts: async (params?: { status?: string; severity?: string }) => ({
    data: [],
  }),
  acknowledgeAlert: async (id: string) => ({ data: null }),
  resolveAlert: async (id: string) => ({ data: null }),
  getAlertChannels: async () => ({ data: [] }),
  createAlertChannel: async (data: Partial<AlertChannel>) => ({
    data: data as AlertChannel,
  }),
  deleteAlertChannel: async (id: string) => ({ data: null }),
  testAlertChannel: async (id: string) => ({ data: null }),

  // Performance Snapshots (Mocked)
  getPerformanceSnapshots: async (params?: any) => ({ data: [] }),
  getSnapshotDetail: async (id: string) => ({
    data: {} as PerformanceSnapshot,
  }),
  triggerSnapshot: async (modelId: string) => ({
    data: {} as PerformanceSnapshot,
  }),
};
