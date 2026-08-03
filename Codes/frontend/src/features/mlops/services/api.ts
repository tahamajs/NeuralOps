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
  getFeatureGroups: async (): Promise<FeatureGroup[]> => [],
  getFeatureGroup: async (id: string): Promise<FeatureGroup> =>
    ({} as FeatureGroup),
  createFeatureGroup: async (
    data: Partial<FeatureGroup>,
  ): Promise<FeatureGroup> => data as FeatureGroup,
  updateFeatureGroup: async (
    id: string,
    data: Partial<FeatureGroup>,
  ): Promise<any> => data,
  deleteFeatureGroup: async (id: string): Promise<null> => null,
  ingestFeatures: (groupId: string, records: any[]) =>
    request.post(`/mlops/features/ingest/`, { records }),
  serveFeatures: (groupId: string, entityKeys: string[]) =>
    request.post(`/mlops/features/serve/`, {
      entity_keys: entityKeys,
    }),

  // Drift Monitoring (Mocked)
  getDriftMonitors: async (): Promise<DriftMonitor[]> => [],
  createDriftMonitor: async (
    data: Partial<DriftMonitor>,
  ): Promise<DriftMonitor> => data as DriftMonitor,
  deleteDriftMonitor: async (id: string): Promise<null> => null,
  getDriftChecks: async (
    monitorId: string,
    limit?: number,
  ): Promise<DriftCheck[]> => [],
  runDriftCheck: async (monitorId: string): Promise<DriftCheck> =>
    ({} as DriftCheck),

  // Alerts (Mocked)
  getAlerts: async (params?: {
    status?: string;
    severity?: string;
  }): Promise<Alert[]> => [],
  acknowledgeAlert: async (id: string): Promise<null> => null,
  resolveAlert: async (id: string): Promise<null> => null,
  getAlertChannels: async (): Promise<AlertChannel[]> => [],
  createAlertChannel: async (
    data: Partial<AlertChannel>,
  ): Promise<AlertChannel> => data as AlertChannel,
  deleteAlertChannel: async (id: string): Promise<null> => null,
  testAlertChannel: async (id: string): Promise<null> => null,

  // Performance Snapshots (Mocked)
  getPerformanceSnapshots: async (params?: any): Promise<PerformanceSnapshot[]> =>
    [],
  getSnapshotDetail: async (id: string): Promise<PerformanceSnapshot> =>
    ({} as PerformanceSnapshot),
  triggerSnapshot: async (modelId: string): Promise<PerformanceSnapshot> =>
    ({} as PerformanceSnapshot),
};
