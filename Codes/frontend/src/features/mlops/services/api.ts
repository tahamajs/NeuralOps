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
  // Feature Store
  getFeatureGroups: () =>
    request.get<FeatureGroup[]>("/mlops/feature-stores/groups/"),
  getFeatureGroup: (id: string) =>
    request.get<FeatureGroup>(`/mlops/feature-stores/groups/${id}/`),
  createFeatureGroup: (data: Partial<FeatureGroup>) =>
    request.post<FeatureGroup>("/mlops/feature-stores/groups/", data),
  updateFeatureGroup: (id: string, data: Partial<FeatureGroup>) =>
    request.patch(`/mlops/feature-stores/groups/${id}/`, data),
  deleteFeatureGroup: (id: string) =>
    request.delete(`/mlops/feature-stores/groups/${id}/`),
  ingestFeatures: (groupId: string, records: any[]) =>
    request.post(`/mlops/feature-stores/groups/${groupId}/ingest/`, { records }),
  serveFeatures: (groupId: string, entityKeys: string[]) =>
    request.post(`/mlops/feature-stores/groups/${groupId}/serve/`, {
      entity_keys: entityKeys,
    }),

  // Drift Monitoring
  getDriftMonitors: () => request.get<DriftMonitor[]>("/mlops/drift/monitors/"),
  createDriftMonitor: (data: Partial<DriftMonitor>) =>
    request.post<DriftMonitor>("/mlops/drift/monitors/", data),
  deleteDriftMonitor: (id: string) =>
    request.delete(`/mlops/drift/monitors/${id}/`),
  getDriftChecks: (monitorId: string, limit?: number) =>
    request.get<DriftCheck[]>(`/mlops/drift/monitors/${monitorId}/checks/`, {
      params: { limit },
    }),
  runDriftCheck: (monitorId: string) =>
    request.post<DriftCheck>(`/mlops/drift/monitors/${monitorId}/check/`),

  // Alerts
  getAlerts: (params?: { status?: string; severity?: string }) =>
    request.get<Alert[]>("/mlops/alerts/", { params }),
  acknowledgeAlert: (id: string) =>
    request.post(`/mlops/alerts/${id}/acknowledge/`),
  resolveAlert: (id: string) => request.post(`/mlops/alerts/${id}/resolve/`),
  getAlertChannels: () => request.get<AlertChannel[]>("/mlops/alerts/channels/"),
  createAlertChannel: (data: Partial<AlertChannel>) =>
    request.post<AlertChannel>("/mlops/alerts/channels/", data),
  deleteAlertChannel: (id: string) =>
    request.delete(`/mlops/alerts/channels/${id}/`),
  testAlertChannel: (id: string) =>
    request.post(`/mlops/alerts/channels/${id}/test/`),

  // Performance Snapshots
  getPerformanceSnapshots: (params?: {
    model_id?: string;
    start_date?: string;
    end_date?: string;
  }) =>
    request.get<PerformanceSnapshot[]>("/mlops/performance/snapshots/", { params }),
  getSnapshotDetail: (id: string) =>
    request.get<PerformanceSnapshot>(`/mlops/performance/snapshots/${id}/`),
  triggerSnapshot: (modelId: string) =>
    request.post<PerformanceSnapshot>("/mlops/performance/snapshots/trigger/", {
      model_id: modelId,
    }),
};
