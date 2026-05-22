import api from '../../../api/client'
import { FeatureGroup, DriftMonitor, DriftCheck, Alert, AlertChannel, PerformanceSnapshot } from '../types'

export const mlopsAPI = {
  // Feature Store
  getFeatureGroups: () => api.get<FeatureGroup[]>('/mlops/feature-stores/groups/'),
  getFeatureGroup: (id: string) => api.get<FeatureGroup>(`/mlops/feature-stores/groups/${id}/`),
  createFeatureGroup: (data: Partial<FeatureGroup>) => api.post<FeatureGroup>('/mlops/feature-stores/groups/', data),
  updateFeatureGroup: (id: string, data: Partial<FeatureGroup>) => api.patch(`/mlops/feature-stores/groups/${id}/`, data),
  deleteFeatureGroup: (id: string) => api.delete(`/mlops/feature-stores/groups/${id}/`),
  ingestFeatures: (groupId: string, records: any[]) => api.post(`/mlops/feature-stores/groups/${groupId}/ingest/`, { records }),
  serveFeatures: (groupId: string, entityKeys: string[]) => api.post(`/mlops/feature-stores/groups/${groupId}/serve/`, { entity_keys: entityKeys }),

  // Drift Monitoring
  getDriftMonitors: () => api.get<DriftMonitor[]>('/mlops/drift/monitors/'),
  createDriftMonitor: (data: Partial<DriftMonitor>) => api.post<DriftMonitor>('/mlops/drift/monitors/', data),
  deleteDriftMonitor: (id: string) => api.delete(`/mlops/drift/monitors/${id}/`),
  getDriftChecks: (monitorId: string, limit?: number) => api.get<DriftCheck[]>(`/mlops/drift/monitors/${monitorId}/checks/`, { params: { limit } }),
  runDriftCheck: (monitorId: string) => api.post<DriftCheck>(`/mlops/drift/monitors/${monitorId}/check/`),

  // Alerts
  getAlerts: (params?: { status?: string; severity?: string }) => api.get<Alert[]>('/mlops/alerts/', { params }),
  acknowledgeAlert: (id: string) => api.post(`/mlops/alerts/${id}/acknowledge/`),
  resolveAlert: (id: string) => api.post(`/mlops/alerts/${id}/resolve/`),
  getAlertChannels: () => api.get<AlertChannel[]>('/mlops/alerts/channels/'),
  createAlertChannel: (data: Partial<AlertChannel>) => api.post<AlertChannel>('/mlops/alerts/channels/', data),
  deleteAlertChannel: (id: string) => api.delete(`/mlops/alerts/channels/${id}/`),
  testAlertChannel: (id: string) => api.post(`/mlops/alerts/channels/${id}/test/`),

  // Performance Snapshots
  getPerformanceSnapshots: (params?: { model_id?: string; start_date?: string; end_date?: string }) =>
    api.get<PerformanceSnapshot[]>('/mlops/performance/snapshots/', { params }),
  getSnapshotDetail: (id: string) => api.get<PerformanceSnapshot>(`/mlops/performance/snapshots/${id}/`),
  triggerSnapshot: (modelId: string) => api.post<PerformanceSnapshot>('/mlops/performance/snapshots/trigger/', { model_id: modelId }),
}