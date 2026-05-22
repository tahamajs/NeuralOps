import api from '../../../api/client'
import {
  DriftMonitor,
  DriftCheck,
  Alert,
  SystemHealth,
  PerformanceMetric,
} from '../types'

export const healthMonitorAPI = {
  // Drift monitors
  getDriftMonitors: () => api.get<DriftMonitor[]>('/health-monitor/drift-monitors/'),
  getDriftMonitor: (id: string) => api.get<DriftMonitor>(`/health-monitor/drift-monitors/${id}/`),
  createDriftMonitor: (data: Partial<DriftMonitor>) => api.post<DriftMonitor>('/health-monitor/drift-monitors/', data),
  updateDriftMonitor: (id: string, data: Partial<DriftMonitor>) => api.patch(`/health-monitor/drift-monitors/${id}/`, data),
  deleteDriftMonitor: (id: string) => api.delete(`/health-monitor/drift-monitors/${id}/`),
  runDriftCheck: (id: string) => api.post<DriftCheck>(`/health-monitor/drift-monitors/${id}/check/`),
  getDriftChecks: (id: string, limit?: number) => api.get<DriftCheck[]>(`/health-monitor/drift-monitors/${id}/checks/`, { params: { limit } }),

  // Alerts
  getAlerts: (params?: { status?: string; severity?: string; limit?: number }) =>
    api.get<Alert[]>('/health-monitor/alerts/', { params }),
  acknowledgeAlert: (id: string) => api.post(`/health-monitor/alerts/${id}/acknowledge/`),
  resolveAlert: (id: string) => api.post(`/health-monitor/alerts/${id}/resolve/`),

  // System health
  getSystemHealth: () => api.get<SystemHealth>('/health-monitor/health/'),
  getPerformanceMetrics: (params?: { model_id?: string; metric_type?: string; hours?: number }) =>
    api.get<PerformanceMetric[]>('/health-monitor/performance/', { params }),
}