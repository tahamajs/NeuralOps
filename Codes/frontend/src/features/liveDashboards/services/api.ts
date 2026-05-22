import api from '../../../api/client'
import { Dashboard, WidgetDefinition, LiveDataPoint } from '../types'

export const liveDashboardsAPI = {
  // Dashboards
  getDashboards: () => api.get<Dashboard[]>('/live-dash/dashboards/'),
  getDashboard: (id: string) => api.get<Dashboard>(`/live-dash/dashboards/${id}/`),
  createDashboard: (data: Partial<Dashboard>) => api.post<Dashboard>('/live-dash/dashboards/', data),
  updateDashboard: (id: string, data: Partial<Dashboard>) => api.patch(`/live-dash/dashboards/${id}/`, data),
  deleteDashboard: (id: string) => api.delete(`/live-dash/dashboards/${id}/`),

  // Widget library
  getWidgetDefinitions: () => api.get<WidgetDefinition[]>('/live-dash/widgets/definitions/'),

  // Live data (REST fallback, WebSocket used for real-time)
  fetchWidgetData: (widgetId: string, params?: any) =>
    api.post<LiveDataPoint[]>(`/live-dash/widgets/${widgetId}/data/`, params),
}