import api from '../../../api/client'
import type {
  KpiMetric,
  ExperimentInsight,
  ExperimentComparison,
  OpportunityScore,
  InterventionSuggestion,
  AutoRefreshConfig,
} from '../types'

export const acceleratorAPI = {
  // KPI & Overview
  getKpis: () => api.get<KpiMetric[]>('/accelerator/kpis/'),
  getOpportunityScore: () => api.get<OpportunityScore>('/accelerator/opportunity/'),

  // Insights
  getInsights: (params?: { status?: string; type?: string; limit?: number }) =>
    api.get<{ results: ExperimentInsight[]; count: number }>('/accelerator/insights/', { params }),
  markInsightRead: (id: string) => api.post(`/accelerator/insights/${id}/read/`),
  dismissInsight: (id: string) => api.post(`/accelerator/insights/${id}/dismiss/`),

  // Experiment Comparisons
  compareExperiments: (experimentIds: string[]) =>
    api.post<ExperimentComparison[]>('/accelerator/compare/', { experiment_ids: experimentIds }),

  // Interventions
  getInterventions: () => api.get<InterventionSuggestion[]>('/accelerator/interventions/'),
  applyIntervention: (id: string) => api.post(`/accelerator/interventions/${id}/apply/`),

  // Auto‑refresh settings
  getAutoRefreshConfig: () => api.get<AutoRefreshConfig>('/accelerator/auto-refresh/'),
  updateAutoRefreshConfig: (enabled: boolean, interval_seconds: number) =>
    api.patch('/accelerator/auto-refresh/', { enabled, interval_seconds }),

  // Generate fresh insights (manual trigger)
  generateInsights: () => api.post('/accelerator/insights/generate/'),
}