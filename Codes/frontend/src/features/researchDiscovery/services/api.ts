import api from '../../../api/client'
import { Paper, LiteratureMap, SynthesisResult, Hypothesis, Grant } from '../types'

export const researchDiscoveryAPI = {
  // Literature synthesis
  synthesizeLiterature: (query: string, limit?: number) =>
    api.post<SynthesisResult>('/research-discovery/literature/synthesize/', { query, limit }),
  getLiteratureMap: (paperId: string, depth: number = 2) =>
    api.get<LiteratureMap>(`/research-discovery/literature/map/${paperId}/`, { params: { depth } }),

  // Hypothesis generation
  generateHypotheses: (goal: string, parameters?: any) =>
    api.post<Hypothesis[]>('/research-discovery/hypotheses/generate/', { goal, parameters }),
  evolveHypothesis: (hypothesisId: string, feedback?: string) =>
    api.post<Hypothesis>(`/research-discovery/hypotheses/${hypothesisId}/evolve/`, { feedback }),
  getHypothesisTimeline: (hypothesisId: string) =>
    api.get<Hypothesis[]>(`/research-discovery/hypotheses/${hypothesisId}/timeline/`),

  // Cross‑database search
  crossSearch: (query: string, databases: string[], limit?: number) =>
    api.post<Paper[]>('/research-discovery/databases/search/', { query, databases, limit }),
  getDatabaseStatus: () => api.get<Record<string, boolean>>('/research-discovery/databases/status/'),

  // Grant matching
  matchGrants: (profile: any) => api.post<Grant[]>('/research-discovery/grants/match/', { profile }),
  getGrantDetail: (id: string) => api.get<Grant>(`/research-discovery/grants/${id}/`),
}