// src/features/coscientist/services/api.ts
import api from '../../../api/client'
import { ResearchSession, Hypothesis, TournamentMatch, CoscientistConfig } from '../types'

export const coscientistAPI = {
  // Sessions
  startSession: (goal: string, config?: Partial<CoscientistConfig>) =>
    api.post<{ session_id: string }>('/coscientist/start/', { goal, config }),
  getSession: (sessionId: string) => api.get<ResearchSession>(`/coscientist/sessions/${sessionId}/`),
  listSessions: () => api.get<{ results: ResearchSession[] }>('/coscientist/sessions/'),
  stopSession: (sessionId: string) => api.post(`/coscientist/sessions/${sessionId}/stop/`),

  // Hypotheses
  getHypotheses: (sessionId: string) => api.get<Hypothesis[]>(`/coscientist/sessions/${sessionId}/hypotheses/`),
  getHypothesisDetail: (hypothesisId: string) => api.get<Hypothesis>(`/coscientist/hypotheses/${hypothesisId}/`),

  // Tournament
  getTournamentBracket: (sessionId: string) => api.get<TournamentMatch[]>(`/coscientist/sessions/${sessionId}/tournament/`),

  // Evolution
  triggerEvolutionCycle: (sessionId: string) => api.post(`/coscientist/sessions/${sessionId}/evolve/`),
}