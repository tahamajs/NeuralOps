import api from '../../../api/client'
import { LLMProviderConfig, Prompt, EvaluationRun, CostEntry, MemoryEntry, ToolDefinition, ArenaBattle, GuardrailRule } from '../types'

export const llmHubAPI = {
  // Providers
  getProviders: () => api.get<LLMProviderConfig[]>('/llm/providers/'),
  updateProvider: (id: string, data: Partial<LLMProviderConfig>) => api.patch(`/llm/providers/${id}/`, data),
  testProvider: (id: string) => api.post(`/llm/providers/${id}/test/`),

  // Prompts
  getPrompts: () => api.get<Prompt[]>('/llm/prompts/'),
  getPrompt: (id: string) => api.get<Prompt>(`/llm/prompts/${id}/`),
  createPrompt: (data: Partial<Prompt>) => api.post<Prompt>('/llm/prompts/', data),
  updatePrompt: (id: string, data: Partial<Prompt>) => api.patch(`/llm/prompts/${id}/`, data),
  deletePrompt: (id: string) => api.delete(`/llm/prompts/${id}/`),

  // Playground / Chat
  chatCompletion: (provider: string, model: string, messages: any[], params?: any) =>
    api.post('/llm/chat/', { provider, model, messages, params }, { responseType: 'stream' }),

  // Evaluations
  getEvaluationRuns: () => api.get<EvaluationRun[]>('/llm/evaluations/'),
  createEvaluation: (data: Partial<EvaluationRun>) => api.post<EvaluationRun>('/llm/evaluations/', data),
  getEvaluationResults: (runId: string) => api.get<EvaluationResult[]>(`/llm/evaluations/${runId}/results/`),

  // Cost Tracking
  getCostEntries: (params?: { start_date?: string; end_date?: string }) => api.get<CostEntry[]>('/llm/costs/', { params }),
  getCostSummary: () => api.get('/llm/costs/summary/'),

  // Memory
  getMemoryEntries: () => api.get<MemoryEntry[]>('/llm/memory/'),
  createMemoryEntry: (data: Partial<MemoryEntry>) => api.post<MemoryEntry>('/llm/memory/', data),
  deleteMemoryEntry: (id: string) => api.delete(`/llm/memory/${id}/`),

  // Tools
  getTools: () => api.get<ToolDefinition[]>('/llm/tools/'),
  updateTool: (name: string, enabled: boolean) => api.patch(`/llm/tools/${name}/`, { enabled }),

  // Arena
  getArenaBattles: () => api.get<ArenaBattle[]>('/llm/arena/'),
  submitVote: (battleId: string, winner: 'A' | 'B' | 'tie') => api.post(`/llm/arena/${battleId}/vote/`, { winner }),

  // Guardrails
  getGuardrails: () => api.get<GuardrailRule[]>('/llm/guardrails/'),
  createGuardrail: (data: Partial<GuardrailRule>) => api.post<GuardrailRule>('/llm/guardrails/', data),
  updateGuardrail: (id: string, data: Partial<GuardrailRule>) => api.patch(`/llm/guardrails/${id}/`, data),
  deleteGuardrail: (id: string) => api.delete(`/llm/guardrails/${id}/`),
  testGuardrail: (id: string, text: string) => api.post(`/llm/guardrails/${id}/test/`, { text }),
}