import { request } from "../../../api/client";
import {
  LLMProviderConfig,
  Prompt,
  EvaluationRun,
  CostEntry,
  MemoryEntry,
  ToolDefinition,
  ArenaBattle,
  GuardrailRule,
} from "../types";

export type EvaluationResult = any;

export const llmHubAPI = {
  // Providers
  getProviders: () => request.get<LLMProviderConfig[]>("/llm/providers/"),
  updateProvider: (id: string, data: Partial<LLMProviderConfig>) =>
    request.patch(`/llm/providers/${id}/`, data),
  testProvider: (id: string) => request.post(`/llm/providers/${id}/test/`),

  // Prompts
  getPrompts: () => request.get<Prompt[]>("/llm/prompts/"),
  getPrompt: (id: string) => request.get<Prompt>(`/llm/prompts/${id}/`),
  createPrompt: (data: Partial<Prompt>) =>
    request.post<Prompt>("/llm/prompts/", data),
  updatePrompt: (id: string, data: Partial<Prompt>) =>
    request.patch(`/llm/prompts/${id}/`, data),
  deletePrompt: (id: string) => request.delete(`/llm/prompts/${id}/`),

  // Playground / Chat
  chatCompletion: (
    provider: string,
    model: string,
    messages: any[],
    params?: any,
  ) =>
    request.post(
      "/llm/chat/",
      { provider, model, messages, params },
      { responseType: "stream" },
    ),

  // Evaluations
  getEvaluationRuns: () => request.get<EvaluationRun[]>("/llm/evaluations/"),
  createEvaluation: (data: Partial<EvaluationRun>) =>
    request.post<EvaluationRun>("/llm/evaluations/", data),
  getEvaluationResults: (runId: string) =>
    request.get<EvaluationResult[]>(`/llm/evaluations/${runId}/results/`),

  // Cost Tracking
  getCostEntries: (params?: { start_date?: string; end_date?: string }) =>
    request.get<CostEntry[]>("/llm/costs/", { params }),
  getCostSummary: () => request.get("/llm/costs/summary/"),

  // Memory
  getMemoryEntries: () => request.get<MemoryEntry[]>("/llm/memory/"),
  createMemoryEntry: (data: Partial<MemoryEntry>) =>
    request.post<MemoryEntry>("/llm/memory/", data),
  deleteMemoryEntry: (id: string) => request.delete(`/llm/memory/${id}/`),

  // Tools
  getTools: () => request.get<ToolDefinition[]>("/llm/tools/"),
  updateTool: (name: string, enabled: boolean) =>
    request.patch(`/llm/tools/${name}/`, { enabled }),

  // Arena
  getArenaBattles: () => request.get<ArenaBattle[]>("/llm/arena/"),
  submitVote: (battleId: string, winner: "A" | "B" | "tie") =>
    request.post(`/llm/arena/${battleId}/vote/`, { winner }),

  // Guardrails
  getGuardrails: () => request.get<GuardrailRule[]>("/llm/guardrails/"),
  createGuardrail: (data: Partial<GuardrailRule>) =>
    request.post<GuardrailRule>("/llm/guardrails/", data),
  updateGuardrail: (id: string, data: Partial<GuardrailRule>) =>
    request.patch(`/llm/guardrails/${id}/`, data),
  deleteGuardrail: (id: string) => request.delete(`/llm/guardrails/${id}/`),
  testGuardrail: (id: string, text: string) =>
    request.post(`/llm/guardrails/${id}/test/`, { text }),
};
