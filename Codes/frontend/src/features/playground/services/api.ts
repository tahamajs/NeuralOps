import api from '../../../api/client'
import { ChatMessage, ModelConfig, ComparisonResult, ToolDefinition, ToolTestResult } from '../types'

export const playgroundAPI = {
  // Chat completion (streaming)
  chatCompletion: (messages: ChatMessage[], config: ModelConfig) =>
    api.post('/playground/chat/', { messages, config }, { responseType: 'stream' }),

  // List available models
  getModels: () => api.get<ModelConfig[]>('/playground/models/'),

  // Comparison
  compareModels: (configA: ModelConfig, configB: ModelConfig, prompt: string) =>
    api.post<ComparisonResult>('/playground/compare/', { config_a: configA, config_b: configB, prompt }),

  // Tools
  getTools: () => api.get<ToolDefinition[]>('/playground/tools/'),
  testTool: (toolName: string, input: any) =>
    api.post<ToolTestResult>(`/playground/tools/${toolName}/test/`, { input }),
}