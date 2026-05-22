export type LLMProvider = 'openai' | 'anthropic' | 'google' | 'cohere' | 'groq' | 'local'
export type MessageRole = 'system' | 'user' | 'assistant' | 'tool'

export interface ChatMessage {
  id: string
  role: MessageRole
  content: string
  tool_calls?: ToolCall[]
  tool_call_id?: string
  name?: string
  createdAt?: Date
}

export interface ToolCall {
  id: string
  type: 'function'
  function: {
    name: string
    arguments: string
  }
}

export interface ModelConfig {
  provider: LLMProvider
  model: string
  temperature: number
  max_tokens: number
  top_p: number
  frequency_penalty: number
  presence_penalty: number
}

export interface ComparisonResult {
  model_a: ModelConfig
  model_b: ModelConfig
  prompt: string
  response_a: string
  response_b: string
  tokens_a: { input: number; output: number }
  tokens_b: { input: number; output: number }
  latency_ms_a: number
  latency_ms_b: number
  created_at: string
}

export interface ToolDefinition {
  name: string
  description: string
  parameters: Record<string, any>
  enabled: boolean
}

export interface ToolTestResult {
  tool_name: string
  input: any
  output: any
  success: boolean
  error?: string
  execution_time_ms: number
}