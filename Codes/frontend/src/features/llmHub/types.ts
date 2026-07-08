export type LLMProvider =
  | "openai"
  | "anthropic"
  | "google"
  | "cohere"
  | "groq"
  | "together"
  | "huggingface"
  | "replicate"
  | "local";
export type ModelCapability =
  "chat" | "completion" | "embedding" | "vision" | "tool_call" | "streaming";
export type EvaluationMetric =
  | "accuracy"
  | "fluency"
  | "coherence"
  | "relevance"
  | "toxicity"
  | "bias"
  | "hallucination";
export type MemoryType = "session" | "user" | "project" | "global";

export interface LLMProviderConfig {
  id: string;
  type: LLMProvider;
  name: string;
  apiKey?: string;
  baseUrl?: string;
  models: string[];
  capabilities: ModelCapability[];
  enabled: boolean;
  icon?: string;
  costPerMillionTokens?: { input: number; output: number };
}

export interface Prompt {
  id: string;
  name: string;
  version: number;
  template: string;
  variables: string[];
  tags: string[];
  created_by: string;
  created_at: string;
  metadata?: Record<string, any>;
}

export interface EvaluationRun {
  id: string;
  name: string;
  prompt_id: string;
  models: string[];
  metrics: EvaluationMetric[];
  status: "pending" | "running" | "completed" | "failed";
  results: EvaluationResult[];
  created_at: string;
}

export interface EvaluationResult {
  model: string;
  scores: Record<EvaluationMetric, number>;
  latency_ms: number;
  token_usage: { input: number; output: number };
  cost_usd: number;
  output: string;
}

export interface CostEntry {
  date: string;
  provider: LLMProvider;
  model: string;
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
  request_id: string;
}

export interface MemoryEntry {
  id: string;
  type: MemoryType;
  key: string;
  value: string;
  created_at: string;
  expires_at?: string;
}

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: Record<string, any>;
  enabled: boolean;
  example?: string;
}

export interface ArenaBattle {
  id: string;
  model_a: string;
  model_b: string;
  prompt: string;
  response_a: string;
  response_b: string;
  votes: { for_a: number; for_b: number; tie: number };
  winner?: "A" | "B" | "tie";
}

export interface GuardrailRule {
  id: string;
  name: string;
  type: "input" | "output" | "both";
  action: "block" | "warn" | "replace";
  pattern: string;
  enabled: boolean;
  severity: "low" | "medium" | "high";
}
