// Agent feature types — fully complete

export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
  result?: unknown;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'tool' | 'experiment_card' | 'system';
  content: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  metadata?: Record<string, unknown>;
  created_at: string;
}

export interface ChatSession {
  id: string;
  title: string;
  status: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface Paper {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  year?: number;
  citation_count: number;
  url: string;
  pdf_url?: string;
  source: string;
  source_id: string;
  tags: string[];
  notes: string;
}

export interface TrainingAnalysisResult {
  run_id: string;
  status: 'healthy' | 'overfitting' | 'plateau' | 'underfitting' | 'diverging';
  issues: string[];
  suggestions: string[];
  metrics_summary: Record<string, unknown>;
}

export interface ExperimentConfig {
  model_name: string;
  task_name: string;
  attention_type: string;
  method: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  max_seq_length: number;
  compression_rank: number;
}

export interface AgentTool {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  category?: string;
}

export interface AgentStatus {
  status: 'operational' | 'degraded' | 'offline';
  model: string;
  tools_count: number;
  active_sessions: number;
  ws_enabled: boolean;
}

export interface AgentSession {
  id: string;
  name?: string;
  objective?: string;
  status: 'active' | 'paused' | 'completed' | 'failed';
  cycles_completed?: number;
  max_cycles?: number;
  experiments_run?: number;
  best_metric_value?: number;
  total_llm_cost_cents?: number;
  memory?: {
    working_memory: string;
    long_term_summary: string;
  };
  actions?: AgentAction[];
  created_at: string;
  updated_at: string;
}

export interface AgentAction {
  id: string;
  cycle: number;
  action_type: string;
  tool_name?: string;
  output_summary?: string;
  created_at: string;
}
