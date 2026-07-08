export interface ChatSession {
  id: string;
  title: string;
  status: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "tool" | "experiment_card" | "system";
  content: string;
  metadata?: any;
  created_at: string;
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
  status: "healthy" | "overfitting" | "plateau" | "underfitting" | "diverging";
  issues: string[];
  suggestions: string[];
  metrics_summary: Record<string, any>;
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
