export interface Experiment {
  id: string;
  name: string;
  description?: string;
  status?: "queued" | "running" | "completed" | "failed" | "stopped";
  job_type?: "training" | "evaluation" | "compression";
  config?: Record<string, any>;
  default_config?: Record<string, any>;
  tags?: string;
  runs_count?: number;
  created_at: string;
  updated_at?: string;
  started_at?: string;
  completed_at?: string;
  current_step?: number;
  total_steps?: number;
  progress?: number;
  logs?: string | string[];
  metrics?: { loss: number[]; accuracy: number[] };
  final_metrics?: Record<string, any>;
  training_history?: Record<string, any>;
}

export interface TrainingRun {
  id: string;
  experiment?: string;
  experiment_name?: string;
  name?: string;
  job_type: "training" | "evaluation" | "compression";
  status: "queued" | "running" | "completed" | "failed" | "stopped";
  config: Record<string, any>;
  final_metrics: Record<string, any>;
  training_history?: Record<string, any>;
  logs?: string;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  current_step: number;
  total_steps: number;
  estimated_remaining_seconds?: number | null;
  cpu_usage?: number;
  ram_usage_mb?: number;
  gpu_usage?: number | null;
  gpu_memory_mb?: number | null;
  artifacts_dir?: string;
}

export interface TrainingSnapshot {
  id?: string | number;
  run: string;
  step: number;
  epoch?: number | null;
  train_loss?: number | null;
  eval_loss?: number | null;
  eval_accuracy?: number | null;
  learning_rate?: number | null;
  cpu_percent?: number | null;
  ram_mb?: number | null;
  timestamp: string;
}

export interface ActivityLog {
  id: string | number;
  username?: string;
  user?: number;
  action: string;
  detail?: string;
  created_at: string;
}

