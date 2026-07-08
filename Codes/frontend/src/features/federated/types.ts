export interface Client {
  id: string;
  status: "active" | "inactive" | "straggler";
  accuracy: number;
  data_samples: number;
  last_update: string;
}

export interface GlobalMetrics {
  final_accuracy: number;
  current_round: number;
  total_rounds: number;
  history: { round: number; accuracy: number }[];
  next_aggregation_progress: number;
}

export interface FLRun {
  id: string;
  name: string;
  status: "running" | "completed" | "failed";
  started_at: string;
  clients: Client[];
  global_metrics: GlobalMetrics;
}
