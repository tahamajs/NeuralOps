export interface Anomaly {
  id: string;
  run_id: string;
  run_name: string;
  anomaly_type:
    | "metric_spike"
    | "metric_drop"
    | "log_error"
    | "system_failure"
    | "data_leak";
  severity: "low" | "medium" | "high" | "critical";
  detected_at: string;
  status: "open" | "investigating" | "resolved" | "false_positive";
  summary: string;
  details: {
    metric_name?: string;
    expected_value?: number;
    actual_value?: number;
    log_snippet?: string;
    error_message?: string;
  };
  suggested_fix: string | null;
  resolved_at?: string;
  resolved_by?: string;
}

export interface FixAction {
  id: string;
  anomaly_id: string;
  description: string;
  command: string;
  parameters: Record<string, any>;
  status: "pending" | "applied" | "failed";
  created_at: string;
}
