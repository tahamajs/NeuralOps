export type FailureSeverity = "low" | "medium" | "high";
export type FailureStatus = "open" | "analyzing" | "resolved" | "ignored";

export interface FailureAnalysis {
  id: string;
  run_id: string;
  run_name?: string;
  error_message: string;
  error_type: string;
  stack_trace?: string;
  severity: FailureSeverity;
  status: FailureStatus;
  root_cause: string;
  suggested_fix: string;
  fix_command?: string;
  created_at: string;
  resolved_at?: string;
}

export interface FixAction {
  id: string;
  analysis_id: string;
  description: string;
  command: string;
  parameters: Record<string, any>;
  status: "pending" | "applied" | "failed";
}

export interface ErrorPattern {
  id: string;
  pattern_name: string;
  regex?: string;
  occurrences: number;
  last_seen: string;
  common_fix: string;
}

export interface FailureStats {
  total_failures: number;
  resolved_failures: number;
  unresolved_failures: number;
  most_common_error: string;
  avg_resolution_time_hours: number;
}
