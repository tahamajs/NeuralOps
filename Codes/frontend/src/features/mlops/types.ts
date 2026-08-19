export type FeatureType = "int" | "float" | "string" | "boolean" | "datetime";
export type AlertSeverity = "info" | "warning" | "critical";
export type AlertStatus = "active" | "acknowledged" | "resolved";
export type DriftStatus = "normal" | "warning" | "critical";

export interface FeatureGroup {
  id: string;
  name: string;
  description?: string;
  features: Feature[];
  entity_key: string;
  online_enabled: boolean;
  offline_enabled: boolean;
  tags?: string[];
  version?: string;
  records_count?: number;
  created_at: string;
  updated_at: string;
}

export interface Feature {
  name: string;
  type: FeatureType;
  description?: string;
  stats?: FeatureStats;
}

export interface FeatureStats {
  min: number;
  max: number;
  mean: number;
  std: number;
  null_count: number;
  unique_count?: number;
  histogram?: number[];
  bins?: string[];
}

export interface DriftMonitor {
  id: string;
  name: string;
  feature_name: string;
  feature_group_id: string;
  baseline_window: string; // e.g. '7d'
  alert_threshold: number; // p-value or divergence threshold
  status: DriftStatus;
  last_checked_at: string;
  created_at: string;
  method?: "psi" | "ks" | "emd" | "js";
  current_p_value?: number;
  current_divergence?: number;
}

export interface DriftCheck {
  id: string;
  monitor_id: string;
  timestamp: string;
  p_value: number;
  divergence: number;
  status: DriftStatus;
  sample_size: number;
  reference_histogram?: number[];
  current_histogram?: number[];
}

export interface Alert {
  id: string;
  monitor_id: string;
  severity: AlertSeverity;
  message: string;
  status: AlertStatus;
  created_at: string;
  acknowledged_at?: string;
  resolved_at?: string;
  metadata?: Record<string, any>;
}

export interface AlertChannel {
  id: string;
  type: "slack" | "email" | "webhook" | "pagerduty";
  name: string;
  config: Record<string, any>;
  enabled: boolean;
}

export interface PerformanceSnapshot {
  id: string;
  model_id: string;
  model_name: string;
  timestamp: string;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    latency_p50: number;
    latency_p99: number;
    throughput: number;
  };
  data_drift_score: number;
  concept_drift_score: number;
  feature_stats: Record<string, FeatureStats>;
}

