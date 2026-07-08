export type DriftStatus = "normal" | "warning" | "critical";
export type AlertSeverity = "info" | "warning" | "critical" | "fatal";
export type AlertStatus = "active" | "acknowledged" | "resolved";
export type MetricType =
  "accuracy" | "precision" | "recall" | "f1" | "latency" | "throughput";

export interface DriftMonitor {
  id: string;
  name: string;
  model_id: string;
  model_name: string;
  feature_name: string;
  method: "psi" | "ks" | "emd" | "js";
  alert_threshold: number;
  status: DriftStatus;
  last_check: string;
  created_at: string;
  enabled: boolean;
}

export interface DriftCheck {
  id: string;
  monitor_id: string;
  timestamp: string;
  drift_score: number;
  status: DriftStatus;
  sample_size: number;
  distribution_comparison?: {
    reference_histogram: number[];
    current_histogram: number[];
  };
}

export interface Alert {
  id: string;
  monitor_id: string;
  monitor_name: string;
  severity: AlertSeverity;
  title: string;
  message: string;
  status: AlertStatus;
  acknowledged_by?: string;
  acknowledged_at?: string;
  resolved_by?: string;
  resolved_at?: string;
  created_at: string;
  metadata?: Record<string, any>;
}

export interface SystemHealth {
  overall_status: "healthy" | "degraded" | "unhealthy";
  services: {
    api: ServiceHealth;
    database: ServiceHealth;
    redis: ServiceHealth;
    gpu_scheduler: ServiceHealth;
  };
  resources: {
    cpu_usage_percent: number;
    memory_usage_gb: number;
    memory_total_gb: number;
    disk_usage_percent: number;
    gpu_utilization: number;
  };
  timestamp: string;
}

export interface ServiceHealth {
  status: "healthy" | "degraded" | "unhealthy";
  response_time_ms: number;
  uptime_percent: number;
  last_check: string;
}

export interface PerformanceMetric {
  timestamp: string;
  value: number;
  metric_type: MetricType;
  model_id: string;
}
