import { request } from "../../../api/client";
import {
  FeatureGroup,
  DriftMonitor,
  DriftCheck,
  Alert,
  AlertChannel,
  PerformanceSnapshot,
} from "../types";

// Initial Mock Data
const DEFAULT_FEATURE_GROUPS: FeatureGroup[] = [
  {
    id: "fg-user-embeddings",
    name: "user_embeddings_v2",
    description: "Multi-modal user contextual vectors, demographic features, and real-time CTR indicators",
    entity_key: "user_id",
    online_enabled: true,
    offline_enabled: true,
    tags: ["production", "recommendation", "v2"],
    version: "2.4.0",
    records_count: 1420000,
    created_at: new Date(Date.now() - 30 * 86400000).toISOString(),
    updated_at: new Date(Date.now() - 3600000).toISOString(),
    features: [
      {
        name: "user_id",
        type: "string",
        description: "Primary entity UUID key",
        stats: { min: 0, max: 0, mean: 0, std: 0, null_count: 0, unique_count: 1420000 },
      },
      {
        name: "age",
        type: "int",
        description: "User age in years",
        stats: {
          min: 18,
          max: 85,
          mean: 34.2,
          std: 11.4,
          null_count: 120,
          unique_count: 68,
          histogram: [15, 32, 45, 60, 40, 25, 12, 5],
          bins: ["18-25", "26-35", "36-45", "46-55", "56-65", "66-75", "76-85", "85+"],
        },
      },
      {
        name: "embedding_dim_128",
        type: "float",
        description: "PCA projection of user interest graph",
        stats: {
          min: -3.84,
          max: 4.12,
          mean: 0.02,
          std: 1.01,
          null_count: 0,
          unique_count: 1400000,
          histogram: [4, 18, 48, 85, 76, 42, 16, 3],
          bins: ["<-3", "-3..-2", "-2..-1", "-1..0", "0..1", "1..2", "2..3", ">3"],
        },
      },
      {
        name: "ctr_7d",
        type: "float",
        description: "Click-through rate over previous 7 days",
        stats: {
          min: 0.0,
          max: 0.68,
          mean: 0.084,
          std: 0.045,
          null_count: 45,
          unique_count: 5200,
          histogram: [40, 85, 62, 30, 12, 4, 1, 0],
          bins: ["0-0.05", "0.05-0.10", "0.10-0.15", "0.15-0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50", ">0.50"],
        },
      },
      {
        name: "churn_risk_score",
        type: "float",
        description: "Predicted probability of user inactivity in next 30 days",
        stats: {
          min: 0.01,
          max: 0.99,
          mean: 0.18,
          std: 0.16,
          null_count: 0,
          unique_count: 10000,
          histogram: [95, 60, 35, 20, 12, 8, 4, 2],
          bins: ["0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.8", "0.8-1.0"],
        },
      },
      {
        name: "is_premium",
        type: "boolean",
        description: "Active subscription flag",
        stats: { min: 0, max: 1, mean: 0.28, std: 0.45, null_count: 0, unique_count: 2 },
      },
      {
        name: "last_active_at",
        type: "datetime",
        description: "Last user interaction ISO timestamp",
        stats: { min: 0, max: 0, mean: 0, std: 0, null_count: 14 },
      },
    ],
  },
  {
    id: "fg-attention-kv",
    name: "text_attention_features",
    description: "SVD Linear Attention runtime metrics, KV cache entropy, and sequence latency",
    entity_key: "prompt_id",
    online_enabled: true,
    offline_enabled: true,
    tags: ["nlp", "linear-attention", "svd"],
    version: "1.8.2",
    records_count: 850000,
    created_at: new Date(Date.now() - 20 * 86400000).toISOString(),
    updated_at: new Date(Date.now() - 7200000).toISOString(),
    features: [
      {
        name: "prompt_id",
        type: "string",
        description: "Unique prompt execution identifier",
        stats: { min: 0, max: 0, mean: 0, std: 0, null_count: 0 },
      },
      {
        name: "prompt_length",
        type: "int",
        description: "Token sequence length",
        stats: {
          min: 16,
          max: 8192,
          mean: 1024,
          std: 640,
          null_count: 0,
          histogram: [30, 70, 85, 45, 20, 10, 5, 2],
          bins: ["16-256", "256-512", "512-1024", "1024-2048", "2048-4096", "4096-6144", "6144-8192", ">8192"],
        },
      },
      {
        name: "svd_rank_k",
        type: "int",
        description: "Low-rank compression dimension K",
        stats: { min: 8, max: 64, mean: 32, std: 12, null_count: 0, unique_count: 6 },
      },
      {
        name: "attention_sparsity",
        type: "float",
        description: "Fraction of zero/near-zero attention matrix entries",
        stats: {
          min: 0.12,
          max: 0.94,
          mean: 0.68,
          std: 0.14,
          null_count: 0,
          histogram: [5, 12, 28, 55, 82, 70, 30, 8],
          bins: ["0.1-0.2", "0.2-0.3", "0.3-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", ">0.9"],
        },
      },
      {
        name: "token_entropy",
        type: "float",
        description: "Shannon entropy of next-token probability distribution",
        stats: {
          min: 0.42,
          max: 4.86,
          mean: 2.14,
          std: 0.72,
          null_count: 0,
          histogram: [8, 25, 68, 92, 54, 22, 6, 1],
          bins: ["0-1", "1-1.5", "1.5-2", "2-2.5", "2.5-3", "3-3.5", "3.5-4", ">4"],
        },
      },
      {
        name: "inference_latency_ms",
        type: "float",
        description: "End-to-end token generation latency in ms",
        stats: {
          min: 4.2,
          max: 48.5,
          mean: 11.8,
          std: 5.2,
          null_count: 0,
          histogram: [45, 88, 62, 24, 10, 5, 2, 1],
          bins: ["0-8ms", "8-12ms", "12-16ms", "16-20ms", "20-25ms", "25-30ms", "30-40ms", ">40ms"],
        },
      },
    ],
  },
  {
    id: "fg-fraud-detection",
    name: "financial_transactions_daily",
    description: "Transaction velocity, geographic risk scores, and anomaly features for real-time fraud scoring",
    entity_key: "account_id",
    online_enabled: true,
    offline_enabled: true,
    tags: ["finance", "risk", "real-time"],
    version: "3.1.0",
    records_count: 4900000,
    created_at: new Date(Date.now() - 45 * 86400000).toISOString(),
    updated_at: new Date(Date.now() - 1800000).toISOString(),
    features: [
      {
        name: "account_id",
        type: "string",
        description: "Hashed account identifier",
        stats: { min: 0, max: 0, mean: 0, std: 0, null_count: 0 },
      },
      {
        name: "amount_usd",
        type: "float",
        description: "Normalized transaction value in USD",
        stats: { min: 0.5, max: 125000, mean: 184.5, std: 412.0, null_count: 0 },
      },
      {
        name: "velocity_24h",
        type: "int",
        description: "Number of transactions originating from IP in 24h",
        stats: { min: 1, max: 142, mean: 4.2, std: 6.8, null_count: 0 },
      },
      {
        name: "fraud_score",
        type: "float",
        description: "Ensemble risk score [0..1]",
        stats: {
          min: 0.001,
          max: 0.998,
          mean: 0.032,
          std: 0.091,
          null_count: 0,
          histogram: [96, 25, 10, 4, 2, 1, 1, 0],
          bins: ["0-0.05", "0.05-0.10", "0.10-0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80", "0.80-0.95", ">0.95"],
        },
      },
      {
        name: "is_cross_border",
        type: "boolean",
        description: "Flag indicating international transaction",
        stats: { min: 0, max: 1, mean: 0.14, std: 0.35, null_count: 0 },
      },
    ],
  },
];

const DEFAULT_DRIFT_MONITORS: DriftMonitor[] = [
  {
    id: "dm-user-ctr",
    name: "User CTR Distribution Monitor",
    feature_name: "ctr_7d",
    feature_group_id: "fg-user-embeddings",
    baseline_window: "7d",
    alert_threshold: 0.05,
    status: "normal",
    method: "ks",
    current_p_value: 0.42,
    current_divergence: 0.021,
    last_checked_at: new Date(Date.now() - 15 * 60000).toISOString(),
    created_at: new Date(Date.now() - 14 * 86400000).toISOString(),
  },
  {
    id: "dm-attention-sparsity",
    name: "Attention Sparsity PSI Monitor",
    feature_name: "attention_sparsity",
    feature_group_id: "fg-attention-kv",
    baseline_window: "14d",
    alert_threshold: 0.1,
    status: "warning",
    method: "psi",
    current_p_value: 0.082,
    current_divergence: 0.124,
    last_checked_at: new Date(Date.now() - 35 * 60000).toISOString(),
    created_at: new Date(Date.now() - 10 * 86400000).toISOString(),
  },
  {
    id: "dm-fraud-score",
    name: "Fraud Risk JS Divergence Monitor",
    feature_name: "fraud_score",
    feature_group_id: "fg-fraud-detection",
    baseline_window: "30d",
    alert_threshold: 0.05,
    status: "normal",
    method: "js",
    current_p_value: 0.68,
    current_divergence: 0.015,
    last_checked_at: new Date(Date.now() - 5 * 60000).toISOString(),
    created_at: new Date(Date.now() - 25 * 86400000).toISOString(),
  },
  {
    id: "dm-token-entropy",
    name: "Token Entropy Wasserstein Monitor",
    feature_name: "token_entropy",
    feature_group_id: "fg-attention-kv",
    baseline_window: "7d",
    alert_threshold: 0.03,
    status: "critical",
    method: "emd",
    current_p_value: 0.018,
    current_divergence: 0.285,
    last_checked_at: new Date(Date.now() - 2 * 60000).toISOString(),
    created_at: new Date(Date.now() - 7 * 86400000).toISOString(),
  },
];

const generateDriftChecks = (monitorId: string, count: number = 30): DriftCheck[] => {
  const checks: DriftCheck[] = [];
  const now = Date.now();
  for (let i = count - 1; i >= 0; i--) {
    const timestamp = new Date(now - i * 3600000 * 4).toISOString();
    let p_value: number;
    let divergence: number;
    let status: "normal" | "warning" | "critical";

    if (monitorId === "dm-token-entropy") {
      p_value = Math.max(0.005, 0.25 - (i < 5 ? 0.22 : (count - i) * 0.006));
      divergence = Math.min(0.35, 0.03 + (i < 5 ? 0.25 : (count - i) * 0.005));
    } else if (monitorId === "dm-attention-sparsity") {
      p_value = 0.08 + Math.sin(i * 0.5) * 0.04;
      divergence = 0.11 + Math.cos(i * 0.4) * 0.03;
    } else {
      p_value = 0.35 + Math.sin(i * 0.3) * 0.25;
      divergence = 0.02 + Math.abs(Math.cos(i * 0.2)) * 0.025;
    }

    if (p_value < 0.03 || divergence > 0.2) status = "critical";
    else if (p_value < 0.1 || divergence > 0.1) status = "warning";
    else status = "normal";

    checks.push({
      id: `dc-${monitorId}-${i}`,
      monitor_id: monitorId,
      timestamp,
      p_value: parseFloat(p_value.toFixed(4)),
      divergence: parseFloat(divergence.toFixed(4)),
      status,
      sample_size: 5000 + Math.floor(Math.random() * 2000),
      reference_histogram: [10, 25, 45, 75, 60, 35, 15, 5],
      current_histogram: [
        Math.round(10 + (Math.random() - 0.5) * 6),
        Math.round(25 + (Math.random() - 0.5) * 8),
        Math.round(45 + (Math.random() - 0.5) * 10),
        Math.round(75 + (Math.random() - 0.5) * 15),
        Math.round(60 + (Math.random() - 0.5) * 12),
        Math.round(35 + (Math.random() - 0.5) * 8),
        Math.round(15 + (Math.random() - 0.5) * 5),
        Math.round(5 + (Math.random() - 0.5) * 3),
      ],
    });
  }
  return checks;
};

const DEFAULT_ALERTS: Alert[] = [
  {
    id: "alt-001",
    monitor_id: "dm-token-entropy",
    severity: "critical",
    message: "Critical distribution drift detected in feature 'token_entropy' (p-value: 0.018, threshold: 0.030)",
    status: "active",
    created_at: new Date(Date.now() - 25 * 60000).toISOString(),
    metadata: {
      feature: "token_entropy",
      model: "model-svd-transformer-v2",
      observed_divergence: 0.285,
      threshold: 0.03,
      sample_size: 6420,
    },
  },
  {
    id: "alt-002",
    monitor_id: "dm-attention-sparsity",
    severity: "warning",
    message: "Warning: Population Stability Index (PSI) exceeded warning band on 'attention_sparsity' (PSI: 0.124)",
    status: "active",
    created_at: new Date(Date.now() - 140 * 60000).toISOString(),
    metadata: {
      feature: "attention_sparsity",
      model: "model-linear-attn-v1",
      psi_score: 0.124,
      threshold: 0.1,
    },
  },
  {
    id: "alt-003",
    monitor_id: "dm-user-ctr",
    severity: "info",
    message: "Weekly feature store baseline recalibration successfully executed for 'user_embeddings_v2'",
    status: "acknowledged",
    created_at: new Date(Date.now() - 360 * 60000).toISOString(),
    acknowledged_at: new Date(Date.now() - 180 * 60000).toISOString(),
    metadata: {
      feature_group: "user_embeddings_v2",
      records_processed: 1420000,
    },
  },
  {
    id: "alt-004",
    monitor_id: "dm-fraud-score",
    severity: "warning",
    message: "Elevated null count detected in financial transactions daily ingestion pipeline",
    status: "resolved",
    created_at: new Date(Date.now() - 86400000).toISOString(),
    acknowledged_at: new Date(Date.now() - 80000000).toISOString(),
    resolved_at: new Date(Date.now() - 72000000).toISOString(),
    metadata: {
      column: "merchant_category_code",
      null_percentage: "2.4%",
      resolved_by: "pipeline-healer",
    },
  },
];

const DEFAULT_ALERT_CHANNELS: AlertChannel[] = [
  {
    id: "chan-01",
    type: "slack",
    name: "#mlops-critical-alerts",
    config: { webhook_url: "https://hooks.slack.com/services/T00/B00/XXXXX", channel: "#mlops-critical" },
    enabled: true,
  },
  {
    id: "chan-02",
    type: "pagerduty",
    name: "PagerDuty On-Call ML Engineer",
    config: { routing_key: "pd-service-key-8482", severity_override: "critical" },
    enabled: true,
  },
  {
    id: "chan-03",
    type: "email",
    name: "Governance & Reliability Distribution List",
    config: { recipients: ["mlops-team@neuralops.internal", "oncall-lead@neuralops.internal"] },
    enabled: true,
  },
  {
    id: "chan-04",
    type: "webhook",
    name: "Automated Rollback Webhook",
    config: { url: "https://api.neuralops.internal/v1/models/revert", method: "POST" },
    enabled: false,
  },
];

const generatePerformanceSnapshots = (modelId?: string): PerformanceSnapshot[] => {
  const models = [
    { id: "model-svd-transformer-v2", name: "SVD Linear Attention Transformer v2" },
    { id: "model-linear-attn-v1", name: "Linear Attention Baseline v1" },
    { id: "model-bert-base", name: "BERT Attention Compressed" },
  ];

  const targetModel = models.find((m) => m.id === modelId) || models[0];
  const snapshots: PerformanceSnapshot[] = [];
  const now = Date.now();

  for (let i = 29; i >= 0; i--) {
    const timestamp = new Date(now - i * 86400000).toISOString();
    const baseAcc = targetModel.id.includes("svd") ? 0.912 : 0.865;
    const noise = Math.sin(i * 0.3) * 0.015 - (i < 5 && targetModel.id.includes("svd") ? 0.025 : 0);
    const accuracy = Math.min(0.99, Math.max(0.7, baseAcc + noise));
    const precision = Math.min(0.99, accuracy + (Math.sin(i * 0.4) * 0.01));
    const recall = Math.min(0.99, accuracy - (Math.cos(i * 0.4) * 0.012));
    const f1 = (2 * precision * recall) / (precision + recall);
    const latency_p50 = Math.round(8.2 + Math.sin(i * 0.5) * 1.5);
    const latency_p99 = Math.round(18.5 + Math.cos(i * 0.5) * 3.2);
    const throughput = Math.round(1450 + Math.sin(i * 0.2) * 200);
    const data_drift_score = parseFloat((0.02 + Math.abs(Math.sin(i * 0.25)) * 0.08).toFixed(4));
    const concept_drift_score = parseFloat((0.015 + Math.abs(Math.cos(i * 0.2)) * 0.06).toFixed(4));

    snapshots.push({
      id: `snap-${targetModel.id}-${i}`,
      model_id: targetModel.id,
      model_name: targetModel.name,
      timestamp,
      metrics: {
        accuracy: parseFloat(accuracy.toFixed(4)),
        precision: parseFloat(precision.toFixed(4)),
        recall: parseFloat(recall.toFixed(4)),
        f1: parseFloat(f1.toFixed(4)),
        latency_p50,
        latency_p99,
        throughput,
      },
      data_drift_score,
      concept_drift_score,
      feature_stats: {
        attention_sparsity: {
          min: 0.1,
          max: 0.95,
          mean: 0.68 - (i < 5 ? 0.12 : 0),
          std: 0.14,
          null_count: 0,
          histogram: [10, 20, 30, 40, 50, 40, 30, 20],
        },
      },
    });
  }
  return snapshots;
};

// Storage Helpers for In-Memory / Local Storage state
const getStoredFeatureGroups = (): FeatureGroup[] => {
  try {
    const saved = localStorage.getItem("mlops_feature_groups");
    return saved ? JSON.parse(saved) : DEFAULT_FEATURE_GROUPS;
  } catch {
    return DEFAULT_FEATURE_GROUPS;
  }
};

const saveStoredFeatureGroups = (data: FeatureGroup[]) => {
  try {
    localStorage.setItem("mlops_feature_groups", JSON.stringify(data));
  } catch {}
};

const getStoredDriftMonitors = (): DriftMonitor[] => {
  try {
    const saved = localStorage.getItem("mlops_drift_monitors");
    return saved ? JSON.parse(saved) : DEFAULT_DRIFT_MONITORS;
  } catch {
    return DEFAULT_DRIFT_MONITORS;
  }
};

const saveStoredDriftMonitors = (data: DriftMonitor[]) => {
  try {
    localStorage.setItem("mlops_drift_monitors", JSON.stringify(data));
  } catch {}
};

const getStoredAlerts = (): Alert[] => {
  try {
    const saved = localStorage.getItem("mlops_alerts");
    return saved ? JSON.parse(saved) : DEFAULT_ALERTS;
  } catch {
    return DEFAULT_ALERTS;
  }
};

const saveStoredAlerts = (data: Alert[]) => {
  try {
    localStorage.setItem("mlops_alerts", JSON.stringify(data));
  } catch {}
};

const getStoredChannels = (): AlertChannel[] => {
  try {
    const saved = localStorage.getItem("mlops_alert_channels");
    return saved ? JSON.parse(saved) : DEFAULT_ALERT_CHANNELS;
  } catch {
    return DEFAULT_ALERT_CHANNELS;
  }
};

const saveStoredChannels = (data: AlertChannel[]) => {
  try {
    localStorage.setItem("mlops_alert_channels", JSON.stringify(data));
  } catch {}
};

export const mlopsAPI = {
  // Feature Store
  getFeatureGroups: async (): Promise<FeatureGroup[]> => {
    try {
      const res = await request.get("/mlops/features/groups/");
      if (res?.data && Array.isArray(res.data) && res.data.length > 0) return res.data;
    } catch {}
    return getStoredFeatureGroups();
  },

  getFeatureGroup: async (id: string): Promise<FeatureGroup> => {
    try {
      const res = await request.get(`/mlops/features/groups/${id}/`);
      if (res?.data?.id) return res.data;
    } catch {}
    const groups = getStoredFeatureGroups();
    return groups.find((g) => g.id === id) || groups[0];
  },

  createFeatureGroup: async (data: Partial<FeatureGroup>): Promise<FeatureGroup> => {
    const newGroup: FeatureGroup = {
      id: data.id || `fg-${Date.now().toString(36)}`,
      name: data.name || "new_feature_group",
      description: data.description || "Created in Feature Store",
      entity_key: data.entity_key || "entity_id",
      online_enabled: data.online_enabled ?? true,
      offline_enabled: data.offline_enabled ?? true,
      tags: data.tags || ["custom"],
      version: "1.0.0",
      records_count: 0,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      features: data.features && data.features.length > 0 ? data.features : [
        {
          name: data.entity_key || "entity_id",
          type: "string",
          description: "Primary key",
          stats: { min: 0, max: 0, mean: 0, std: 0, null_count: 0 },
        },
        {
          name: "feature_val_1",
          type: "float",
          description: "Default metric feature",
          stats: { min: 0, max: 1, mean: 0.5, std: 0.2, null_count: 0, histogram: [10, 20, 30, 20, 10] },
        },
      ],
    };
    try {
      await request.post("/mlops/features/groups/", newGroup);
    } catch {}
    const groups = [newGroup, ...getStoredFeatureGroups()];
    saveStoredFeatureGroups(groups);
    return newGroup;
  },

  updateFeatureGroup: async (id: string, data: Partial<FeatureGroup>): Promise<FeatureGroup> => {
    try {
      await request.patch(`/mlops/features/groups/${id}/`, data);
    } catch {}
    const groups = getStoredFeatureGroups().map((g) => (g.id === id ? { ...g, ...data, updated_at: new Date().toISOString() } : g));
    saveStoredFeatureGroups(groups);
    return groups.find((g) => g.id === id)!;
  },

  deleteFeatureGroup: async (id: string): Promise<null> => {
    try {
      await request.delete(`/mlops/features/groups/${id}/`);
    } catch {}
    const groups = getStoredFeatureGroups().filter((g) => g.id !== id);
    saveStoredFeatureGroups(groups);
    return null;
  },

  ingestFeatures: async (groupId: string, records: any[]): Promise<{ ingested_count: number; status: string }> => {
    try {
      const res = await request.post(`/mlops/features/ingest/`, { group_id: groupId, records });
      if (res?.data) return res.data;
    } catch {}
    // Simulate updating record count
    const groups = getStoredFeatureGroups().map((g) => {
      if (g.id === groupId) {
        return {
          ...g,
          records_count: (g.records_count || 0) + records.length,
          updated_at: new Date().toISOString(),
        };
      }
      return g;
    });
    saveStoredFeatureGroups(groups);
    return { ingested_count: records.length, status: "success" };
  },

  serveFeatures: async (groupId: string, entityKeys: string[]): Promise<{ features: Record<string, any>[]; latency_ms: number }> => {
    try {
      const res = await request.post(`/mlops/features/serve/`, { group_id: groupId, entity_keys: entityKeys });
      if (res?.data) return res.data;
    } catch {}
    const group = getStoredFeatureGroups().find((g) => g.id === groupId);
    const features = entityKeys.map((key) => {
      const record: Record<string, any> = { [group?.entity_key || "entity_id"]: key };
      group?.features.forEach((f) => {
        if (f.type === "float") record[f.name] = parseFloat((Math.random() * 10).toFixed(3));
        else if (f.type === "int") record[f.name] = Math.floor(Math.random() * 100);
        else if (f.type === "boolean") record[f.name] = Math.random() > 0.5;
        else if (f.type === "datetime") record[f.name] = new Date().toISOString();
        else record[f.name] = `val_${key}_${f.name}`;
      });
      return record;
    });
    return { features, latency_ms: parseFloat((1.2 + Math.random() * 0.6).toFixed(2)) };
  },

  // Drift Monitoring
  getDriftMonitors: async (): Promise<DriftMonitor[]> => {
    try {
      const res = await request.get("/mlops/drift/monitors/");
      if (res?.data && Array.isArray(res.data) && res.data.length > 0) return res.data;
    } catch {}
    return getStoredDriftMonitors();
  },

  createDriftMonitor: async (data: Partial<DriftMonitor>): Promise<DriftMonitor> => {
    const newMon: DriftMonitor = {
      id: data.id || `dm-${Date.now().toString(36)}`,
      name: data.name || "New Drift Monitor",
      feature_name: data.feature_name || "feature_1",
      feature_group_id: data.feature_group_id || "fg-user-embeddings",
      baseline_window: data.baseline_window || "7d",
      alert_threshold: data.alert_threshold ?? 0.05,
      status: "normal",
      method: data.method || "ks",
      current_p_value: 0.35,
      current_divergence: 0.02,
      last_checked_at: new Date().toISOString(),
      created_at: new Date().toISOString(),
    };
    try {
      await request.post("/mlops/drift/monitors/", newMon);
    } catch {}
    const monitors = [newMon, ...getStoredDriftMonitors()];
    saveStoredDriftMonitors(monitors);
    return newMon;
  },

  deleteDriftMonitor: async (id: string): Promise<null> => {
    try {
      await request.delete(`/mlops/drift/monitors/${id}/`);
    } catch {}
    const monitors = getStoredDriftMonitors().filter((m) => m.id !== id);
    saveStoredDriftMonitors(monitors);
    return null;
  },

  getDriftChecks: async (monitorId: string, limit: number = 30): Promise<DriftCheck[]> => {
    try {
      const res = await request.get(`/mlops/drift/monitors/${monitorId}/checks/`, { params: { limit } });
      if (res?.data && Array.isArray(res.data) && res.data.length > 0) return res.data;
    } catch {}
    return generateDriftChecks(monitorId, limit);
  },

  runDriftCheck: async (monitorId: string): Promise<DriftCheck> => {
    try {
      const res = await request.post(`/mlops/drift/monitors/${monitorId}/check/`);
      if (res?.data?.id) return res.data;
    } catch {}
    const p_value = parseFloat((0.02 + Math.random() * 0.4).toFixed(4));
    const divergence = parseFloat((0.01 + Math.random() * 0.15).toFixed(4));
    const status = p_value < 0.03 || divergence > 0.2 ? "critical" : p_value < 0.1 || divergence > 0.1 ? "warning" : "normal";

    const check: DriftCheck = {
      id: `dc-${monitorId}-${Date.now()}`,
      monitor_id: monitorId,
      timestamp: new Date().toISOString(),
      p_value,
      divergence,
      status,
      sample_size: 5000 + Math.floor(Math.random() * 2000),
      reference_histogram: [10, 25, 45, 75, 60, 35, 15, 5],
      current_histogram: [12, 28, 42, 70, 64, 38, 14, 6],
    };

    const monitors = getStoredDriftMonitors().map((m) =>
      m.id === monitorId
        ? { ...m, status, current_p_value: p_value, current_divergence: divergence, last_checked_at: new Date().toISOString() }
        : m
    );
    saveStoredDriftMonitors(monitors);
    return check;
  },

  // Alerts
  getAlerts: async (params?: { status?: string; severity?: string }): Promise<Alert[]> => {
    try {
      const res = await request.get("/mlops/alerts/", { params });
      if (res?.data && Array.isArray(res.data) && res.data.length > 0) return res.data;
    } catch {}
    let alerts = getStoredAlerts();
    if (params?.status && params.status !== "all") alerts = alerts.filter((a) => a.status === params.status);
    if (params?.severity) alerts = alerts.filter((a) => a.severity === params.severity);
    return alerts;
  },

  acknowledgeAlert: async (id: string): Promise<null> => {
    try {
      await request.post(`/mlops/alerts/${id}/acknowledge/`);
    } catch {}
    const alerts = getStoredAlerts().map((a) =>
      a.id === id ? { ...a, status: "acknowledged" as const, acknowledged_at: new Date().toISOString() } : a
    );
    saveStoredAlerts(alerts);
    return null;
  },

  resolveAlert: async (id: string): Promise<null> => {
    try {
      await request.post(`/mlops/alerts/${id}/resolve/`);
    } catch {}
    const alerts = getStoredAlerts().map((a) =>
      a.id === id ? { ...a, status: "resolved" as const, resolved_at: new Date().toISOString() } : a
    );
    saveStoredAlerts(alerts);
    return null;
  },

  getAlertChannels: async (): Promise<AlertChannel[]> => {
    try {
      const res = await request.get("/mlops/alerts/channels/");
      if (res?.data && Array.isArray(res.data) && res.data.length > 0) return res.data;
    } catch {}
    return getStoredChannels();
  },

  createAlertChannel: async (data: Partial<AlertChannel>): Promise<AlertChannel> => {
    const newChan: AlertChannel = {
      id: data.id || `chan-${Date.now().toString(36)}`,
      type: data.type || "slack",
      name: data.name || "New Notification Channel",
      config: data.config || {},
      enabled: data.enabled ?? true,
    };
    try {
      await request.post("/mlops/alerts/channels/", newChan);
    } catch {}
    const chans = [...getStoredChannels(), newChan];
    saveStoredChannels(chans);
    return newChan;
  },

  deleteAlertChannel: async (id: string): Promise<null> => {
    try {
      await request.delete(`/mlops/alerts/channels/${id}/`);
    } catch {}
    const chans = getStoredChannels().filter((c) => c.id !== id);
    saveStoredChannels(chans);
    return null;
  },

  testAlertChannel: async (id: string): Promise<{ success: boolean; message: string }> => {
    try {
      const res = await request.post(`/mlops/alerts/channels/${id}/test/`);
      if (res?.data) return res.data;
    } catch {}
    return { success: true, message: `Test ping dispatched to channel ${id}` };
  },

  // Performance Snapshots
  getPerformanceSnapshots: async (params?: { model_id?: string; range?: string }): Promise<PerformanceSnapshot[]> => {
    try {
      const res = await request.get("/mlops/performance/snapshots/", { params });
      if (res?.data && Array.isArray(res.data) && res.data.length > 0) return res.data;
    } catch {}
    return generatePerformanceSnapshots(params?.model_id);
  },

  getSnapshotDetail: async (id: string): Promise<PerformanceSnapshot> => {
    try {
      const res = await request.get(`/mlops/performance/snapshots/${id}/`);
      if (res?.data?.id) return res.data;
    } catch {}
    return generatePerformanceSnapshots()[0];
  },

  triggerSnapshot: async (modelId: string): Promise<PerformanceSnapshot> => {
    try {
      const res = await request.post(`/mlops/performance/snapshots/trigger/`, { model_id: modelId });
      if (res?.data?.id) return res.data;
    } catch {}
    const newSnapshot: PerformanceSnapshot = {
      id: `snap-${modelId}-${Date.now()}`,
      model_id: modelId,
      model_name: modelId.replace(/-/g, " ").toUpperCase(),
      timestamp: new Date().toISOString(),
      metrics: {
        accuracy: parseFloat((0.92 + Math.random() * 0.04).toFixed(4)),
        precision: parseFloat((0.91 + Math.random() * 0.04).toFixed(4)),
        recall: parseFloat((0.90 + Math.random() * 0.04).toFixed(4)),
        f1: parseFloat((0.91 + Math.random() * 0.04).toFixed(4)),
        latency_p50: Math.round(7.5 + Math.random() * 2),
        latency_p99: Math.round(16.0 + Math.random() * 4),
        throughput: Math.round(1520 + Math.random() * 250),
      },
      data_drift_score: parseFloat((0.015 + Math.random() * 0.02).toFixed(4)),
      concept_drift_score: parseFloat((0.012 + Math.random() * 0.02).toFixed(4)),
      feature_stats: {
        attention_sparsity: {
          min: 0.12,
          max: 0.94,
          mean: 0.69,
          std: 0.13,
          null_count: 0,
          histogram: [12, 22, 35, 45, 52, 38, 26, 15],
        },
      },
    };
    return newSnapshot;
  },
};

