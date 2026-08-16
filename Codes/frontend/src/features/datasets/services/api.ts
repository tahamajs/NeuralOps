import { request } from "../../../api/client";

export interface DatasetVersion {
  id: string;
  version_number: number;
  description: string;
  file_path: string;
  size_bytes: number;
  row_count: number;
  column_count: number;
  schema: Record<string, string>;
  sample_data: any[];
  dvc_hash: string;
  dvc_lock: string;
  sha256: string;
  md5: string;
  created_at: string;
}

export interface Dataset {
  id: string;
  name: string;
  description: string;
  status: "draft" | "published" | "deprecated" | "archived";
  is_public: boolean;
  dvc_remote: string;
  dvc_revision: string;
  tags: string[];
  license: string;
  citation: string;
  created_at: string;
  updated_at: string;
  latest_version?: DatasetVersion;
  versions?: DatasetVersion[];
  size_bytes?: number;
  row_count?: number;
}

export interface Artifact {
  id: string;
  name: string;
  artifact_type:
    | "dataset"
    | "model"
    | "checkpoint"
    | "log"
    | "plot"
    | "report"
    | "config"
    | "code"
    | "other";
  run?: string;
  file_path: string;
  file_url: string;
  size_bytes: number;
  sha256: string;
  md5: string;
  metadata: Record<string, any>;
  tags: string[];
  created_at: string;
}

export interface AuditLog {
  id: string;
  action_type: string;
  resource_type: string;
  resource_id: string;
  description: string;
  severity: "info" | "warning" | "error" | "critical";
  ip_address: string;
  user_agent: string;
  created_at: string;
}

export const datasetsAPI = {
  getDatasets: (params?: { q?: string; tag?: string }) =>
    request.get<{ results: Dataset[] }>("/datasets/", { params }),
  getDataset: (id: string) => request.get<Dataset>(`/datasets/${id}/`),
  createDataset: (data: Partial<Dataset>) =>
    request.post<Dataset>("/datasets/", data),
  updateDataset: (id: string, data: Partial<Dataset>) =>
    request.patch<Dataset>(`/datasets/${id}/`, data),
  deleteDataset: (id: string) => request.delete(`/datasets/${id}/`),
  exportDatasetUrl: (id: string) =>
    `${(import.meta as any).env.VITE_API_URL || "http://localhost:8000/api"}/datasets/${id}/export/`,

  // Artifacts & Lineage
  getArtifacts: (params?: { type?: string; run_id?: string; q?: string }) =>
    request.get<{ results: Artifact[] }>("/datasets/artifacts/", { params }),
  getArtifactLineage: (id: string) =>
    request.get<{ nodes: any[]; edges: any[] }>(
      `/datasets/artifacts/${id}/lineage/`,
    ),

  // Audit Logs
  getAuditLogs: (params?: {
    action_type?: string;
    resource_type?: string;
    severity?: string;
  }) => request.get<{ results: AuditLog[] }>("/datasets/audit-logs/", { params }),
  requestAuditExport: (data: {
    format: "csv" | "json" | "pdf";
    filters?: Record<string, any>;
  }) => request.post("/datasets/audit-logs/export/", data),
};
