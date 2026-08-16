import { request } from "../api/client";
import { ENDPOINTS } from "../api/endpoints";

export interface JobItem {
  id: string;
  name: string;
  model_name?: string;
  task_name?: string;
  attention_type?: string;
  method?: string;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  max_seq_length?: number;
  compression_rank?: number;
  gpu_requirements?: number;
  memory_requirements?: number;
  priority?: number;
  status:
    | "queued"
    | "running"
    | "completed"
    | "failed"
    | "stopped"
    | "paused"
    | "preempted";
  notes?: string;
  checkpoint_path?: string;
  progress?: number;
  current_step?: number;
  total_steps?: number;
  logs?: string;
  config?: Record<string, any>;
  metrics?: Record<string, any>;
  final_metrics?: Record<string, any>;
  training_history?: Record<string, any>;
  estimated_duration?: number | null;
  actual_duration?: number | null;
  git_commit_hash?: string | null;
  git_branch?: string | null;
  git_repository?: string | null;
  git_dirty?: boolean;
  group?: string;
  preemptible?: boolean;
  aging_priority?: number;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  updated_at?: string;
}

export interface CustomMetric {
  id?: string;
  run: string;
  step: number;
  epoch?: number | null;
  name: string;
  type: "scalar" | "image" | "table" | "figure" | "text";
  value: any;
  media_url?: string | null;
  created_at?: string;
}

export interface VersionedConfigItem {
  id?: string;
  run: string;
  config_snapshot: Record<string, any>;
  version_number: number;
  change_description?: string;
  git_commit_hash?: string | null;
  git_branch?: string | null;
  python_version?: string | null;
  created_at: string;
}

export const jobsService = {
  getJobs: (params?: { status?: string; search?: string }) =>
    request.get<JobItem[]>(ENDPOINTS.JOBS.LIST, { params }),

  getJob: (jobId: string) =>
    request.get<JobItem>(ENDPOINTS.JOBS.DETAIL(jobId)),

  pauseJob: (jobId: string) =>
    request.post<{ success: boolean }>(ENDPOINTS.JOBS.PAUSE(jobId)),

  resumeJob: (jobId: string) =>
    request.post<{ success: boolean }>(ENDPOINTS.JOBS.RESUME(jobId)),

  stopJob: (jobId: string) =>
    request.post<{ success: boolean }>(ENDPOINTS.JOBS.STOP(jobId)),

  retryJob: (jobId: string) =>
    request.post<{ success: boolean }>(ENDPOINTS.JOBS.RETRY(jobId)),

  getJobHistory: (jobId: string) =>
    request.get<any[]>(ENDPOINTS.JOBS.HISTORY(jobId)),

  getJobNotes: (jobId: string) =>
    request.get<any>(ENDPOINTS.JOBS.NOTES(jobId)),

  getJobConfigHistory: (jobId: string) =>
    request.get<any[]>(ENDPOINTS.JOBS.CONFIG_HISTORY(jobId)),

  compareJobConfigs: (jobId: string, versionA: number, versionB: number) =>
    request.get<{ diff: Record<string, any> }>(
      ENDPOINTS.JOBS.COMPARE_CONFIGS(jobId, versionA, versionB),
    ),

  getJobCustomMetrics: (jobId: string) =>
    request.get<Record<string, any>>(ENDPOINTS.JOBS.CUSTOM_METRICS(jobId)),

  getJobEnvironment: (jobId: string) =>
    request.get<any>(ENDPOINTS.JOBS.ENVIRONMENT(jobId)),

  getJobScheduling: (jobId: string) =>
    request.get<any>(ENDPOINTS.JOBS.SCHEDULING(jobId)),

  getJobStats: () =>
    request.get<any>(ENDPOINTS.JOBS.STATS),

  bulkDeleteJobs: (jobIds: string[]) =>
    request.post<{ success: boolean }>(ENDPOINTS.JOBS.BULK_DELETE, {
      job_ids: jobIds,
    }),
};
