import { request } from "../api/client";
import { ENDPOINTS } from "../api/endpoints";

export interface JobItem {
  id: string;
  name: string;
  status: "queued" | "running" | "completed" | "failed" | "stopped" | "paused";
  created_at: string;
  updated_at?: string;
  config?: Record<string, any>;
  metrics?: Record<string, any>;
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
