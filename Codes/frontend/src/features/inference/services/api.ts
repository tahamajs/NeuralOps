import api from '../../../api/client'
import {
  InferenceEndpoint,
  PipelineRun,
  CTCTrigger,
  DistillationConfig,
  MultiNodeConfig,
} from '../types'

export const inferenceAPI = {
  // Endpoints
  getEndpoints: () => api.get<InferenceEndpoint[]>('/inference/endpoints/'),
  getEndpoint: (id: string) => api.get<InferenceEndpoint>(`/inference/endpoints/${id}/`),
  createEndpoint: (data: Partial<InferenceEndpoint>) => api.post<InferenceEndpoint>('/inference/endpoints/', data),
  updateEndpoint: (id: string, data: Partial<InferenceEndpoint>) => api.patch(`/inference/endpoints/${id}/`, data),
  deleteEndpoint: (id: string) => api.delete(`/inference/endpoints/${id}/`),
  redeployEndpoint: (id: string) => api.post(`/inference/endpoints/${id}/redeploy/`),

  // Pipelines
  getPipelineRuns: () => api.get<PipelineRun[]>('/inference/pipelines/'),
  getPipelineRun: (id: string) => api.get<PipelineRun>(`/inference/pipelines/${id}/`),
  triggerPipeline: (pipelineId: string, params?: any) => api.post(`/inference/pipelines/${pipelineId}/trigger/`, params),
  stopPipeline: (id: string) => api.post(`/inference/pipelines/${id}/stop/`),

  // CTC (Continuous Training / Continuous Deployment)
  getCTCTriggers: () => api.get<CTCTrigger[]>('/inference/ctc/triggers/'),
  createCTCTrigger: (data: Partial<CTCTrigger>) => api.post<CTCTrigger>('/inference/ctc/triggers/', data),
  updateCTCTrigger: (id: string, data: Partial<CTCTrigger>) => api.patch(`/inference/ctc/triggers/${id}/`, data),
  deleteCTCTrigger: (id: string) => api.delete(`/inference/ctc/triggers/${id}/`),
  testCTCTrigger: (id: string) => api.post(`/inference/ctc/triggers/${id}/test/`),

  // Distillation
  startDistillation: (config: DistillationConfig) => api.post<PipelineRun>('/inference/distillation/start/', config),
  getDistillationStatus: (runId: string) => api.get(`/inference/distillation/${runId}/status/`),

  // Multi‑node training
  launchMultiNodeTraining: (config: MultiNodeConfig) => api.post<PipelineRun>('/inference/multi-node/launch/', config),
  getTrainingLogs: (runId: string) => api.get(`/inference/multi-node/${runId}/logs/`),
}