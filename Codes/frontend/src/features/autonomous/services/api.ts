import api from '../../../api/client'
import { PipelineRun, WorkflowDefinition, StepTemplate } from '../types'

export const autonomousAPI = {
  // Pipeline runs
  getPipelineRuns: (params?: { status?: string; limit?: number }) =>
    api.get<{ results: PipelineRun[]; count: number }>('/research/pipelines/', { params }),
  getPipelineRun: (runId: string) => api.get<PipelineRun>(`/research/pipelines/${runId}/`),
  startPipeline: (definitionId: string, inputs?: Record<string, any>) =>
    api.post<{ run_id: string }>('/research/pipelines/start/', { definition_id: definitionId, inputs }),
  pausePipeline: (runId: string) => api.post(`/research/pipelines/${runId}/pause/`),
  resumePipeline: (runId: string) => api.post(`/research/pipelines/${runId}/resume/`),
  cancelPipeline: (runId: string) => api.post(`/research/pipelines/${runId}/cancel/`),
  retryStep: (runId: string, stepId: string) => api.post(`/research/pipelines/${runId}/steps/${stepId}/retry/`),

  // Workflow definitions
  getWorkflowDefinitions: () => api.get<WorkflowDefinition[]>('/research/definitions/'),
  getWorkflowDefinition: (id: string) => api.get<WorkflowDefinition>(`/research/definitions/${id}/`),
  createWorkflowDefinition: (data: Partial<WorkflowDefinition>) =>
    api.post<WorkflowDefinition>('/research/definitions/', data),
  updateWorkflowDefinition: (id: string, data: Partial<WorkflowDefinition>) =>
    api.patch<WorkflowDefinition>(`/research/definitions/${id}/`, data),
  deleteWorkflowDefinition: (id: string) => api.delete(`/research/definitions/${id}/`),

  // Step templates
  getStepTemplates: () => api.get<StepTemplate[]>('/research/templates/'),
}