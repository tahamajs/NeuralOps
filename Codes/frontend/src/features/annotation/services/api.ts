import api from "../../../api/client";
import {
  AnnotationProject,
  AnnotationItem,
  ReviewQueueItem,
  ReviewResult,
  ProjectConfig,
} from "../types";

export const annotationAPI = {
  // Projects
  getProjects: () => api.get<AnnotationProject[]>("/annotation/projects/"),
  getProject: (id: string) =>
    api.get<AnnotationProject>(`/annotation/projects/${id}/`),
  createProject: (data: Partial<AnnotationProject>) =>
    api.post<AnnotationProject>("/annotation/projects/", data),
  updateProject: (id: string, data: Partial<AnnotationProject>) =>
    api.patch(`/annotation/projects/${id}/`, data),
  deleteProject: (id: string) => api.delete(`/annotation/projects/${id}/`),

  // Annotation items
  getItems: (projectId: string, params?: { status?: string; page?: number }) =>
    api.get<{ results: AnnotationItem[]; count: number }>(
      `/annotation/projects/${projectId}/items/`,
      { params },
    ),
  getItem: (itemId: string) =>
    api.get<AnnotationItem>(`/annotation/items/${itemId}/`),
  saveAnnotation: (itemId: string, annotation: Partial<any>) =>
    api.post(`/annotation/items/${itemId}/annotations/`, annotation),
  submitItem: (itemId: string) =>
    api.post(`/annotation/items/${itemId}/submit/`),

  // AI pre-labeling
  triggerPreLabeling: (projectId: string, model?: string) =>
    api.post(`/annotation/projects/${projectId}/pre-label/`, { model }),
  getPreLabelStatus: (projectId: string) =>
    api.get(`/annotation/projects/${projectId}/pre-label/status/`),
  applyPreLabel: (itemId: string, preLabelId: string) =>
    api.post(`/annotation/items/${itemId}/apply-pre-label/`, {
      pre_label_id: preLabelId,
    }),

  // Review queue
  getReviewQueue: (projectId: string) =>
    api.get<ReviewQueueItem[]>(
      `/annotation/projects/${projectId}/review/queue/`,
    ),
  submitReview: (projectId: string, result: ReviewResult) =>
    api.post(`/annotation/projects/${projectId}/review/`, result),
};
