import api from "../../../api/client";
import {
  ExpertProfile,
  Project,
  Application,
  FilterOptions,
  PaginatedResponse,
} from "../types";

export const expertWallAPI = {
  // Experts
  getExperts: (params?: FilterOptions) =>
    api.get<PaginatedResponse<ExpertProfile>>("/experts/", { params }),
  getExpertDetail: (id: string) => api.get<ExpertProfile>(`/experts/${id}/`),

  // Projects
  getProjects: (params?: { status?: string; expertise?: string[] }) =>
    api.get<PaginatedResponse<Project>>("/projects/", { params }),
  getProjectDetail: (id: string) => api.get<Project>(`/projects/${id}/`),
  createProject: (data: Partial<Project>) =>
    api.post<Project>("/projects/", data),

  // Applications
  submitApplication: (projectId: string, data: Partial<Application>) =>
    api.post<Application>(`/projects/${projectId}/apply/`, data),
  getUserApplications: () => api.get<Application[]>("/applications/me/"),
};
