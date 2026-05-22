import api from '../../../api/client'
import type { PaperProject, PaperFile, CompilationResult } from '../types'

export const paperAPI = {
  // Project CRUD
  getProjects: () =>
    api.get('/paper/projects/').then(res => res.data.results),

  getProject: (id: string) =>
    api.get(`/paper/projects/${id}/`).then(res => res.data),

  createProject: (data: { name: string; description?: string; template?: string }) =>
    api.post('/paper/projects/', data).then(res => res.data),

  updateProject: (id: string, data: Partial<PaperProject>) =>
    api.patch(`/paper/projects/${id}/`, data).then(res => res.data),

  deleteProject: (id: string) =>
    api.delete(`/paper/projects/${id}/`),

  // File management
  listFiles: (projectId: string) =>
    api.get(`/paper/projects/${projectId}/files/`).then(res => res.data),

  getFile: (projectId: string, path: string) =>
    api.get(`/paper/projects/${projectId}/files/${encodeURIComponent(path)}/`).then(res => res.data),

  createFile: (projectId: string, path: string, content: string) =>
    api.post(`/paper/projects/${projectId}/files/`, { path, content }).then(res => res.data),

  updateFile: (projectId: string, path: string, content: string) =>
    api.put(`/paper/projects/${projectId}/files/${encodeURIComponent(path)}/`, { content }).then(res => res.data),

  deleteFile: (projectId: string, path: string) =>
    api.delete(`/paper/projects/${projectId}/files/${encodeURIComponent(path)}/`),

  renameFile: (projectId: string, oldPath: string, newPath: string) =>
    api.post(`/paper/projects/${projectId}/files/${encodeURIComponent(oldPath)}/rename/`, { newPath }),

  createDirectory: (projectId: string, path: string) =>
    api.post(`/paper/projects/${projectId}/directories/`, { path }).then(res => res.data),

  // Compilation
  compile: (projectId: string) =>
    api.post(`/paper/projects/${projectId}/compile/`).then(res => res.data),

  getCompilationStatus: (projectId: string, jobId: string) =>
    api.get(`/paper/projects/${projectId}/compile/${jobId}/status/`).then(res => res.data),

  getPdf: (projectId: string) =>
    api.get(`/paper/projects/${projectId}/output/pdf/`, { responseType: 'blob' }).then(res => URL.createObjectURL(res.data)),

  getCompilationLogs: (projectId: string) =>
    api.get(`/paper/projects/${projectId}/output/log/`).then(res => res.data),

  // Templates
  getTemplates: () =>
    api.get('/paper/templates/').then(res => res.data),

  useTemplate: (templateId: string, projectId: string) =>
    api.post(`/paper/templates/${templateId}/use/`, { project_id: projectId }).then(res => res.data),

  // Export
  exportProject: (projectId: string) =>
    api.get(`/paper/projects/${projectId}/export/`, { responseType: 'blob' }).then(res => URL.createObjectURL(res.data)),

  // Git sync
  gitPush: (projectId: string) =>
    api.post(`/paper/projects/${projectId}/git/push/`),

  gitPull: (projectId: string) =>
    api.post(`/paper/projects/${projectId}/git/pull/`),
}