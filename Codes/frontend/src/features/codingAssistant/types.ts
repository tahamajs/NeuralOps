import api from "../../../api/client";
import {
  Message,
  GeneratedFile,
  TestResult,
  DocumentationItem,
} from "../types";

export const codingAssistantAPI = {
  // Sessions
  createSession: (config?: { language?: string; framework?: string }) =>
    api.post<{ session_id: string }>("/code/sessions/", config),

  getSession: (sessionId: string) => api.get(`/code/sessions/${sessionId}/`),

  // Chat & Code generation
  sendMessage: (sessionId: string, message: string) =>
    api.post<Message>(`/code/sessions/${sessionId}/send/`, { message }),

  streamMessage: (sessionId: string, message: string) =>
    api.post(
      `/code/sessions/${sessionId}/stream/`,
      { message },
      { responseType: "stream" },
    ),

  // File management
  listFiles: (sessionId: string) =>
    api.get<GeneratedFile[]>(`/code/sessions/${sessionId}/files/`),
  getFileContent: (fileId: string) =>
    api.get<{ content: string }>(`/code/files/${fileId}/`),
  saveFileContent: (fileId: string, content: string) =>
    api.put(`/code/files/${fileId}/`, { content }),
  deleteFile: (fileId: string) => api.delete(`/code/files/${fileId}/`),

  // Code review
  requestReview: (fileId: string) =>
    api.post<CodeSuggestion[]>(`/code/files/${fileId}/review/`),
  getReviews: (fileId: string) =>
    api.get<CodeSuggestion[]>(`/code/files/${fileId}/reviews/`),

  // Testing
  generateTests: (fileId: string) => api.post(`/code/files/${fileId}/tests/`),
  getTests: (fileId: string) =>
    api.get<TestResult[]>(`/code/files/${fileId}/tests/get/`),

  // Documentation
  generateDocs: (fileId: string) =>
    api.post<DocumentationItem>(`/code/files/${fileId}/document/`),

  // Experiment config generation
  generateExperimentConfig: (params: any) =>
    api.post("/code/generate-config/", params),
};
