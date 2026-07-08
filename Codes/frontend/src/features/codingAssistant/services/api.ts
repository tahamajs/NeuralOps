import api from "../../../api/client";

export const codingAssistantAPI = {
  createSession: (data: any = {}) => api.post("/code/sessions/", data),
  saveFile: (fileId: string, content: string) =>
    api.patch(`/code/files/${fileId}/`, { content }),
  getFileContent: (fileId: string) => api.get(`/code/files/${fileId}/`),
  generateExperimentConfig: (params: any) =>
    api.post("/code/generate-config/", params),
  getHistory: () => api.get("/code/sessions/"),
  sendMessage: (sessionId: string, content: string, language: string = "python") =>
    api.post(`/code/sessions/${sessionId}/send/`, { content, language }),
};
