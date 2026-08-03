import api from "../../../api/client";

export const codingAssistantAPI = {
  createSession: (data: any = {}) => api.post("/code/sessions/", data),
  getSession: (sessionId: string) => api.get(`/code/sessions/${sessionId}/`),
  listFiles: async (sessionId: string) => {
    const res = await api.get(`/code/sessions/${sessionId}/`);
    return res.data.generated_files || [];
  },
  getFileContent: (fileId: string) => api.get(`/code/files/${fileId}/`),
  saveFileContent: (fileId: string, content: string) =>
    api.patch(`/code/files/${fileId}/`, { content }),
  saveFile: (
    sessionId: string,
    data: { name: string; content: string; language: string },
  ) => api.post(`/code/sessions/${sessionId}/files/`, data),
  deleteFile: (fileId: string) => api.delete(`/code/files/${fileId}/`),
  createFile: (sessionId: string, filename: string) =>
    api.post(`/code/sessions/${sessionId}/files/`, { filename, content: "" }),
  generateExperimentConfig: (params: any) =>
    api.post("/code/generate-config/", params),
  getHistory: () => api.get("/code/sessions/"),
  sendMessage: (
    sessionId: string,
    content: string,
    language: string = "python",
  ) => api.post(`/code/sessions/${sessionId}/send/`, { content, language }),
};
