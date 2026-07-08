import api from "../../../api/client";

export const communityLabAPI = {
  getLabs: () => api.get("/community-lab/labs/"),
  createLab: (data: any) => api.post("/community-lab/labs/", data),
  getLab: (labId: string) => api.get(`/community-lab/labs/${labId}/`),
  getLabAgents: (labId: string) => api.get(`/community-lab/labs/${labId}/agents/`),
  createAgent: (labId: string, data: any) => api.post(`/community-lab/labs/${labId}/agents/`, data),
  startAgent: (labId: string, agentId: string) => Promise.resolve({ status: "active" }),
  stopAgent: (labId: string, agentId: string) => Promise.resolve({ status: "inactive" }),
  getChatrooms: (labId: string) => api.get(`/community-lab/labs/${labId}/chatrooms/`),
  createChatroom: (labId: string, name: string) => api.post(`/community-lab/labs/${labId}/chatrooms/`, { name }),
  getDiscoveries: (labId: string | null, params: any = {}) => {
    const url = labId ? `/community-lab/labs/${labId}/discoveries/` : "/community-lab/discoveries/";
    return api.get(url, { params });
  },
  getReviewQueue: (labId: string) => api.get("/community-lab/discoveries/"),
  submitReview: (labId: string, submissionId: string, data: any) =>
    api.post(`/community-lab/discoveries/${submissionId}/review/`, data),
};
