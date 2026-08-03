import api from "../../../api/client";

export const communityLabAPI = {
  getLabs: () => api.get("/community-lab/labs/").then((res) => res.data),
  createLab: (data: any) =>
    api.post("/community-lab/labs/", data).then((res) => res.data),
  getLab: (labId: string) =>
    api.get(`/community-lab/labs/${labId}/`).then((res) => res.data),
  getLabAgents: (labId: string) =>
    api.get(`/community-lab/labs/${labId}/agents/`).then((res) => res.data),
  createAgent: (labId: string, data: any) =>
    api
      .post(`/community-lab/labs/${labId}/agents/`, data)
      .then((res) => res.data),
  startAgent: (labId: string, agentId: string) =>
    api
      .post(`/community-lab/labs/${labId}/agents/${agentId}/start/`)
      .then((res) => res.data),
  stopAgent: (labId: string, agentId: string) =>
    api
      .post(`/community-lab/labs/${labId}/agents/${agentId}/stop/`)
      .then((res) => res.data),
  getChatrooms: (labId: string) =>
    api.get(`/community-lab/labs/${labId}/chatrooms/`).then((res) => res.data),
  createChatroom: (labId: string, name: string) =>
    api
      .post(`/community-lab/labs/${labId}/chatrooms/`, { name })
      .then((res) => res.data),
  getDiscoveries: (labId: string | null, params: any = {}) => {
    const url = labId
      ? `/community-lab/labs/${labId}/discoveries/`
      : "/community-lab/discoveries/";
    return api.get(url, { params }).then((res) => res.data);
  },
  getReviewQueue: (labId: string) =>
    api.get("/community-lab/discoveries/").then((res) => res.data),
  submitReview: (labId: string, submissionId: string, data: any) =>
    api
      .post(`/community-lab/discoveries/${submissionId}/review/`, data)
      .then((res) => res.data),
};
