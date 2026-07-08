import api from "../../../api/client";
import { Question, Answer, Comment } from "../types";
type VoteType = any;

export const communityAPI: any = {
  // Questions
  getQuestions: (params?: Record<string, unknown>) =>
    api.get("/questions/", { params }).then((res) => res.data),

  getQuestion: (id: string) =>
    api.get(`/questions/${id}/`).then((res) => res.data),

  createQuestion: (data: { title: string; body: string; tags: string[] }) =>
    api.post("/questions/", data).then((res) => res.data),

  updateQuestion: (id: string, data: Partial<Question>) =>
    api.patch(`/questions/${id}/`, data).then((res) => res.data),

  deleteQuestion: (id: string) => api.delete(`/questions/${id}/`),

  closeQuestion: (id: string, reason: string) =>
    api.post(`/questions/${id}/close/`, { reason }).then((res) => res.data),

  // Answers
  getAnswers: (questionId: string) =>
    api.get(`/questions/${questionId}/answer/`).then((res) => res.data),

  postAnswer: (questionId: string, body: string) =>
    api
      .post(`/questions/${questionId}/answer/`, { body })
      .then((res) => res.data),

  updateAnswer: (answerId: string, body: string) =>
    api.patch(`/answers/${answerId}/`, { body }).then((res) => res.data),

  deleteAnswer: (answerId: string) => api.delete(`/answers/${answerId}/`),

  acceptAnswer: (answerId: string) =>
    api.post(`/answers/${answerId}/accept/`).then((res) => res.data),

  // Voting
  voteQuestion: (questionId: string, value: VoteType) =>
    api
      .post(`/questions/${questionId}/vote/`, { value })
      .then((res) => res.data),

  voteAnswer: (answerId: string, value: VoteType) =>
    api.post(`/answers/${answerId}/vote/`, { value }).then((res) => res.data),

  // Comments
  getComments: (parentType: string, parentId: string) =>
    api.get(`/${parentType}s/${parentId}/comment/`).then((res) => res.data),

  postComment: (parentType: string, parentId: string, content: string) =>
    api
      .post(`/${parentType}s/${parentId}/comment/`, { content })
      .then((res) => res.data),

  deleteComment: (commentId: string) => api.delete(`/comments/${commentId}/`),

  // Bookmarks
  bookmarkQuestion: (questionId: string) =>
    api.post(`/questions/${questionId}/bookmark/`).then((res) => res.data),

  unbookmarkQuestion: (questionId: string) =>
    api.post(`/questions/${questionId}/unbookmark/`).then((res) => res.data),

  // Tags
  getTags: () => api.get("/tags/").then((res) => res.data),

  // Bounties
  startBounty: (questionId: string, amount: number) =>
    api
      .post(`/questions/${questionId}/bounty/`, { amount })
      .then((res) => res.data),
};
