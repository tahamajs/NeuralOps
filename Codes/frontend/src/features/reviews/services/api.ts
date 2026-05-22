import api from '../../../api/client'

export const reviewsAPI = {
  // Request endpoints
  getRequests: (params?: Record<string, any>) =>
    api.get('/requests/', { params }).then(res => res.data),

  getRequest: (id: string) =>
    api.get(`/requests/${id}/`).then(res => res.data),

  createRequest: (data: any) =>
    api.post('/requests/', data).then(res => res.data),

  updateRequest: (id: string, data: any) =>
    api.patch(`/requests/${id}/`, data).then(res => res.data),

  deleteRequest: (id: string) =>
    api.delete(`/requests/${id}/`),

  submitForReview: (id: string) =>
    api.post(`/requests/${id}/submit/`, {}).then(res => res.data),

  openForReview: (id: string) =>
    api.post(`/requests/${id}/open/`, {}).then(res => res.data),

  completeReviewRound: (id: string) =>
    api.post(`/requests/${id}/complete/`, {}).then(res => res.data),

  getReviewSummary: (id: string) =>
    api.get(`/requests/${id}/summary/`).then(res => res.data),

  // Review endpoints
  getReviewsForRequest: (requestId: string) =>
    api.get(`/requests/${requestId}/review/`).then(res => res.data),

  submitReview: (requestId: string, content: string, rating: number) =>
    api.post(`/requests/${requestId}/review/`, { content, rating }).then(res => res.data),

  updateReview: (reviewId: string, content: string, rating: number) =>
    api.patch(`/reviews/${reviewId}/`, { content, rating }).then(res => res.data),

  deleteReview: (reviewId: string) =>
    api.delete(`/reviews/${reviewId}/`),

  // Stats
  getStats: () =>
    api.get('/reviews/stats/').then(res => res.data),

  // Comments
  getCommentsForReview: (reviewId: string) =>
    api.get(`/reviews/${reviewId}/comment/`).then(res => res.data),

  addComment: (reviewId: string, content: string) =>
    api.post(`/reviews/${reviewId}/comment/`, { content }).then(res => res.data),

  deleteComment: (commentId: string) =>
    api.delete(`/comments/${commentId}/`),

  resolveComment: (commentId: string) =>
    api.post(`/comments/${commentId}/resolve/`, {}).then(res => res.data),

  // Voting
  voteReview: (reviewId: string, helpful: boolean) =>
    api.post(`/reviews/${reviewId}/vote/`, { helpful }).then(res => res.data),
}