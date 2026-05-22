// Add to existing socialAPI object

export const socialAPI = {
  // ... existing methods (follow/unfollow, groups, feed, etc.)

  // Group discussions
  getDiscussionThreads: (groupId: string) =>
    api.get<DiscussionThread[]>(`/social/groups/${groupId}/discussions/`),
  getDiscussion: (discussionId: string) =>
    api.get<Discussion>(`/social/discussions/${discussionId}/`),
  createDiscussion: (groupId: string, title: string, content: string) =>
    api.post(`/social/groups/${groupId}/discussions/`, { title, content }),
  postReply: (discussionId: string, content: string) =>
    api.post(`/social/discussions/${discussionId}/reply/`, { content }),
  likeDiscussion: (discussionId: string) =>
    api.post(`/social/discussions/${discussionId}/like/`),

  // Alerts
  getAlerts: () => api.get<Alert[]>('/social/alerts/'),
  markAlertRead: (id: string) => api.post(`/social/alerts/${id}/read/`),
  markAllAlertsRead: () => api.post('/social/alerts/mark-all-read/'),

  // Suggested researchers
  getSuggestedResearchers: () => api.get<SuggestedResearcher[]>('/social/suggested/'),
  dismissSuggestion: (id: string) => api.post(`/social/suggested/${id}/dismiss/`),
}