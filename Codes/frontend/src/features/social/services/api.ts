import { request } from "../../../api/client";
import {
  DiscussionThread,
  Discussion,
  Alert,
  SuggestedResearcher,
} from "../types";

export const socialAPI = {
  // Group discussions
  getDiscussionThreads: (groupId: string) =>
    request.get<DiscussionThread[]>(`/social/groups/${groupId}/discussions/`),
  getDiscussion: (discussionId: string) =>
    request.get<Discussion>(`/social/discussions/${discussionId}/`),
  createDiscussion: (groupId: string, title: string, content: string) =>
    request.post(`/social/groups/${groupId}/discussions/`, { title, content }),
  postReply: (discussionId: string, content: string) =>
    request.post(`/social/discussions/${discussionId}/reply/`, { content }),
  likeDiscussion: (discussionId: string) =>
    request.post(`/social/discussions/${discussionId}/like/`),

  // Alerts
  getAlerts: () => request.get<Alert[]>("/social/alerts/"),
  markAlertRead: (id: string) => request.post(`/social/alerts/${id}/read/`),
  markAllAlertsRead: () => request.post("/social/alerts/mark-all-read/"),

  // Suggested researchers
  getSuggestedResearchers: () =>
    request.get<SuggestedResearcher[]>("/social/suggested/"),
  dismissSuggestion: (id: string) =>
    request.post(`/social/suggested/${id}/dismiss/`),
};
