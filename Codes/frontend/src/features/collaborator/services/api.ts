import api from "../../../api/client";
import {
  SuggestedCollaborator,
  MatchHistoryEntry,
  MatchFilters,
  PaginatedResponse,
} from "../types";

export const collaboratorAPI: any = {
  // Suggestion endpoints
  getSuggestions: (params?: MatchFilters) =>
    api.get<PaginatedResponse<SuggestedCollaborator>>(
      "/collaborators/suggestions/",
      { params },
    ),
  refreshSuggestions: () => api.post("/collaborators/suggestions/refresh/"),
  getSuggestionDetail: (id: string) =>
    api.get<SuggestedCollaborator>(`/collaborators/suggestions/${id}/`),
  acceptSuggestion: (id: string) =>
    api.post(`/collaborators/suggestions/${id}/accept/`),
  rejectSuggestion: (id: string) =>
    api.post(`/collaborators/suggestions/${id}/reject/`),

  // Match history
  getMatchHistory: (params?: {
    status?: string;
    page?: number;
    limit?: number;
  }) =>
    api.get<PaginatedResponse<MatchHistoryEntry>>("/collaborators/matches/", {
      params,
    }),
  getMatchDetail: (id: string) =>
    api.get<MatchHistoryEntry>(`/collaborators/matches/${id}/`),

  // Connection graph / network
  getNetworkGraph: (userId?: string) =>
    api.get("/collaborators/network/", { params: { user_id: userId } }),

  // Feedback
  rateMatch: (id: string, rating: number) =>
    api.post(`/collaborators/matches/${id}/rate/`, { rating }),
};
