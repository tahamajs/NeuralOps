import { request as api } from "../../../api/client";
import type {
  CitationGraph,
  PathResult,
  Recommendation,
  ResearchLandscape,
  FeedItem,
} from "../types";

export const literatureAPI = {
  // Citation graph
  getGraphForPaper: (paperId: string, depth: number = 2) =>
    api.get<CitationGraph>(`/literature/discovery/graph/${paperId}/`, {
      params: { depth },
    }),
  getGraphForTopic: (topic: string, limit: number = 50) =>
    api.get<CitationGraph>(`/literature/discovery/graph/topic/`, {
      params: { topic, limit },
    }),
  getGraphVisualisation: (paperId: string) =>
    api.get(`/literature/discovery/graph/${paperId}/visualise/`, {
      responseType: "blob",
    }),

  // Path finding
  findPath: (fromId: string, toId: string) =>
    api.get<PathResult>("/literature/discovery/find-path/", {
      params: { from: fromId, to: toId },
    }),
  getSavedPaths: () => api.get<PathResult[]>("/literature/discovery/paths/"),

  // Recommendations
  getRecommendations: (paperId: string) =>
    api.get<Recommendation[]>(`/literature/discovery/recommendations/`, {
      params: { paper_id: paperId },
    }),
  generateRecommendations: (paperId: string) =>
    api.post(`/literature/discovery/recommendations/generate/`, {
      paper_id: paperId,
    }),
  dismissRecommendation: (recId: string) =>
    api.post(`/literature/discovery/recommendations/${recId}/dismiss/`),
  saveRecommendation: (recId: string) =>
    api.post(`/literature/discovery/recommendations/${recId}/save/`),

  // Research landscapes
  getLandscapes: () =>
    api.get<ResearchLandscape[]>("/literature/discovery/landscapes/"),
  getLandscapeDetail: (landscapeId: string) =>
    api.get<any>(
      `/literature/discovery/landscapes/${landscapeId}/`,
    ),
  generateLandscape: (topic: string) =>
    api.post("/literature/discovery/landscapes/generate/", { topic }),

  // Feed
  getFeed: () => api.get<FeedItem[]>("/literature/discovery/feed/"),
  markFeedSeen: (feedId: string) =>
    api.post(`/literature/discovery/feed/${feedId}/seen/`),
};
