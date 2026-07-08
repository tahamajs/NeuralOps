import { request } from "../../../api/client";
import { Recommendation, CreativeVariant, OpportunityIndex } from "../types";

export const recommenderAPI = {
  // Recommendations
  getRecommendations: (params?: { limit?: number; type?: string }) =>
    request.get<{ results: Recommendation[]; count: number }>(
      "/recommender/suggestions/",
      { params },
    ),

  acceptRecommendation: (id: string) =>
    request.post(`/recommender/suggestions/${id}/accept/`),
  dismissRecommendation: (id: string) =>
    request.post(`/recommender/suggestions/${id}/dismiss/`),

  // Opportunity Index
  getOpportunityIndex: () =>
    request.get<OpportunityIndex>("/recommender/opportunity/"),
  recalculateOpportunity: () =>
    request.post("/recommender/opportunity/recalculate/"),

  // Creative Variants
  getVariants: (runId: string) =>
    request.get<CreativeVariant[]>(`/recommender/variants/${runId}/`),
  acceptVariant: (variantId: string) =>
    request.post(`/recommender/variants/accept/${variantId}/`),

  // Regenerate all recommendations
  regenerateAll: () => request.post("/recommender/suggestions/regenerate/"),
};
