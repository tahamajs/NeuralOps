import api from '../../../api/client'
import { Recommendation, CreativeVariant, OpportunityIndex } from '../types'

export const recommenderAPI = {
  // Recommendations
  getRecommendations: (params?: { limit?: number; type?: string }) =>
    api.get<{ results: Recommendation[]; count: number }>('/recommender/suggestions/', { params }),

  acceptRecommendation: (id: string) => api.post(`/recommender/suggestions/${id}/accept/`),
  dismissRecommendation: (id: string) => api.post(`/recommender/suggestions/${id}/dismiss/`),

  // Opportunity Index
  getOpportunityIndex: () => api.get<OpportunityIndex>('/recommender/opportunity/'),
  recalculateOpportunity: () => api.post('/recommender/opportunity/recalculate/'),

  // Creative Variants
  getVariants: (runId: string) => api.get<CreativeVariant[]>(`/recommender/variants/${runId}/`),
  acceptVariant: (variantId: string) => api.post(`/recommender/variants/accept/${variantId}/`),

  // Regenerate all recommendations
  regenerateAll: () => api.post('/recommender/suggestions/regenerate/'),
}