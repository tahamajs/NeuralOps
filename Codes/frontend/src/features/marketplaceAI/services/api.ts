import api from "../../../api/client";
import {
  MarketplaceListing,
  ComparisonGroup,
  ComplianceCheck,
  ConversationState,
} from "../types";

export const marketplaceAIApi = {
  // Agent mode endpoints
  startAgentSession: () =>
    api.post<{ session_id: string }>("/marketplace-ai/agent/start/"),
  sendAgentMessage: (sessionId: string, message: string) =>
    api.post<{
      reply: string;
      listings: MarketplaceListing[];
      state: ConversationState;
    }>(`/marketplace-ai/agent/${sessionId}/send/`, { message }),
  getAgentState: (sessionId: string) =>
    api.get<ConversationState>(`/marketplace-ai/agent/${sessionId}/state/`),

  // Listings
  getListings: (params?: Record<string, any>) =>
    api.get<{ results: MarketplaceListing[]; count: number }>(
      "/marketplace-ai/listings/",
      { params },
    ),

  // Comparison builder
  getComparisonGroups: () =>
    api.get<ComparisonGroup[]>("/marketplace-ai/comparisons/"),
  createComparisonGroup: (name: string, listingIds: string[]) =>
    api.post<ComparisonGroup>("/marketplace-ai/comparisons/", {
      name,
      listing_ids: listingIds,
    }),
  deleteComparisonGroup: (id: string) =>
    api.delete(`/marketplace-ai/comparisons/${id}/`),
  addToListingsToComparison: (groupId: string, listingIds: string[]) =>
    api.post(`/marketplace-ai/comparisons/${groupId}/add/`, {
      listing_ids: listingIds,
    }),

  // Compliance
  checkListingCompliance: (listingId: string) =>
    api.get<ComplianceCheck>(
      `/marketplace-ai/listings/${listingId}/compliance/`,
    ),
};
