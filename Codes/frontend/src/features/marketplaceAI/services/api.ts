import { request } from "../../../api/client";
import {
  MarketplaceListing,
  ComparisonGroup,
  ComplianceCheck,
  ConversationState,
} from "../types";

export const marketplaceAIApi = {
  // Agent mode endpoints
  startAgentSession: () =>
    request.post<{ session_id: string }>("/marketplace-ai/agent/start/"),
  sendAgentMessage: (sessionId: string, message: string) =>
    request.post<{
      reply: string;
      listings: MarketplaceListing[];
      state: ConversationState;
    }>(`/marketplace-ai/agent/${sessionId}/send/`, { message }),
  getAgentState: (sessionId: string) =>
    request.get<ConversationState>(`/marketplace-ai/agent/${sessionId}/state/`),

  // Listings
  getListings: (params?: Record<string, any>) =>
    request.get<{ results: MarketplaceListing[]; count: number }>(
      "/marketplace-ai/listings/",
      { params },
    ),

  // Comparison builder
  getComparisonGroups: () =>
    request.get<ComparisonGroup[]>("/marketplace-ai/comparisons/"),
  createComparisonGroup: (name: string, listingIds: string[]) =>
    request.post<ComparisonGroup>("/marketplace-ai/comparisons/", {
      name,
      listing_ids: listingIds,
    }),
  deleteComparisonGroup: (id: string) =>
    request.delete(`/marketplace-ai/comparisons/${id}/`),
  addToListingsToComparison: (groupId: string, listingIds: string[]) =>
    request.post(`/marketplace-ai/comparisons/${groupId}/add/`, {
      listing_ids: listingIds,
    }),

  // Compliance
  checkListingCompliance: (listingId: string) =>
    request.get<ComplianceCheck>(
      `/marketplace-ai/listings/${listingId}/compliance/`,
    ),
};
