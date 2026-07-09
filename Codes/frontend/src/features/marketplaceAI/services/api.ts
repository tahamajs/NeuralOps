import { request } from "../../../api/client";
import {
  MarketplaceListing,
  ComparisonGroup,
  ComplianceCheck,
  ConversationState,
} from "../types";

export const marketplaceAIApi = {
  // Agent mode endpoints
  // Agent mode endpoints (Mocked)
  startAgentSession: async () => ({ data: { session_id: "dummy-session-123" } }),
  sendAgentMessage: async (sessionId: string, message: string) => ({
    data: { reply: "I am a mocked agent response.", listings: [], state: { step: "idle", current_task: "Mocking", active_filters: {}, results: [] } }
  }),
  getAgentState: async (sessionId: string) => ({
    data: { step: "idle", current_task: "Mocking", active_filters: {}, results: [] }
  }),

  // Listings (Mocked)
  getListings: async (params?: Record<string, any>) => ({
    data: { results: [], count: 0 }
  }),

  // Comparison builder (Mocked)
  getComparisonGroups: async () => ({ data: [] }),
  createComparisonGroup: async (name: string, listingIds: string[]) => ({
    data: { id: "123", name, items: listingIds } as any
  }),
  deleteComparisonGroup: async (id: string) => ({ data: null }),
  addToListingsToComparison: async (groupId: string, listingIds: string[]) => ({ data: null }),

  // Compliance (Mocked)
  checkListingCompliance: async (listingId: string) => ({
    data: { score: 100, certifications: [], last_checked: new Date().toISOString() } as any
  }),
};
