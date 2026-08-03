import { request } from "../../../api/client";
import {
  MarketplaceListing,
  ComparisonGroup,
  ComplianceCheck,
  ConversationState,
} from "../types";

export const marketplaceAIApi = {
  // Agent mode endpoints (Mocked)
  startAgentSession: async () => ({ session_id: "dummy-session-123" }),
  sendAgentMessage: async (sessionId: string, message: string) => ({
    reply: "I am a mocked agent response.",
    listings: [] as MarketplaceListing[],
    state: {
      step: "idle",
      current_task: "Mocking",
      active_filters: {},
      results: [] as MarketplaceListing[],
    },
  }),
  getAgentState: async (sessionId: string) => ({
    step: "idle",
    current_task: "Mocking",
    active_filters: {},
    results: [] as MarketplaceListing[],
  }),

  // Listings (Mocked)
  getListings: async (
    params?: Record<string, any>,
  ): Promise<{ results: MarketplaceListing[]; count: number }> => ({
    results: [],
    count: 0,
  }),

  // Comparison builder (Mocked)
  getComparisonGroups: async (): Promise<ComparisonGroup[]> => [],
  createComparisonGroup: async (
    name: string,
    listingIds: string[],
  ): Promise<ComparisonGroup> =>
    ({
      id: "123",
      name,
      listings: [],
      created_at: new Date().toISOString(),
    }) as ComparisonGroup,
  deleteComparisonGroup: async (id: string): Promise<null> => null,
  addToListingsToComparison: async (
    groupId: string,
    listingIds: string[],
  ): Promise<null> => null,

  // Compliance (Mocked)
  checkListingCompliance: async (
    listingId: string,
  ): Promise<ComplianceCheck> =>
    ({
      id: "comp-1",
      listing_id: listingId,
      status: "compliant",
      passed_checks: [],
      failed_checks: [],
      details: {},
      checked_at: new Date().toISOString(),
    }) as ComplianceCheck,
};
