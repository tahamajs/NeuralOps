import { request } from "../../../api/client";
import { ProviderStats, RevenuePoint, UsageData, Dispute, OrderBookDepth } from "../types";

export const marketplaceAPI = {
  // Provider dashboard
  getProviderStats: () =>
    request.get<ProviderStats>("/marketplace/provider/stats/"),
  getProviderRevenueHistory: (days?: number) =>
    request.get<RevenuePoint[]>("/marketplace/provider/revenue/", {
      params: { days },
    }),
  getProviderUsage: (period?: string) =>
    request.get<UsageData>("/marketplace/provider/usage/", { params: { period } }),

  // Disputes
  getDisputes: (params?: { status?: string }) =>
    request.get<Dispute[]>("/marketplace/disputes/", { params }),
  getDispute: (id: string) => request.get<Dispute>(`/marketplace/disputes/${id}/`),
  raiseDispute: (jobId: string, reason: string, description: string) =>
    request.post("/marketplace/disputes/", { job_id: jobId, reason, description }),
  respondToDispute: (id: string, response: string) =>
    request.post(`/marketplace/disputes/${id}/respond/`, { response }),
  resolveDispute: (id: string, resolution: string, refundAmount?: number) =>
    request.post(`/marketplace/disputes/${id}/resolve/`, {
      resolution,
      refund_amount: refundAmount,
    }),

  // Order book (live)
  getOrderBookDepth: (listingId: string) =>
    request.get<OrderBookDepth>(`/marketplace/listings/${listingId}/depth/`),
};
