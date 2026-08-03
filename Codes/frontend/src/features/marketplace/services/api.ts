import { request } from "../../../api/client";
import {
  ProviderStats,
  RevenuePoint,
  UsageData,
  Dispute,
  OrderBookDepth,
} from "../types";

export const marketplaceAPI = {
  // Provider dashboard
  getProviderStats: () =>
    request.get<ProviderStats>("/marketplace/provider/stats/"),
  getProviderRevenueHistory: (days?: number) =>
    request.get<RevenuePoint[]>("/marketplace/provider/revenue/", {
      params: { days },
    }),
  getProviderUsage: (period?: string) =>
    request.get<UsageData>("/marketplace/provider/usage/", {
      params: { period },
    }),

  // Disputes
  getDisputes: (params?: any) =>
    request.get<Dispute[]>("/gpu/reputation/disputes/", { params }),
  getDispute: (id: string) =>
    request.get<Dispute>(`/gpu/reputation/disputes/${id}/`),
  raiseDispute: (jobId: string, reason: string, description: string) =>
    request.post(`/gpu/reputation/disputes/raise/${jobId}/`, {
      reason,
      description,
    }),
  respondToDispute: (id: string, response: string) =>
    request.post(`/gpu/reputation/disputes/${id}/respond/`, { response }),
  resolveDispute: (id: string, resolution: string, refundAmount?: number) =>
    request.post(`/gpu/reputation/disputes/${id}/resolve/`, {
      resolution,
      refund_amount: refundAmount,
    }),

  // Order book (live)
  getOrderBookDepth: (listingId: string) =>
    request.get<OrderBookDepth>(`/marketplace/listings/${listingId}/depth/`),
};
