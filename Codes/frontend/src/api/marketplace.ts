// src/api/marketplace.ts
import { request as api } from "./client";
import type {
  GPUListing,
  MarketOrder,
  Trade,
  Portfolio,
  OrderBookDepth,
  ProviderStats,
  RevenuePoint,
  UsageData,
  Dispute,
  DisputeMessage,
} from "@/types/marketplace";

export const marketplaceApi = {
  // ========================
  // Core marketplace (consumer)
  // ========================
  getListings: () => api.get<GPUListing[]>("/marketplace/listings/"),
  getListingDetail: (id: string) =>
    api.get<GPUListing>(`/marketplace/listings/${id}/`),

  placeOrder: (
    listingId: string,
    side: "BUY" | "SELL",
    quantity: number,
    price?: number,
  ) =>
    api.post(`/marketplace/listings/${listingId}/place_order/`, {
      side,
      quantity,
      price,
    }),

  getOrderBook: (listingId: string) =>
    api.get<{ bids: MarketOrder[]; asks: MarketOrder[] }>(
      `/marketplace/listings/${listingId}/order_book/`,
    ),

  getOrderBookDepth: (listingId: string) =>
    api.get<OrderBookDepth>(`/marketplace/listings/${listingId}/depth/`),

  getPriceHistory: (listingId: string) =>
    api.get<{ time: number; price: number; volume: number }[]>(
      `/marketplace/listings/${listingId}/price_history/`,
    ),

  // Portfolio / consumer account
  getPortfolio: () => api.get<Portfolio>("/marketplace/portfolio/"),
  deposit: (amount: number) =>
    api.post("/marketplace/portfolio/deposit/", { amount }),
  withdraw: (amount: number) =>
    api.post("/marketplace/portfolio/withdraw/", { amount }),

  // ========================
  // Provider dashboard
  // ========================
  getProviderStats: () =>
    api.get<ProviderStats>("/marketplace/provider/stats/"),
  getProviderRevenueHistory: (days?: number) =>
    api.get<RevenuePoint[]>("/marketplace/provider/revenue/", {
      params: { days },
    }),
  getProviderUsage: (period?: string) =>
    api.get<UsageData>("/marketplace/provider/usage/", { params: { period } }),

  // ========================
  // Disputes
  // ========================
  getDisputes: (params?: { status?: string }) =>
    api.get<Dispute[]>("/marketplace/disputes/", { params }),
  getDispute: (id: string) => api.get<Dispute>(`/marketplace/disputes/${id}/`),
  raiseDispute: (jobId: string, reason: string, description: string) =>
    api.post("/marketplace/disputes/", { job_id: jobId, reason, description }),
  respondToDispute: (id: string, response: string) =>
    api.post(`/marketplace/disputes/${id}/respond/`, { response }),
  resolveDispute: (id: string, resolution: string, refundAmount?: number) =>
    api.post(`/marketplace/disputes/${id}/resolve/`, {
      resolution,
      refund_amount: refundAmount,
    }),

  // ========================
  // Trading / orders (extended)
  // ========================
  cancelOrder: (orderId: string) =>
    api.post(`/marketplace/orders/${orderId}/cancel/`),
  getMyOrders: () => api.get<MarketOrder[]>("/marketplace/orders/mine/"),
  getMyTrades: () => api.get<Trade[]>("/marketplace/trades/"),
};
