import { request as api } from "../../../api/client";
import {
  FuturesContract,
  FuturesOrder,
  FuturesPosition,
  ComputeIndex,
  Invoice,
  PayoutRequest,
  SubscriptionPlan,
} from "../types";

export const gpuFinanceAPI = {
  // Futures contracts
  getContracts: () => api.get<FuturesContract[]>("/gpu-finance/futures/"),
  getContract: (id: string) =>
    api.get<FuturesContract>(`/gpu-finance/futures/${id}/`),

  // Orders
  getOrders: () => api.get<FuturesOrder[]>("/gpu-finance/orders/"),
  placeOrder: (data: Partial<FuturesOrder>) =>
    api.post<FuturesOrder>("/gpu-finance/orders/", data),
  cancelOrder: (id: string) => api.post(`/gpu-finance/orders/${id}/cancel/`),

  // Positions
  getPositions: () => api.get<FuturesPosition[]>("/gpu-finance/positions/"),

  // Indices
  getIndices: () => api.get<ComputeIndex[]>("/gpu-finance/indices/"),
  getIndex: (id: string) =>
    api.get<ComputeIndex>(`/gpu-finance/indices/${id}/`),

  // Invoices
  getInvoices: () => api.get<Invoice[]>("/gpu-finance/invoices/"),
  downloadInvoice: (id: string) =>
    api.get(`/gpu-finance/invoices/${id}/download/`, { responseType: "blob" }),

  // Payouts
  getPayoutRequests: () => api.get<PayoutRequest[]>("/gpu-finance/payouts/"),
  requestPayout: (data: Partial<PayoutRequest>) =>
    api.post<PayoutRequest>("/gpu-finance/payouts/", data),

  // Subscriptions
  getPlans: () => api.get<SubscriptionPlan[]>("/gpu-finance/plans/"),
  getCurrentSubscription: () =>
    api.get<SubscriptionPlan>("/gpu-finance/subscription/"),
  subscribe: (planId: string, interval: "monthly" | "yearly") =>
    api.post("/gpu-finance/subscription/", { plan_id: planId, interval }),
  cancelSubscription: () => api.post("/gpu-finance/subscription/cancel/"),
};
