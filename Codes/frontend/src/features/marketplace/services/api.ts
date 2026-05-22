// Add to existing marketplaceAPI object

export const marketplaceAPI = {
  // ... existing methods (getListings, placeOrder, etc.)

  // Provider dashboard
  getProviderStats: () => api.get<ProviderStats>('/marketplace/provider/stats/'),
  getProviderRevenueHistory: (days?: number) => api.get<RevenuePoint[]>('/marketplace/provider/revenue/', { params: { days } }),
  getProviderUsage: (period?: string) => api.get<UsageData>('/marketplace/provider/usage/', { params: { period } }),

  // Disputes
  getDisputes: (params?: { status?: string }) => api.get<Dispute[]>('/marketplace/disputes/', { params }),
  getDispute: (id: string) => api.get<Dispute>(`/marketplace/disputes/${id}/`),
  raiseDispute: (jobId: string, reason: string, description: string) =>
    api.post('/marketplace/disputes/', { job_id: jobId, reason, description }),
  respondToDispute: (id: string, response: string) => api.post(`/marketplace/disputes/${id}/respond/`, { response }),
  resolveDispute: (id: string, resolution: string, refundAmount?: number) =>
    api.post(`/marketplace/disputes/${id}/resolve/`, { resolution, refund_amount: refundAmount }),

  // Order book (live)
  getOrderBookDepth: (listingId: string) => api.get<OrderBookDepth>(`/marketplace/listings/${listingId}/depth/`),
}