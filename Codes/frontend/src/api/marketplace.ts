import api from './client';
import { GPUListing, MarketOrder, Trade, Portfolio } from '@/types/marketplace';

export const marketplaceApi = {
  getListings: () => api.get<GPUListing[]>('/marketplace/listings/'),
  getListingDetail: (id: string) => api.get<GPUListing>(`/marketplace/listings/${id}/`),
  placeOrder: (listingId: string, side: 'BUY' | 'SELL', quantity: number, price: number) =>
    api.post(`/marketplace/listings/${listingId}/place_order/`, { side, quantity, price }),
  getOrderBook: (listingId: string) => api.get<{ bids: MarketOrder[]; asks: MarketOrder[] }>(`/marketplace/listings/${listingId}/order_book/`),
  getPriceHistory: (listingId: string) => api.get<{ time: number; price: number; volume: number }[]>(`/marketplace/listings/${listingId}/price_history/`),
  getPortfolio: () => api.get<Portfolio>('/marketplace/portfolio/'),
  deposit: (amount: number) => api.post('/marketplace/portfolio/deposit/', { amount }),
  withdraw: (amount: number) => api.post('/marketplace/portfolio/withdraw/', { amount }),
};