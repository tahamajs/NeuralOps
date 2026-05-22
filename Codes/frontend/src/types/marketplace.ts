export interface GPUListing {
  id: string;
  provider: string;
  provider_username: string;
  gpu_model_detail: { name: string; memory_gb: number };
  price_per_hour: number;
  change_percent: number;
  volume_24h_hours: number;
  market_cap: number;
  available_units: number;
  status: string;
}

export interface MarketOrder {
  id: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  status: string;
  created_at: string;
}

export interface Trade {
  id: string;
  buyer_username: string;
  seller_username: string;
  listing_name: string;
  quantity: number;
  price: number;
  total_amount: number;
  executed_at: string;
}

export interface Holding {
  listing_name: string;
  quantity: number;
  average_buy_price: number;
  current_price: number;
  current_value: number;
  pnl: number;
}

export interface Portfolio {
  balance_usd: number;
  total_value_usd_display: number;
  total_invested_usd: number;
  unrealized_pnl_display: number;
  holdings: Holding[];
}