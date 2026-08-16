export interface GPUListing {
  id: string;
  provider: string | number;
  provider_username: string;
  provider_reputation?: number;
  gpu_model_spec?: string | null;
  gpu_model_spec_detail?: {
    id?: string | number;
    name?: string;
    manufacturer?: string;
    vram_gb?: number;
    architecture?: string;
    tf32_tflops?: number;
    fp16_tflops?: number;
    memory_bandwidth_gbs?: number;
    tdp_watts?: number;
    market_price_usd?: number;
    listing_count?: number;
    avg_price?: number;
  };
  gpu_model_detail?: { name: string; memory_gb: number };
  gpu_model: string;
  gpu_count?: number;
  vram_gb?: number;
  cpu_cores?: number;
  ram_gb?: number;
  storage_gb?: number;
  bandwidth_mbps?: number;
  location?: string;
  country_code?: string;
  listing_type?: "on_demand" | "spot" | "reserved" | string;
  base_price_per_hour?: number | string;
  min_price_per_hour?: number | string;
  current_price_per_hour?: number | string;
  storage_price_per_gb_hour?: number | string;
  price_per_hour?: number;
  change_percent?: number;
  volume_24h_hours?: number;
  market_cap?: number;
  available_units?: number;
  status: string;
  verification_level?: string;
  dlperf_score?: number;
  reliability_score?: number;
  uptime_percentage?: number;
  reputation_score?: number;
  total_hours_rented?: number;
  total_earnings_usd?: number;
  completed_jobs?: number;
  failed_jobs?: number;
  total_rentals?: number;
  available_from?: string;
  available_until?: string;
  max_rental_hours?: number;
  supports_ssh?: boolean;
  supports_jupyter?: boolean;
  created_at?: string;
  updated_at?: string;
}

export interface MarketOrder {
  id: string;
  user?: number | string;
  username?: string;
  order_type?: "market" | "limit" | "auction" | string;
  order_mode?: "buy" | "sell" | string;
  side?: "BUY" | "SELL";
  gpu_model_preference?: string;
  min_vram_gb?: number;
  min_gpu_count?: number;
  max_price_per_hour?: number | string;
  required_hours?: number;
  quantity?: number;
  price?: number;
  status: string;
  matched_listing?: string | null;
  final_price_per_hour?: number | string | null;
  bid_count?: number;
  created_at: string;
  updated_at?: string;
  filled_at?: string | null;
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

export interface OrderBookDepth {
  asks: [number, number][];
  bids: [number, number][];
}

export interface ProviderStats {
  active_gpus: number;
  total_gpus: number;
  uptime_percent: number;
  total_earned: number;
  monthly_earned: number;
  pending_earnings?: number;
  total_hours_provided?: number;
  active_listings?: number;
  average_rating?: number;
  dispute_rate?: number;
}

export interface RevenuePoint {
  date: string;
  amount: number;
}

export interface UsageData {
  labels?: string[];
  datasets?: { label: string; data: number[] }[];
  hourly_distribution?: Array<{ hour: number; hours: number }>;
  daily_distribution?: Array<{ day: string; hours: number }>;
  top_gpu_models?: Array<{ model: string; hours: number }>;
}

export interface Dispute {
  id: string;
  job?: string;
  job_id?: string;
  raised_by?: number | string;
  raised_by_username?: string;
  against?: number | string;
  against_username?: string;
  reason: string;
  description?: string;
  evidence?: string;
  status: string;
  required_jurors?: number;
  resolution?: string;
  resolution_reason?: string;
  dispute_fee_usd?: number | string;
  jury_reward_usd?: number | string;
  jury_count?: number;
  vote_summary?: Record<string, any>;
  created_at: string;
  resolved_at?: string | null;
  updated_at?: string;
  messages?: DisputeMessage[];
}

export interface DisputeMessage {
  id: string;
  sender: string;
  message: string;
  created_at: string;
}

