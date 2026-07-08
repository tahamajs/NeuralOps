export interface ProviderStats {
  total_earned: number;
  pending_earnings: number;
  total_hours_provided: number;
  active_listings: number;
  average_rating: number;
  dispute_rate: number;
}

export interface RevenuePoint {
  date: string;
  amount: number;
}

export interface UsageData {
  hourly_distribution: { hour: number; hours: number }[];
  daily_distribution: { day: string; hours: number }[];
  top_gpu_models: { model: string; hours: number }[];
}

export interface Dispute {
  id: string;
  job_id: string;
  raised_by: string;
  raised_by_role: "consumer" | "provider";
  reason: string;
  description: string;
  status: "open" | "in_review" | "resolved" | "closed";
  resolution?: string;
  refund_amount?: number;
  created_at: string;
  updated_at: string;
  messages: DisputeMessage[];
}

export interface DisputeMessage {
  id: string;
  user: string;
  content: string;
  created_at: string;
}

export interface OrderBookDepth {
  bids: { price: number; quantity: number; total: number }[];
  asks: { price: number; quantity: number; total: number }[];
  spread: number;
  last_price: number;
}
