export type OrderSide = 'buy' | 'sell'
export type OrderType = 'market' | 'limit'
export type OrderStatus = 'open' | 'filled' | 'cancelled' | 'expired'
export type PositionSide = 'long' | 'short'
export type ContractStatus = 'active' | 'expired' | 'settled'

export interface FuturesContract {
  id: string
  underlying_index: string      // e.g., "GPU_H100"
  expiry_date: string
  status: ContractStatus
  settlement_price?: number
  created_at: string
}

export interface FuturesOrder {
  id: string
  contract_id: string
  side: OrderSide
  type: OrderType
  quantity: number
  price?: number        // for limit orders
  status: OrderStatus
  filled_quantity: number
  average_price?: number
  created_at: string
}

export interface FuturesPosition {
  contract_id: string
  side: PositionSide
  quantity: number
  entry_price: number
  mark_price: number
  unrealized_pnl: number
  realized_pnl: number
}

export interface ComputeIndex {
  id: string
  name: string           // e.g., "NeuralOps GPU Index"
  symbol: string         // e.g., "NGI"
  current_value: number
  change_24h: number
  volume_24h: number
  components: IndexComponent[]
  history: IndexHistoryPoint[]
}

export interface IndexComponent {
  gpu_model: string
  weight: number
  price: number
}

export interface IndexHistoryPoint {
  timestamp: string
  value: number
}

export interface Invoice {
  id: string
  amount: number
  currency: string
  status: 'pending' | 'paid' | 'failed'
  issued_at: string
  paid_at?: string
  due_date: string
  description: string
  download_url: string
}

export interface PayoutRequest {
  id: string
  amount: number
  currency: string
  method: 'bank_transfer' | 'crypto' | 'paypal'
  status: 'pending' | 'processing' | 'completed' | 'rejected'
  requested_at: string
  processed_at?: string
  notes?: string
}

export interface SubscriptionPlan {
  id: string
  name: string
  tier: 'free' | 'pro' | 'enterprise'
  price_monthly: number
  price_yearly: number
  features: string[]
  limits: {
    max_gpu_hours: number
    max_concurrent_jobs: number
    priority_support: boolean
  }
}