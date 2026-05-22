export type ListingStatus = 'available' | 'limited' | 'booked'
export type ComplianceStatus = 'compliant' | 'non_compliant' | 'unknown'
export type MessageRole = 'user' | 'assistant' | 'system'

export interface GPUSpecs {
  model: string
  vram_gb: number
  tdp_watts: number
  architecture: string
  interconnects: string[]
}

export interface MarketplaceListing {
  id: string
  provider: string
  provider_avatar?: string
  gpu_specs: GPUSpecs
  price_per_hour: number
  price_per_hour_spot?: number
  available_units: number
  total_units: number
  region: string
  status: ListingStatus
  reliability_score: number
  compliance_badges: string[]
  created_at: string
}

export interface ComparisonGroup {
  id: string
  name: string
  listings: MarketplaceListing[]
  created_at: string
}

export interface ComplianceCheck {
  id: string
  listing_id: string
  status: ComplianceStatus
  passed_checks: string[]
  failed_checks: string[]
  details: Record<string, any>
  checked_at: string
}

export interface Message {
  id: string
  role: MessageRole
  content: string
  suggestions?: string[]
  listings?: MarketplaceListing[]
  timestamp: Date
}

export interface ConversationState {
  filters: {
    min_price?: number
    max_price?: number
    region?: string
    gpu_model?: string
    min_vram?: number
    compliance_required?: string[]
  }
  context: string[]
}