export type InsightType = 'opportunity' | 'warning' | 'recommendation' | 'anomaly'
export type InterventionPriority = 'high' | 'medium' | 'low'
export type InsightStatus = 'new' | 'read' | 'applied'

export interface KpiMetric {
  label: string
  value: number
  unit: string
  change: number // percentage
  trend: 'up' | 'down' | 'stable'
  description?: string
}

export interface ExperimentInsight {
  id: string
  type: InsightType
  title: string
  description: string
  impact_score: number // 0-100
  suggested_action?: string
  confidence: number // 0-1
  created_at: string
  status: InsightStatus
  metadata?: Record<string, any>
  experiment_id?: string
  experiment_name?: string
}

export interface ExperimentComparison {
  id: string
  name: string
  accuracy: number
  loss: number
  epochs: number
  params_count: number
  training_time_seconds: number
  created_at: string
}

export interface OpportunityScore {
  total: number
  breakdown: {
    hyperparameter_tuning: number
    architecture_change: number
    data_augmentation: number
    early_stopping: number
    learning_rate_adjustment: number
  }
  recommended_focus: string
}

export interface InterventionSuggestion {
  id: string
  title: string
  description: string
  priority: InterventionPriority
  expected_improvement: number // percentage
  effort_estimate: 'low' | 'medium' | 'high'
  confidence: number
  parameters?: Record<string, any>
  related_insight_id?: string
}

export interface AutoRefreshConfig {
  enabled: boolean
  interval_seconds: number
}