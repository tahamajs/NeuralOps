export interface ExperimentRun {
  id: string
  name: string
  parameters: Record<string, number | string | boolean>
  metrics: Record<string, number>
  status: 'running' | 'completed' | 'failed'
  created_at: string
  duration_seconds?: number
}

export interface ParameterMetricCorrelation {
  parameter: string
  metric: string
  correlation: number
}

export interface DashboardWidget {
  id: string
  type: 'scatter' | 'parallel_coords' | 'correlation_matrix' | 'metric_history' | 'parameter_distribution'
  title: string
  config: Record<string, any>
  x: number
  y: number
  w: number
  h: number
}

export interface DashboardLayout {
  id: string
  name: string
  widgets: DashboardWidget[]
  created_at: string
}

export interface SQLConnection {
  id: string
  name: string
  dialect: 'postgresql' | 'mysql' | 'sqlite' | 'bigquery'
  host?: string
  database?: string
  schema?: string
}

export interface SQLQueryResult {
  columns: string[]
  rows: Record<string, any>[]
  rowCount: number
  executionTimeMs: number
}