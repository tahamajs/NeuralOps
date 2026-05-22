export type EndpointStatus = 'deploying' | 'running' | 'stopped' | 'failed'
export type PipelineStatus = 'idle' | 'running' | 'completed' | 'failed'
export type CTCTriggerType = 'cron' | 'webhook' | 'new_data' | 'git_push'

export interface InferenceEndpoint {
  id: string
  name: string
  model_id: string
  model_version: string
  status: EndpointStatus
  url: string
  created_at: string
  updated_at: string
  config: {
    replicas: number
    max_batch_size: number
    timeout_ms: number
    gpu: number
    cpu: number
    memory_mb: number
  }
  metrics?: {
    requests_per_second: number
    avg_latency_ms: number
    error_rate: number
  }
}

export interface PipelineRun {
  id: string
  name: string
  status: PipelineStatus
  started_at: string
  completed_at?: string
  steps: PipelineStep[]
  logs_url?: string
}

export interface PipelineStep {
  id: string
  name: string
  status: PipelineStatus
  duration_seconds?: number
  logs?: string[]
}

export interface CTCTrigger {
  id: string
  name: string
  type: CTCTriggerType
  config: {
    cron_expression?: string
    webhook_url?: string
    branch?: string
    dataset_path?: string
  }
  pipeline_id: string
  enabled: boolean
  last_triggered_at?: string
  created_at: string
}

export interface DistillationConfig {
  teacher_model_id: string
  student_model_id: string
  temperature: number
  alpha: number
  epochs: number
  batch_size: number
  learning_rate: number
  dataset_path: string
}

export interface MultiNodeConfig {
  script_path: string
  num_nodes: number
  gpus_per_node: number
  cpu_per_node: number
  memory_per_node_gb: number
  max_wall_time_seconds: number
  environment_variables: Record<string, string>
}