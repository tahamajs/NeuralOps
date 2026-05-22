export type PipelineStatus = 'idle' | 'running' | 'paused' | 'completed' | 'failed'
export type StepStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped'

export interface PipelineStep {
  id: string
  name: string
  description?: string
  status: StepStatus
  duration?: number      // seconds
  started_at?: string
  completed_at?: string
  retry_count?: number
  dependencies: string[] // step IDs this step depends on
  input_artifacts?: Artifact[]
  output_artifacts?: Artifact[]
  logs_url?: string
}

export interface Artifact {
  name: string
  type: string
  uri: string
  size_bytes?: number
}

export interface PipelineRun {
  id: string
  name: string
  status: PipelineStatus
  started_at: string
  completed_at?: string
  steps: PipelineStep[]
  current_step_id?: string
  progress: number   // 0-100
  error_message?: string
}

export interface WorkflowDefinition {
  id: string
  name: string
  description?: string
  steps: {
    id: string
    name: string
    type: string
    config: Record<string, any>
    next_step_ids: string[]
  }[]
  created_at: string
  updated_at: string
}

export interface StepTemplate {
  id: string
  name: string
  type: string
  description: string
  default_config: Record<string, any>
  icon?: string
}