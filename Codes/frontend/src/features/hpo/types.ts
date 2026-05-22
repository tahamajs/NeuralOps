export type SweepStatus = 'running' | 'completed' | 'failed' | 'cancelled'
export type TrialStatus = 'running' | 'completed' | 'pruned' | 'failed'

export interface Hyperparameter {
  name: string
  type: 'float' | 'int' | 'categorical'
  min?: number
  max?: number
  values?: string[]
}

export interface Trial {
  id: string
  number: number
  status: TrialStatus
  params: Record<string, number | string>
  objective_value: number
  intermediate_values: { step: number; value: number }[]
  duration_seconds: number
  created_at: string
  completed_at?: string
}

export interface Sweep {
  id: string
  name: string
  algorithm: 'grid' | 'random' | 'bayesian' | 'hyperband'
  objective: 'minimize' | 'maximize'
  hyperparameters: Hyperparameter[]
  total_trials: number
  completed_trials: number
  best_trial_id?: string
  best_value?: number
  status: SweepStatus
  created_at: string
  updated_at: string
}

export interface SweepStats {
  total_trials: number
  completed_trials: number
  pruned_trials: number
  failed_trials: number
  running_trials: number
  best_value: number
  best_trial_id: string
  average_duration_seconds: number
}

export interface ParameterImportance {
  name: string
  importance: number
}