export interface Experiment {
  id: string;
  name: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'stopped';
  job_type: 'training' | 'evaluation' | 'compression';
  config: Record<string, any>;
  created_at: string;
  current_step: number;
  total_steps: number;
  progress: number;
  logs?: string[];
  metrics?: { loss: number[]; accuracy: number[] };
}