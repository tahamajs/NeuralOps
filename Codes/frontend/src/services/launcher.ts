import { request } from "../api/client";
import { ENDPOINTS } from "../api/endpoints";

export interface LaunchConfig {
  name: string;
  model_name: string;
  task_name: string;
  attention_type: string;
  method?: string;
  compression_rank?: number;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
  max_seq_length?: number;
  fp16?: boolean;
  seed?: number;
  [key: string]: any;
}

export const launcherService = {
  launchTraining: (config: LaunchConfig) =>
    request.post<{ id: string; name: string; status: string }>(
      ENDPOINTS.EXPERIMENTS.START_TRAIN,
      config,
    ),

  launchEvaluation: (config: LaunchConfig) =>
    request.post<{ id: string; name: string; status: string }>(
      ENDPOINTS.EXPERIMENTS.START_EVALUATE,
      config,
    ),

  launchCompression: (config: LaunchConfig) =>
    request.post<{ id: string; name: string; status: string }>(
      ENDPOINTS.EXPERIMENTS.START_COMPRESS,
      config,
    ),

  getConfigOptions: () =>
    request.get<any>(ENDPOINTS.LAUNCHER.CONFIG_OPTIONS),

  getAttentions: () =>
    request.get<string[]>(ENDPOINTS.CONFIG.ATTENTIONS),

  getModels: () =>
    request.get<string[]>(ENDPOINTS.CONFIG.MODELS),

  getTasks: () =>
    request.get<string[]>(ENDPOINTS.CONFIG.TASKS),

  getPresets: () =>
    request.get<any[]>(ENDPOINTS.CONFIG.PRESETS),
};
