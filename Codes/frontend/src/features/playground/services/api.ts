import { request } from "../../../api/client";
import {
  ChatMessage,
  ModelConfig,
  ComparisonResult,
  ToolDefinition,
  ToolTestResult,
} from "../types";

export const playgroundAPI = {
  // Chat completion (streaming)
  chatCompletion: (messages: ChatMessage[], config: ModelConfig) =>
    request.post<any>(
      "/playground/chat/",
      { messages, config },
      { responseType: "stream" },
    ),

  // List available models
  getModels: () => request.get<ModelConfig[]>("/playground/models/"),

  // Comparison
  compareModels: (configA: ModelConfig, configB: ModelConfig, prompt: string) =>
    request.post<ComparisonResult>("/playground/compare/", {
      config_a: configA,
      config_b: configB,
      prompt,
    }),

  // Tools
  getTools: () => request.get<ToolDefinition[]>("/playground/tools/"),
  testTool: (toolName: string, input: any) =>
    request.post<ToolTestResult>(`/playground/tools/${toolName}/test/`, {
      input,
    }),
};
