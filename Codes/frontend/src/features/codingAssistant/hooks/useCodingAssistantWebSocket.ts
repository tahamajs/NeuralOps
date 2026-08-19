// src/features/codingAssistant/hooks/useCodingAssistantWebSocket.ts
import { useState, useEffect, useRef, useCallback } from "react";
import { CodeSuggestion } from "../types";

export function useCodingAssistantWebSocket(sessionId: string | null = null) {
  const [isConnected, setIsConnected] = useState(true);
  const [streamingContent, setStreamingContent] = useState("");
  const [suggestions, setSuggestions] = useState<CodeSuggestion[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const streamIntervalRef = useRef<any>(null);

  const sendMessage = useCallback((prompt: string) => {
    setStreamingContent("");

    // If real WebSocket is open, send through socket
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "prompt", prompt }));
      return;
    }

    // Otherwise, simulate streaming response locally
    const lower = prompt.toLowerCase();
    let responseText = "";
    let mockSuggestions: CodeSuggestion[] = [];

    if (lower.includes("svd") || lower.includes("linear") || lower.includes("attention")) {
      responseText = `Here is an optimized **SVD Linear Attention Module** with low-rank projection factors:\n\n\`\`\`python\nimport torch\nimport torch.nn as nn\n\nclass SVDLinearAttention(nn.Module):\n    def __init__(self, dim=768, rank=64):\n        super().__init__()\n        self.dim = dim\n        self.rank = rank\n        self.q_proj = nn.Linear(dim, dim, bias=False)\n        self.u_proj = nn.Linear(dim, rank, bias=False)\n        self.v_proj = nn.Linear(rank, dim, bias=False)\n\n    def forward(self, x):\n        q = torch.relu(self.q_proj(x))\n        k_comp = self.u_proj(x)\n        kv = torch.einsum("btr, btd -> brd", k_comp, x)\n        return torch.einsum("btd, brd -> btd", q, kv)\n\`\`\`\n\nThis factors the key projection through rank $r=64$, reducing $O(N^2)$ softmax compute to $O(N)$ linear tensor contraction.`;
      mockSuggestions = [
        {
          title: "SVD Linear Attention Module",
          modifiedCode: `import torch\nimport torch.nn as nn\n\nclass SVDLinearAttention(nn.Module):\n    def __init__(self, dim=768, rank=64):\n        super().__init__()\n        self.dim = dim\n        self.rank = rank\n        self.q_proj = nn.Linear(dim, dim, bias=False)\n        self.u_proj = nn.Linear(dim, rank, bias=False)\n        self.v_proj = nn.Linear(rank, dim, bias=False)\n\n    def forward(self, x):\n        q = torch.relu(self.q_proj(x))\n        k_comp = self.u_proj(x)\n        kv = torch.einsum("btr, btd -> brd", k_comp, x)\n        return torch.einsum("btd, brd -> btd", q, kv)\n`,
          explanation: "Replaces standard O(N^2) quadratic softmax attention with linear SVD projection factor.",
        },
      ];
    } else if (lower.includes("optuna") || lower.includes("sweep") || lower.includes("hpo")) {
      responseText = `Configured **Optuna Hyperparameter Sweep** searching over SVD rank sizes:\n\n\`\`\`python\ndef objective(trial):\n    rank = trial.suggest_categorical("svd_rank", [32, 64, 128])\n    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)\n    return train_and_eval(rank, lr)\n\`\`\``;
      mockSuggestions = [
        {
          title: "Optuna Sweep Script",
          modifiedCode: `import optuna\n\ndef objective(trial):\n    rank = trial.suggest_categorical("svd_rank", [32, 64, 128])\n    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)\n    return train_and_eval(rank, lr)\n\nstudy = optuna.create_study(direction="minimize")\nstudy.optimize(objective, n_trials=20)\n`,
          explanation: "Automated Optuna search over singular value decomposition rank sizes.",
        },
      ];
    } else {
      responseText = `I have analyzed your code. Applying memory bandwidth optimization and vectorization:\n\n\`\`\`python\n# Optimized tensor implementation\nimport torch\n\ndef compute_kernel_projection(x, weight_u, weight_v):\n    return torch.matmul(torch.matmul(x, weight_u), weight_v)\n\`\`\``;
    }

    let charIdx = 0;
    if (streamIntervalRef.current) clearInterval(streamIntervalRef.current);

    streamIntervalRef.current = setInterval(() => {
      if (charIdx < responseText.length) {
        setStreamingContent(responseText.slice(0, charIdx + 6));
        charIdx += 6;
      } else {
        clearInterval(streamIntervalRef.current);
        setStreamingContent("");
        if (mockSuggestions.length > 0) {
          setSuggestions(mockSuggestions);
        }
      }
    }, 20);
  }, []);

  const applySuggestion = useCallback((suggestion: CodeSuggestion) => {
    // Handled by parent
  }, []);

  useEffect(() => {
    return () => {
      if (streamIntervalRef.current) clearInterval(streamIntervalRef.current);
    };
  }, []);

  return {
    isConnected,
    streamingContent,
    suggestions,
    sendMessage,
    applySuggestion,
  };
}
