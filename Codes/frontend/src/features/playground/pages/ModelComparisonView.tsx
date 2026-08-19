import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { playgroundAPI } from "../services/api";
import { ComparisonTable } from "../components/ComparisonTable";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Textarea } from "../../../components/ui/Input";
import { ModelSelector } from "../components/ModelSelector";
import { ParameterControls } from "../components/ParameterControls";
import { Spinner } from "../../../components/ui/Spinner";
import { GitBranch, Play } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";
import { ModelConfig } from "../types";

const DEFAULT_MODELS = [
  { provider: "openai", model: "gpt-4o" },
  { provider: "openai", model: "gpt-4o-mini" },
  { provider: "anthropic", model: "claude-3-5-sonnet-20240620" },
  { provider: "google", model: "gemini-1.5-pro" },
  { provider: "deepseek", model: "deepseek-chat" },
  { provider: "meta", model: "llama-3.1-70b-instruct" },
];

export default function ModelComparisonView({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [prompt, setPrompt] = useState(
    "Explain how linear attention reduces compute complexity from O(N^2) to O(N).",
  );
  const [configA, setConfigA] = useState<ModelConfig>({
    provider: "openai",
    model: "gpt-4o-mini",
    temperature: 0.7,
    max_tokens: 500,
    top_p: 0.9,
    frequency_penalty: 0,
    presence_penalty: 0,
  });
  const [configB, setConfigB] = useState<ModelConfig>({
    provider: "anthropic",
    model: "claude-3-5-sonnet-20240620",
    temperature: 0.7,
    max_tokens: 500,
    top_p: 0.9,
    frequency_penalty: 0,
    presence_penalty: 0,
  });
  const [result, setResult] = useState<any>(null);
  const { success, error: toastError } = useToast();

  const { data: modelsData } = useQuery({
    queryKey: ["playground-models"],
    queryFn: () => playgroundAPI.getModels(),
  });

  const availableModels = (modelsData && modelsData.length > 0) ? modelsData : DEFAULT_MODELS;

  const compareMutation = useMutation({
    mutationFn: () => playgroundAPI.compareModels(configA, configB, prompt),
    onSuccess: (res) => setResult(res),
    onError: () => toastError("Comparison failed"),
  });

  const handleCompare = () => {
    if (prompt.trim()) compareMutation.mutate();
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <GitBranch size={20} className="text-neon-mint" /> Model Comparison
          </h1>
          <p className="text-xs text-slate-400">
            Side‑by‑side LLM output comparison
          </p>
        </div>
        {onClose && (
          <Button size="sm" variant="ghost" onClick={onClose}>
            ✕
          </Button>
        )}
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <TerminalCard className="p-4">
          <Textarea
            rows={4}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter prompt to test both models..."
          />
        </TerminalCard>
        <div className="grid lg:grid-cols-2 gap-4">
          <TerminalCard className="p-4">
            <h3 className="text-sm font-bold text-white mb-2">Model A</h3>
            <ModelSelector models={availableModels} value={configA} onChange={setConfigA} />
            <ParameterControls config={configA} onChange={setConfigA} />
          </TerminalCard>
          <TerminalCard className="p-4">
            <h3 className="text-sm font-bold text-white mb-2">Model B</h3>
            <ModelSelector models={availableModels} value={configB} onChange={setConfigB} />
            <ParameterControls config={configB} onChange={setConfigB} />
          </TerminalCard>
        </div>
        <Button
          onClick={handleCompare}
          disabled={!prompt.trim() || compareMutation.isPending}
        >
          <Play size={14} className="mr-1" /> Compare Models
        </Button>
        {compareMutation.isPending && <Spinner />}
        {result && <ComparisonTable result={result} />}
      </div>
    </div>
  );
}
