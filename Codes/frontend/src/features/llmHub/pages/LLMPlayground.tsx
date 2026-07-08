// src/features/llmHub/pages/LLMPlayground.tsx
import { useState, useRef, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { llmHubAPI } from "../services/api";
import { ModelSelector } from "../components/ModelSelector";
import { ParameterControls } from "../components/ParameterControls";
import { StreamingOutput } from "../components/StreamingOutput";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Textarea } from "../../../components/ui/Input";
import { Tabs, TabList, Tab, TabPanel } from "../../../components/ui/Tabs";
import { Send, GitBranch, Settings, Zap } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function LLMPlayground({ onClose }: { onClose?: () => void }) {
  const [prompt, setPrompt] = useState("");
  const [selectedProvider, setSelectedProvider] = useState("openai");
  const [selectedModel, setSelectedModel] = useState("gpt-4o-mini");
  const [parameters, setParameters] = useState({
    temperature: 0.7,
    max_tokens: 1000,
    top_p: 0.9,
  });
  const [response, setResponse] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const { success, error } = useToast();

  const { data: providers } = useQuery({
    queryKey: ["llm-providers"],
    queryFn: () => llmHubAPI.getProviders(),
  });
  const availableModels =
    providers?.find((p) => p.type === selectedProvider)?.models || [];

  const handleSend = async () => {
    if (!prompt.trim()) return;
    setResponse("");
    setIsStreaming(true);
    try {
      const response = await llmHubAPI.chatCompletion(
        selectedProvider,
        selectedModel,
        [{ role: "user", content: prompt }],
        parameters,
      );
      const reader = (response as any).data ? (response as any).data.getReader() : (response as any).getReader();
      const decoder = new TextDecoder();
      let fullText = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        fullText += chunk;
        setResponse(fullText);
      }
      success("Response generated");
    } catch (err) {
      error("Streaming failed");
    } finally {
      setIsStreaming(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Zap size={20} className="text-yellow-400" /> LLM Playground
          </h1>
          <p className="text-xs text-slate-400">
            Test prompts across multiple providers
          </p>
        </div>
        <div className="flex gap-2">
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-hidden p-4">
        <div className="grid lg:grid-cols-3 gap-4 h-full">
          <div className="lg:col-span-2 flex flex-col gap-4">
            <TerminalCard className="p-4">
              <div className="flex gap-2 mb-3">
                <ModelSelector
                  providers={providers || []}
                  selectedProvider={selectedProvider}
                  setSelectedProvider={setSelectedProvider}
                  selectedModel={selectedModel}
                  setSelectedModel={setSelectedModel}
                  models={availableModels}
                />
                <ParameterControls
                  parameters={parameters}
                  onChange={setParameters}
                />
              </div>
              <Textarea
                rows={8}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter your prompt..."
              />
              <Button
                onClick={handleSend}
                disabled={isStreaming || !prompt.trim()}
                className="mt-2"
              >
                <Send size={14} className="mr-1" />{" "}
                {isStreaming ? "Streaming..." : "Send"}
              </Button>
            </TerminalCard>
            <TerminalCard className="p-4 flex-1 overflow-auto">
              <StreamingOutput content={response} isStreaming={isStreaming} />
            </TerminalCard>
          </div>
          <div>
            <TerminalCard className="p-4 h-full overflow-auto">
              <h3 className="text-sm font-bold text-white mb-2">
                Quick Examples
              </h3>
              <div className="space-y-2">
                <button
                  onClick={() =>
                    setPrompt("Explain quantum computing in simple terms.")
                  }
                  className="w-full text-left text-sm text-slate-400 hover:text-white p-2 rounded bg-white/5"
                >
                  🔬 Explain quantum computing
                </button>
                <button
                  onClick={() =>
                    setPrompt(
                      "Write a Python function to calculate fibonacci numbers.",
                    )
                  }
                  className="w-full text-left text-sm text-slate-400 hover:text-white p-2 rounded bg-white/5"
                >
                  🐍 Fibonacci function
                </button>
                <button
                  onClick={() =>
                    setPrompt(
                      'Summarize the key findings of the paper "Attention is All You Need".',
                    )
                  }
                  className="w-full text-left text-sm text-slate-400 hover:text-white p-2 rounded bg-white/5"
                >
                  📄 Paper summary
                </button>
              </div>
            </TerminalCard>
          </div>
        </div>
      </div>
    </div>
  );
}
