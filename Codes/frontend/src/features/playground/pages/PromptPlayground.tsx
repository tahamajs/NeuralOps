// src/features/playground/pages/PromptPlayground.tsx
import { useState, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { playgroundAPI } from "../services/api";
import { useLLMStream } from "../hooks/useLLMStream";
import { PromptEditor } from "../components/PromptEditor";
import { ModelSelector } from "../components/ModelSelector";
import { ParameterControls } from "../components/ParameterControls";
import { StreamingResponse } from "../components/StreamingResponse";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Settings, Play, Save, Trash2 } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function PromptPlayground({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { id: "1", role: "system", content: "You are a helpful assistant." },
    { id: "2", role: "user", content: "" },
  ]);
  const [config, setConfig] = useState<ModelConfig>({
    provider: "openai",
    model: "gpt-4o-mini",
    temperature: 0.7,
    max_tokens: 1000,
    top_p: 0.9,
    frequency_penalty: 0,
    presence_penalty: 0,
  });
  const { isStreaming, streamingContent, error, sendMessage } = useLLMStream();
  const { data: models } = useQuery({
    queryKey: ["playground-models"],
    queryFn: () => playgroundAPI.getModels(),
  });
  const { success, error: toastError } = useToast();

  const handleSend = () => {
    const lastUserMsg = messages.findLast((m) => m.role === "user");
    if (!lastUserMsg?.content.trim()) return;
    sendMessage(messages, config);
  };

  const updateMessage = (id: string, content: string) => {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, content } : m)),
    );
  };

  const addUserMessage = () => {
    setMessages((prev) => [
      ...prev,
      { id: Date.now().toString(), role: "user", content: "" },
    ]);
  };

  const clearMessages = () => {
    setMessages([
      { id: "sys", role: "system", content: "You are a helpful assistant." },
    ]);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Settings size={20} className="text-neon-mint" /> Prompt Playground
          </h1>
          <p className="text-xs text-slate-400">
            Test prompts across multiple providers
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="ghost" onClick={clearMessages}>
            <Trash2 size={14} /> Clear
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-hidden p-4">
        <div className="grid lg:grid-cols-3 gap-4 h-full">
          <div className="lg:col-span-2 flex flex-col gap-4 overflow-auto">
            <TerminalCard className="flex-1 overflow-auto p-3">
              <div className="space-y-3">
                {messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[80%] rounded-lg p-3 ${msg.role === "user" ? "bg-neon-blue/20 text-white" : "bg-white/5 text-slate-300"}`}
                    >
                      <div className="text-xs text-slate-400 mb-1">
                        {msg.role}
                      </div>
                      {msg.role === "user" ? (
                        <textarea
                          value={msg.content}
                          onChange={(e) =>
                            updateMessage(msg.id, e.target.value)
                          }
                          className="w-full bg-transparent outline-none resize-none"
                          rows={3}
                        />
                      ) : (
                        <div className="whitespace-pre-wrap">{msg.content}</div>
                      )}
                    </div>
                  </div>
                ))}
                <div className="flex justify-center">
                  <Button size="sm" variant="ghost" onClick={addUserMessage}>
                    + Add User Message
                  </Button>
                </div>
              </div>
            </TerminalCard>
            <div className="flex gap-2">
              <Button onClick={handleSend} disabled={isStreaming}>
                <Play size={14} className="mr-1" /> Send
              </Button>
              <span className="text-xs text-slate-500 self-center">
                Streaming response appears below
              </span>
            </div>
            <StreamingResponse
              content={streamingContent}
              isStreaming={isStreaming}
              error={error}
            />
          </div>
          <div className="space-y-4 overflow-auto">
            <TerminalCard className="p-4">
              <h3 className="text-sm font-bold text-white mb-2">Model</h3>
              <ModelSelector
                models={models || []}
                value={config}
                onChange={setConfig}
              />
            </TerminalCard>
            <TerminalCard className="p-4">
              <h3 className="text-sm font-bold text-white mb-2">Parameters</h3>
              <ParameterControls config={config} onChange={setConfig} />
            </TerminalCard>
          </div>
        </div>
      </div>
    </div>
  );
}
