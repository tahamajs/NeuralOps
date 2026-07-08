// src/features/llmHub/pages/LLMProviderList.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { llmHubAPI } from "../services/api";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Modal } from "../../../components/ui/Modal";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { Switch } from "../../../components/ui/Switch";
import { Server, Key, CheckCircle, XCircle, Settings } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function LLMProviderList({ onClose }: { onClose?: () => void }) {
  const [selectedProvider, setSelectedProvider] = useState<any>(null);
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: providers, isLoading } = useQuery({
    queryKey: ["llm-providers"],
    queryFn: () => llmHubAPI.getProviders(),
  });

  const updateProvider = useMutation({
    mutationFn: ({ id, data }: { id: string; data: any }) =>
      llmHubAPI.updateProvider(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["llm-providers"] });
      setShowConfigModal(false);
      success("Provider updated");
    },
  });

  const testProvider = useMutation({
    mutationFn: (id: string) => llmHubAPI.testProvider(id),
    onSuccess: (_, id) => {
      const provider = providers?.find((p) => p.id === id);
      success(`${provider?.name} connection test passed`);
    },
    onError: () => error("Connection test failed"),
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Server size={20} className="text-neon-mint" /> LLM Providers
          </h1>
          <p className="text-xs text-slate-400">
            Manage API connections to all LLM services
          </p>
        </div>
        {onClose && (
          <Button size="sm" variant="ghost" onClick={onClose}>
            ✕
          </Button>
        )}
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-3">
        {providers?.map((provider) => (
          <TerminalCard key={provider.id} className="p-4">
            <div className="flex justify-between items-start">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-lg">{provider.icon || "🤖"}</span>
                  <h3 className="font-bold text-white">{provider.name}</h3>
                  <Badge variant={provider.enabled ? "success" : "secondary"}>
                    {provider.enabled ? "Enabled" : "Disabled"}
                  </Badge>
                </div>
                <div className="flex flex-wrap gap-1 mt-2">
                  {provider.models.slice(0, 4).map((m) => (
                    <Badge key={m} variant="info" className="text-[10px]">
                      {m}
                    </Badge>
                  ))}
                </div>
                <div className="flex gap-3 mt-2 text-xs text-slate-500">
                  <span>Capabilities: {provider.capabilities.join(", ")}</span>
                  {provider.costPerMillionTokens && (
                    <span>~${provider.costPerMillionTokens.input}/1M in</span>
                  )}
                </div>
              </div>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => testProvider.mutate(provider.id)}
                >
                  <CheckCircle size={14} /> Test
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    setSelectedProvider(provider);
                    setApiKey(provider.apiKey || "");
                    setBaseUrl(provider.baseUrl || "");
                    setShowConfigModal(true);
                  }}
                >
                  <Settings size={14} />
                </Button>
              </div>
            </div>
          </TerminalCard>
        ))}
      </div>
      <Modal
        open={showConfigModal}
        onClose={() => setShowConfigModal(false)}
        title={`Configure ${selectedProvider?.name}`}
      >
        <div className="space-y-4">
          <Input
            label="API Key"
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
          />
          <Input
            label="Base URL"
            value={baseUrl}
            onChange={(e) => setBaseUrl(e.target.value)}
            placeholder="https://api.openai.com/v1"
          />
          <div className="flex justify-between items-center">
            <span className="text-sm text-slate-400">Enable provider</span>
            <Switch
              checked={selectedProvider?.enabled}
              onChange={(checked) =>
                setSelectedProvider({ ...selectedProvider, enabled: checked })
              }
            />
          </div>
          <Button
            onClick={() =>
              updateProvider.mutate({
                id: selectedProvider.id,
                data: {
                  api_key: apiKey,
                  base_url: baseUrl,
                  enabled: selectedProvider.enabled,
                },
              })
            }
          >
            Save
          </Button>
        </div>
      </Modal>
    </div>
  );
}
