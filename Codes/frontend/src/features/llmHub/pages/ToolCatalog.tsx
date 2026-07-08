// src/features/llmHub/pages/ToolCatalog.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { llmHubAPI } from "../services/api";
import { ToolCard } from "../components/ToolCard";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Spinner } from "../../../components/ui/Spinner";
import { Wrench, Search } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function ToolCatalog({ onClose }: { onClose?: () => void }) {
  const [search, setSearch] = useState("");
  const queryClient = useQueryClient();
  const { success } = useToast();

  const { data: tools, isLoading } = useQuery({
    queryKey: ["llm-tools"],
    queryFn: () => llmHubAPI.getTools(),
  });
  const updateTool = useMutation({
    mutationFn: ({ name, enabled }: { name: string; enabled: boolean }) =>
      llmHubAPI.updateTool(name, enabled),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["llm-tools"] });
      success("Tool updated");
    },
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Wrench size={20} className="text-neon-mint" /> Tool Catalog
          </h1>
          <p className="text-xs text-slate-400">
            Function calling tools for LLM agents
          </p>
        </div>
        {onClose && (
          <Button size="sm" variant="ghost" onClick={onClose}>
            ✕
          </Button>
        )}
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <Input
          placeholder="Search tools..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="max-w-sm"
        />
        <div className="grid md:grid-cols-2 gap-3">
          {tools
            ?.filter((t) => t.name.includes(search))
            .map((tool) => (
              <ToolCard
                key={tool.name}
                tool={tool}
                onToggle={(enabled) =>
                  updateTool.mutate({ name: tool.name, enabled })
                }
              />
            ))}
        </div>
      </div>
    </div>
  );
}
