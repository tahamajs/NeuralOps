// src/features/llmHub/pages/PromptLibrary.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { llmHubAPI } from "../services/api";
import { PromptEditor } from "../components/PromptEditor";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Modal } from "../../../components/ui/Modal";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { Tabs, TabList, Tab, TabPanel } from "../../../components/ui/Tabs";
import { Plus, FileText, GitBranch, Trash2 } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function PromptLibrary({ onClose }: { onClose?: () => void }) {
  const [selectedPrompt, setSelectedPrompt] = useState<any>(null);
  const [showEditor, setShowEditor] = useState(false);
  const [search, setSearch] = useState("");
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: prompts, isLoading } = useQuery({
    queryKey: ["prompts"],
    queryFn: () => llmHubAPI.getPrompts(),
  });
  const createPrompt = useMutation({
    mutationFn: (data: any) => llmHubAPI.createPrompt(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["prompts"] });
      setShowEditor(false);
      setSelectedPrompt(null);
      success("Prompt created");
    },
    onError: () => error("Failed to create prompt"),
  });

  const updatePrompt = useMutation({
    mutationFn: ({ id, data }: { id: string; data: any }) =>
      llmHubAPI.updatePrompt(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["prompts"] });
      setShowEditor(false);
      setSelectedPrompt(null);
      success("Prompt updated");
    },
    onError: () => error("Failed to update prompt"),
  });

  const deletePrompt = useMutation({
    mutationFn: (id: string) => llmHubAPI.deletePrompt(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["prompts"] });
      success("Prompt deleted");
    },
    onError: () => error("Failed to delete prompt"),
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <FileText size={20} className="text-neon-mint" /> Prompt Library
          </h1>
          <p className="text-xs text-slate-400">
            Version‑controlled prompt templates
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" onClick={() => setShowEditor(true)}>
            <Plus size={14} className="mr-1" /> New Prompt
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <Input
          placeholder="Search prompts..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="max-w-sm"
        />
        <div className="grid md:grid-cols-2 gap-3">
          {prompts
            ?.filter((p) => p.name.includes(search))
            .map((prompt) => (
              <TerminalCard key={prompt.id} className="p-3">
                <div className="flex justify-between items-start">
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="font-bold text-white">{prompt.name}</h3>
                      <Badge variant="secondary">v{prompt.version}</Badge>
                    </div>
                    <div className="text-xs text-slate-400 mt-1">
                      by {prompt.created_by}
                    </div>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {prompt.tags.map((t) => (
                        <Badge key={t} variant="info">
                          {t}
                        </Badge>
                      ))}
                    </div>
                    <div className="text-xs text-slate-500 mt-2 line-clamp-2">
                      {prompt.template.slice(0, 100)}
                    </div>
                  </div>
                  <div className="flex gap-1">
                    <button
                      onClick={() => {
                        setSelectedPrompt(prompt);
                        setShowEditor(true);
                      }}
                    >
                      <GitBranch size={14} className="text-slate-400" />
                    </button>
                    <button onClick={() => deletePrompt.mutate(prompt.id)}>
                      <Trash2 size={14} className="text-red-400" />
                    </button>
                  </div>
                </div>
              </TerminalCard>
            ))}
        </div>
      </div>
      <Modal
        open={showEditor}
        onClose={() => setShowEditor(false)}
        title={selectedPrompt ? `Edit ${selectedPrompt.name}` : "New Prompt"}
      >
        <PromptEditor
          initialPrompt={selectedPrompt}
          onSave={(data) =>
            selectedPrompt
              ? updatePrompt.mutate({ id: selectedPrompt.id, data })
              : createPrompt.mutate(data)
          }
        />
      </Modal>
    </div>
  );
}
