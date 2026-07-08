// src/features/llmHub/pages/MemoryManager.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { llmHubAPI } from "../services/api";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Modal } from "../../../components/ui/Modal";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { Select } from "../../../components/ui/Select";
import { Brain, Plus, Trash2, Clock } from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { useToast } from "../../../components/ui/Toast";

export default function MemoryManager({ onClose }: { onClose?: () => void }) {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newKey, setNewKey] = useState("");
  const [newValue, setNewValue] = useState("");
  const [newType, setNewType] = useState("session");
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: memories, isLoading } = useQuery({
    queryKey: ["llm-memory"],
    queryFn: () => llmHubAPI.getMemoryEntries(),
  });
  const createMemory = useMutation({
    mutationFn: (data: any) => llmHubAPI.createMemoryEntry(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["llm-memory"] });
      setShowCreateModal(false);
      success("Memory created");
    },
  });
  const deleteMemory = useMutation({
    mutationFn: (id: string) => llmHubAPI.deleteMemoryEntry(id),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["llm-memory"] }),
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Brain size={20} className="text-neon-mint" /> Memory Manager
          </h1>
          <p className="text-xs text-slate-400">
            Persistent knowledge for LLM interactions
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" onClick={() => setShowCreateModal(true)}>
            <Plus size={14} className="mr-1" /> Add Memory
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-3">
        {memories?.map((m) => (
          <TerminalCard key={m.id} className="p-3">
            <div className="flex justify-between items-start">
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-mono font-bold text-white">
                    {m.key}
                  </span>
                  <Badge variant="info">{m.type}</Badge>
                  <span className="text-xs text-slate-400 flex items-center gap-1">
                    <Clock size={10} />{" "}
                    {formatDistanceToNow(new Date(m.created_at), {
                      addSuffix: true,
                    })}
                  </span>
                </div>
                <div className="text-sm text-slate-300 mt-1">{m.value}</div>
              </div>
              <button onClick={() => deleteMemory.mutate(m.id)}>
                <Trash2 size={14} className="text-red-400" />
              </button>
            </div>
          </TerminalCard>
        ))}
      </div>
      <Modal
        open={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        title="Add Memory"
      >
        <div className="space-y-4">
          <Input
            label="Key"
            value={newKey}
            onChange={(e) => setNewKey(e.target.value)}
            placeholder="user.preferences"
          />
          <Input
            label="Value"
            value={newValue}
            onChange={(e) => setNewValue(e.target.value)}
            placeholder="Prefers concise answers with code examples"
          />
          <Select
            label="Type"
            options={[
              { value: "session", label: "Session" },
              { value: "user", label: "User" },
              { value: "project", label: "Project" },
              { value: "global", label: "Global" },
            ]}
            value={newType}
            onChange={(e) => setNewType(e.target.value)}
          />
          <Button
            onClick={() =>
              createMemory.mutate({
                key: newKey,
                value: newValue,
                type: newType,
              })
            }
            disabled={!newKey || !newValue}
          >
            Save
          </Button>
        </div>
      </Modal>
    </div>
  );
}
