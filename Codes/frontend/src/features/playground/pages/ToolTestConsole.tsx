// src/features/playground/pages/ToolTestConsole.tsx
import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { playgroundAPI } from "../services/api";
import { ToolCallPanel } from "../components/ToolCallPanel";
import { ToolResultCard } from "../components/ToolResultCard";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Wrench, Play } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function ToolTestConsole({ onClose }: { onClose?: () => void }) {
  const [selectedTool, setSelectedTool] = useState("");
  const [inputJson, setInputJson] = useState("{}");
  const [result, setResult] = useState<any>(null);
  const { success, error: toastError } = useToast();

  const { data: tools, isLoading } = useQuery({
    queryKey: ["playground-tools"],
    queryFn: () => playgroundAPI.getTools(),
  });
  const testMutation = useMutation({
    mutationFn: () =>
      playgroundAPI.testTool(selectedTool, JSON.parse(inputJson)),
    onSuccess: (res) => setResult(res),
    onError: () => toastError("Test failed"),
  });

  const handleTest = () => {
    if (selectedTool && inputJson) testMutation.mutate();
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Wrench size={20} className="text-neon-mint" /> Tool Test Console
          </h1>
          <p className="text-xs text-slate-400">
            Test LLM function‑calling tools
          </p>
        </div>
        {onClose && (
          <Button size="sm" variant="ghost" onClick={onClose}>
            ✕
          </Button>
        )}
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {isLoading ? (
          <Spinner />
        ) : (
          <TerminalCard className="p-4 space-y-4">
            <Select
              label="Tool"
              options={
                tools?.map((t) => ({ value: t.name, label: t.name })) || []
              }
              value={selectedTool}
              onChange={(e) => {
                setSelectedTool(e.target.value);
                setResult(null);
              }}
            />
            <div>
              <label className="text-xs text-slate-400 block mb-1">
                Input (JSON)
              </label>
              <textarea
                rows={6}
                value={inputJson}
                onChange={(e) => setInputJson(e.target.value)}
                className="w-full bg-white/5 border border-white/10 rounded p-2 font-mono text-sm"
              />
            </div>
            <Button
              onClick={handleTest}
              disabled={!selectedTool || !inputJson || testMutation.isPending}
            >
              <Play size={14} className="mr-1" /> Run Tool
            </Button>
          </TerminalCard>
        )}
        {result && <ToolResultCard result={result} />}
      </div>
    </div>
  );
}
