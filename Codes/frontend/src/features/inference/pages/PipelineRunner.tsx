// src/features/inference/pages/PipelineRunner.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { inferenceAPI } from "../services/api";
import { useInferenceWebSocket } from "../hooks/useInferenceWebSocket";
import { PipelineStep } from "../components/PipelineStep";
import { LogViewer } from "../components/LogViewer";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Play, Square, GitBranch } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function PipelineRunner({ onClose }: { onClose?: () => void }) {
  const [selectedPipeline, setSelectedPipeline] = useState("");
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const queryClient = useQueryClient();
  const { success, error } = useToast();
  const { logs, status, isConnected } = useInferenceWebSocket(activeRunId);

  const { data: runs, isLoading } = useQuery({
    queryKey: ["pipeline-runs"],
    queryFn: () => inferenceAPI.getPipelineRuns(),
    refetchInterval: 5000,
  });

  const triggerPipeline = useMutation({
    mutationFn: () => inferenceAPI.triggerPipeline(selectedPipeline),
    onSuccess: (res) => {
      setActiveRunId(res.data.id);
      success("Pipeline triggered");
    },
  });

  const stopPipeline = useMutation({
    mutationFn: () => inferenceAPI.stopPipeline(activeRunId!),
    onSuccess: () => success("Pipeline stopped"),
  });

  const activeRun = runs?.find((r) => r.id === activeRunId);

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <GitBranch size={20} className="text-neon-mint" /> Pipeline Runner
          </h1>
          <p className="text-xs text-slate-400">
            Execute and monitor ML pipelines
          </p>
        </div>
        {onClose && (
          <Button size="sm" variant="ghost" onClick={onClose}>
            ✕
          </Button>
        )}
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <TerminalCard className="p-4 flex gap-2 items-end">
          <Select
            options={[
              { value: "train", label: "Training Pipeline" },
              { value: "evaluate", label: "Evaluation Pipeline" },
              { value: "deploy", label: "Deployment Pipeline" },
            ]}
            value={selectedPipeline}
            onChange={(e) => setSelectedPipeline(e.target.value)}
            className="flex-1"
          />
          <Button
            onClick={() => triggerPipeline.mutate()}
            disabled={!selectedPipeline || triggerPipeline.isPending}
          >
            <Play size={14} className="mr-1" /> Run Pipeline
          </Button>
        </TerminalCard>
        {activeRun && (
          <div>
            <div className="flex justify-between items-center">
              <div>
                <span className="text-sm text-slate-400">
                  Run {activeRun.id.slice(0, 8)} · {activeRun.status}
                </span>
                <div
                  className={`text-xs ${isConnected ? "text-green-400" : "text-red-400"}`}
                >
                  {isConnected ? "● Live logs" : "○ Disconnected"}
                </div>
              </div>
              {activeRun.status === "running" && (
                <Button
                  size="sm"
                  variant="danger"
                  onClick={() => stopPipeline.mutate()}
                >
                  <Square size={14} /> Stop
                </Button>
              )}
            </div>
            <div className="space-y-2 mt-2">
              {activeRun.steps.map((s) => (
                <PipelineStep key={s.id} step={s} />
              ))}
            </div>
            <LogViewer logs={logs} />
          </div>
        )}
      </div>
    </div>
  );
}
