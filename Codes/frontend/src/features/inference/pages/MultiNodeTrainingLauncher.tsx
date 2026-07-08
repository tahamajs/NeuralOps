// src/features/inference/pages/MultiNodeTrainingLauncher.tsx
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { inferenceAPI } from "../services/api";
import { TrainingConfigForm } from "../components/TrainingConfigForm";
import { LogViewer } from "../components/LogViewer";
import { useInferenceWebSocket } from "../hooks/useInferenceWebSocket";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Cpu, Rocket, Network, Clock } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function MultiNodeTrainingLauncher({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [config, setConfig] = useState({
    script_path: "",
    num_nodes: 1,
    gpus_per_node: 1,
    cpu_per_node: 4,
    memory_per_node_gb: 16,
    max_wall_time_seconds: 3600,
    environment_variables: { CUDA_VISIBLE_DEVICES: "0,1" },
  });
  const [runId, setRunId] = useState<string | null>(null);
  const { success, error } = useToast();
  const { logs, status, isConnected } = useInferenceWebSocket(runId);

  const launchTraining = useMutation({
    mutationFn: () => inferenceAPI.launchMultiNodeTraining(config),
    onSuccess: (res) => {
      setRunId(res.id);
      success("Training launched");
    },
  });

  if (runId) {
    return (
      <div className="flex flex-col h-full">
        <div className="flex justify-between items-center p-4 border-b border-white/10">
          <h1 className="text-xl font-bold text-white">
            Training Run: {runId.slice(0, 8)}
          </h1>
          <div
            className={`text-xs ${isConnected ? "text-green-400" : "text-red-400"}`}
          >
            {isConnected ? "● Live" : "○ Offline"}
          </div>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
        <div className="flex-1 overflow-auto p-4">
          <LogViewer logs={logs} />
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Network size={20} className="text-neon-mint" /> Multi‑Node Training
            Launcher
          </h1>
          <p className="text-xs text-slate-400">
            Distributed training across clusters
          </p>
        </div>
        {onClose && (
          <Button size="sm" variant="ghost" onClick={onClose}>
            ✕
          </Button>
        )}
      </div>
      <div className="flex-1 overflow-auto p-4 max-w-2xl mx-auto">
        <TerminalCard className="p-6 space-y-4">
          <TrainingConfigForm config={config} onChange={setConfig} />
          <Button
            onClick={() => launchTraining.mutate()}
            disabled={!config.script_path || launchTraining.isPending}
            className="w-full"
          >
            <Rocket size={14} className="mr-1" /> Launch Training
          </Button>
        </TerminalCard>
      </div>
    </div>
  );
}
