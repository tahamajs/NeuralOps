import { useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { useWebSocket, WSStatus } from "../../../hooks/useWebSocket";
import { useLiveStore } from "../../../stores/experimentStore";
import {
  Play,
  Square,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  Wifi,
  WifiOff,
} from "lucide-react";

interface TrainingControlPanelProps {
  runId: string;
  status: string;
  progress: { current: number; total: number; eta: number };
}

export default function TrainingControlPanel({
  runId,
  status,
  progress,
}: TrainingControlPanelProps) {
  const queryClient = useQueryClient();
  const liveStore = useLiveStore();
  const { wsStatus } = useWebSocket(runId);

  const pct =
    progress.total > 0
      ? Math.min(100, (progress.current / progress.total) * 100)
      : 0;

  // Stop mutation with error handling
  const stopMut = useMutation({
    mutationFn: () => api.post(`/api/jobs/${runId}/stop/`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["run", runId] });
      liveStore.addLog("Stop request sent.");
    },
    onError: (err: any) => {
      liveStore.addLog(`Failed to stop: ${err.message}`);
    },
  });

  // Retry mutation
  const retryMut = useMutation({
    mutationFn: () => api.post(`/api/jobs/${runId}/retry/`),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["run", runId] });
      liveStore.addLog("Job re-queued for retry.");
    },
    onError: (err: any) => {
      liveStore.addLog(`Retry failed: ${err.message}`);
    },
  });

  const wsStatusIcon = {
    connecting: <Wifi className="animate-pulse text-yellow-400" size={14} />,
    open: <Wifi className="text-green-400" size={14} />,
    closed: <WifiOff className="text-red-400" size={14} />,
    error: <AlertTriangle className="text-red-400" size={14} />,
  }[wsStatus];

  return (
    <Card className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-display font-bold text-white flex items-center gap-2">
          {wsStatusIcon}
          Training Control
        </h3>
        <Badge status={status}>{status}</Badge>
      </div>

      {/* Progress bar with ETA */}
      <div className="space-y-1">
        <div className="flex justify-between text-xs text-slate-400">
          <span>
            Step {progress.current || 0} / {progress.total || 0}
          </span>
          {progress.eta > 0 && <span>ETA: {Math.round(progress.eta)}s</span>}
        </div>
        <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-neon-blue to-purple-500 transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        {status === "running" && (
          <Button
            variant="danger"
            size="sm"
            onClick={() => stopMut.mutate()}
            disabled={stopMut.isPending}
          >
            <Square size={14} className="mr-1" />
            {stopMut.isPending ? "Stopping..." : "Stop Training"}
          </Button>
        )}
        {(status === "failed" ||
          status === "stopped" ||
          status === "preempted") && (
          <Button
            variant="primary"
            size="sm"
            onClick={() => retryMut.mutate()}
            disabled={retryMut.isPending}
          >
            <RefreshCw size={14} className="mr-1" />
            {retryMut.isPending ? "Retrying..." : "Retry / Resume"}
          </Button>
        )}
      </div>

      {/* Connection status hint */}
      {wsStatus !== "open" && (
        <div className="text-[10px] text-slate-500 flex items-center gap-1">
          {wsStatusIcon}
          {wsStatus === "connecting"
            ? "Connecting to live stream..."
            : wsStatus === "closed"
              ? "Reconnecting..."
              : wsStatus === "error"
                ? "Connection error – some updates may be delayed."
                : ""}
        </div>
      )}
    </Card>
  );
}
