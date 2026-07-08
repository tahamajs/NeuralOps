import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { autonomousAPI } from "../services/api";
import { usePipelineWebSocket } from "../hooks/usePipelineWebSocket";
import { PipelineDAG } from "../components/PipelineDAG";
import { NodeDetailsPanel } from "../components/NodeDetailsPanel";
import { ExecutionLogs } from "../components/ExecutionLogs";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { Tabs, TabList, Tab, TabPanel } from "../../../components/ui/Tabs";
import { Play, Pause, Square, RefreshCw, GitBranch } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function AutonomousPipelineMonitor({
  onClose,
}: {
  onClose?: () => void;
}) {
  const { runId: urlRunId } = useParams();
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("graph");
  const queryClient = useQueryClient();
  const { success, error: toastError } = useToast();

  const {
    data: pipelineData,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ["pipeline-run", urlRunId],
    queryFn: () => autonomousAPI.getPipelineRun(urlRunId!),
    enabled: !!urlRunId,
  });

  const {
    pipeline: wsPipeline,
    logs,
    currentStep,
    isConnected,
    error: wsError,
  } = usePipelineWebSocket(urlRunId || null);
  const pipeline = (wsPipeline as any) || pipelineData;

  const pauseMutation = useMutation({
    mutationFn: () => autonomousAPI.pausePipeline(urlRunId!),
    onSuccess: () => success("Pipeline paused"),
    onError: () => toastError("Failed to pause"),
  });
  const resumeMutation = useMutation({
    mutationFn: () => autonomousAPI.resumePipeline(urlRunId!),
    onSuccess: () => success("Pipeline resumed"),
    onError: () => toastError("Failed to resume"),
  });
  const cancelMutation = useMutation({
    mutationFn: () => autonomousAPI.cancelPipeline(urlRunId!),
    onSuccess: () => success("Pipeline cancelled"),
    onError: () => toastError("Failed to cancel"),
  });
  const retryStepMutation = useMutation({
    mutationFn: (stepId: string) => autonomousAPI.retryStep(urlRunId!, stepId),
    onSuccess: () => refetch(),
  });

  if (isLoading) return <Spinner />;
  if (!pipeline)
    return <div className="p-4 text-red-400">Pipeline run not found</div>;

  const statusColor = {
    idle: "secondary",
    running: "info",
    paused: "warning",
    completed: "success",
    failed: "danger",
  }[pipeline.status];

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white">
            Autonomous Pipeline Monitor
          </h1>
          <p className="text-xs text-slate-400">
            {pipeline.name} · {pipeline.id.slice(0, 8)}
          </p>
        </div>
        <div className="flex gap-2">
          <Badge variant={statusColor as any}>{pipeline.status}</Badge>
          <span
            className={`text-xs ${isConnected ? "text-green-400" : "text-yellow-400"}`}
          >
            {isConnected ? "● Live" : "○ Reconnecting"}
          </span>
          <div className="flex gap-1 ml-2">
            {pipeline.status === "running" && (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => pauseMutation.mutate()}
              >
                <Pause size={14} />
              </Button>
            )}
            {pipeline.status === "paused" && (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => resumeMutation.mutate()}
              >
                <Play size={14} />
              </Button>
            )}
            {(pipeline.status === "running" ||
              pipeline.status === "paused") && (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => cancelMutation.mutate()}
              >
                <Square size={14} />
              </Button>
            )}
            <Button size="sm" variant="ghost" onClick={() => refetch()}>
              <RefreshCw size={14} />
            </Button>
            {onClose && (
              <Button size="sm" variant="ghost" onClick={onClose}>
                ✕
              </Button>
            )}
          </div>
        </div>
      </div>
      <div className="flex-1 overflow-hidden p-4">
        <div className="grid lg:grid-cols-3 gap-4 h-full">
          <div className="lg:col-span-2">
            <TerminalCard className="h-full p-2">
              <Tabs value={activeTab} onValueChange={setActiveTab}>
                <TabList>
                  <Tab value="graph">Graph View</Tab>
                  <Tab value="logs">Execution Logs</Tab>
                </TabList>
                <TabPanel value="graph" className="h-[500px]">
                  <PipelineDAG
                    steps={pipeline.steps}
                    onNodeClick={setSelectedNodeId}
                  />
                </TabPanel>
                <TabPanel value="logs" className="h-[500px]">
                  <ExecutionLogs logs={logs} />
                </TabPanel>
              </Tabs>
            </TerminalCard>
          </div>
          <div className="space-y-4">
            <NodeDetailsPanel
              step={pipeline.steps.find((s) => s.id === selectedNodeId)}
              currentStep={currentStep}
              onRetry={() =>
                selectedNodeId && retryStepMutation.mutate(selectedNodeId)
              }
            />
            <TerminalCard className="p-3">
              <div className="text-sm font-bold text-white mb-2">
                Overall Progress
              </div>
              <div className="w-full bg-white/10 rounded-full h-2">
                <div
                  className="bg-neon-blue h-2 rounded-full"
                  style={{ width: `${pipeline.progress}%` }}
                />
              </div>
              <div className="flex justify-between mt-2 text-xs">
                <span>Progress</span>
                <span className="text-neon-mint">{pipeline.progress}%</span>
              </div>
              {pipeline.error_message && (
                <div className="mt-3 p-2 rounded bg-red-500/10 border border-red-500/30 text-red-400 text-xs">
                  {pipeline.error_message}
                </div>
              )}
            </TerminalCard>
          </div>
        </div>
      </div>
    </div>
  );
}
