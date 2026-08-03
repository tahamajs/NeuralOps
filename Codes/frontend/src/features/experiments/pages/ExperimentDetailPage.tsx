import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { useWebSocket } from "../../../hooks/useWebSocket";
import { useLiveStore } from "../../../stores/experimentStore";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";
import { Card } from "../../../components/ui/Card";
import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { LiveLossPlot } from "../../../components/widgets/LiveLossPlot";
import { LiveLogViewer } from "../../../components/widgets/LiveLogViewer";
import { ProgressBar } from "../../../components/widgets/ProgressBar";
import { ConfigSummary } from "../../../components/widgets/ConfigSummary";
import { ArtifactList } from "../../../components/widgets/ArtifactList";
import { DownloadModelButton } from "../../../components/widgets/DownloadModelButton";
import { DownloadPaperButton } from "../../../components/widgets/DownloadPaperButton";
import TrainingControlPanel from "../components/TrainingControlPanel";
import CustomMetricsDashboard from "../components/CustomMetricsDashboard";
import ExperimentTimeline from "../components/ExperimentTimeline";
import { Experiment } from "../../../types/experiment";
import { useState } from "react";
import TrainingMonitorPanel from "../components/TrainingMonitorPanel";
import PaperRecommendationsPanel from "../../agent/components/PaperRecommendationsPanel";

export default function ExperimentDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { wsStatus } = useWebSocket(id);
  const [showAllMetrics, setShowAllMetrics] = useState(false);

  const { data: experiment, isLoading } = useQuery<Experiment>({
    queryKey: ["experiment", id],
    queryFn: async () =>
      (await api.get(ENDPOINTS.EXPERIMENTS.DETAIL(id!))).data,
    enabled: !!id,
    refetchInterval: 10000,
  });

  const live = useLiveStore();

  if (isLoading || !experiment) return <Spinner className="py-16" />;

  // Merge live data
  const status = live.status || experiment.status;
  const progress = live.progress || {
    current: experiment.current_step,
    total: experiment.total_steps,
    eta: 0,
  };
  const logs = live.logs.length > 0 ? live.logs : experiment.logs || [];
  const metrics =
    live.metrics.loss.length > 0
      ? live.metrics
      : experiment.metrics || { loss: [], accuracy: [] };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-display font-bold text-white">
            {experiment.name || "Unnamed Session"}
          </h1>
          <p className="text-xs text-slate-400 font-mono mt-1">
            {experiment.id}
          </p>
        </div>
        <Badge status={status} />
      </div>

      {/* ────── Training Control Panel (stop / retry / connection) ────── */}
      <TrainingControlPanel runId={id!} status={status} progress={progress} />

      {/* Progress Bar (if running) */}
      {(status === "running" || status === "queued") && (
        <Card>
          <ProgressBar progress={progress} />
        </Card>
      )}
      <TrainingMonitorPanel runId={id!} status={status} />
      <PaperRecommendationsPanel runId={id!} />

      {/* Metrics & Configuration */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <h3 className="text-sm font-display font-bold text-white mb-3 flex items-center gap-2">
            <span className="w-1 h-4 bg-neon-blue rounded-full" />
            Training Loss
          </h3>
          <LiveLossPlot metrics={metrics} />
        </Card>
        <Card>
          <h3 className="text-sm font-display font-bold text-white mb-3 flex items-center gap-2">
            <span className="w-1 h-4 bg-purple-500 rounded-full" />
            Configuration
          </h3>
          <ConfigSummary config={experiment.config} />
        </Card>
      </div>

      {/* Custom Metrics Dashboard */}
      <CustomMetricsDashboard runId={id!} />

      {/* Live Logs */}
      <Card>
        <h3 className="text-sm font-display font-bold text-white mb-3 flex items-center gap-2">
          <span className="w-1 h-4 bg-green-500 rounded-full" />
          Live Logs
        </h3>
        <LiveLogViewer logs={logs} />
      </Card>

      {/* Experiment Timeline */}
      <ExperimentTimeline run={experiment} />

      {/* Environment Info (collapsible) */}
      <Card>
        <details>
          <summary className="text-sm font-display font-bold text-white cursor-pointer">
            📌 Code & Data Versioning / Environment
          </summary>
          <div className="mt-4 grid grid-cols-2 gap-4 text-xs">
            <div>
              <span className="text-slate-400">Python:</span>{" "}
              <span className="text-white">
                {(experiment as any).python_version || "—"}
              </span>
            </div>
            <div>
              <span className="text-slate-400">Git Commit:</span>{" "}
              <span className="text-white font-mono">
                {(experiment as any).git_commit_hash?.slice(0, 7) || "—"}
              </span>
            </div>
            <div>
              <span className="text-slate-400">Branch:</span>{" "}
              <span className="text-white">
                {(experiment as any).git_branch || "—"}
              </span>
            </div>
            <div>
              <span className="text-slate-400">DVC Hash:</span>{" "}
              <span className="text-white font-mono">
                {(experiment as any).dvc_data_hash?.slice(0, 12) || "—"}
              </span>
            </div>
          </div>
        </details>
      </Card>

      {/* Config History (quick diff) */}
      <Card>
        <details>
          <summary className="text-sm font-display font-bold text-white cursor-pointer">
            ⚙️ Configuration History & Diff
          </summary>
          {/* This can be replaced with the full ConfigHistoryDiff component if desired */}
          <div className="mt-4 text-xs text-slate-500">
            Use the dedicated config history page for version‑by‑version diffs.
            <br />
            <Button
              size="sm"
              variant="ghost"
              className="mt-2"
              onClick={() =>
                window.open(`/jobs/${id}/config-history`, "_blank")
              }
            >
              Open Config History
            </Button>
          </div>
        </details>
      </Card>

      {/* Artifacts, Model, Paper */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <Card>
          <h3 className="text-sm font-display font-bold text-white mb-3">
            Artifacts
          </h3>
          {(ArtifactList as any)({ experimentId: id! })}
        </Card>
        <Card>
          <h3 className="text-sm font-display font-bold text-white mb-3">
            Model
          </h3>
          {(DownloadModelButton as any)({ experimentId: id! })}
        </Card>
        <Card>
          <h3 className="text-sm font-display font-bold text-white mb-3">
            Paper
          </h3>
          {(DownloadPaperButton as any)({ experimentId: id! })}
        </Card>
      </div>
    </div>
  );
}
