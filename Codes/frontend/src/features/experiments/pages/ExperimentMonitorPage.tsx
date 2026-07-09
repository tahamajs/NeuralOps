import { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import Plot from "react-plotly.js";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Badge } from "../../../components/ui/Badge";
import { Spinner } from "../../../components/ui/Spinner";
import {
  useFloatingWindowStore,
  WINDOW_TYPES,
} from "../../../components/floating/FloatingWindowManager";
import { useWebSocket } from "../../../hooks/useWebSocket";
import { useLiveStore } from "../../../stores/experimentStore";

// ---------- Sub-components ----------

/** Progress bar with step/ETA display */
const ProgressSection = ({ progress }: { progress: any }) => {
  const pct =
    progress?.total > 0
      ? Math.min(100, (progress.current / progress.total) * 100)
      : 0;
  return (
    <Card>
      <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
        <span className="w-1 h-4 bg-neon-blue rounded-full" />
        Progress
      </h3>
      <div className="space-y-2">
        <div className="flex justify-between text-xs text-slate-400">
          <span>
            Step {progress?.current || 0} / {progress?.total || 0}
          </span>
          {progress?.eta > 0 && <span>ETA: {Math.round(progress.eta)}s</span>}
        </div>
        <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-neon-blue to-purple-500 transition-all duration-500"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
      <div className="grid grid-cols-4 gap-3 mt-4 text-center">
        {[
          {
            label: "Step",
            value: `${progress?.current || 0}/${progress?.total || 0}`,
          },
          {
            label: "ETA",
            value: progress?.eta > 0 ? `${Math.round(progress.eta)}s` : "--",
          },
          { label: "Loss", value: "--" },
          { label: "Accuracy", value: "--" },
        ].map((s) => (
          <div key={s.label} className="bg-cyber-800 rounded-lg p-2">
            <div className="text-[10px] text-slate-500 uppercase tracking-wider">
              {s.label}
            </div>
            <div className="text-sm font-bold text-white">{s.value}</div>
          </div>
        ))}
      </div>
    </Card>
  );
};

/** Live logs viewer */
const LogViewer = ({ logs }: { logs: string[] }) => {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [logs]);
  return (
    <div
      ref={ref}
      className="bg-cyber-800 rounded-lg p-3 text-green-400 font-mono text-xs h-48 overflow-y-auto scrollbar-thin"
    >
      {logs.length === 0 && (
        <span className="text-slate-500">Waiting for logs...</span>
      )}
      {logs.map((line, i) => (
        <div key={i} className="whitespace-pre-wrap break-all">
          {line}
        </div>
      ))}
    </div>
  );
};

/** Final metrics grid */
const FinalMetrics = ({ metrics }: { metrics: Record<string, any> }) => {
  if (!Object.keys(metrics).length)
    return <p className="text-slate-500 text-xs">No final metrics yet.</p>;
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {Object.entries(metrics).map(([k, v]) => (
        <div key={k} className="bg-cyber-800 rounded-lg p-3 text-center">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider">
            {k.replace(/_/g, " ")}
          </div>
          <div className="text-lg font-bold text-white">
            {typeof v === "number" ? v.toFixed(4) : v}
          </div>
        </div>
      ))}
    </div>
  );
};

/** Code & Data Versioning */
const VersioningInfo = ({ run }: { run: any }) => (
  <div className="grid grid-cols-2 gap-4 text-xs">
    <div className="bg-cyber-800 p-3 rounded-lg">
      <div className="text-slate-500 mb-1">Git Commit</div>
      <div className="font-mono text-neon-blue">
        {run.git_commit_hash ? (
          <a
            href={`${run.git_repository}/commit/${run.git_commit_hash}`}
            target="_blank"
            rel="noopener noreferrer"
            className="hover:underline"
          >
            {run.git_commit_hash.slice(0, 8)}
          </a>
        ) : (
          "—"
        )}
      </div>
      <div className="text-slate-400 mt-1">Branch: {run.git_branch || "—"}</div>
      {run.git_dirty && <Badge status="warning">Uncommitted changes</Badge>}
    </div>
    <div className="bg-cyber-800 p-3 rounded-lg">
      <div className="text-slate-500 mb-1">Data Version (DVC)</div>
      <div className="font-mono text-neon-blue">
        {run.data_git_commit_id?.slice(0, 8) || "—"}
      </div>
      <div className="text-slate-400 mt-1">
        Hash: {run.dvc_data_hash || "—"}
      </div>
    </div>
  </div>
);

/** Config History Diff */
const ConfigHistoryDiff = ({ runId }: { runId: string }) => {
  const [va, setVa] = useState("");
  const [vb, setVb] = useState("");
  const [diff, setDiff] = useState<any>(null);
  const { data: history } = useQuery({
    queryKey: ["configHistory", runId],
    queryFn: async () =>
      (await api.get(`/runs/${runId}/config/history/`)).data,
  });
  const versions = history?.versions || [];
  const options = versions.map((v: any) => ({
    value: v.version.toString(),
    label: `v${v.version} (${new Date(v.created_at).toLocaleString()})`,
  }));

  const handleCompare = async () => {
    if (!va || !vb) return;
    const res = await api.get(`/runs/${runId}/config/compare/${va}/${vb}/`);
    setDiff(res.data);
  };

  return (
    <div className="space-y-3">
      <div className="flex gap-3">
        <Select
          label="Version A"
          options={options}
          value={va}
          onChange={(e) => setVa(e.target.value)}
        />
        <Select
          label="Version B"
          options={options}
          value={vb}
          onChange={(e) => setVb(e.target.value)}
        />
        <Button size="sm" onClick={handleCompare} className="self-end">
          Compare
        </Button>
      </div>
      {diff && (
        <pre className="bg-cyber-800 p-3 rounded-lg text-xs text-slate-400 max-h-64 overflow-auto">
          {JSON.stringify(diff.diff || diff, null, 2)}
        </pre>
      )}
    </div>
  );
};

/** Training Analytics Panel */
const TrainingAnalytics = ({
  trainSteps,
  trainLosses,
  evalAccuracies,
  evalSteps,
}: any) => {
  if (!trainSteps?.length)
    return <p className="text-slate-500 text-xs">No training data yet.</p>;
  const lastStep = trainSteps[trainSteps.length - 1] || 0;
  const minLoss = Math.min(...trainLosses).toFixed(4);
  const maxLoss = Math.max(...trainLosses).toFixed(4);
  const avgLast10 = (
    trainLosses.slice(-10).reduce((a: number, b: number) => a + b, 0) /
    Math.min(10, trainLosses.length)
  ).toFixed(4);
  const reduction = (
    ((trainLosses[0] - trainLosses[trainLosses.length - 1]) / trainLosses[0]) *
    100
  ).toFixed(1);
  const bestEval = evalAccuracies?.length
    ? Math.max(...evalAccuracies.filter((v: any) => v !== null)).toFixed(4)
    : "--";

  // Convergence warning
  let showWarning = false;
  if (trainLosses.length > 50) {
    const recent = trainLosses.slice(-50);
    showWarning = recent[recent.length - 1] >= recent[0] - 0.01;
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          { label: "Step", value: lastStep },
          { label: "Min Loss", value: minLoss },
          { label: "Max Loss", value: maxLoss },
          { label: "Avg Loss (last 10)", value: avgLast10 },
          { label: "Reduction", value: `${reduction}%` },
          { label: "Best Eval", value: bestEval },
          { label: "Eval Count", value: evalSteps?.length || 0 },
        ].map((s) => (
          <div key={s.label} className="bg-cyber-800 p-2 rounded text-center">
            <div className="text-[10px] text-slate-500 uppercase">
              {s.label}
            </div>
            <div className="text-sm font-bold text-white">{s.value}</div>
          </div>
        ))}
      </div>
      {showWarning && (
        <div className="bg-yellow-500/10 border border-yellow-500/20 text-yellow-400 p-2 rounded text-xs text-center">
          ⚠️ Loss has not decreased for the last 50 steps. Consider early
          stopping.
        </div>
      )}
    </div>
  );
};

/** Custom Metrics Dashboard (Plotly) */
const CustomMetricsDashboard = ({ runId }: { runId: string }) => {
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([]);
  const { data: metrics, isLoading } = useQuery({
    queryKey: ["customMetrics", runId],
    queryFn: async () => (await api.get(`/runs/${runId}/metrics/`)).data,
    refetchInterval: 10000,
  });
  if (isLoading) return <Spinner className="py-4" />;
  if (!metrics || !Object.keys(metrics).length)
    return <p className="text-slate-500 text-xs">No custom metrics logged.</p>;

  const scalarMetrics = Object.entries(metrics).filter(
    ([_, d]: any) => d.type === "scalar",
  );
  const imageMetrics = Object.entries(metrics).filter(
    ([_, d]: any) => d.type === "image",
  );
  const tableMetrics = Object.entries(metrics).filter(
    ([_, d]: any) => d.type === "table",
  );

  const chartData = selectedMetrics
    .filter((name) => metrics[name]?.type === "scalar")
    .map((name) => ({
      x: metrics[name].steps,
      y: metrics[name].values,
      type: "scatter",
      mode: "lines+markers",
      name,
    }));

  return (
    <div className="space-y-4">
      {scalarMetrics.length > 0 && (
        <>
          <div className="flex items-center gap-2">
            <select
              multiple
              className="bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-white text-xs h-32 w-64 scrollbar-thin"
              value={selectedMetrics}
              onChange={(e) =>
                setSelectedMetrics(
                  Array.from(e.target.selectedOptions, (o) => o.value),
                )
              }
            >
              {scalarMetrics.map(([name]) => (
                <option key={name} value={name} className="bg-cyber-800">
                  {name}
                </option>
              ))}
            </select>
            <Button size="sm" onClick={() => {}}>
              Update Chart
            </Button>
          </div>
          {chartData.length > 0 && (
            <Plot
              data={chartData}
              layout={{
                autosize: true,
                height: 350,
                margin: { t: 20, r: 20, b: 40, l: 50 },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                font: { color: "#94a3b8" },
              }}
              useResizeHandler
              style={{ width: "100%" }}
            />
          )}
        </>
      )}

      {imageMetrics.length > 0 && (
        <div>
          <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
            Logged Images
          </h4>
          <div className="grid grid-cols-3 gap-2">
            {imageMetrics.map(([name, data]: any) => {
              const lastImg = data.values[data.values.length - 1];
              const step = data.steps[data.steps.length - 1];
              return (
                <div
                  key={name}
                  className="bg-cyber-800 p-2 rounded text-center"
                >
                  <img
                    src={lastImg}
                    alt={`${name} step ${step}`}
                    className="max-h-24 mx-auto rounded"
                  />
                  <div className="text-[10px] text-slate-400 mt-1">
                    {name} (step {step})
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {tableMetrics.length > 0 && (
        <div>
          <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
            Logged Tables
          </h4>
          {tableMetrics.map(([name, data]: any) => {
            const last = data.values[data.values.length - 1];
            const headers = Object.keys(last);
            return (
              <div key={name} className="mb-3">
                <div className="text-xs text-white mb-1">{name}</div>
                <table className="w-full text-xs border border-white/10 rounded-lg overflow-hidden">
                  <thead>
                    <tr className="bg-cyber-700">
                      {headers.map((h) => (
                        <th key={h} className="p-2 text-left text-slate-400">
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-t border-white/5">
                      {headers.map((h) => (
                        <td key={h} className="p-2 text-white">
                          {String(last[h])}
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

// ---------- Main Page Component ----------
export default function ExperimentMonitorPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { addWindow } = useFloatingWindowStore();

  // WebSocket connection
  useWebSocket(id);

  const liveStore = useLiveStore();

  const { data: run, isLoading } = useQuery({
    queryKey: ["run", id],
    queryFn: async () =>
      (await api.get(ENDPOINTS.EXPERIMENTS.DETAIL(id!))).data,
    enabled: !!id,
    refetchInterval: 5000,
  });

  const { data: history } = useQuery({
    queryKey: ["runHistory", id],
    queryFn: async () => (await api.get(`/runs/${id}/history/`)).data,
    enabled: !!id,
  });

  // Merge live state with fetched data
  const status = liveStore.status || run?.status || "unknown";
  const logs =
    liveStore.logs.length > 0 ? liveStore.logs : run?.logs?.split("\n") || [];
  const progress = liveStore.progress || {
    current: run?.current_step || 0,
    total: run?.total_steps || 0,
    eta: run?.estimated_remaining_seconds || 0,
  };
  const trainSteps = history?.steps || [];
  const trainLosses = history?.losses || [];
  const evalSteps = history?.eval_steps || [];
  const evalLosses = history?.eval_losses || [];
  const evalAccuracies = history?.eval_accuracies || [];

  // Action mutations
  const stopMut = useMutation({
    mutationFn: () => api.post(`/api/jobs/${id}/stop/`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["run", id] }),
  });
  const retryMut = useMutation({
    mutationFn: () => api.post(`/api/jobs/${id}/retry/`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["run", id] }),
  });
  const deleteMut = useMutation({
    mutationFn: () => api.delete(`/runs/${id}/`),
    onSuccess: () => navigate("/experiments"),
  });
  const evalMut = useMutation({
    mutationFn: () => api.post(`/runs/${id}/evaluate/`),
  });
  const paperMut = useMutation({
    mutationFn: async () => {
      const res = await api.get(`/runs/${id}/paper/`, {
        responseType: "blob",
      });
      const url = URL.createObjectURL(res.data);
      const a = document.createElement("a");
      a.href = url;
      a.download = String(res.headers["content-type"])?.includes("pdf")
        ? `paper_${id}.pdf`
        : `paper_${id}.tex`;
      a.click();
      URL.revokeObjectURL(url);
    },
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-black text-white tracking-tighter">
            {run.name || "Experiment"}{" "}
            <span className="text-slate-500 text-sm font-mono">
              ({id?.slice(0, 8)})
            </span>
          </h1>
          <Badge status={status} />
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="ghost" size="sm" onClick={() => evalMut.mutate()}>
            📊 Evaluate
          </Button>
          <Button variant="danger" size="sm" onClick={() => stopMut.mutate()}>
            ⏹ Stop
          </Button>
          <Button variant="ghost" size="sm" onClick={() => retryMut.mutate()}>
            🔄 Retry
          </Button>
          <Button variant="danger" size="sm" onClick={() => deleteMut.mutate()}>
            🗑 Delete
          </Button>
          <Button variant="ghost" size="sm" onClick={() => paperMut.mutate()}>
            📄 Generate Paper
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => window.open(`/runs/${id}/report/`, "_blank")}
          >
            📑 Full Report
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() =>
              window.open(`/runs/${id}/paper_preview/`, "_blank")
            }
          >
            📄 PDF Preview
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() =>
              addWindow({
                id: WINDOW_TYPES.TERMINAL,
                title: "Terminal",
                icon: ">_",
                content: WINDOW_TYPES.TERMINAL,
              })
            }
          >
            🖥️ Terminal
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => navigate("/experiments")}
          >
            ← Dashboard
          </Button>
        </div>
      </div>

      {/* Progress Section */}
      <ProgressSection progress={progress} />

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
            <span className="w-1 h-4 bg-orange-500 rounded-full" />
            Training Loss
          </h3>
          {trainSteps.length > 0 ? (
            <Plot
              data={[
                {
                  x: trainSteps,
                  y: trainLosses,
                  type: "scatter",
                  mode: "lines",
                  line: { color: "#f97316", width: 2 },
                  fill: "tozeroy",
                  fillcolor: "rgba(249,115,22,0.1)",
                },
              ]}
              layout={{
                autosize: true,
                height: 250,
                margin: { t: 10, r: 10, b: 30, l: 40 },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                font: { color: "#94a3b8" },
              }}
              useResizeHandler
              style={{ width: "100%" }}
            />
          ) : (
            <p className="text-slate-500 text-xs">No training data yet.</p>
          )}
        </Card>
        <Card>
          <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
            <span className="w-1 h-4 bg-purple-500 rounded-full" />
            Evaluation Metrics
          </h3>
          {evalSteps.length > 0 ? (
            <Plot
              data={[
                {
                  x: evalSteps,
                  y: evalLosses,
                  type: "scatter",
                  mode: "lines",
                  name: "Eval Loss",
                  line: { color: "#c084fc", width: 2 },
                },
                {
                  x: evalSteps,
                  y: evalAccuracies,
                  type: "scatter",
                  mode: "lines",
                  name: "Accuracy",
                  yaxis: "y2",
                  line: { color: "#10b981", width: 2 },
                },
              ]}
              layout={{
                autosize: true,
                height: 250,
                margin: { t: 10, r: 10, b: 30, l: 40 },
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                font: { color: "#94a3b8" },
                yaxis: { title: "Loss" },
                yaxis2: {
                  title: "Accuracy",
                  overlaying: "y",
                  side: "right",
                  range: [0, 1],
                },
              }}
              useResizeHandler
              style={{ width: "100%" }}
            />
          ) : (
            <p className="text-slate-500 text-xs">No evaluation data yet.</p>
          )}
        </Card>
      </div>

      {/* Live Logs */}
      <Card>
        <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
          <span className="w-1 h-4 bg-green-500 rounded-full" />
          Live Logs
        </h3>
        <LogViewer logs={logs} />
      </Card>

      {/* Final Metrics */}
      <Card>
        <h3 className="text-sm font-bold text-white mb-3">📊 Final Metrics</h3>
        <FinalMetrics metrics={run?.final_metrics || {}} />
      </Card>

      {/* Code & Data Versioning */}
      <Card>
        <h3 className="text-sm font-bold text-white mb-3">
          📌 Code & Data Versioning
        </h3>
        <VersioningInfo run={run} />
        <Button
          variant="ghost"
          size="sm"
          className="mt-3"
          onClick={async () => {
            const env = await api.get(`/runs/${id}/environment/`);
            alert(JSON.stringify(env.data, null, 2));
          }}
        >
          📦 Capture Environment
        </Button>
      </Card>

      {/* Custom Metrics Dashboard */}
      <Card>
        <h3 className="text-sm font-bold text-white mb-3">📈 Custom Metrics</h3>
        <CustomMetricsDashboard runId={id!} />
      </Card>

      {/* Config History Diff */}
      <Card>
        <h3 className="text-sm font-bold text-white mb-3">
          ⚙️ Configuration History
        </h3>
        <ConfigHistoryDiff runId={id!} />
      </Card>

      {/* Training Analytics */}
      <Card>
        <h3 className="text-sm font-bold text-white mb-3">
          📈 Training Analytics
        </h3>
        <TrainingAnalytics
          trainSteps={trainSteps}
          trainLosses={trainLosses}
          evalAccuracies={evalAccuracies}
          evalSteps={evalSteps}
        />
      </Card>

      {/* Artifacts & Downloads */}
      <Card>
        <h3 className="text-sm font-bold text-white mb-3">
          📁 Artifacts & Downloads
        </h3>
        <div className="flex gap-2">
          <a href={`/runs/${id}/model/`} download>
            <Button variant="secondary" size="sm">
              🤖 Download Model
            </Button>
          </a>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              const blob = new Blob([JSON.stringify(run.config, null, 2)], {
                type: "application/json",
              });
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url;
              a.download = `config_${id}.json`;
              a.click();
              URL.revokeObjectURL(url);
            }}
          >
            📄 Config JSON
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              const blob = new Blob(
                [JSON.stringify(run.final_metrics, null, 2)],
                { type: "application/json" },
              );
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url;
              a.download = `metrics_${id}.json`;
              a.click();
              URL.revokeObjectURL(url);
            }}
          >
            📊 Metrics JSON
          </Button>
        </div>
      </Card>
    </div>
  );
}
