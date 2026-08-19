// src/features/mlops/pages/PerformanceSnapshotView.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { mlopsAPI } from "../services/api";
import { SnapshotTable } from "../components/SnapshotTable";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Input } from "../../../components/ui/Input";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import {
  Activity,
  Camera,
  Download,
  CheckCircle,
  TrendingUp,
  Cpu,
  Layers,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

const PRESET_MODELS = [
  { id: "model-svd-transformer-v2", label: "SVD Linear Attention Transformer v2" },
  { id: "model-linear-attn-v1", label: "Linear Attention Baseline v1" },
  { id: "model-bert-base", label: "BERT Attention Compressed" },
];

export default function PerformanceSnapshotView({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [selectedModel, setSelectedModel] = useState("model-svd-transformer-v2");
  const [timeRange, setTimeRange] = useState("30d");
  const [visibleMetrics, setVisibleMetrics] = useState({
    accuracy: true,
    precision: true,
    recall: true,
    f1: true,
    drift: true,
    latency: false,
  });

  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: snapshots, isLoading } = useQuery({
    queryKey: ["performance-snapshots", selectedModel, timeRange],
    queryFn: () =>
      mlopsAPI.getPerformanceSnapshots({
        model_id: selectedModel || undefined,
        range: timeRange,
      }),
  });

  const triggerSnapshot = useMutation({
    mutationFn: (modelId: string) => mlopsAPI.triggerSnapshot(modelId),
    onSuccess: (newSnap) => {
      queryClient.invalidateQueries({ queryKey: ["performance-snapshots"] });
      success(
        "Snapshot captured",
        `Accuracy: ${(newSnap.metrics.accuracy * 100).toFixed(2)}% · F1: ${newSnap.metrics.f1} · Latency: ${newSnap.metrics.latency_p99}ms`,
      );
    },
  });

  const handleExportCSV = () => {
    if (!snapshots || snapshots.length === 0) return;
    const header = "Timestamp,Model_ID,Accuracy,Precision,Recall,F1,Latency_P50,Latency_P99,Throughput,Data_Drift\n";
    const rows = snapshots
      .map(
        (s) =>
          `"${s.timestamp}","${s.model_id}",${s.metrics.accuracy},${s.metrics.precision},${s.metrics.recall},${s.metrics.f1},${s.metrics.latency_p50},${s.metrics.latency_p99},${s.metrics.throughput},${s.data_drift_score}`,
      )
      .join("\n");
    const blob = new Blob([header + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `performance_snapshots_${selectedModel}_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    success("Export complete", "Performance snapshots downloaded as CSV");
  };

  // Prepare chart data
  const chartData =
    snapshots?.map((s) => ({
      date: new Date(s.timestamp).toLocaleDateString([], {
        month: "short",
        day: "numeric",
      }),
      accuracy: s.metrics.accuracy,
      precision: s.metrics.precision,
      recall: s.metrics.recall,
      f1: s.metrics.f1,
      latency_p99: s.metrics.latency_p99,
      drift: s.data_drift_score,
    })) || [];

  const latestSnapshot = snapshots?.[snapshots.length - 1];

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full bg-slate-950">
      {/* Header */}
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Activity size={20} className="text-neon-mint" /> Performance Snapshots
          </h1>
          <p className="text-xs text-slate-400">
            Historical validation snapshots, accuracy metrics, and inference latency telemetry
          </p>
        </div>
        <div className="flex gap-2">
          {snapshots && snapshots.length > 0 && (
            <Button
              size="sm"
              variant="ghost"
              onClick={handleExportCSV}
              title="Download CSV"
            >
              <Download size={14} className="mr-1" /> Export CSV
            </Button>
          )}
          {selectedModel && (
            <Button
              size="sm"
              onClick={() => triggerSnapshot.mutate(selectedModel)}
              disabled={triggerSnapshot.isPending}
              className="bg-neon-mint text-slate-950 font-bold hover:bg-neon-mint/90"
            >
              <Camera size={14} className="mr-1" />{" "}
              {triggerSnapshot.isPending ? "Capturing..." : "Capture Snapshot"}
            </Button>
          )}
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto p-4 space-y-5">
        {/* Model Selector and Range Bar */}
        <div className="flex flex-wrap gap-3 items-center justify-between">
          <div className="flex gap-2 flex-wrap items-center">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="bg-slate-900 border border-white/10 rounded px-3 py-1.5 text-xs text-white focus:outline-none focus:border-neon-mint w-64"
            >
              {PRESET_MODELS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.label}
                </option>
              ))}
            </select>
            <Select
              options={[
                { value: "7d", label: "Last 7 days" },
                { value: "30d", label: "Last 30 days" },
                { value: "90d", label: "Last 90 days" },
              ]}
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="w-36 text-xs"
            />
          </div>

          {/* Metric Toggle Chips */}
          <div className="flex flex-wrap gap-1.5 items-center text-xs">
            <span className="text-slate-400 mr-1">Metrics:</span>
            {[
              { key: "accuracy", label: "Accuracy", color: "#3b82f6" },
              { key: "f1", label: "F1 Score", color: "#10b981" },
              { key: "precision", label: "Precision", color: "#8b5cf6" },
              { key: "recall", label: "Recall", color: "#f59e0b" },
              { key: "drift", label: "Data Drift", color: "#ef4444" },
              { key: "latency", label: "Latency P99", color: "#ec4899" },
            ].map((m) => {
              const active = visibleMetrics[m.key as keyof typeof visibleMetrics];
              return (
                <button
                  key={m.key}
                  onClick={() =>
                    setVisibleMetrics((prev) => ({
                      ...prev,
                      [m.key]: !prev[m.key as keyof typeof visibleMetrics],
                    }))
                  }
                  className={`px-2 py-0.5 rounded text-[11px] font-mono border transition-all ${
                    active
                      ? "border-white/30 bg-white/10 text-white font-bold"
                      : "border-white/5 bg-transparent text-slate-500 opacity-60"
                  }`}
                  style={{ color: active ? m.color : undefined }}
                >
                  ● {m.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Top Summary Banner */}
        {latestSnapshot && (
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            <TerminalCard className="p-3 text-center">
              <div className="text-xs text-slate-400">Latest Accuracy</div>
              <div className="text-xl font-bold text-blue-400 font-mono mt-0.5">
                {(latestSnapshot.metrics.accuracy * 100).toFixed(2)}%
              </div>
            </TerminalCard>
            <TerminalCard className="p-3 text-center">
              <div className="text-xs text-slate-400">Latest F1 Score</div>
              <div className="text-xl font-bold text-neon-mint font-mono mt-0.5">
                {latestSnapshot.metrics.f1.toFixed(4)}
              </div>
            </TerminalCard>
            <TerminalCard className="p-3 text-center">
              <div className="text-xs text-slate-400">Latency P50 / P99</div>
              <div className="text-xl font-bold text-yellow-400 font-mono mt-0.5">
                {latestSnapshot.metrics.latency_p50}ms / {latestSnapshot.metrics.latency_p99}ms
              </div>
            </TerminalCard>
            <TerminalCard className="p-3 text-center">
              <div className="text-xs text-slate-400">Throughput</div>
              <div className="text-xl font-bold text-white font-mono mt-0.5">
                {latestSnapshot.metrics.throughput.toLocaleString()} req/s
              </div>
            </TerminalCard>
            <TerminalCard className="p-3 text-center">
              <div className="text-xs text-slate-400">Data Drift Score</div>
              <div className="text-xl font-bold text-purple-400 font-mono mt-0.5">
                {latestSnapshot.data_drift_score.toFixed(4)}
              </div>
            </TerminalCard>
          </div>
        )}

        {snapshots && snapshots.length > 0 ? (
          <>
            {/* Trend Chart */}
            <TerminalCard className="p-4 space-y-2">
              <div className="flex justify-between items-center">
                <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
                  <TrendingUp size={15} className="text-neon-mint" /> Metric Trends Over Time ({timeRange})
                </h3>
                <span className="text-xs text-slate-400">
                  {snapshots.length} evaluation points
                </span>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={chartData}>
                  <XAxis dataKey="date" stroke="#64748b" tick={{ fontSize: 10 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 10 }} domain={[0, "auto"]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      borderColor: "#334155",
                      fontSize: "11px",
                      borderRadius: "4px",
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {visibleMetrics.accuracy && (
                    <Line
                      type="monotone"
                      name="Accuracy"
                      dataKey="accuracy"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={{ r: 2 }}
                    />
                  )}
                  {visibleMetrics.f1 && (
                    <Line
                      type="monotone"
                      name="F1 Score"
                      dataKey="f1"
                      stroke="#10b981"
                      strokeWidth={2}
                      dot={{ r: 2 }}
                    />
                  )}
                  {visibleMetrics.precision && (
                    <Line
                      type="monotone"
                      name="Precision"
                      dataKey="precision"
                      stroke="#8b5cf6"
                      strokeWidth={1.5}
                      strokeDasharray="4 4"
                      dot={false}
                    />
                  )}
                  {visibleMetrics.recall && (
                    <Line
                      type="monotone"
                      name="Recall"
                      dataKey="recall"
                      stroke="#f59e0b"
                      strokeWidth={1.5}
                      strokeDasharray="4 4"
                      dot={false}
                    />
                  )}
                  {visibleMetrics.drift && (
                    <Line
                      type="monotone"
                      name="Data Drift Score"
                      dataKey="drift"
                      stroke="#ef4444"
                      strokeWidth={1.5}
                      dot={false}
                    />
                  )}
                  {visibleMetrics.latency && (
                    <Line
                      type="monotone"
                      name="Latency P99 (ms)"
                      dataKey="latency_p99"
                      stroke="#ec4899"
                      strokeWidth={2}
                      dot={{ r: 2 }}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </TerminalCard>

            {/* Snapshots Table */}
            <TerminalCard className="p-4 space-y-2">
              <div className="flex justify-between items-center">
                <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
                  <Layers size={15} className="text-neon-blue" /> Snapshot History Log
                </h3>
                <span className="text-xs text-slate-400">
                  Select any 2 checkboxes to compare side-by-side
                </span>
              </div>
              <SnapshotTable snapshots={snapshots} />
            </TerminalCard>
          </>
        ) : (
          <div className="text-center text-slate-500 py-16">
            <Activity size={32} className="mx-auto mb-2 opacity-40" />
            <p>No snapshots found for this model. Click "Capture Snapshot" to record current telemetry.</p>
          </div>
        )}
      </div>
    </div>
  );
}

