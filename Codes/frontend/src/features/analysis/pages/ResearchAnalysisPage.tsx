import { useState, useMemo, useCallback } from "react";
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
import { BarChart3, Download, FileDown } from "lucide-react";

// ---------- Types ----------
interface Run {
  id: string;
  name?: string;
  status: string;
  config: Record<string, any>;
  final_metrics: Record<string, any>;
  created_at: string;
}

// ---------- Helpers ----------
function getMetric(run: Run, metric: string): number {
  const m = run.final_metrics || {};
  if (metric === "accuracy") return m.eval_accuracy ?? m.accuracy ?? 0;
  if (metric === "loss") return m.eval_loss ?? m.loss ?? 0;
  if (metric === "f1") return m.eval_f1 ?? m.f1 ?? 0;
  return 0;
}

function uniqueValues(runs: Run[], key: string) {
  return Array.from(new Set(runs.map((r) => r.config?.[key]).filter(Boolean)));
}

// ---------- Main Component ----------
export default function ResearchAnalysisPage() {
  const queryClient = useQueryClient();
  const [selectedRuns, setSelectedRuns] = useState<Set<string>>(new Set());
  const [filterTask, setFilterTask] = useState("");
  const [filterAttention, setFilterAttention] = useState("");
  const [filterStatus, setFilterStatus] = useState("");
  const [metricToChart, setMetricToChart] = useState("accuracy");
  const [showDetails, setShowDetails] = useState(false);
  const [detailRun, setDetailRun] = useState<Run | null>(null);
  const [noteText, setNoteText] = useState("");
  const [sortBy, setSortBy] = useState<"date" | "accuracy" | "loss">("date");
  const [sortAsc, setSortAsc] = useState(false);
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 20;

  // --- Queries ---
  const { data: runs = [], isLoading: runsLoading } = useQuery<Run[]>({
    queryKey: ["runs"],
    queryFn: async () => {
      const res = await api.get("/api/runs/?all=true");
      return res.data;
    },
    refetchInterval: 15000,
  });

  const { data: notes = [] } = useQuery({
    queryKey: ["notes"],
    queryFn: async () =>
      (await api.get(ENDPOINTS.NOTES.LIST)).data?.results ?? [],
  });

  const createNoteMut = useMutation({
    mutationFn: async (text: string) => {
      const runId =
        selectedRuns.size === 1 ? Array.from(selectedRuns)[0] : undefined;
      await api.post(ENDPOINTS.NOTES.LIST, {
        content: text,
        title: "Research insight",
        tags: ["analysis"],
        run: runId || null,
      });
    },
    onSuccess: () => {
      setNoteText("");
      queryClient.invalidateQueries({ queryKey: ["notes"] });
    },
  });

  const deleteNoteMut = useMutation({
    mutationFn: async (id: string) => api.delete(`/api/notes/v1/notes/${id}/`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["notes"] }),
  });

  // --- Derived data ---
  const tasks = uniqueValues(runs, "task_name");
  const attentions = uniqueValues(runs, "attention_type");

  const filteredRuns = useMemo(() => {
    let filtered = runs;
    if (filterTask)
      filtered = filtered.filter((r) => r.config?.task_name === filterTask);
    if (filterAttention)
      filtered = filtered.filter(
        (r) => r.config?.attention_type === filterAttention,
      );
    if (filterStatus)
      filtered = filtered.filter((r) => r.status === filterStatus);
    return filtered;
  }, [runs, filterTask, filterAttention, filterStatus]);

  const sortedRuns = useMemo(() => {
    const arr = [...filteredRuns];
    arr.sort((a, b) => {
      let cmp = 0;
      if (sortBy === "date")
        cmp =
          new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      else if (sortBy === "accuracy")
        cmp = getMetric(b, "accuracy") - getMetric(a, "accuracy");
      else if (sortBy === "loss")
        cmp = getMetric(a, "loss") - getMetric(b, "loss");
      return sortAsc ? -cmp : cmp;
    });
    return arr;
  }, [filteredRuns, sortBy, sortAsc]);

  // Pagination
  const totalPages = Math.ceil(sortedRuns.length / PAGE_SIZE);
  const pagedRuns = sortedRuns.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  const toggleSelect = useCallback((id: string) => {
    setSelectedRuns((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const selectAll = (checked: boolean) => {
    if (checked) setSelectedRuns(new Set(pagedRuns.map((r) => r.id)));
    else setSelectedRuns(new Set());
  };

  const selectedArray = Array.from(selectedRuns)
    .map((id) => runs.find((r) => r.id === id)!)
    .filter(Boolean);

  // Chart data
  const chartData = selectedArray.map((r) => ({
    x: [
      `${r.config?.task_name?.toUpperCase() ?? "?"} (${r.config?.attention_type})`,
    ],
    y: [getMetric(r, metricToChart)],
    type: "bar" as const,
    name: r.name || r.id.slice(0, 8),
    marker: { color: "#3b82f6" },
  }));

  const exportCSV = () => {
    const headers = [
      "ID",
      "Task",
      "Model",
      "Attention",
      "Accuracy",
      "Loss",
      "F1",
      "Status",
      "Date",
    ];
    const rows = runs.map((r) => [
      r.id,
      r.config?.task_name || "",
      r.config?.model_name || "",
      r.config?.attention_type || "",
      getMetric(r, "accuracy"),
      getMetric(r, "loss"),
      getMetric(r, "f1"),
      r.status,
      new Date(r.created_at).toLocaleDateString(),
    ]);
    const csv = [headers, ...rows].map((row) => row.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `experiments_${new Date().toISOString().slice(0, 19)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (runsLoading) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-display font-bold text-white flex items-center gap-2">
            <BarChart3 size={24} className="text-neon-blue" />
            Research <span className="text-neon-blue">Analysis</span>
          </h1>
          <p className="text-slate-400 text-sm">
            Compare experiments, annotate findings, and export reports
          </p>
        </div>
        <Button variant="ghost" size="sm" onClick={exportCSV}>
          <FileDown size={14} className="mr-1" /> Export CSV
        </Button>
      </div>

      {/* Filters */}
      <Card className="flex flex-wrap gap-4 items-end">
        <Select
          label="Task"
          options={[
            { value: "", label: "All" },
            ...tasks.map((t) => ({ value: t, label: t.toUpperCase() })),
          ]}
          value={filterTask}
          onChange={(e) => setFilterTask(e.target.value)}
        />
        <Select
          label="Attention"
          options={[
            { value: "", label: "All" },
            ...attentions.map((a) => ({ value: a, label: a })),
          ]}
          value={filterAttention}
          onChange={(e) => setFilterAttention(e.target.value)}
        />
        <Select
          label="Status"
          options={[
            { value: "", label: "All" },
            { value: "completed", label: "Completed" },
            { value: "running", label: "Running" },
            { value: "failed", label: "Failed" },
          ]}
          value={filterStatus}
          onChange={(e) => setFilterStatus(e.target.value)}
        />
      </Card>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card>
            <div className="flex justify-between items-center mb-3">
              <h2 className="text-sm font-bold text-white">
                Experiments ({filteredRuns.length})
              </h2>
              <Button
                size="sm"
                disabled={selectedRuns.size < 2}
                onClick={() => {}}
              >
                Compare {selectedRuns.size > 0 ? `(${selectedRuns.size})` : ""}
              </Button>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left text-slate-400 uppercase tracking-wider border-b border-white/5">
                    <th className="py-2 px-2">
                      <input
                        type="checkbox"
                        onChange={(e) => selectAll(e.target.checked)}
                        checked={
                          pagedRuns.length > 0 &&
                          selectedRuns.size === pagedRuns.length
                        }
                        className="accent-neon-blue"
                      />
                    </th>
                    <th
                      className="py-2 px-2 cursor-pointer"
                      onClick={() => {
                        setSortBy("date");
                        setSortAsc(!sortAsc);
                      }}
                    >
                      ID
                    </th>
                    <th className="py-2 px-2">Task</th>
                    <th className="py-2 px-2">Attention</th>
                    <th
                      className="py-2 px-2 cursor-pointer"
                      onClick={() => {
                        setSortBy("accuracy");
                        setSortAsc(!sortAsc);
                      }}
                    >
                      Acc
                    </th>
                    <th
                      className="py-2 px-2 cursor-pointer"
                      onClick={() => {
                        setSortBy("loss");
                        setSortAsc(!sortAsc);
                      }}
                    >
                      Loss
                    </th>
                    <th className="py-2 px-2">Status</th>
                    <th className="py-2 px-2">Date</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {pagedRuns.map((run) => (
                    <tr
                      key={run.id}
                      className="hover:bg-white/5 cursor-pointer"
                      onDoubleClick={() => {
                        setDetailRun(run);
                        setShowDetails(true);
                      }}
                    >
                      <td className="py-2 px-2">
                        <input
                          type="checkbox"
                          checked={selectedRuns.has(run.id)}
                          onChange={() => toggleSelect(run.id)}
                          className="accent-neon-blue"
                        />
                      </td>
                      <td className="py-2 px-2 font-mono text-neon-blue">
                        {run.id.slice(0, 8)}
                      </td>
                      <td className="py-2 px-2">
                        {(run.config?.task_name || "?").toUpperCase()}
                      </td>
                      <td className="py-2 px-2">
                        <span className="px-2 py-0.5 rounded bg-purple-500/10 text-purple-300 text-[10px] font-bold">
                          {run.config?.attention_type || "full"}
                        </span>
                      </td>
                      <td className="py-2 px-2">
                        {getMetric(run, "accuracy").toFixed(4)}
                      </td>
                      <td className="py-2 px-2">
                        {getMetric(run, "loss").toFixed(4)}
                      </td>
                      <td className="py-2 px-2">
                        <Badge status={run.status}>{run.status}</Badge>
                      </td>
                      <td className="py-2 px-2 text-slate-400">
                        {new Date(run.created_at).toLocaleDateString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex justify-center gap-2 mt-4">
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={page <= 1}
                  onClick={() => setPage(page - 1)}
                >
                  Prev
                </Button>
                <span className="py-2 text-xs text-slate-400">
                  {page} / {totalPages}
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={page >= totalPages}
                  onClick={() => setPage(page + 1)}
                >
                  Next
                </Button>
              </div>
            )}
          </Card>
        </div>

        {/* Right Side: Chart + Notes */}
        <div className="space-y-6">
          <Card>
            <h3 className="text-sm font-bold text-white mb-3">
              Comparison Chart
            </h3>
            {selectedRuns.size === 0 ? (
              <p className="text-slate-500 text-xs">Select runs to compare.</p>
            ) : (
              <Plot
                data={chartData}
                layout={{
                  autosize: true,
                  height: 300,
                  margin: { t: 20, r: 20, b: 40, l: 50 },
                  paper_bgcolor: "transparent",
                  plot_bgcolor: "transparent",
                  font: { color: "#94a3b8" },
                  barmode: "group",
                }}
                useResizeHandler
                style={{ width: "100%" }}
              />
            )}
            <Select
              label="Metric"
              options={[
                { value: "accuracy", label: "Accuracy" },
                { value: "loss", label: "Loss" },
                { value: "f1", label: "F1 Score" },
              ]}
              value={metricToChart}
              onChange={(e) => setMetricToChart(e.target.value)}
              className="mt-2"
            />
          </Card>

          <Card>
            <h3 className="text-sm font-bold text-white mb-3">Quick Notes</h3>
            <textarea
              className="w-full bg-white/5 border border-white/10 rounded-xl p-3 text-white text-sm min-h-[80px] focus:border-neon-blue/50 resize-none"
              placeholder="Write observations..."
              value={noteText}
              onChange={(e) => setNoteText(e.target.value)}
            />
            <div className="flex gap-2 mt-2">
              <Button
                size="sm"
                onClick={() => createNoteMut.mutate(noteText)}
                disabled={!noteText.trim()}
              >
                Save Note
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() =>
                  queryClient.invalidateQueries({ queryKey: ["notes"] })
                }
              >
                Refresh
              </Button>
            </div>
            <div className="mt-3 space-y-2 max-h-48 overflow-y-auto scrollbar-thin">
              {notes.slice(0, 10).map((note: any) => (
                <div key={note.id} className="bg-cyber-800 p-3 rounded-lg">
                  <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                    <span>{new Date(note.created_at).toLocaleString()}</span>
                    <button
                      onClick={() => deleteNoteMut.mutate(note.id)}
                      className="text-red-400 hover:text-red-300"
                    >
                      🗑
                    </button>
                  </div>
                  <p className="text-xs text-white">{note.content}</p>
                  {note.run && (
                    <span className="text-[9px] text-slate-500">
                      Run: {note.run.slice(0, 8)}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>

      {/* Detail Panel (shown on double-click) */}
      {showDetails && detailRun && (
        <Card>
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-sm font-bold text-white">Run Details</h3>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowDetails(false)}
            >
              ✕
            </Button>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
            <div>
              <span className="text-slate-400">ID:</span>{" "}
              <span className="text-white font-mono">{detailRun.id}</span>
            </div>
            <div>
              <span className="text-slate-400">Status:</span>{" "}
              <Badge status={detailRun.status}>{detailRun.status}</Badge>
            </div>
            <div>
              <span className="text-slate-400">Model:</span>{" "}
              <span className="text-white">{detailRun.config?.model_name}</span>
            </div>
            <div>
              <span className="text-slate-400">Task:</span>{" "}
              <span className="text-white">
                {detailRun.config?.task_name?.toUpperCase()}
              </span>
            </div>
            <div>
              <span className="text-slate-400">Attention:</span>{" "}
              <span className="text-white">
                {detailRun.config?.attention_type}
              </span>
            </div>
            <div>
              <span className="text-slate-400">Epochs:</span>{" "}
              <span className="text-white">{detailRun.config?.epochs}</span>
            </div>
            <div>
              <span className="text-slate-400">Batch:</span>{" "}
              <span className="text-white">{detailRun.config?.batch_size}</span>
            </div>
            <div>
              <span className="text-slate-400">LR:</span>{" "}
              <span className="text-white">
                {detailRun.config?.learning_rate}
              </span>
            </div>
          </div>
          <div className="mt-4">
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
              Metrics
            </h4>
            <pre className="bg-cyber-800 p-3 rounded-lg text-xs text-slate-400 overflow-auto max-h-32">
              {JSON.stringify(detailRun.final_metrics, null, 2)}
            </pre>
          </div>
        </Card>
      )}
    </div>
  );
}
