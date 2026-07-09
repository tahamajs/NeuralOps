import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import Plot from "react-plotly.js";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { Spinner } from "../../../components/ui/Spinner";

export default function ExperimentReportPage() {
  const { id } = useParams<{ id: string }>();

  // Fetch run data
  const { data: run, isLoading: runLoading } = useQuery({
    queryKey: ["run", id],
    queryFn: async () => (await api.get(`/runs/${id}/`)).data,
    enabled: !!id,
  });

  // Fetch training history
  const { data: history, isLoading: historyLoading } = useQuery({
    queryKey: ["history", id],
    queryFn: async () => {
      const res = await api.get(`/runs/${id}/history/`);
      return res.data;
    },
    enabled: !!id,
  });

  if (runLoading || !run) return <Spinner />;

  const metrics = run.final_metrics || {};
  const accuracy = metrics.eval_accuracy ?? "—";
  const f1 = metrics.eval_f1 ?? "—";
  const loss = metrics.eval_loss ?? metrics.loss ?? "—";
  const config = run.config || {};

  // History processing for chart
  const steps = history?.steps || [];
  const losses = history?.losses || [];
  const evalSteps = history?.eval_steps || [];
  const evalLosses = history?.eval_losses || [];
  const evalAccuracies = history?.eval_accuracies || [];

  // Training summary
  const minLoss = losses.length ? Math.min(...losses).toFixed(4) : null;
  const maxLoss = losses.length ? Math.max(...losses).toFixed(4) : null;
  const finalLoss = losses.length ? losses[losses.length - 1].toFixed(4) : null;
  const bestAcc = evalAccuracies.length
    ? Math.max(...evalAccuracies.filter((v: any) => v !== null)).toFixed(4)
    : null;

  // Chart traces
  const chartData = [];
  if (steps.length && losses.length) {
    chartData.push({
      x: steps,
      y: losses,
      mode: "lines+markers",
      name: "Train Loss",
      line: { color: "#f97316", width: 2 },
      marker: { size: 4 },
    });
  }
  if (evalSteps.length && evalLosses.length) {
    chartData.push({
      x: evalSteps,
      y: evalLosses,
      mode: "lines+markers",
      name: "Eval Loss",
      line: { color: "#c084fc", width: 2 },
      marker: { size: 4 },
    });
  }
  if (evalSteps.length && evalAccuracies.length) {
    chartData.push({
      x: evalSteps,
      y: evalAccuracies,
      mode: "lines+markers",
      name: "Eval Accuracy",
      line: { color: "#10b981", width: 2 },
      marker: { size: 4 },
      yaxis: "y2",
    });
  }

  // Download helpers
  const downloadJSON = (data: any, filename: string) => {
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handlePrint = () => window.print();

  return (
    <div className="space-y-6 animate-fade-in max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-black text-white">
            📄 Experiment Report
          </h1>
          <div className="mt-2 flex items-center gap-3">
            <Badge status={run.status}>{run.status}</Badge>
            <span className="text-sm text-slate-400">ID: {run.id}</span>
          </div>
        </div>
        <div className="flex gap-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => window.history.back()}
          >
            ← Back
          </Button>
          <Button
            size="sm"
            onClick={() => window.open(`/runs/${id}/paper/`, "_blank")}
          >
            📄 LaTeX Source
          </Button>
        </div>
      </div>

      {/* Quick Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="text-center">
          <div className="text-2xl font-bold text-neon-blue">{accuracy}</div>
          <div className="text-[10px] uppercase tracking-wider text-slate-400 mt-1">
            Accuracy
          </div>
        </Card>
        <Card className="text-center">
          <div className="text-2xl font-bold text-neon-blue">{f1}</div>
          <div className="text-[10px] uppercase tracking-wider text-slate-400 mt-1">
            F1 Score
          </div>
        </Card>
        <Card className="text-center">
          <div className="text-2xl font-bold text-neon-blue">{loss}</div>
          <div className="text-[10px] uppercase tracking-wider text-slate-400 mt-1">
            Loss
          </div>
        </Card>
        <Card className="text-center">
          <div className="text-2xl font-bold text-neon-blue">
            {metrics.eval_matthews_correlation ?? "—"}
          </div>
          <div className="text-[10px] uppercase tracking-wider text-slate-400 mt-1">
            Matthews Corr.
          </div>
        </Card>
      </div>

      {/* Training Curves */}
      <Card>
        <h2 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
          <span className="w-1 h-4 bg-orange-500 rounded-full" />
          Training Curves
        </h2>
        {chartData.length > 0 ? (
          <Plot
            data={chartData}
            layout={{
              autosize: true,
              height: 400,
              margin: { t: 20, r: 70, b: 40, l: 50 },
              paper_bgcolor: "transparent",
              plot_bgcolor: "transparent",
              font: { color: "#94a3b8" },
              legend: {
                x: 1,
                y: 1,
                xanchor: "right",
                bgcolor: "rgba(0,0,0,0.5)",
              },
              hovermode: "closest",
              yaxis: { title: "Loss", gridcolor: "#334155" },
              yaxis2: {
                title: "Accuracy",
                overlaying: "y",
                side: "right",
                range: [0, 1],
                gridcolor: "#334155",
              },
            }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        ) : (
          <p className="text-slate-500 text-sm text-center py-8">
            No training history available.
          </p>
        )}
      </Card>

      {/* Configuration */}
      <Card>
        <h2 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
          <span className="w-1 h-4 bg-purple-500 rounded-full" />
          Configuration
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div className="text-xs">
            <span className="text-slate-400">Model</span>
            <div className="text-white font-semibold">
              {config.model_name || "—"}
            </div>
          </div>
          <div className="text-xs">
            <span className="text-slate-400">Task</span>
            <div className="text-white font-semibold uppercase">
              {config.task_name || config.task || "—"}
            </div>
          </div>
          <div className="text-xs">
            <span className="text-slate-400">Attention</span>
            <div className="text-white font-semibold">
              {config.attention_type || "—"}
            </div>
          </div>
          <div className="text-xs">
            <span className="text-slate-400">Method</span>
            <div className="text-white font-semibold">
              {config.method || "—"}
            </div>
          </div>
          <div className="text-xs">
            <span className="text-slate-400">Epochs</span>
            <div className="text-white font-semibold">
              {config.epochs ?? "—"}
            </div>
          </div>
          <div className="text-xs">
            <span className="text-slate-400">Batch size</span>
            <div className="text-white font-semibold">
              {config.batch_size ?? "—"}
            </div>
          </div>
          <div className="text-xs">
            <span className="text-slate-400">Learning rate</span>
            <div className="text-white font-semibold">
              {config.learning_rate ?? "—"}
            </div>
          </div>
          <div className="text-xs">
            <span className="text-slate-400">Max length</span>
            <div className="text-white font-semibold">
              {config.max_seq_length ?? "—"}
            </div>
          </div>
          <div className="text-xs">
            <span className="text-slate-400">Compression rank</span>
            <div className="text-white font-semibold">
              {config.compression_rank ?? "—"}
            </div>
          </div>
          <div className="text-xs">
            <span className="text-slate-400">GPU required</span>
            <div className="text-white font-semibold">
              {config.gpu_requirements ?? "—"}
            </div>
          </div>
        </div>
      </Card>

      {/* Timeline */}
      <Card>
        <h2 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
          <span className="w-1 h-4 bg-cyan-500 rounded-full" />
          Timeline
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-slate-400">Created</span>
            <div className="text-white">
              {new Date(run.created_at).toLocaleString()}
            </div>
          </div>
          <div>
            <span className="text-slate-400">Started</span>
            <div className="text-white">
              {run.started_at ? new Date(run.started_at).toLocaleString() : "—"}
            </div>
          </div>
          <div>
            <span className="text-slate-400">Completed</span>
            <div className="text-white">
              {run.completed_at
                ? new Date(run.completed_at).toLocaleString()
                : "—"}
            </div>
          </div>
          <div>
            <span className="text-slate-400">Total steps</span>
            <div className="text-white">{run.total_steps || "—"}</div>
          </div>
        </div>
      </Card>

      {/* Artifacts & Downloads */}
      <Card>
        <h2 className="text-sm font-bold text-white mb-3">
          📁 Artifacts & Downloads
        </h2>
        <div className="flex flex-wrap gap-3">
          <a href={`/runs/${id}/model/`} download>
            <Button variant="secondary" size="sm">
              🤖 Download Model
            </Button>
          </a>
          <Button
            variant="ghost"
            size="sm"
            onClick={() =>
              downloadJSON(run.config, `config_${id?.slice(0, 8)}.json`)
            }
          >
            📄 Config JSON
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() =>
              downloadJSON(run.final_metrics, `metrics_${id?.slice(0, 8)}.json`)
            }
          >
            📊 Metrics JSON
          </Button>
          <a
            href={`/runs/${id}/history/?format=csv`}
            target="_blank"
            rel="noopener noreferrer"
          >
            <Button variant="ghost" size="sm">
              📈 History CSV
            </Button>
          </a>
          <a
            href={`/runs/${id}/paper/`}
            target="_blank"
            rel="noopener noreferrer"
          >
            <Button variant="ghost" size="sm">
              📝 LaTeX Source
            </Button>
          </a>
        </div>
      </Card>

      {/* Full JSON (collapsible) */}
      <Card>
        <details>
          <summary className="text-sm font-bold text-white cursor-pointer">
            📜 Full Configuration (JSON)
          </summary>
          <pre className="mt-2 bg-cyber-800 p-3 rounded-lg text-xs text-slate-400 overflow-auto max-h-64">
            {JSON.stringify(run.config, null, 2)}
          </pre>
        </details>
      </Card>

      <Card>
        <details>
          <summary className="text-sm font-bold text-white cursor-pointer">
            📊 Complete Metrics
          </summary>
          <pre className="mt-2 bg-cyber-800 p-3 rounded-lg text-xs text-slate-400 overflow-auto max-h-64">
            {JSON.stringify(run.final_metrics, null, 2)}
          </pre>
        </details>
      </Card>

      {/* Training Summary Stats */}
      {(minLoss || bestAcc) && (
        <Card>
          <h2 className="text-sm font-bold text-white mb-3">
            📈 Training Summary
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {minLoss && (
              <div className="text-center">
                <div className="text-lg font-bold text-white">{minLoss}</div>
                <div className="text-[10px] text-slate-400 uppercase">
                  Min Loss
                </div>
              </div>
            )}
            {maxLoss && (
              <div className="text-center">
                <div className="text-lg font-bold text-white">{maxLoss}</div>
                <div className="text-[10px] text-slate-400 uppercase">
                  Max Loss
                </div>
              </div>
            )}
            {finalLoss && (
              <div className="text-center">
                <div className="text-lg font-bold text-white">{finalLoss}</div>
                <div className="text-[10px] text-slate-400 uppercase">
                  Final Loss
                </div>
              </div>
            )}
            {bestAcc && (
              <div className="text-center">
                <div className="text-lg font-bold text-white">{bestAcc}</div>
                <div className="text-[10px] text-slate-400 uppercase">
                  Best Accuracy
                </div>
              </div>
            )}
          </div>
        </Card>
      )}

      {/* Footer */}
      <div className="text-center text-xs text-slate-500 pt-6 border-t border-white/5 flex justify-between items-center">
        <span>
          Generated by Transformer Toolkit • {new Date().toLocaleString()}
        </span>
        <Button variant="ghost" size="sm" onClick={handlePrint}>
          🖨️ Print / Save as PDF
        </Button>
      </div>
    </div>
  );
}
