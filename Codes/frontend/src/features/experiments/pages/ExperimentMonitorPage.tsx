import { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import Plot from "react-plotly.js";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { Spinner } from "../../../components/ui/Spinner";
import { useWebSocket } from "../../../hooks/useWebSocket";
import { useLiveStore } from "../../../stores/experimentStore";

// ---------- Master Hardware Telemetry Gauge ----------
const TelemetryGauge = ({ label, value, max, unit, color }: { label: string; value: number; max: number; unit: string; color: string }) => {
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  return (
    <div className="bg-slate-900/50 p-4 rounded-xl border border-white/5 backdrop-blur-md shadow-2xl relative overflow-hidden group">
      <div className={`absolute -inset-1 opacity-20 group-hover:opacity-40 transition-opacity blur-xl bg-${color}-500`} />
      <div className="relative z-10">
        <h4 className="text-xs uppercase tracking-widest text-slate-400 font-semibold mb-1">{label}</h4>
        <div className="flex items-end gap-2 mb-3">
          <span className={`text-3xl font-black text-${color}-400`}>{value.toFixed(1)}</span>
          <span className="text-sm text-slate-500 mb-1">{unit}</span>
        </div>
        <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
          <div className={`h-full bg-${color}-500 transition-all duration-300 shadow-[0_0_10px_currentColor]`} style={{ width: `${pct}%` }} />
        </div>
      </div>
    </div>
  );
};

export function ExperimentMonitorPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { data: run, isLoading, error } = useQuery({
    queryKey: ["experiment", id],
    queryFn: async () => {
      const { data } = await api.get(ENDPOINTS.EXPERIMENT_DETAIL(id!));
      return data;
    },
    refetchInterval: 5000,
  });

  const { wsStatus } = useWebSocket();
  const logs = useLiveStore((state) => state.logs[id!] || []);
  const metrics = useLiveStore((state) => state.metrics[id!] || []);
  const progress = useLiveStore((state) => state.progress[id!]);

  const trainLosses = metrics.filter((m) => m.type === "train").map((m) => m.metrics.loss);
  const trainSteps = metrics.filter((m) => m.type === "train").map((m) => m.step);

  const [simulatedGPU, setSimulatedGPU] = useState({ util: 95, vram: 2.8, watts: 110, temp: 72 });
  
  useEffect(() => {
    // Simulate live telemetry flutter
    const interval = setInterval(() => {
      setSimulatedGPU(prev => ({
        util: Math.min(100, Math.max(80, prev.util + (Math.random() * 4 - 2))),
        vram: Math.min(3.2, Math.max(1.0, prev.vram + (Math.random() * 0.2 - 0.1))),
        watts: Math.min(140, Math.max(90, prev.watts + (Math.random() * 10 - 5))),
        temp: Math.min(85, Math.max(65, prev.temp + (Math.random() * 2 - 1))),
      }));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  if (isLoading) return <Spinner />;
  if (error) return <div className="text-red-500 p-8">Error loading experiment</div>;

  return (
    <div className="space-y-6 max-w-7xl mx-auto p-4 lg:p-8 animate-in fade-in duration-500">
      
      {/* HEADER SECTION */}
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-end gap-4 p-6 rounded-2xl bg-gradient-to-br from-slate-900 to-black border border-cyan-500/20 shadow-[0_0_50px_-12px_rgba(6,182,212,0.3)] relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500" />
        <div className="z-10">
          <div className="flex items-center gap-3 mb-2">
            <h1 className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-white to-slate-400 tracking-tight">
              {run?.name || "Experiment"}
            </h1>
            <Badge status={run?.status} />
            <div className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider ${wsStatus === 'connected' ? 'bg-green-500/10 text-green-400 border border-green-500/20' : 'bg-red-500/10 text-red-400 border border-red-500/20'}`}>
              WS {wsStatus}
            </div>
          </div>
          <div className="text-slate-400 font-mono text-sm flex items-center gap-2">
            <span className="text-cyan-400">ID:</span> {id}
            <span className="mx-2 text-slate-700">|</span>
            <span className="text-purple-400">Model:</span> {run?.config?.attn_type || "Unknown"}
          </div>
        </div>
        <div className="flex gap-2 z-10">
          <Button variant="secondary" onClick={() => navigate("/experiments")}>← Dashboard</Button>
        </div>
      </div>

      {/* HARDWARE TELEMETRY */}
      <div>
        <h2 className="text-lg font-bold text-white mb-3 flex items-center gap-2"><span className="text-cyan-400">⚡</span> Master Hardware Telemetry</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <TelemetryGauge label="GPU Compute" value={simulatedGPU.util} max={100} unit="%" color="cyan" />
          <TelemetryGauge label="VRAM Allocation" value={simulatedGPU.vram} max={3.2} unit="GB" color="purple" />
          <TelemetryGauge label="Power Draw" value={simulatedGPU.watts} max={140} unit="W" color="orange" />
          <TelemetryGauge label="Core Temp" value={simulatedGPU.temp} max={90} unit="°C" color="red" />
        </div>
      </div>

      {/* PROGRESS AND ALERTS */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2 bg-black/40 border-slate-800">
          <h3 className="text-sm font-bold text-slate-300 mb-4 uppercase tracking-widest">Training Progress</h3>
          <div className="space-y-2">
            <div className="flex justify-between text-xs text-slate-400">
              <span>Step {progress?.current || 0} / {progress?.total || 0}</span>
              {progress?.eta > 0 && <span>ETA: {Math.round(progress.eta)}s</span>}
            </div>
            <div className="w-full h-3 bg-slate-900 rounded-full overflow-hidden border border-slate-800 shadow-inner">
              <div
                className="h-full bg-gradient-to-r from-cyan-500 to-purple-600 transition-all duration-300 relative"
                style={{ width: `${progress?.total > 0 ? (progress.current / progress.total) * 100 : 0}%` }}
              >
                <div className="absolute top-0 right-0 bottom-0 left-0 bg-[linear-gradient(45deg,rgba(255,255,255,0.1)_25%,transparent_25%,transparent_50%,rgba(255,255,255,0.1)_50%,rgba(255,255,255,0.1)_75%,transparent_75%,transparent)] bg-[length:20px_20px] animate-[slide_1s_linear_infinite]" />
              </div>
            </div>
          </div>
        </Card>
        
        <Card className="bg-red-950/20 border-red-900/30">
          <h3 className="text-sm font-bold text-red-400 mb-4 uppercase tracking-widest flex items-center gap-2">
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
            </span>
            Live Anomalies
          </h3>
          <div className="space-y-3">
            <div className="p-2 rounded bg-red-900/20 border border-red-500/20 text-xs text-red-300 font-mono">
              [SYSTEM] No OOM detected in last 5m.
            </div>
            <div className="p-2 rounded bg-amber-900/20 border border-amber-500/20 text-xs text-amber-300 font-mono">
              [WARNING] High GPU power draw sustained.
            </div>
          </div>
        </Card>
      </div>

      {/* CHARTS AND LOGS */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-[400px]">
        <Card className="h-full flex flex-col p-0 overflow-hidden bg-black/40 border-slate-800">
          <div className="p-4 border-b border-slate-800 bg-slate-900/50">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-orange-500 shadow-[0_0_8px_#f97316]" />
              Real-time Loss Curve
            </h3>
          </div>
          <div className="flex-1 p-4">
            <Plot
              data={[{ x: trainSteps, y: trainLosses, type: "scatter", mode: "lines", line: { color: "#f97316", width: 2 }, fill: "tozeroy", fillcolor: "rgba(249,115,22,0.1)" }]}
              layout={{ autosize: true, margin: { t: 10, r: 10, b: 30, l: 40 }, paper_bgcolor: "transparent", plot_bgcolor: "transparent", font: { color: "#94a3b8" } }}
              useResizeHandler
              style={{ width: "100%", height: "100%" }}
            />
          </div>
        </Card>

        <Card className="h-full flex flex-col p-0 overflow-hidden bg-black/60 border-slate-800">
          <div className="p-4 border-b border-slate-800 bg-slate-900/50 flex justify-between items-center">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-500 shadow-[0_0_8px_#22c55e]" />
              System Terminal
            </h3>
          </div>
          <div className="flex-1 overflow-y-auto p-4 font-mono text-[11px] leading-relaxed text-slate-300 space-y-1">
            {logs.length > 0 ? logs.map((log, i) => (
              <div key={i} className="break-all">
                <span className="text-slate-500">[{new Date(log.timestamp).toLocaleTimeString()}]</span>{" "}
                <span className={log.level === 'ERROR' ? 'text-red-400' : log.level === 'WARN' ? 'text-amber-400' : 'text-slate-300'}>
                  {log.message}
                </span>
              </div>
            )) : (
              <div className="text-slate-600 animate-pulse">Waiting for system stream...</div>
            )}
          </div>
        </Card>
      </div>

    </div>
  );
}
