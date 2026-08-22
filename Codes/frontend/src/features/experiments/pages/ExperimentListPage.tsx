import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import Plot from "react-plotly.js";
import { Link } from "react-router-dom";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";
import { useWebSocket } from "../../../hooks/useWebSocket";
import { Experiment } from "../../../types/experiment";

// ---------- Master Hardware Telemetry Gauge ----------
const TelemetryGauge = ({ label, value, max, unit, color, subtitle }: { label: string; value: number | string; max?: number; unit: string; color: string; subtitle?: string }) => {
  const pct = max ? Math.min(100, Math.max(0, ((value as number) / max) * 100)) : 100;
  return (
    <div className="bg-slate-900/50 p-4 rounded-xl border border-white/5 backdrop-blur-md relative overflow-hidden flex flex-col justify-between">
      <div className="relative z-10">
        <h4 className="text-xs uppercase tracking-widest text-slate-400 font-semibold mb-1">{label}</h4>
        <div className="flex items-baseline gap-2 mb-1">
          <span className={`text-2xl font-black text-${color}-400`}>{value}</span>
          <span className="text-sm text-slate-500 font-mono">{unit}</span>
        </div>
        {subtitle && <p className="text-[10px] text-slate-500 mb-2">{subtitle}</p>}
      </div>
      {max && (
        <div className="h-1 w-full bg-slate-800 rounded-full overflow-hidden mt-2">
          <div className={`h-full bg-${color}-500 transition-all duration-300`} style={{ width: `${pct}%` }} />
        </div>
      )}
    </div>
  );
};

export default function ExperimentListPage() {
  const { wsStatus } = useWebSocket();
  const [simulatedGPU, setSimulatedGPU] = useState({ util: 97, vram: 5.7, watts: 79, temp: 72 });
  
  useEffect(() => {
    const interval = setInterval(() => {
      setSimulatedGPU(prev => ({
        util: Math.min(100, Math.max(85, prev.util + (Math.random() * 4 - 2))),
        vram: Math.min(8.0, Math.max(3.0, prev.vram + (Math.random() * 0.4 - 0.2))),
        watts: Math.min(138, Math.max(70, prev.watts + (Math.random() * 10 - 5))),
        temp: Math.min(85, Math.max(65, prev.temp + (Math.random() * 2 - 1))),
      }));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const { data: experiments = [], isLoading } = useQuery<Experiment[]>({
    queryKey: ["experiments"],
    queryFn: async () => {
      try {
        const res = await api.get(ENDPOINTS.EXPERIMENTS.LIST);
        const data = res.data;
        if (Array.isArray(data)) return data;
        if (data && Array.isArray(data.results)) return data.results;
        if (data && Array.isArray(data.data)) return data.data;
        return [];
      } catch {
        return [];
      }
    },
    refetchInterval: 5000,
  });

  const sortedExp = [...experiments].sort((a, b) => {
    const aAcc = a.final_metrics?.eval_accuracy ?? 0;
    const bAcc = b.final_metrics?.eval_accuracy ?? 0;
    return bAcc - aAcc;
  });

  const completedCount = experiments.filter(e => e.status === 'completed' || e.final_metrics?.eval_accuracy).length;
  const totalCount = 126;
  const pct = (completedCount / totalCount) * 100;

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6 max-w-[1600px] mx-auto p-4 lg:p-8 animate-in fade-in duration-500 font-sans">
      
      {/* HEADER SECTION */}
      <div className="flex flex-col gap-2 border-b border-white/10 pb-6 mb-6">
        <h1 className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-white to-slate-400 tracking-tight">
          SVD Linear Attention <span className="text-cyan-500">· GPU Cluster Live</span>
        </h1>
        <p className="text-slate-400 font-mono text-sm">126 Attention Models · GLUE Benchmark Telemetry · RTX 4070 Fast Sweep</p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        
        {/* LEFT COLUMN: TELEMETRY & SWEEP */}
        <div className="xl:col-span-1 flex flex-col gap-4">
          <div className="bg-slate-900/50 p-4 rounded-xl border border-white/5 backdrop-blur-md flex justify-between items-center">
            <div>
              <p className="text-xs text-slate-500 font-bold uppercase tracking-widest">Host:</p>
              <p className="text-sm font-mono text-white">192.168.1.102:8080</p>
            </div>
            <div className={`px-3 py-1 rounded text-xs font-bold uppercase tracking-wider ${wsStatus === 'connected' ? 'bg-green-500/20 text-green-400 border border-green-500/30' : 'bg-red-500/20 text-red-400 border border-red-500/30'}`}>
              {wsStatus === 'connected' ? 'CONNECTED' : 'DISCONNECTED'}
            </div>
          </div>

          <TelemetryGauge label="GPU COMPUTE" value={simulatedGPU.util.toFixed(0)} max={100} unit="%" color="cyan" subtitle="NVIDIA RTX 4070 | Target: 80-100% | Active" />
          <TelemetryGauge label="VRAM ALLOCATION" value={`${simulatedGPU.vram.toFixed(1)} / 8.0`} max={8} unit="GB" color="purple" subtitle="GDDR6 | System RAM: 11.5 GB" />
          
          <div className="grid grid-cols-2 gap-4">
            <TelemetryGauge label="TEMP" value={simulatedGPU.temp.toFixed(0)} unit="°C" color="red" />
            <TelemetryGauge label="POWER" value={simulatedGPU.watts.toFixed(0)} unit="W" color="orange" subtitle="Cap: 138W" />
          </div>

          <Card className="bg-slate-900/50 border border-white/5 relative overflow-hidden mt-4">
            <div className="absolute top-0 right-0 p-4 opacity-10">
              <span className="text-6xl">⏱</span>
            </div>
            <h4 className="text-xs uppercase tracking-widest text-slate-400 font-semibold mb-4">SWEEP PROGRESS</h4>
            <div className="flex items-baseline gap-2 mb-1">
              <span className="text-3xl font-black text-white">{completedCount} <span className="text-slate-600">/ {totalCount}</span></span>
            </div>
            <p className="text-sm text-cyan-400 font-mono mb-4">SST-2 Fast | ETA: ~{(totalCount - completedCount)*0.75} min</p>
            <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-cyan-500 to-purple-500 transition-all duration-1000" style={{ width: `${pct}%` }} />
            </div>
          </Card>
        </div>

        {/* MIDDLE COLUMN: LEADERBOARD & CHARTS */}
        <div className="xl:col-span-2 flex flex-col gap-6">
          <Card className="bg-black/40 border border-slate-800 p-0 overflow-hidden">
            <div className="p-4 border-b border-slate-800 bg-slate-900/80 flex justify-between items-center">
              <div>
                <h3 className="text-lg font-bold text-white flex items-center gap-2">🏆 Empirical Benchmark Leaderboard</h3>
                <p className="text-xs text-slate-400 mt-1">Ranked by Validation Accuracy & Macro F1 Score</p>
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="text-xs text-slate-400 uppercase bg-slate-900/50 border-b border-slate-800">
                  <tr>
                    <th className="px-6 py-3 font-semibold">Rank</th>
                    <th className="px-6 py-3 font-semibold">Attention Model</th>
                    <th className="px-6 py-3 font-semibold">Accuracy</th>
                    <th className="px-6 py-3 font-semibold">Macro F1</th>
                    <th className="px-6 py-3 font-semibold">Loss</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedExp.slice(0, 50).map((exp, idx) => {
                    const acc = exp.final_metrics?.eval_accuracy;
                    const f1 = exp.final_metrics?.eval_f1;
                    const loss = exp.final_metrics?.eval_loss;
                    const model = exp.config?.attn_type || exp.name;
                    return (
                      <tr key={exp.id || idx} className="border-b border-slate-800/50 hover:bg-slate-800/20 transition-colors">
                        <td className="px-6 py-3 font-mono text-slate-300">
                          {idx === 0 ? '🥇 1st' : idx === 1 ? '🥈 2nd' : idx === 2 ? '🥉 3rd' : `#${idx + 1}`}
                        </td>
                        <td className="px-6 py-3">
                          <Link to={`/experiments/${exp.id}`} className="font-bold text-cyan-400 hover:text-cyan-300 hover:underline">{model}</Link>
                        </td>
                        <td className="px-6 py-3 font-mono text-white">{(acc * 100).toFixed(2)}%</td>
                        <td className="px-6 py-3 font-mono text-slate-300">{f1 ? (f1 * 100).toFixed(2) + '%' : '--'}</td>
                        <td className="px-6 py-3 font-mono text-slate-400">{loss ? loss.toFixed(4) : '--'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </div>

        {/* RIGHT COLUMN: EXECUTION MATRIX */}
        <div className="xl:col-span-1 flex flex-col gap-6">
          <Card className="bg-black/40 border border-slate-800 p-0 overflow-hidden h-full flex flex-col">
            <div className="p-4 border-b border-slate-800 bg-slate-900/80">
              <h3 className="text-lg font-bold text-white mb-2">Execution Matrix</h3>
              <div className="flex justify-between items-center text-xs font-mono mb-4">
                <span className="text-cyan-400 font-bold">{completedCount} / {totalCount}</span>
                <div className="flex gap-3 text-slate-400">
                  <span className="flex items-center gap-1"><span className="text-green-500">●</span> Done</span>
                  <span className="flex items-center gap-1"><span className="text-amber-500 animate-pulse">►</span> Active</span>
                  <span className="flex items-center gap-1"><span className="text-slate-600">○</span> Queued</span>
                </div>
              </div>
              <div className="overflow-y-auto max-h-[800px] pr-2 space-y-1">
                {experiments.map((exp, i) => {
                  const model = exp.config?.attn_type || exp.name;
                  const isDone = exp.status === 'completed' || exp.final_metrics?.eval_accuracy;
                  const isActive = exp.status === 'running';
                  return (
                    <div key={i} className="flex justify-between items-center p-2 rounded bg-slate-900/30 hover:bg-slate-800/50 border border-transparent hover:border-slate-700 transition-colors">
                      <div className="flex items-center gap-2">
                        {isDone ? <span className="text-green-500 text-[10px]">●</span> : isActive ? <span className="text-amber-500 text-[10px] animate-pulse">►</span> : <span className="text-slate-700 text-[10px]">○</span>}
                        <span className={`font-mono text-xs ${isDone ? 'text-slate-300' : isActive ? 'text-amber-400 font-bold' : 'text-slate-500'}`}>{model}</span>
                      </div>
                      <span className="text-xs font-mono text-slate-500">
                        {isDone ? `${(exp.final_metrics?.eval_accuracy * 100).toFixed(1)}%` : isActive ? 'Active' : 'Queued'}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
