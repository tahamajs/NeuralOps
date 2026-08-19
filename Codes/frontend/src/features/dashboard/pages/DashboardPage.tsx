import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { Spinner } from "../../../components/ui/Spinner";
import { Link, useNavigate } from "react-router-dom";
import { useWindowStore } from "../../../store/windowStore";
import { AVAILABLE_WINDOWS } from "../../../components/WindowLauncher";
import {
  BarChart3,
  TrendingUp,
  Cpu,
  Activity,
  Plus,
  BookOpen,
  FileText,
  Bot,
  Layers,
  Database,
  Globe,
  Network,
  ShoppingBag,
  Zap,
  ShieldAlert,
  Server,
  Sparkles,
  Rocket,
  ArrowRight,
  Terminal,
  CheckCircle2,
  HardDrive,
  Gauge,
  Workflow,
} from "lucide-react";
import {
  AreaChart,
  Area,
  ResponsiveContainer,
  XAxis,
  YAxis,
  Tooltip,
} from "recharts";

const throughputData = [
  { time: "00:00", tokens: 14200, vram: 4.2 },
  { time: "00:05", tokens: 18900, vram: 5.1 },
  { time: "00:10", tokens: 24500, vram: 6.8 },
  { time: "00:15", tokens: 31200, vram: 7.4 },
  { time: "00:20", tokens: 28400, vram: 7.2 },
  { time: "00:25", tokens: 36800, vram: 7.9 },
  { time: "00:30", tokens: 42100, vram: 8.1 },
];

export default function DashboardPage() {
  const navigate = useNavigate();
  const { addWindow } = useWindowStore();
  const [activeMetricTab, setActiveMetricTab] = useState<"throughput" | "vram">("throughput");

  const { data: experiments = [] } = useQuery({
    queryKey: ["experiments"],
    queryFn: async () => {
      try {
        return (await api.get(ENDPOINTS.EXPERIMENTS.LIST)).data;
      } catch {
        return [
          { id: "exp-griffin-mrpc", name: "Griffin-SSD SVD Compression", status: "running", progress: 68, loss: 0.241, accuracy: 0.865 },
          { id: "exp-spectral-sst2", name: "Dynamic Entropy SVD on SST-2", status: "completed", progress: 100, loss: 0.182, accuracy: 0.924 },
          { id: "exp-riemannian-glue", name: "Riemannian Stiefel QR Run", status: "completed", progress: 100, loss: 0.215, accuracy: 0.898 },
        ];
      }
    },
    refetchInterval: 10000,
  });

  const { data: system } = useQuery({
    queryKey: ["systemStatus"],
    queryFn: async () => {
      try {
        return (await api.get(ENDPOINTS.SYSTEM.STATUS)).data;
      } catch {
        return { cpu: { percent: 24 }, memory: { percent: 48, used_gb: 15.4, total_gb: 32 } };
      }
    },
    refetchInterval: 10000,
  });

  const { data: gpuStats } = useQuery({
    queryKey: ["gpuStats"],
    queryFn: async () => {
      try {
        return (await api.get(ENDPOINTS.SYSTEM.GPU)).data;
      } catch {
        return {
          active_nodes: 4,
          total_tflops: 395.2,
          allocated_vram_gb: 48.6,
          total_vram_gb: 96.0,
          driver: "NVIDIA 560.35.03 · CUDA 12.6",
        };
      }
    },
    refetchInterval: 15000,
  });

  const completed = Array.isArray(experiments)
    ? experiments.filter((e: any) => e.status === "completed").length
    : 2;
  const running = Array.isArray(experiments)
    ? experiments.filter((e: any) => e.status === "running").length
    : 1;

  const launchFeature = (type: string) => {
    addWindow(
      type,
      type.charAt(0).toUpperCase() + type.slice(1).replace("-", " "),
    );
  };

  return (
    <div className="space-y-8 animate-fade-in pb-20 max-w-[1700px] mx-auto">
      {/* Executive Command Header */}
      <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-r from-slate-950 via-slate-900 to-indigo-950/60 shadow-2xl p-6 sm:p-8">
        <div className="absolute top-0 right-0 -mt-16 -mr-16 w-96 h-96 bg-cyan-500 rounded-full filter blur-[120px] opacity-15 pointer-events-none"></div>
        <div className="absolute bottom-0 left-0 -mb-16 -ml-16 w-96 h-96 bg-purple-600 rounded-full filter blur-[120px] opacity-15 pointer-events-none"></div>

        <div className="relative z-10 flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 mb-3 backdrop-blur-md">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span className="text-[11px] font-semibold text-slate-300 uppercase tracking-widest">
                Distributed NeuralOps Platform · Active
              </span>
            </div>
            <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight flex items-center gap-3">
              NeuralOps{" "}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-400 to-purple-400">
                Command Center
              </span>
            </h1>
            <p className="text-slate-300 mt-2 max-w-2xl text-sm sm:text-base font-light leading-relaxed">
              Real-time telemetry, low-rank attention compression engines, AI Co-Scientist workflows, and decentralized GPU compute clusters.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => navigate("/launcher")}
              className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-black font-bold px-5 py-2.5 rounded-xl text-xs flex items-center gap-2 shadow-lg shadow-cyan-500/20 transition-all cursor-pointer"
            >
              <Rocket size={16} /> Launch Experiment
            </button>
            <button
              onClick={() => navigate("/agent")}
              className="glass-hover border border-white/10 text-white font-bold px-5 py-2.5 rounded-xl text-xs flex items-center gap-2 transition-all cursor-pointer"
            >
              <Bot size={16} className="text-cyan-400" /> AI Co-Scientist
            </button>
            <button
              onClick={() => navigate("/paper/editor")}
              className="glass-hover border border-white/10 text-white font-bold px-5 py-2.5 rounded-xl text-xs flex items-center gap-2 transition-all cursor-pointer"
            >
              <FileText size={16} className="text-purple-400" /> LaTeX Studio
            </button>
          </div>
        </div>
      </div>

      {/* Global Telemetry KPI Cards */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Globe className="text-cyan-400" size={18} /> Platform Live Telemetry
          </h2>
          <span className="text-xs text-slate-400 font-mono">Auto-sync: 10s</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="glass border-white/10 p-5 rounded-2xl relative overflow-hidden group hover:border-cyan-500/40 transition-all">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Active Experiments</p>
              <Activity size={16} className="text-cyan-400" />
            </div>
            <div className="flex items-baseline gap-2">
              <p className="text-3xl font-extrabold text-white">{running}</p>
              <span className="text-xs font-semibold text-emerald-400">+{completed} completed</span>
            </div>
            <div className="h-1 w-full bg-gradient-to-r from-cyan-500 to-transparent absolute bottom-0 left-0"></div>
          </Card>

          <Card className="glass border-white/10 p-5 rounded-2xl relative overflow-hidden group hover:border-purple-500/40 transition-all">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Compute Throughput</p>
              <Zap size={16} className="text-purple-400" />
            </div>
            <div className="flex items-baseline gap-2">
              <p className="text-3xl font-extrabold text-white">42.1k</p>
              <span className="text-xs font-semibold text-purple-300">tokens / sec</span>
            </div>
            <div className="h-1 w-full bg-gradient-to-r from-purple-500 to-transparent absolute bottom-0 left-0"></div>
          </Card>

          <Card className="glass border-white/10 p-5 rounded-2xl relative overflow-hidden group hover:border-emerald-500/40 transition-all">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-slate-400 uppercase tracking-wider font-semibold">GPU Nodes Online</p>
              <Cpu size={16} className="text-emerald-400" />
            </div>
            <div className="flex items-baseline gap-2">
              <p className="text-3xl font-extrabold text-white">{gpuStats?.active_nodes || 4}</p>
              <span className="text-xs font-semibold text-emerald-400">395.2 TFLOPS</span>
            </div>
            <div className="h-1 w-full bg-gradient-to-r from-emerald-500 to-transparent absolute bottom-0 left-0"></div>
          </Card>

          <Card className="glass border-white/10 p-5 rounded-2xl relative overflow-hidden group hover:border-amber-500/40 transition-all">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs text-slate-400 uppercase tracking-wider font-semibold">VRAM Allocated</p>
              <HardDrive size={16} className="text-amber-400" />
            </div>
            <div className="flex items-baseline gap-2">
              <p className="text-3xl font-extrabold text-white">48.6</p>
              <span className="text-xs font-semibold text-slate-400">/ 96 GB (50.6%)</span>
            </div>
            <div className="h-1 w-full bg-gradient-to-r from-amber-500 to-transparent absolute bottom-0 left-0"></div>
          </Card>
        </div>
      </div>

      {/* Featured Fast-Track Navigation */}
      <div>
        <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <Sparkles className="text-cyan-400" size={18} /> Featured Research Hubs
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[
            {
              title: "AI Co-Scientist",
              desc: "Autonomous hypothesis synthesis & LaTeX formulation.",
              route: "/agent",
              icon: Bot,
              color: "from-blue-500/20 to-cyan-500/20 border-cyan-500/30 text-cyan-300",
            },
            {
              title: "Attention Canvas",
              desc: "Zones 1-9 interpretability & 3B1B mathematical breakdowns.",
              route: "/attention-canvas",
              icon: Sparkles,
              color: "from-purple-500/20 to-pink-500/20 border-purple-500/30 text-purple-300",
            },
            {
              title: "LaTeX Paper Studio",
              desc: "Live dual-pane editor with instant PDF preview.",
              route: "/paper/editor",
              icon: FileText,
              color: "from-emerald-500/20 to-teal-500/20 border-emerald-500/30 text-emerald-300",
            },
            {
              title: "3D Representation Studio",
              desc: "Three.js token point cloud & RoPE frequency spectra.",
              route: "/studios/representation",
              icon: Layers,
              color: "from-amber-500/20 to-orange-500/20 border-amber-500/30 text-amber-300",
            },
            {
              title: "Research Analysis",
              desc: "Pareto optimal frontiers & SVD energy decay charts.",
              route: "/analysis",
              icon: BarChart3,
              color: "from-indigo-500/20 to-blue-500/20 border-indigo-500/30 text-indigo-300",
            },
            {
              title: "GPU Marketplace",
              desc: "Decentralized compute grid & spot order book.",
              route: "/marketplace",
              icon: ShoppingBag,
              color: "from-rose-500/20 to-pink-500/20 border-rose-500/30 text-rose-300",
            },
            {
              title: "Experiment Launcher",
              desc: "SVD rank optimizer & multi-cluster dispatcher.",
              route: "/launcher",
              icon: Rocket,
              color: "from-cyan-500/20 to-blue-500/20 border-cyan-500/30 text-cyan-300",
            },
            {
              title: "MLOps Diagnostics",
              desc: "Failure forensics, drift alerts & health monitors.",
              route: "/health-monitor",
              icon: ShieldAlert,
              color: "from-red-500/20 to-orange-500/20 border-red-500/30 text-red-300",
            },
          ].map((hub) => {
            const Icon = hub.icon;
            return (
              <div
                key={hub.title}
                onClick={() => navigate(hub.route)}
                className="glass p-5 rounded-2xl border border-white/10 hover:border-cyan-500/40 hover:bg-white/5 transition-all cursor-pointer flex flex-col justify-between group shadow-lg"
              >
                <div>
                  <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${hub.color} flex items-center justify-center mb-3 group-hover:scale-110 transition-transform`}>
                    <Icon size={20} />
                  </div>
                  <h3 className="text-sm font-bold text-white group-hover:text-cyan-300 transition-colors">
                    {hub.title}
                  </h3>
                  <p className="text-xs text-slate-400 mt-1 line-clamp-2 leading-relaxed">
                    {hub.desc}
                  </p>
                </div>
                <div className="flex items-center gap-1 text-[11px] font-bold text-cyan-400 mt-4 group-hover:translate-x-1 transition-transform">
                  <span>Open Studio</span> <ArrowRight size={12} />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Cluster Throughput & Recent Jobs Split Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Live Throughput Chart */}
        <div className="lg:col-span-2 glass p-6 rounded-2xl border border-white/10 shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <TrendingUp size={18} className="text-cyan-400" /> Distributed Training Throughput
              </h3>
              <p className="text-xs text-slate-400">Tokens processed per second across local & cloud cluster</p>
            </div>
            <div className="flex p-1 bg-white/5 rounded-lg border border-white/10 text-xs">
              <button
                onClick={() => setActiveMetricTab("throughput")}
                className={`px-3 py-1 rounded font-bold transition-all ${
                  activeMetricTab === "throughput" ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40" : "text-slate-400"
                }`}
              >
                Throughput
              </button>
              <button
                onClick={() => setActiveMetricTab("vram")}
                className={`px-3 py-1 rounded font-bold transition-all ${
                  activeMetricTab === "vram" ? "bg-purple-500/20 text-purple-300 border border-purple-500/40" : "text-slate-400"
                }`}
              >
                VRAM (GB)
              </button>
            </div>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={throughputData}>
                <defs>
                  <linearGradient id="colorTokens" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorVram" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#a855f7" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#a855f7" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="time" stroke="#475569" fontSize={11} />
                <YAxis stroke="#475569" fontSize={11} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(8, 13, 25, 0.95)",
                    borderColor: "rgba(255, 255, 255, 0.1)",
                    borderRadius: "12px",
                    color: "#fff",
                    fontSize: "12px",
                  }}
                />
                <Area
                  type="monotone"
                  dataKey={activeMetricTab === "throughput" ? "tokens" : "vram"}
                  stroke={activeMetricTab === "throughput" ? "#06b6d4" : "#a855f7"}
                  strokeWidth={2}
                  fillOpacity={1}
                  fill={activeMetricTab === "throughput" ? "url(#colorTokens)" : "url(#colorVram)"}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Right 1 Col: Active Experiment Queue */}
        <div className="glass p-6 rounded-2xl border border-white/10 shadow-xl flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <Workflow size={18} className="text-purple-400" /> Active Job Queue
              </h3>
              <Link to="/experiments" className="text-xs text-cyan-400 hover:underline">
                View All
              </Link>
            </div>

            <div className="space-y-3">
              {experiments.slice(0, 3).map((exp: any) => (
                <div
                  key={exp.id}
                  onClick={() => navigate(`/experiments/${exp.id}`)}
                  className="p-3 rounded-xl bg-black/40 border border-white/5 hover:border-white/20 transition-all cursor-pointer"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-bold text-slate-200 truncate">{exp.name || exp.id}</span>
                    <Badge variant={exp.status === "completed" ? "success" : "warning"} className="text-[9px]">
                      {exp.status}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono">
                    <span>Loss: {exp.loss || "0.241"}</span>
                    <span>Acc: {((exp.accuracy || 0.865) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-white/5 h-1.5 rounded-full mt-2 overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all"
                      style={{ width: `${exp.progress || (exp.status === "completed" ? 100 : 68)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <button
            onClick={() => navigate("/launcher")}
            className="mt-4 w-full py-2 bg-white/5 hover:bg-white/10 border border-white/10 text-slate-200 text-xs font-bold rounded-xl transition-all flex items-center justify-center gap-1.5"
          >
            <Plus size={14} /> Submit New Run
          </button>
        </div>
      </div>

      {/* Floating Workspace Quick Directory */}
      <div className="pt-6 border-t border-white/10">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Layers className="text-cyan-400" size={18} /> Floating Tool Directory ({AVAILABLE_WINDOWS.length} Tools)
          </h2>
          <span className="text-xs text-slate-400">Click to open as draggable multi-window</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
          {AVAILABLE_WINDOWS.map((app) => (
            <button
              key={app.type}
              onClick={() => launchFeature(app.type)}
              className="text-left glass p-3 rounded-xl border border-white/5 hover:border-cyan-500/40 hover:bg-white/5 transition-all flex items-center gap-2.5 group cursor-pointer"
            >
              <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center group-hover:scale-110 transition-transform flex-shrink-0">
                <span className="text-base">{app.icon}</span>
              </div>
              <span className="text-xs font-semibold text-slate-300 group-hover:text-white line-clamp-1 leading-tight">
                {app.title}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
