import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { Spinner } from "../../../components/ui/Spinner";
import { Link, useNavigate } from "react-router-dom";
import { useWindowStore } from "../../../store/windowStore";
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
  Cpu as CpuIcon,
  ShoppingBag,
  Zap,
  ShieldAlert,
  Server
} from "lucide-react";

export default function DashboardPage() {
  const navigate = useNavigate();
  const { openWindow } = useWindowStore();

  // Fetch multiple metrics for the CEO dashboard
  const { data: experiments = [], isLoading: expsLoading } = useQuery({
    queryKey: ["experiments"],
    queryFn: async () => (await api.get(ENDPOINTS.EXPERIMENTS.LIST)).data,
    refetchInterval: 10000,
  });

  const { data: system, isLoading: sysLoading } = useQuery({
    queryKey: ["systemStatus"],
    queryFn: async () => (await api.get(ENDPOINTS.SYSTEM.STATUS)).data,
    refetchInterval: 10000,
  });

  const { data: gpuStats } = useQuery({
    queryKey: ["gpuStats"],
    queryFn: async () => (await api.get(ENDPOINTS.SYSTEM.GPU)).data,
    refetchInterval: 15000,
  });

  const { data: marketplaceStats } = useQuery({
    queryKey: ["marketplaceStats"],
    queryFn: async () => (await api.get(ENDPOINTS.GPU_MARKETPLACE.STATS)).data,
    refetchInterval: 15000,
  });

  const isLoading = expsLoading || sysLoading;

  const completed = Array.isArray(experiments) ? experiments.filter((e: any) => e.status === "completed").length : 0;
  const running = Array.isArray(experiments) ? experiments.filter((e: any) => e.status === "running").length : 0;
  
  const launchFeature = (type: string) => {
    openWindow(type, { title: type.charAt(0).toUpperCase() + type.slice(1).replace('-', ' ') });
  };

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-8 animate-fade-in pb-20">
      {/* Executive Welcome Banner */}
      <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 shadow-2xl p-8 sm:p-10">
        <div className="absolute inset-0 bg-[url('/grid.svg')] bg-center [mask-image:linear-gradient(180deg,white,rgba(255,255,255,0))] opacity-20"></div>
        <div className="absolute top-0 right-0 -mt-16 -mr-16 w-96 h-96 bg-neon-blue rounded-full filter blur-[120px] opacity-20 pointer-events-none"></div>
        <div className="absolute bottom-0 left-0 -mb-16 -ml-16 w-96 h-96 bg-purple-600 rounded-full filter blur-[120px] opacity-20 pointer-events-none"></div>
        
        <div className="relative z-10 flex flex-col sm:flex-row sm:items-center justify-between gap-6">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 mb-4">
              <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span>
              <span className="text-xs font-medium text-white/80 uppercase tracking-wider">Executive Overview</span>
            </div>
            <h1 className="text-4xl font-display font-bold text-white tracking-tight">
              NeuralOps <span className="text-transparent bg-clip-text bg-gradient-to-r from-neon-blue to-purple-400">Command Center</span>
            </h1>
            <p className="text-slate-300 mt-3 max-w-xl text-lg font-light leading-relaxed">
              Global platform telemetry, autonomous research agents, and distributed GPU network status at a glance.
            </p>
          </div>
          <div className="flex flex-col gap-3">
            <Button
              onClick={() => navigate("/launcher")}
              className="flex items-center justify-center gap-2 shadow-[0_0_20px_rgba(56,189,248,0.3)] hover:shadow-[0_0_30px_rgba(56,189,248,0.5)] transition-all bg-gradient-to-r from-neon-blue to-blue-600 border-0"
              size="lg"
            >
              <Zap size={18} /> Launch Run
            </Button>
            <Button
              onClick={() => launchFeature('research')}
              variant="outline"
              className="flex items-center justify-center gap-2 border-white/10 hover:bg-white/5 bg-black/20 backdrop-blur-md"
              size="lg"
            >
              <Bot size={18} /> AI Co-Scientist
            </Button>
          </div>
        </div>
      </div>

      {/* Global Telemetry */}
      <div>
        <h2 className="text-xl font-display font-bold text-white mb-4 flex items-center gap-2">
          <Globe className="text-neon-blue" size={20} /> Global Telemetry
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="bg-black/40 backdrop-blur-xl border-white/5 hover:border-white/10 transition-all overflow-hidden relative group">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="p-5">
              <div className="flex items-center justify-between mb-2">
                <p className="text-xs text-slate-400 uppercase tracking-widest font-medium">Active Experiments</p>
                <Activity size={16} className="text-neon-blue" />
              </div>
              <div className="flex items-baseline gap-2">
                <p className="text-4xl font-display font-bold text-white tracking-tight">{running}</p>
                <span className="text-sm font-medium text-green-400">+{completed} done</span>
              </div>
            </div>
            <div className="h-1 w-full bg-gradient-to-r from-neon-blue to-transparent absolute bottom-0 left-0"></div>
          </Card>

          <Card className="bg-black/40 backdrop-blur-xl border-white/5 hover:border-white/10 transition-all overflow-hidden relative group">
            <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="p-5">
              <div className="flex items-center justify-between mb-2">
                <p className="text-xs text-slate-400 uppercase tracking-widest font-medium">Platform Load</p>
                <Server size={16} className="text-purple-400" />
              </div>
              <div className="flex items-baseline gap-2">
                <p className="text-4xl font-display font-bold text-white tracking-tight">{system?.cpu?.percent || 0}%</p>
                <span className="text-sm font-medium text-slate-400">avg</span>
              </div>
            </div>
            <div className="h-1 w-full bg-gradient-to-r from-purple-500 to-transparent absolute bottom-0 left-0"></div>
          </Card>

          <Card className="bg-black/40 backdrop-blur-xl border-white/5 hover:border-white/10 transition-all overflow-hidden relative group">
            <div className="absolute inset-0 bg-gradient-to-br from-green-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="p-5">
              <div className="flex items-center justify-between mb-2">
                <p className="text-xs text-slate-400 uppercase tracking-widest font-medium">GPUs Online</p>
                <CpuIcon size={16} className="text-green-400" />
              </div>
              <div className="flex items-baseline gap-2">
                <p className="text-4xl font-display font-bold text-white tracking-tight">{gpuStats?.active_nodes || 0}</p>
                <span className="text-sm font-medium text-green-400">nodes</span>
              </div>
            </div>
            <div className="h-1 w-full bg-gradient-to-r from-green-500 to-transparent absolute bottom-0 left-0"></div>
          </Card>

          <Card className="bg-black/40 backdrop-blur-xl border-white/5 hover:border-white/10 transition-all overflow-hidden relative group">
            <div className="absolute inset-0 bg-gradient-to-br from-amber-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="p-5">
              <div className="flex items-center justify-between mb-2">
                <p className="text-xs text-slate-400 uppercase tracking-widest font-medium">Market Volume</p>
                <ShoppingBag size={16} className="text-amber-400" />
              </div>
              <div className="flex items-baseline gap-2">
                <p className="text-4xl font-display font-bold text-white tracking-tight">${marketplaceStats?.volume_24h || 0}</p>
                <span className="text-sm font-medium text-green-400">+12%</span>
              </div>
            </div>
            <div className="h-1 w-full bg-gradient-to-r from-amber-500 to-transparent absolute bottom-0 left-0"></div>
          </Card>
        </div>
      </div>

      {/* Ecosystem Hubs */}
      <div>
        <h2 className="text-xl font-display font-bold text-white mb-4 flex items-center gap-2">
          <Layers className="text-neon-blue" size={20} /> Neural Ecosystem
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
          
          <button onClick={() => launchFeature('marketplace')} className="group text-left h-full">
            <div className="h-full glass p-5 rounded-2xl border border-white/5 hover:border-neon-blue/50 hover:bg-white/5 transition-all flex flex-col gap-3">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-neon-blue/20 to-blue-600/20 flex items-center justify-center border border-neon-blue/20 group-hover:scale-110 transition-transform">
                <ShoppingBag size={24} className="text-neon-blue" />
              </div>
              <div>
                <h3 className="font-semibold text-white">GPU Marketplace</h3>
                <p className="text-xs text-slate-400 mt-1 line-clamp-2">Rent & lease decentralized GPU compute power.</p>
              </div>
            </div>
          </button>

          <button onClick={() => launchFeature('playground')} className="group text-left h-full">
            <div className="h-full glass p-5 rounded-2xl border border-white/5 hover:border-purple-500/50 hover:bg-white/5 transition-all flex flex-col gap-3">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500/20 to-pink-500/20 flex items-center justify-center border border-purple-500/20 group-hover:scale-110 transition-transform">
                <Bot size={24} className="text-purple-400" />
              </div>
              <div>
                <h3 className="font-semibold text-white">LLM Hub</h3>
                <p className="text-xs text-slate-400 mt-1 line-clamp-2">Prompt library, arenas, & model evaluations.</p>
              </div>
            </div>
          </button>

          <button onClick={() => launchFeature('labs')} className="group text-left h-full">
            <div className="h-full glass p-5 rounded-2xl border border-white/5 hover:border-green-500/50 hover:bg-white/5 transition-all flex flex-col gap-3">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-green-500/20 to-emerald-500/20 flex items-center justify-center border border-green-500/20 group-hover:scale-110 transition-transform">
                <Network size={24} className="text-green-400" />
              </div>
              <div>
                <h3 className="font-semibold text-white">Community Lab</h3>
                <p className="text-xs text-slate-400 mt-1 line-clamp-2">Collaborative research spaces and projects.</p>
              </div>
            </div>
          </button>

          <button onClick={() => launchFeature('datasets')} className="group text-left h-full">
            <div className="h-full glass p-5 rounded-2xl border border-white/5 hover:border-orange-500/50 hover:bg-white/5 transition-all flex flex-col gap-3">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-orange-500/20 to-amber-500/20 flex items-center justify-center border border-orange-500/20 group-hover:scale-110 transition-transform">
                <Database size={24} className="text-orange-400" />
              </div>
              <div>
                <h3 className="font-semibold text-white">Dataset Hub</h3>
                <p className="text-xs text-slate-400 mt-1 line-clamp-2">Curated data, lineage tracking & versioning.</p>
              </div>
            </div>
          </button>

          <button onClick={() => launchFeature('drift-monitors')} className="group text-left h-full">
            <div className="h-full glass p-5 rounded-2xl border border-white/5 hover:border-red-500/50 hover:bg-white/5 transition-all flex flex-col gap-3">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-red-500/20 to-rose-500/20 flex items-center justify-center border border-red-500/20 group-hover:scale-110 transition-transform">
                <ShieldAlert size={24} className="text-red-400" />
              </div>
              <div>
                <h3 className="font-semibold text-white">MLOps Monitor</h3>
                <p className="text-xs text-slate-400 mt-1 line-clamp-2">Drift detection, alerts & health monitoring.</p>
              </div>
            </div>
          </button>

        </div>
      </div>

      {/* Advanced Research Tools */}
      <div>
        <h2 className="text-xl font-display font-bold text-white mb-4 flex items-center gap-2">
          <CpuIcon className="text-neon-blue" size={20} /> Deep Tech Stack
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
          <Card onClick={() => launchFeature('coding-assistant')} className="cursor-pointer hover:bg-white/5 transition-colors p-4 flex items-center gap-4">
            <div className="w-10 h-10 rounded-full bg-cyan-500/10 flex items-center justify-center">
              <Bot size={20} className="text-cyan-400" />
            </div>
            <div>
              <h4 className="text-sm font-semibold text-white">Code Assistant</h4>
              <p className="text-xs text-slate-400">AI pair programming</p>
            </div>
          </Card>
          
          <Card onClick={() => launchFeature('federated')} className="cursor-pointer hover:bg-white/5 transition-colors p-4 flex items-center gap-4">
            <div className="w-10 h-10 rounded-full bg-indigo-500/10 flex items-center justify-center">
              <Globe size={20} className="text-indigo-400" />
            </div>
            <div>
              <h4 className="text-sm font-semibold text-white">Federated Learning</h4>
              <p className="text-xs text-slate-400">Decentralized training</p>
            </div>
          </Card>
          
          <Card onClick={() => launchFeature('hpo')} className="cursor-pointer hover:bg-white/5 transition-colors p-4 flex items-center gap-4">
            <div className="w-10 h-10 rounded-full bg-fuchsia-500/10 flex items-center justify-center">
              <Activity size={20} className="text-fuchsia-400" />
            </div>
            <div>
              <h4 className="text-sm font-semibold text-white">Hyperparameter Ops</h4>
              <p className="text-xs text-slate-400">Advanced sweeps</p>
            </div>
          </Card>

          <Card onClick={() => launchFeature('model-registry')} className="cursor-pointer hover:bg-white/5 transition-colors p-4 flex items-center gap-4">
            <div className="w-10 h-10 rounded-full bg-emerald-500/10 flex items-center justify-center">
              <Layers size={20} className="text-emerald-400" />
            </div>
            <div>
              <h4 className="text-sm font-semibold text-white">Model Registry</h4>
              <p className="text-xs text-slate-400">Versioning & promotion</p>
            </div>
          </Card>
        </div>
      </div>
      
    </div>
  );
}
