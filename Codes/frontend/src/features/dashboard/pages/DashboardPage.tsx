import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { Spinner } from "../../../components/ui/Spinner";
import { Link, useNavigate } from "react-router-dom";
import { Experiment } from "../../../types/experiment";
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
} from "lucide-react";

export default function DashboardPage() {
  const navigate = useNavigate();

  // Fetch experiments
  const {
    data: experiments = [],
    isLoading: expsLoading,
    error: expsError,
  } = useQuery<Experiment[]>({
    queryKey: ["experiments"],
    queryFn: async () => (await api.get(ENDPOINTS.EXPERIMENTS.LIST)).data,
    refetchInterval: 5000,
  });

  // Fetch system status
  const { data: system, isLoading: sysLoading } = useQuery({
    queryKey: ["systemStatus"],
    queryFn: async () => (await api.get("/system/status/")).data,
    refetchInterval: 10000,
  });

  // Fetch job stats (optional, if endpoint exists)
  const { data: jobStats } = useQuery({
    queryKey: ["jobStats"],
    queryFn: async () => (await api.get("/jobs/stats/")).data,
    refetchInterval: 15000,
    retry: 1,
  });

  const isLoading = expsLoading || sysLoading;
  if (isLoading) return <Spinner />;

  if (expsError) {
    return (
      <Card className="p-6 text-center">
        <p className="text-red-400">
          Failed to load experiments. Please try again later.
        </p>
        <Button
          variant="primary"
          className="mt-4"
          onClick={() => window.location.reload()}
        >
          Retry
        </Button>
      </Card>
    );
  }

  const completed = experiments.filter((e) => e.status === "completed").length;
  const running = experiments.filter((e) => e.status === "running").length;
  const failed = experiments.filter((e) => e.status === "failed").length;

  const avgAccuracy =
    completed > 0
      ? (
          experiments
            .filter((e) => e.status === "completed")
            .reduce(
              (sum, e) => sum + ((e as any).final_metrics?.eval_accuracy || 0),
              0,
            ) / completed
        ).toFixed(4)
      : "—";

  const lastFailed = experiments.find((e) => e.status === "failed");

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Welcome Banner */}
      <div className="glass p-6 rounded-2xl bg-gradient-to-r from-neon-blue/10 to-purple-500/10 border border-white/10">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-display font-bold text-white">
              Welcome back, <span className="text-neon-blue">Researcher</span>
            </h1>
            <p className="text-slate-400 mt-1 text-sm">
              Here's what's happening across your experiments.
            </p>
          </div>
          <Button
            onClick={() => navigate("/launcher")}
            className="flex items-center gap-2"
          >
            <Plus size={16} /> New Experiment
          </Button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="card-hover-effect">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-slate-400 uppercase tracking-wider">
                Total Runs
              </p>
              <p className="text-3xl font-display font-bold text-white mt-1">
                {experiments.length}
              </p>
            </div>
            <div className="w-10 h-10 rounded-xl bg-neon-blue/10 flex items-center justify-center">
              <BarChart3 size={20} className="text-neon-blue" />
            </div>
          </div>
        </Card>
        <Card className="card-hover-effect">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-slate-400 uppercase tracking-wider">
                Running
              </p>
              <p className="text-3xl font-display font-bold text-orange-400 mt-1">
                {running}
              </p>
            </div>
            <div className="w-10 h-10 rounded-xl bg-orange-500/10 flex items-center justify-center">
              <Activity size={20} className="text-orange-400" />
            </div>
          </div>
        </Card>
        <Card className="card-hover-effect">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-slate-400 uppercase tracking-wider">
                Avg Accuracy
              </p>
              <p className="text-3xl font-display font-bold text-green-400 mt-1">
                {avgAccuracy}
              </p>
            </div>
            <div className="w-10 h-10 rounded-xl bg-green-500/10 flex items-center justify-center">
              <TrendingUp size={20} className="text-green-400" />
            </div>
          </div>
        </Card>
        <Card className="card-hover-effect">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-slate-400 uppercase tracking-wider">
                CPU Load
              </p>
              <p className="text-3xl font-display font-bold text-purple-400 mt-1">
                {system?.cpu?.percent || 0}%
              </p>
            </div>
            <div className="w-10 h-10 rounded-xl bg-purple-500/10 flex items-center justify-center">
              <Cpu size={20} className="text-purple-400" />
            </div>
          </div>
        </Card>
      </div>

      {/* Recent Experiments Table */}
      <Card>
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 mb-4">
          <h2 className="text-lg font-display font-bold text-white">
            Recent Experiments
          </h2>
          <Link
            to="/experiments"
            className="text-neon-blue text-sm hover:underline"
          >
            View all →
          </Link>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-slate-400 border-b border-white/5">
                <th className="py-3 px-4 font-medium">Name</th>
                <th className="py-3 px-4 font-medium">Task</th>
                <th className="py-3 px-4 font-medium">Status</th>
                <th className="py-3 px-4 font-medium">Accuracy</th>
                <th className="py-3 px-4 font-medium">Created</th>
                <th className="py-3 px-4 font-medium text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {experiments.slice(0, 8).map((exp) => (
                <tr
                  key={exp.id}
                  className="hover:bg-white/5 cursor-pointer transition-colors"
                  onClick={() => navigate(`/experiments/${exp.id}`)}
                >
                  <td className="py-3 px-4 text-white">
                    {exp.name || exp.id.slice(0, 8)}
                  </td>
                  <td className="py-3 px-4 uppercase text-xs text-slate-300">
                    {exp.config?.task_name || "—"}
                  </td>
                  <td className="py-3 px-4">
                    <Badge variant={exp.status}>{exp.status}</Badge>
                  </td>
                  <td className="py-3 px-4 font-mono">
                    {(exp as any).final_metrics?.eval_accuracy?.toFixed(4) || "—"}
                  </td>
                  <td className="py-3 px-4 text-slate-400 text-xs">
                    {new Date(exp.created_at).toLocaleDateString()}
                  </td>
                  <td className="py-3 px-4 text-right">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        navigate(`/experiments/${exp.id}`);
                      }}
                    >
                      View
                    </Button>
                  </td>
                </tr>
              ))}
              {experiments.length === 0 && (
                <tr>
                  <td colSpan={6} className="py-8 text-center text-slate-500">
                    No experiments found. Start one from the Launcher.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Quick Actions */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Link
          to="/analysis"
          className="glass p-4 rounded-xl hover:bg-white/10 transition-all flex items-center gap-3"
        >
          <BarChart3 size={24} className="text-neon-blue" />
          <span className="text-sm font-medium text-white">Analysis</span>
        </Link>
        <Link
          to="/notes"
          className="glass p-4 rounded-xl hover:bg-white/10 transition-all flex items-center gap-3"
        >
          <BookOpen size={24} className="text-purple-400" />
          <span className="text-sm font-medium text-white">Notebook</span>
        </Link>
        <Link
          to="/paper/new"
          className="glass p-4 rounded-xl hover:bg-white/10 transition-all flex items-center gap-3"
        >
          <FileText size={24} className="text-pink-400" />
          <span className="text-sm font-medium text-white">Papers</span>
        </Link>
        <Link
          to="/agent"
          className="glass p-4 rounded-xl hover:bg-white/10 transition-all flex items-center gap-3"
        >
          <Bot size={24} className="text-cyan-400" />
          <span className="text-sm font-medium text-white">AI Assistant</span>
        </Link>
      </div>

      {/* Failed Alert */}
      {lastFailed && (
        <Card className="border border-red-500/20 bg-red-500/5">
          <div className="flex flex-col sm:flex-row justify-between gap-3">
            <div>
              <p className="text-sm text-red-400 font-medium">
                ⚠️ Last failed: {lastFailed.name || lastFailed.id.slice(0, 8)}
              </p>
              <p className="text-xs text-slate-400 mt-1">
                Task: {lastFailed.config?.task_name || "—"} ·{" "}
                {new Date(lastFailed.created_at).toLocaleString()}
              </p>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate(`/experiments/${lastFailed.id}`)}
            >
              Investigate
            </Button>
          </div>
        </Card>
      )}
    </div>
  );
}
