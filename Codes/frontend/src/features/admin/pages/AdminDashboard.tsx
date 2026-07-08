import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";
import {
  Shield,
  Activity,
  Cpu,
  Layers,
  Server,
  Box,
  Wrench,
  Database,
} from "lucide-react";

export default function AdminDashboard() {
  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: ["health"],
    queryFn: async () => (await api.get("/api/health/")).data,
  });
  const { data: version, isLoading: versionLoading } = useQuery({
    queryKey: ["version"],
    queryFn: async () => (await api.get("/api/version/")).data,
  });
  const { data: jobStats, isLoading: jobsLoading } = useQuery({
    queryKey: ["jobStats"],
    queryFn: async () => (await api.get("/api/jobs/stats/")).data,
  });

  const isLoading = healthLoading || versionLoading || jobsLoading;
  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-display font-bold text-white flex items-center gap-2">
        <Shield size={24} className="text-neon-blue" /> Admin Dashboard
      </h1>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="flex items-center gap-4 p-5">
          <Activity size={28} className="text-green-400" />
          <div>
            <div className="text-xs text-slate-400 uppercase tracking-wider">
              System
            </div>
            <div className="text-xl font-bold text-white">
              {health?.status || "Unknown"}
            </div>
          </div>
        </Card>
        <Card className="flex items-center gap-4 p-5">
          <Layers size={28} className="text-neon-purple" />
          <div>
            <div className="text-xs text-slate-400 uppercase tracking-wider">
              API Version
            </div>
            <div className="text-xl font-bold text-white">
              {version?.api_version || "—"}
            </div>
          </div>
        </Card>
        <Card className="flex items-center gap-4 p-5">
          <Server size={28} className="text-neon-blue" />
          <div>
            <div className="text-xs text-slate-400 uppercase tracking-wider">
              Total Jobs
            </div>
            <div className="text-xl font-bold text-white">
              {jobStats?.total || 0}
            </div>
          </div>
        </Card>
      </div>

      {/* Quick Links */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Link to="/admin/jobs">
          <Card hover className="p-5 text-center">
            <Wrench size={24} className="mx-auto text-neon-orange mb-2" />
            <p className="font-bold text-white">Jobs Management</p>
            <p className="text-xs text-slate-400">Bulk delete, stop all</p>
          </Card>
        </Link>
        <Link to="/admin/model-versions">
          <Card hover className="p-5 text-center">
            <Box size={24} className="mx-auto text-neon-purple mb-2" />
            <p className="font-bold text-white">Model Versions</p>
            <p className="text-xs text-slate-400">Publish, archive</p>
          </Card>
        </Link>
        <Link to="/cluster">
          <Card hover className="p-5 text-center">
            <Cpu size={24} className="mx-auto text-neon-green mb-2" />
            <p className="font-bold text-white">Cluster Status</p>
            <p className="text-xs text-slate-400">GPU nodes & queue</p>
          </Card>
        </Link>
        <a href="/admin/" target="_blank" rel="noopener noreferrer">
          <Card hover className="p-5 text-center">
            <Database size={24} className="mx-auto text-neon-blue mb-2" />
            <p className="font-bold text-white">Django Admin</p>
            <p className="text-xs text-slate-400">Full database access</p>
          </Card>
        </a>
      </div>

      {/* Job Breakdown */}
      {jobStats?.by_status && (
        <Card>
          <h2 className="text-sm font-display font-bold text-white mb-4">
            Job Breakdown
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-center text-xs">
            {Object.entries(jobStats.by_status).map(([status, count]: any) => (
              <div
                key={status}
                className="bg-cyber-800 p-3 rounded-xl border border-white/5"
              >
                <div className="text-xl font-bold text-white">{count}</div>
                <div className="text-slate-400 capitalize mt-1">{status}</div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
