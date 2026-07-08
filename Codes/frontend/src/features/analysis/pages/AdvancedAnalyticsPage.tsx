import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Link } from "react-router-dom";
import {
  Users,
  Server,
  Activity,
  AlertTriangle,
  RefreshCw,
} from "lucide-react";

export default function AdminDashboard() {
  const queryClient = useQueryClient();
  const { data, isLoading, refetch } = useQuery({
    queryKey: ["admin-stats"],
    queryFn: async () => (await api.get("/admin/stats/")).data,
  });

  const refreshAll = useMutation({
    mutationFn: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["admin-stats"] }),
        queryClient.invalidateQueries({ queryKey: ["jobs"] }),
        queryClient.invalidateQueries({ queryKey: ["experiments"] }),
      ]);
      return refetch();
    },
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-white">Admin Dashboard</h1>
        <Button
          size="sm"
          onClick={() => refreshAll.mutate()}
          disabled={refreshAll.isPending}
        >
          <RefreshCw size={14} className="mr-1" /> Refresh All
        </Button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="text-center">
          <Users className="mx-auto text-neon-blue mb-2" size={32} />
          <div className="text-3xl font-bold text-white">
            {data?.total_users || 0}
          </div>
          <div className="text-xs text-slate-400">Total Users</div>
        </Card>
        <Card className="text-center">
          <Server className="mx-auto text-neon-blue mb-2" size={32} />
          <div className="text-3xl font-bold text-white">
            {data?.total_jobs || 0}
          </div>
          <div className="text-xs text-slate-400">Total Jobs</div>
        </Card>
        <Card className="text-center">
          <Activity className="mx-auto text-green-400 mb-2" size={32} />
          <div className="text-3xl font-bold text-white">
            {data?.active_jobs || 0}
          </div>
          <div className="text-xs text-slate-400">Active Jobs</div>
        </Card>
        <Card className="text-center">
          <AlertTriangle className="mx-auto text-red-400 mb-2" size={32} />
          <div className="text-3xl font-bold text-white">
            {data?.failed_jobs || 0}
          </div>
          <div className="text-xs text-slate-400">Failed Jobs</div>
        </Card>
      </div>
      <div className="flex flex-wrap gap-3">
        <Link to="/admin/jobs">
          <Button variant="secondary">Manage Jobs</Button>
        </Link>
        <Link to="/admin/users">
          <Button variant="secondary">Manage Users</Button>
        </Link>
        <Link to="/admin/model-versions">
          <Button variant="secondary">Model Versions</Button>
        </Link>
        <Link to="/admin/stop-all-jobs">
          <Button variant="danger">Stop All Jobs</Button>
        </Link>
      </div>
    </div>
  );
}
