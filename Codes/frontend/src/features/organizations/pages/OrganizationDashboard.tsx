import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card, Button } from "../../../components/ui";
import { Users, CreditCard, HardDrive, BarChart3 } from "lucide-react";

export function OrganizationDashboard({ orgId }: { orgId: string }) {
  const { data: stats } = useQuery({
    queryKey: ["org-stats", orgId],
    queryFn: async () => (await api.get(`/orgs/${orgId}/stats/`)).data,
  });

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="text-center">
          <Users className="mx-auto text-neon-blue mb-2" size={24} />
          <div className="text-2xl font-bold text-white">
            {stats?.member_count || 0}
          </div>
          <div className="text-xs text-slate-400">Members</div>
        </Card>
        <Card className="text-center">
          <BarChart3 className="mx-auto text-green-400 mb-2" size={24} />
          <div className="text-2xl font-bold text-white">
            {stats?.experiment_count || 0}
          </div>
          <div className="text-xs text-slate-400">Experiments</div>
        </Card>
        <Card className="text-center">
          <HardDrive className="mx-auto text-yellow-400 mb-2" size={24} />
          <div className="text-2xl font-bold text-white">
            {stats?.storage_used_gb || 0} GB
          </div>
          <div className="text-xs text-slate-400">Storage Used</div>
        </Card>
        <Card className="text-center">
          <CreditCard className="mx-auto text-purple-400 mb-2" size={24} />
          <div className="text-2xl font-bold text-white">
            {stats?.monthly_usage || 0} hrs
          </div>
          <div className="text-xs text-slate-400">GPU Hours</div>
        </Card>
      </div>

      {/* Recent Activity */}
      <Card>
        <h3 className="text-sm font-bold text-white mb-3">Recent Activity</h3>
        <div className="space-y-2">
          {stats?.recent_activity?.map((activity: any) => (
            <div
              key={activity.id}
              className="flex justify-between text-sm border-b border-white/5 py-2"
            >
              <span className="text-slate-300">{activity.action}</span>
              <span className="text-xs text-slate-500">
                {new Date(activity.created_at).toLocaleString()}
              </span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
