// src/features/marketplace/pages/ProviderDashboardPage.tsx
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { marketplaceAPI } from "../services/api";
import { RevenueChart } from "../components/RevenueChart";
import { UsageHeatmap } from "../components/UsageHeatmap";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import {
  DollarSign,
  Clock,
  Server,
  Star,
  AlertTriangle,
  TrendingUp,
} from "lucide-react";

export default function ProviderDashboardPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [days, setDays] = useState(30);
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ["provider-stats"],
    queryFn: () => marketplaceAPI.getProviderStats(),
    refetchInterval: 30000,
  });
  const { data: revenue, isLoading: revenueLoading } = useQuery({
    queryKey: ["provider-revenue", days],
    queryFn: () => marketplaceAPI.getProviderRevenueHistory(days),
  });
  const { data: usage, isLoading: usageLoading } = useQuery({
    queryKey: ["provider-usage"],
    queryFn: () => marketplaceAPI.getProviderUsage(),
  });

  if (statsLoading || revenueLoading || usageLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Server size={20} className="text-neon-mint" /> Provider Dashboard
          </h1>
          <p className="text-xs text-slate-400">
            Monitor your GPU marketplace performance
          </p>
        </div>
        <div className="flex gap-2">
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <div className="grid md:grid-cols-2 lg:grid-cols-5 gap-3">
          <StatCard
            icon={<DollarSign size={20} />}
            label="Total Earned"
            value={`$${stats?.total_earned.toFixed(2)}`}
          />
          <StatCard
            icon={<Clock size={20} />}
            label="Pending Earnings"
            value={`$${stats?.pending_earnings.toFixed(2)}`}
          />
          <StatCard
            icon={<Server size={20} />}
            label="GPU Hours"
            value={`${stats?.total_hours_provided.toLocaleString()} hrs`}
          />
          <StatCard
            icon={<Star size={20} />}
            label="Avg Rating"
            value={stats?.average_rating.toFixed(1)}
            suffix="/5"
          />
          <StatCard
            icon={<AlertTriangle size={20} />}
            label="Dispute Rate"
            value={`${(stats?.dispute_rate * 100).toFixed(1)}%`}
          />
        </div>
        <div className="grid md:grid-cols-2 gap-4">
          <TerminalCard className="p-4">
            <div className="flex justify-between items-center mb-2">
              <h3 className="text-sm font-bold text-white">Revenue History</h3>
              <Select
                options={[
                  { value: "7", label: "7 days" },
                  { value: "30", label: "30 days" },
                  { value: "90", label: "90 days" },
                ]}
                value={days.toString()}
                onChange={(e) => setDays(parseInt(e.target.value))}
                className="w-28"
              />
            </div>
            <RevenueChart data={revenue || []} />
          </TerminalCard>
          <TerminalCard className="p-4">
            <h3 className="text-sm font-bold text-white mb-2">Usage Heatmap</h3>
            <UsageChart usage={usage} />
          </TerminalCard>
        </div>
        <TerminalCard className="p-4">
          <h3 className="text-sm font-bold text-white mb-2">Top GPU Models</h3>
          <div className="space-y-2">
            {usage?.top_gpu_models.map((m) => (
              <div key={m.model} className="flex justify-between items-center">
                <span>{m.model}</span>
                <div className="flex items-center gap-2">
                  <div className="w-32 h-1.5 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-neon-blue rounded-full"
                      style={{
                        width: `${(m.hours / usage.top_gpu_models[0].hours) * 100}%`,
                      }}
                    />
                  </div>
                  <span className="text-sm text-white">
                    {m.hours.toLocaleString()} hrs
                  </span>
                </div>
              </div>
            ))}
          </div>
        </TerminalCard>
      </div>
    </div>
  );
}

const StatCard = ({ icon, label, value, suffix = "" }: any) => (
  <TerminalCard className="p-3 text-center">
    <div className="flex justify-center text-slate-400 mb-1">{icon}</div>
    <div className="text-2xl font-bold text-white">
      {value}
      {suffix}
    </div>
    <div className="text-xs text-slate-500 uppercase tracking-wider">
      {label}
    </div>
  </TerminalCard>
);
