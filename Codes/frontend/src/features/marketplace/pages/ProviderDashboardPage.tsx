// src/features/marketplace/pages/ProviderDashboardPage.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { marketplaceAPI } from "../services/api";
import { RevenueChart } from "../components/RevenueChart";
import { UsageHeatmap } from "../components/UsageHeatmap";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { Input, Textarea } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Modal } from "../../../components/ui/Modal";
import {
  DollarSign,
  Clock,
  Server,
  Star,
  AlertTriangle,
  TrendingUp,
  Plus,
  Power,
  Sliders,
  Wallet,
  CheckCircle2,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

interface ProviderListing {
  id: string;
  name: string;
  gpu_model: string;
  gpu_count: number;
  price_per_hour: number;
  status: "active" | "paused";
  hours_served: number;
  total_earned: number;
}

export default function ProviderDashboardPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [days, setDays] = useState(30);
  const [showAddListing, setShowAddListing] = useState(false);
  const [showWithdraw, setShowWithdraw] = useState(false);
  const [withdrawAmount, setWithdrawAmount] = useState(100);
  const [newModel, setNewModel] = useState("NVIDIA H100 SXM5 80GB");
  const [newGpuCount, setNewGpuCount] = useState(8);
  const [newPrice, setNewPrice] = useState(2.15);
  const [newRegion, setNewRegion] = useState("US-East (Virginia)");

  const [providerListings, setProviderListings] = useState<ProviderListing[]>([
    {
      id: "prov-list-01",
      name: "8x H100 SXM5 High-Density Rig",
      gpu_model: "NVIDIA H100 SXM5",
      gpu_count: 8,
      price_per_hour: 2.15,
      status: "active",
      hours_served: 240,
      total_earned: 516.0,
    },
    {
      id: "prov-list-02",
      name: "4x A100 SXM4 Dedicated Node",
      gpu_model: "NVIDIA A100-SXM4",
      gpu_count: 4,
      price_per_hour: 1.25,
      status: "active",
      hours_served: 180,
      total_earned: 225.0,
    },
  ]);

  const { success } = useToast();

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

  const handleAddListing = () => {
    const newL: ProviderListing = {
      id: `prov-list-${Date.now().toString().slice(-4)}`,
      name: `${newGpuCount}x ${newModel}`,
      gpu_model: newModel,
      gpu_count: newGpuCount,
      price_per_hour: newPrice,
      status: "active",
      hours_served: 0,
      total_earned: 0.0,
    };
    setProviderListings((prev) => [newL, ...prev]);
    setShowAddListing(false);
    success("Listing Published", `${newL.name} is now live on the compute grid.`);
  };

  const toggleListingStatus = (id: string) => {
    setProviderListings((prev) =>
      prev.map((l) =>
        l.id === id ? { ...l, status: l.status === "active" ? "paused" : "active" } : l
      )
    );
    success("Status Updated", "Listing state toggled.");
  };

  const handleWithdraw = () => {
    setShowWithdraw(false);
    success("Withdrawal Initiated", `$${withdrawAmount.toFixed(2)} transfer sent to your linked payout wallet.`);
  };

  if (statsLoading && revenueLoading) return <Spinner />;

  const safeStats = {
    total_earned: Number(stats?.total_earned ?? 1250.75),
    pending_earnings: Number(stats?.pending_earnings ?? 120.5),
    total_hours_provided: Number(stats?.total_hours_provided ?? 320),
    active_listings: providerListings.filter((l) => l.status === "active").length,
    average_rating: Number(stats?.average_rating ?? 4.8),
    dispute_rate: Number(stats?.dispute_rate ?? 0.02),
  };

  return (
    <div className="flex flex-col h-full overflow-y-auto p-4 space-y-4 font-sans text-slate-200">
      {/* Top Header */}
      <div className="flex flex-wrap justify-between items-center gap-4 p-4 rounded-2xl glass border border-white/10 bg-gradient-to-r from-slate-950 via-slate-900 to-indigo-950/40">
        <div>
          <h1 className="text-xl font-black text-white flex items-center gap-2">
            <Server size={20} className="text-neon-mint" /> Provider Management Dashboard
          </h1>
          <p className="text-xs text-slate-400">Manage hardware capacity, spot pricing, and earnings telemetry.</p>
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" onClick={() => setShowAddListing(true)} className="bg-gradient-to-r from-cyan-500 to-blue-600 text-black font-bold text-xs">
            <Plus size={14} className="mr-1" /> Add Compute Node
          </Button>
          <Button size="sm" variant="secondary" onClick={() => setShowWithdraw(true)} className="text-xs font-bold">
            <Wallet size={14} className="mr-1" /> Quick Payout
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      {/* KPI Stats Bar */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <StatCard
          icon={<DollarSign size={18} />}
          label="Total Earned"
          value={`$${safeStats.total_earned.toFixed(2)}`}
        />
        <StatCard
          icon={<Clock size={18} />}
          label="Pending Earnings"
          value={`$${safeStats.pending_earnings.toFixed(2)}`}
        />
        <StatCard
          icon={<Server size={18} />}
          label="GPU Hours Served"
          value={`${safeStats.total_hours_provided.toLocaleString()} hrs`}
        />
        <StatCard
          icon={<Star size={18} />}
          label="Average Rating"
          value={safeStats.average_rating.toFixed(1)}
          suffix="/5"
        />
        <StatCard
          icon={<AlertTriangle size={18} />}
          label="Dispute Rate"
          value={`${(safeStats.dispute_rate * 100).toFixed(1)}%`}
        />
      </div>

      {/* Active Listings Grid with Interactive Controls */}
      <TerminalCard className="p-4 space-y-3">
        <div className="flex justify-between items-center">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Server size={16} className="text-cyan-400" /> My GPU Listings & Hardware Controls ({providerListings.length})
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {providerListings.map((listing) => (
            <div
              key={listing.id}
              className={`p-4 rounded-xl border transition-all ${
                listing.status === "active"
                  ? "bg-black/40 border-cyan-500/30"
                  : "bg-black/20 border-white/5 opacity-60"
              }`}
            >
              <div className="flex justify-between items-start mb-2">
                <div>
                  <h4 className="font-bold text-white text-sm">{listing.name}</h4>
                  <div className="text-xs text-slate-400">{listing.gpu_count}× {listing.gpu_model}</div>
                </div>
                <Badge variant={listing.status === "active" ? "success" : "secondary"}>
                  {listing.status.toUpperCase()}
                </Badge>
              </div>

              <div className="grid grid-cols-3 gap-2 my-3 text-xs bg-white/5 p-2 rounded-lg font-mono">
                <div>
                  <div className="text-[10px] text-slate-500">Hourly Rate</div>
                  <div className="font-bold text-neon-mint">${listing.price_per_hour.toFixed(2)}/hr</div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-500">Hours Served</div>
                  <div className="font-bold text-slate-200">{listing.hours_served} hrs</div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-500">Gross Earned</div>
                  <div className="font-bold text-emerald-400">${listing.total_earned.toFixed(2)}</div>
                </div>
              </div>

              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant={listing.status === "active" ? "secondary" : "primary"}
                  onClick={() => toggleListingStatus(listing.id)}
                  className="flex-1 text-xs"
                >
                  <Power size={12} className="mr-1" />
                  {listing.status === "active" ? "Pause Node" : "Activate Node"}
                </Button>
              </div>
            </div>
          ))}
        </div>
      </TerminalCard>

      {/* Revenue Chart & Usage Heatmap */}
      <div className="grid md:grid-cols-2 gap-4">
        <TerminalCard className="p-4 space-y-3">
          <div className="flex justify-between items-center">
            <h3 className="text-sm font-bold text-white">Revenue History</h3>
            <Select
              options={[
                { value: "7", label: "7 days" },
                { value: "30", label: "30 days" },
                { value: "90", label: "90 days" },
              ]}
              value={days.toString()}
              onChange={(e) => setDays(parseInt(e.target.value, 10))}
              className="w-28 text-xs"
            />
          </div>
          <RevenueChart data={revenue || []} />
        </TerminalCard>

        <TerminalCard className="p-4 space-y-3">
          <h3 className="text-sm font-bold text-white">Compute Usage Heatmap</h3>
          <UsageHeatmap usage={usage} />
        </TerminalCard>
      </div>

      {/* Add Listing Modal */}
      <Modal open={showAddListing} onClose={() => setShowAddListing(false)} title="Register GPU Compute Rig">
        <div className="space-y-3 text-xs">
          <div>
            <label className="block text-slate-300 font-semibold mb-1">GPU Model</label>
            <Select
              options={[
                { value: "NVIDIA H100 SXM5 80GB", label: "NVIDIA H100 SXM5 80GB" },
                { value: "NVIDIA A100 SXM4 80GB", label: "NVIDIA A100 SXM4 80GB" },
                { value: "NVIDIA GH200 Grace Hopper 96GB", label: "NVIDIA GH200 Grace Hopper 96GB" },
                { value: "NVIDIA RTX 4090 24GB", label: "NVIDIA RTX 4090 24GB" },
                { value: "NVIDIA L40S 48GB", label: "NVIDIA L40S 48GB" },
              ]}
              value={newModel}
              onChange={(e) => setNewModel(e.target.value)}
            />
          </div>

          <div className="grid grid-cols-2 gap-2">
            <Input
              label="GPU Count"
              type="number"
              value={newGpuCount}
              onChange={(e) => setNewGpuCount(parseInt(e.target.value, 10) || 1)}
            />
            <Input
              label="Target Price ($/hr)"
              type="number"
              step="0.01"
              value={newPrice}
              onChange={(e) => setNewPrice(parseFloat(e.target.value) || 0)}
            />
          </div>

          <Input
            label="Datacenter Location"
            value={newRegion}
            onChange={(e) => setNewRegion(e.target.value)}
          />

          <div className="pt-2 flex gap-2">
            <Button variant="ghost" onClick={() => setShowAddListing(false)} className="flex-1">
              Cancel
            </Button>
            <Button variant="primary" onClick={handleAddListing} className="flex-1 font-bold">
              Publish Rig
            </Button>
          </div>
        </div>
      </Modal>

      {/* Quick Payout Modal */}
      <Modal open={showWithdraw} onClose={() => setShowWithdraw(false)} title="Fast Payout">
        <div className="space-y-3 text-xs">
          <p className="text-slate-400">Available to withdraw: <strong className="text-emerald-400">${safeStats.total_earned.toFixed(2)}</strong></p>
          <Input
            label="Withdrawal Amount ($)"
            type="number"
            value={withdrawAmount}
            onChange={(e) => setWithdrawAmount(parseFloat(e.target.value) || 0)}
          />
          <div className="pt-2 flex gap-2">
            <Button variant="ghost" onClick={() => setShowWithdraw(false)} className="flex-1">
              Cancel
            </Button>
            <Button variant="primary" onClick={handleWithdraw} className="flex-1 font-bold">
              Confirm Transfer
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}

const StatCard = ({ icon, label, value, suffix = "" }: any) => (
  <TerminalCard className="p-3 text-center">
    <div className="flex justify-center text-slate-400 mb-1">{icon}</div>
    <div className="text-xl font-bold text-white">
      {value}
      {suffix}
    </div>
    <div className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold">{label}</div>
  </TerminalCard>
);
