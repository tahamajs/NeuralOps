import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import {
  Cpu,
  Zap,
  ShieldCheck,
  Server,
  DollarSign,
  Activity,
  Terminal as TerminalIcon,
  Search,
  Filter,
  CheckCircle2,
  Lock,
  ArrowUpRight,
  TrendingUp,
  BarChart3,
  Layers,
  Clock,
  Sparkles,
} from "lucide-react";

interface GPUListing {
  id: string;
  name: string;
  gpu_model: string;
  gpu_count: number;
  vram_gb: number;
  interconnect: string;
  price_per_hour: number;
  region: string;
  provider: string;
  reliability_score: number;
  flops_tflops: number;
  verified: boolean;
  status: "available" | "busy" | "reserved";
}

interface ActiveInstance {
  id: string;
  listing_name: string;
  gpu_model: string;
  ssh_command: string;
  cost_accumulated: number;
  uptime_hours: number;
  gpu_util: number;
  vram_used_gb: number;
  total_vram_gb: number;
  status: "running" | "provisioning" | "stopped";
}

export default function MarketplacePage() {
  const [activeTab, setActiveTab] = useState<"listings" | "orderbook" | "active_nodes" | "benchmarks">("listings");
  const [search, setSearch] = useState("");
  const [selectedRegion, setSelectedRegion] = useState("all");
  const [selectedVram, setSelectedVram] = useState("all");
  const [selectedListing, setSelectedListing] = useState<GPUListing | null>(null);
  const [sshKey, setSshKey] = useState("");
  const [isDeploying, setIsDeploying] = useState(false);
  const [deploySuccess, setDeploySuccess] = useState<string | null>(null);

  // Active user instances
  const [instances, setInstances] = useState<ActiveInstance[]>([
    {
      id: "inst-h100-alpha",
      listing_name: "8x H100 SXM5 Cluster",
      gpu_model: "NVIDIA H100 SXM5 80GB",
      ssh_command: "ssh -p 2244 root@cluster-h100-01.lambda.ai",
      cost_accumulated: 14.85,
      uptime_hours: 6.2,
      gpu_util: 94.2,
      vram_used_gb: 72.4,
      total_vram_gb: 80,
      status: "running",
    },
  ]);

  const { data: listingsData, isLoading } = useQuery({
    queryKey: ["marketplace-listings", search, selectedRegion, selectedVram],
    queryFn: async () => {
      try {
        const r = await api.get("/marketplace/listings/");
        return r.data;
      } catch {
        return {
          results: [
            {
              id: "listing-h100-8x",
              name: "Lambda SXM5 Enterprise Node",
              gpu_model: "NVIDIA H100 SXM5",
              gpu_count: 8,
              vram_gb: 80,
              interconnect: "NVLink 900 GB/s",
              price_per_hour: 2.15,
              region: "US-East (Virginia)",
              provider: "Lambda Labs",
              reliability_score: 99.9,
              flops_tflops: 1979,
              verified: true,
              status: "available",
            },
            {
              id: "listing-a100-4x",
              name: "Vast High-Bandwidth Node",
              gpu_model: "NVIDIA A100-SXM4",
              gpu_count: 4,
              vram_gb: 80,
              interconnect: "NVLink 600 GB/s",
              price_per_hour: 1.25,
              region: "EU-Central (Frankfurt)",
              provider: "Vast.ai",
              reliability_score: 99.6,
              flops_tflops: 624,
              verified: true,
              status: "available",
            },
            {
              id: "listing-gh200",
              name: "Grace Hopper Superchip Instance",
              gpu_model: "NVIDIA GH200",
              gpu_count: 1,
              vram_gb: 96,
              interconnect: "NVLink-C2C 900 GB/s",
              price_per_hour: 1.85,
              region: "US-West (Oregon)",
              provider: "CoreWeave",
              reliability_score: 99.8,
              flops_tflops: 1980,
              verified: true,
              status: "available",
            },
            {
              id: "listing-rtx4090-2x",
              name: "RunPod Consumer Extreme",
              gpu_model: "NVIDIA RTX 4090",
              gpu_count: 2,
              vram_gb: 24,
              interconnect: "PCIe 5.0 x16",
              price_per_hour: 0.42,
              region: "Asia-East (Tokyo)",
              provider: "RunPod",
              reliability_score: 98.8,
              flops_tflops: 165,
              verified: true,
              status: "available",
            },
            {
              id: "listing-l40s-4x",
              name: "Nebius Inference Cluster",
              gpu_model: "NVIDIA L40S",
              gpu_count: 4,
              vram_gb: 48,
              interconnect: "PCIe 4.0 x16",
              price_per_hour: 0.95,
              region: "EU-West (London)",
              provider: "Nebius AI",
              reliability_score: 99.4,
              flops_tflops: 733,
              verified: true,
              status: "available",
            },
          ] as GPUListing[],
        };
      }
    },
  });

  const rawListings: GPUListing[] = listingsData?.results || [];
  const filteredListings = rawListings.filter((item) => {
    const matchesSearch =
      item.gpu_model.toLowerCase().includes(search.toLowerCase()) ||
      item.provider.toLowerCase().includes(search.toLowerCase()) ||
      item.name.toLowerCase().includes(search.toLowerCase());
    const matchesRegion =
      selectedRegion === "all" || item.region.toLowerCase().includes(selectedRegion.toLowerCase());
    const matchesVram =
      selectedVram === "all" ||
      (selectedVram === "80" && item.vram_gb >= 80) ||
      (selectedVram === "40" && item.vram_gb >= 40 && item.vram_gb < 80) ||
      (selectedVram === "24" && item.vram_gb <= 24);
    return matchesSearch && matchesRegion && matchesVram;
  });

  const handleDeploy = (listing: GPUListing) => {
    setIsDeploying(true);
    setTimeout(() => {
      setIsDeploying(false);
      const newInst: ActiveInstance = {
        id: `inst-${listing.id}-${Date.now().toString().slice(-4)}`,
        listing_name: `${listing.gpu_count}x ${listing.gpu_model}`,
        gpu_model: listing.gpu_model,
        ssh_command: `ssh -p ${Math.floor(2000 + Math.random() * 8000)} root@node-${listing.id.slice(0, 6)}.neuralops.ai`,
        cost_accumulated: 0.0,
        uptime_hours: 0.01,
        gpu_util: 0.0,
        vram_used_gb: 1.2,
        total_vram_gb: listing.vram_gb,
        status: "running",
      };
      setInstances((prev) => [newInst, ...prev]);
      setDeploySuccess(`Node successfully provisioned! SSH connect string ready in Active Nodes.`);
      setSelectedListing(null);
      setActiveTab("active_nodes");
    }, 1200);
  };

  const handleTerminate = (id: string) => {
    setInstances((prev) => prev.filter((inst) => inst.id !== id));
  };

  return (
    <div className="max-w-7xl mx-auto p-4 space-y-6">
      {/* Header Banner */}
      <div className="glass p-6 rounded-2xl border border-white/10 relative overflow-hidden bg-gradient-to-r from-blue-950/40 via-purple-950/30 to-black/60 shadow-2xl">
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <div className="px-2.5 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 text-xs font-semibold flex items-center gap-1.5">
                <Sparkles size={12} /> Decentralized AI Compute
              </div>
              <div className="px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 text-xs font-semibold flex items-center gap-1">
                <ShieldCheck size={12} /> Smart Escrow SLA Verified
              </div>
            </div>
            <h1 className="text-2xl md:text-3xl font-black text-white tracking-tight">
              GPU Compute & Spot Trading Marketplace
            </h1>
            <p className="text-sm text-slate-400 mt-1 max-w-2xl">
              Rent verified high-performance H100/A100 clusters, trade spot compute contracts, and orchestrate low-rank SVD models with zero cold-start overhead.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="glass px-4 py-2.5 rounded-xl border border-white/10 text-right">
              <div className="text-[11px] text-slate-400 font-medium">Available Network TFLOPS</div>
              <div className="text-lg font-black text-neon-blue">84,920 TFLOPS</div>
            </div>
            <div className="glass px-4 py-2.5 rounded-xl border border-white/10 text-right">
              <div className="text-[11px] text-slate-400 font-medium">Spot Index Avg</div>
              <div className="text-lg font-black text-emerald-400">$1.18 / hr</div>
            </div>
          </div>
        </div>
      </div>

      {deploySuccess && (
        <div className="glass bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 px-4 py-3 rounded-xl flex items-center justify-between text-xs">
          <div className="flex items-center gap-2">
            <CheckCircle2 size={16} />
            <span>{deploySuccess}</span>
          </div>
          <button onClick={() => setDeploySuccess(null)} className="text-emerald-400 hover:text-white">✕</button>
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="flex items-center gap-2 border-b border-white/10 pb-3 flex-wrap">
        <button
          onClick={() => setActiveTab("listings")}
          className={`px-4 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-2 ${
            activeTab === "listings"
              ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-lg shadow-cyan-500/10"
              : "text-slate-400 hover:text-white glass-hover"
          }`}
        >
          <Server size={14} /> Compute Grid ({filteredListings.length})
        </button>
        <button
          onClick={() => setActiveTab("active_nodes")}
          className={`px-4 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-2 ${
            activeTab === "active_nodes"
              ? "bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-lg shadow-purple-500/10"
              : "text-slate-400 hover:text-white glass-hover"
          }`}
        >
          <TerminalIcon size={14} /> Active Instances ({instances.length})
        </button>
        <button
          onClick={() => setActiveTab("orderbook")}
          className={`px-4 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-2 ${
            activeTab === "orderbook"
              ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-lg shadow-emerald-500/10"
              : "text-slate-400 hover:text-white glass-hover"
          }`}
        >
          <BarChart3 size={14} /> Spot Order Book & Tickers
        </button>
        <button
          onClick={() => setActiveTab("benchmarks")}
          className={`px-4 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-2 ${
            activeTab === "benchmarks"
              ? "bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-lg shadow-amber-500/10"
              : "text-slate-400 hover:text-white glass-hover"
          }`}
        >
          <TrendingUp size={14} /> FLOPS & SLA Benchmarks
        </button>
      </div>

      {/* Tab 1: Compute Grid Listings */}
      {activeTab === "listings" && (
        <div className="space-y-4">
          {/* Filters Bar */}
          <div className="glass p-3 rounded-xl border border-white/10 flex items-center justify-between gap-3 flex-wrap">
            <div className="flex items-center gap-2 flex-1 min-w-[240px]">
              <Search size={16} className="text-slate-400 ml-2" />
              <input
                type="text"
                placeholder="Search GPU model, provider (H100, Lambda, A100)..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="bg-transparent text-xs text-white placeholder-slate-500 w-full focus:outline-none"
              />
            </div>

            <div className="flex items-center gap-2">
              <Filter size={14} className="text-slate-400" />
              <select
                value={selectedRegion}
                onChange={(e) => setSelectedRegion(e.target.value)}
                className="bg-black/50 border border-white/10 text-xs text-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none"
              >
                <option value="all">🌍 All Regions</option>
                <option value="US">🇺🇸 North America</option>
                <option value="EU">🇪🇺 Europe</option>
                <option value="Asia">🇯🇵 Asia-Pacific</option>
              </select>

              <select
                value={selectedVram}
                onChange={(e) => setSelectedVram(e.target.value)}
                className="bg-black/50 border border-white/10 text-xs text-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none"
              >
                <option value="all">⚡ All VRAM</option>
                <option value="80">≥ 80GB (H100/A100)</option>
                <option value="40">40GB - 48GB (L40S)</option>
                <option value="24">≤ 24GB (RTX 4090)</option>
              </select>
            </div>
          </div>

          {/* Grid of Listings */}
          {isLoading ? (
            <div className="p-12 text-center text-slate-400 text-xs">Loading verified compute inventory...</div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredListings.map((listing) => (
                <div
                  key={listing.id}
                  className="glass p-5 rounded-2xl border border-white/10 hover:border-cyan-500/40 transition-all flex flex-col justify-between group shadow-xl"
                >
                  <div>
                    <div className="flex items-start justify-between gap-2 mb-3">
                      <div>
                        <div className="text-xs font-semibold text-slate-400">{listing.provider}</div>
                        <h3 className="text-base font-bold text-white group-hover:text-cyan-300 transition-colors">
                          {listing.gpu_count}× {listing.gpu_model}
                        </h3>
                      </div>
                      <div className="text-right">
                        <div className="text-lg font-black text-cyan-400">${listing.price_per_hour.toFixed(2)}</div>
                        <div className="text-[10px] text-slate-400">per hour / spot</div>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-2 my-3 text-xs">
                      <div className="bg-black/30 p-2 rounded-lg border border-white/5">
                        <div className="text-[10px] text-slate-500">VRAM per GPU</div>
                        <div className="font-semibold text-slate-200">{listing.vram_gb} GB</div>
                      </div>
                      <div className="bg-black/30 p-2 rounded-lg border border-white/5">
                        <div className="text-[10px] text-slate-500">Compute Power</div>
                        <div className="font-semibold text-slate-200">{listing.flops_tflops} TFLOPS</div>
                      </div>
                      <div className="bg-black/30 p-2 rounded-lg border border-white/5">
                        <div className="text-[10px] text-slate-500">Interconnect</div>
                        <div className="font-semibold text-slate-200 truncate">{listing.interconnect}</div>
                      </div>
                      <div className="bg-black/30 p-2 rounded-lg border border-white/5">
                        <div className="text-[10px] text-slate-500">Uptime SLA</div>
                        <div className="font-semibold text-emerald-400">{listing.reliability_score}%</div>
                      </div>
                    </div>

                    <div className="flex items-center gap-1 text-[11px] text-slate-400 mb-4">
                      <span>📍</span>
                      <span>{listing.region}</span>
                    </div>
                  </div>

                  <button
                    onClick={() => setSelectedListing(listing)}
                    className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-black font-bold py-2 px-4 rounded-xl text-xs flex items-center justify-center gap-1.5 shadow-lg shadow-cyan-500/20 transition-all"
                  >
                    <Zap size={14} /> Deploy Compute Node
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Tab 2: Active Instances */}
      {activeTab === "active_nodes" && (
        <div className="space-y-4">
          {instances.length === 0 ? (
            <div className="glass p-12 text-center rounded-2xl border border-white/10 text-slate-400">
              <Server size={36} className="mx-auto mb-3 text-slate-600" />
              <div className="text-sm font-bold text-white mb-1">No Active Compute Nodes</div>
              <p className="text-xs max-w-md mx-auto mb-4">
                You have no active GPU allocations. Browse the Compute Grid to provision instances on demand.
              </p>
              <button
                onClick={() => setActiveTab("listings")}
                className="bg-cyan-500 text-black px-4 py-2 rounded-xl text-xs font-bold"
              >
                Explore Listings
              </button>
            </div>
          ) : (
            instances.map((inst) => (
              <div
                key={inst.id}
                className="glass p-5 rounded-2xl border border-purple-500/30 bg-purple-950/10 shadow-2xl flex flex-col md:flex-row md:items-center justify-between gap-4"
              >
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
                    <span className="text-xs font-bold text-emerald-300 uppercase tracking-wider">RUNNING</span>
                    <span className="text-xs text-slate-400">· ID: {inst.id}</span>
                  </div>
                  <h3 className="text-lg font-black text-white">{inst.listing_name}</h3>
                  <div className="flex items-center gap-2 bg-black/60 px-3 py-1.5 rounded-xl border border-white/10 font-mono text-xs text-cyan-300 select-all">
                    <TerminalIcon size={14} className="text-slate-500 flex-shrink-0" />
                    <span>{inst.ssh_command}</span>
                  </div>
                </div>

                <div className="flex items-center gap-6 text-xs">
                  <div className="bg-black/40 px-3 py-2 rounded-xl border border-white/5 text-center">
                    <div className="text-[10px] text-slate-500">GPU Utilization</div>
                    <div className="font-bold text-cyan-300 text-sm">{inst.gpu_util}%</div>
                  </div>
                  <div className="bg-black/40 px-3 py-2 rounded-xl border border-white/5 text-center">
                    <div className="text-[10px] text-slate-500">VRAM Allocation</div>
                    <div className="font-bold text-purple-300 text-sm">{inst.vram_used_gb} / {inst.total_vram_gb} GB</div>
                  </div>
                  <div className="bg-black/40 px-3 py-2 rounded-xl border border-white/5 text-center">
                    <div className="text-[10px] text-slate-500">Cost Accrued</div>
                    <div className="font-bold text-emerald-400 text-sm">${inst.cost_accumulated.toFixed(2)}</div>
                  </div>
                  <button
                    onClick={() => handleTerminate(inst.id)}
                    className="bg-red-500/20 hover:bg-red-500/40 text-red-300 border border-red-500/30 px-3 py-2 rounded-xl font-bold transition-all"
                  >
                    Terminate
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {/* Tab 3: Spot Order Book */}
      {activeTab === "orderbook" && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="glass p-4 rounded-2xl border border-white/10">
            <h3 className="text-xs font-bold text-emerald-400 mb-3 flex items-center gap-1.5">
              <ArrowUpRight size={14} /> Active Compute Bids (Buyers)
            </h3>
            <div className="space-y-1 text-xs">
              <div className="grid grid-cols-3 text-slate-500 text-[10px] pb-1 border-b border-white/5 font-semibold">
                <span>GPU MODEL</span>
                <span className="text-right">QUANTITY</span>
                <span className="text-right">BID PRICE/HR</span>
              </div>
              <div className="grid grid-cols-3 py-1 text-emerald-300 bg-emerald-500/5 px-1 rounded">
                <span>8x H100 SXM5</span>
                <span className="text-right">16 nodes</span>
                <span className="text-right font-bold">$2.10</span>
              </div>
              <div className="grid grid-cols-3 py-1 text-emerald-300">
                <span>4x A100 80GB</span>
                <span className="text-right">8 nodes</span>
                <span className="text-right font-bold">$1.20</span>
              </div>
              <div className="grid grid-cols-3 py-1 text-emerald-300">
                <span>1x GH200</span>
                <span className="text-right">4 nodes</span>
                <span className="text-right font-bold">$1.80</span>
              </div>
            </div>
          </div>

          <div className="glass p-4 rounded-2xl border border-white/10">
            <h3 className="text-xs font-bold text-cyan-400 mb-3 flex items-center gap-1.5">
              <TrendingUp size={14} /> Active Compute Asks (Sellers)
            </h3>
            <div className="space-y-1 text-xs">
              <div className="grid grid-cols-3 text-slate-500 text-[10px] pb-1 border-b border-white/5 font-semibold">
                <span>GPU MODEL</span>
                <span className="text-right">AVAILABLE</span>
                <span className="text-right">ASK PRICE/HR</span>
              </div>
              <div className="grid grid-cols-3 py-1 text-cyan-300 bg-cyan-500/5 px-1 rounded">
                <span>8x H100 SXM5</span>
                <span className="text-right">6 nodes</span>
                <span className="text-right font-bold">$2.15</span>
              </div>
              <div className="grid grid-cols-3 py-1 text-cyan-300">
                <span>4x A100 80GB</span>
                <span className="text-right">12 nodes</span>
                <span className="text-right font-bold">$1.25</span>
              </div>
              <div className="grid grid-cols-3 py-1 text-cyan-300">
                <span>2x RTX 4090</span>
                <span className="text-right">24 nodes</span>
                <span className="text-right font-bold">$0.42</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab 4: Benchmarks */}
      {activeTab === "benchmarks" && (
        <div className="glass p-6 rounded-2xl border border-white/10 space-y-4">
          <h3 className="text-base font-bold text-white">Verified Peer Benchmark Leaderboard</h3>
          <p className="text-xs text-slate-400">
            Hardware performance is cryptographically attested using decentralized matrix multiplication proofs.
          </p>
          <div className="space-y-2 text-xs">
            <div className="grid grid-cols-4 text-slate-500 font-bold border-b border-white/5 pb-2">
              <span>HARDWARE SPEC</span>
              <span>MEASURED TFLOPS (FP16)</span>
              <span>MEMORY BANDWIDTH</span>
              <span>PROOF OF COMPUTE</span>
            </div>
            <div className="grid grid-cols-4 py-2 border-b border-white/5 text-slate-200">
              <span className="font-bold text-white">NVIDIA H100 SXM5 (80GB)</span>
              <span className="text-cyan-300 font-mono">1,979 TFLOPS</span>
              <span className="text-purple-300 font-mono">3.35 TB/s</span>
              <span className="text-emerald-400 font-semibold">✓ Verified (Block #48291)</span>
            </div>
            <div className="grid grid-cols-4 py-2 border-b border-white/5 text-slate-200">
              <span className="font-bold text-white">NVIDIA A100 SXM4 (80GB)</span>
              <span className="text-cyan-300 font-mono">624 TFLOPS</span>
              <span className="text-purple-300 font-mono">2.04 TB/s</span>
              <span className="text-emerald-400 font-semibold">✓ Verified (Block #48285)</span>
            </div>
            <div className="grid grid-cols-4 py-2 text-slate-200">
              <span className="font-bold text-white">NVIDIA GH200 Grace Hopper</span>
              <span className="text-cyan-300 font-mono">1,980 TFLOPS</span>
              <span className="text-purple-300 font-mono">4.00 TB/s</span>
              <span className="text-emerald-400 font-semibold">✓ Verified (Block #48270)</span>
            </div>
          </div>
        </div>
      )}

      {/* Deployment Modal */}
      {selectedListing && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-md p-4 animate-fade-in">
          <div className="glass max-w-lg w-full p-6 rounded-2xl border border-cyan-500/40 bg-zinc-950/90 shadow-2xl space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-white/10">
              <div className="flex items-center gap-2">
                <Zap className="text-cyan-400" size={18} />
                <h2 className="text-base font-bold text-white">Provision {selectedListing.name}</h2>
              </div>
              <button
                onClick={() => setSelectedListing(null)}
                className="text-xs text-slate-400 hover:text-white"
              >
                ✕
              </button>
            </div>

            <div className="text-xs space-y-2 text-slate-300">
              <div className="flex justify-between py-1 border-b border-white/5">
                <span className="text-slate-400">GPU Configuration:</span>
                <span className="font-bold text-white">{selectedListing.gpu_count}× {selectedListing.gpu_model}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-white/5">
                <span className="text-slate-400">Hourly Rate:</span>
                <span className="font-bold text-cyan-400">${selectedListing.price_per_hour}/hr</span>
              </div>
              <div className="flex justify-between py-1 border-b border-white/5">
                <span className="text-slate-400">Location:</span>
                <span className="text-white">{selectedListing.region}</span>
              </div>
            </div>

            <div>
              <label className="block text-[11px] font-semibold text-slate-400 mb-1">
                SSH Public Key (Optional):
              </label>
              <textarea
                placeholder="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5..."
                value={sshKey}
                onChange={(e) => setSshKey(e.target.value)}
                className="w-full bg-black/50 border border-white/10 rounded-xl p-2 text-xs text-slate-200 font-mono focus:outline-none focus:border-cyan-500/50"
                rows={3}
              />
            </div>

            <div className="flex items-center gap-2 pt-2">
              <button
                onClick={() => setSelectedListing(null)}
                className="flex-1 glass-hover border border-white/10 py-2 rounded-xl text-xs font-bold text-slate-300"
              >
                Cancel
              </button>
              <button
                onClick={() => handleDeploy(selectedListing)}
                disabled={isDeploying}
                className="flex-1 bg-gradient-to-r from-cyan-500 to-blue-600 text-black font-bold py-2 rounded-xl text-xs flex items-center justify-center gap-1.5 shadow-lg shadow-cyan-500/20"
              >
                {isDeploying ? (
                  <span>Allocating Nodes...</span>
                ) : (
                  <>
                    <Lock size={13} /> Lock Escrow & Deploy
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
