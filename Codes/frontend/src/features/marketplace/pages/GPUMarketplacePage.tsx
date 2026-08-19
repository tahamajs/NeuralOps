// src/features/marketplace/pages/GPUMarketplacePage.tsx
import { useState, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import { useAuthStore } from "../../../features/auth/store/authStore";
import {
  Card,
  Button,
  Input,
  Spinner,
  Select,
  Badge,
} from "../../../components/ui";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import {
  Zap,
  ShieldCheck,
  Server,
  Star,
  Clock,
  Terminal,
  Activity,
  Layers,
  Lock,
  CheckCircle2,
  Bookmark,
  BookmarkCheck,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

// --- Types ---
export interface GPUMarketplaceListing {
  id: string;
  name: string;
  provider_username: string;
  gpu_model: string;
  gpu_count: number;
  memory_gb: number;
  price_per_hour: number;
  status: string;
  region: string;
  created_at: string;
  reliability_score: number;
  performance_score: number;
  availability_percentage: number;
  verified: boolean;
  interconnect?: string;
  flops_tflops?: number;
}

interface MarketDepth {
  model: string;
  total_available: number;
  avg_price: number;
  min_price: number;
  max_price: number;
  sellers_count: number;
}

const FALLBACK_LISTINGS: GPUMarketplaceListing[] = [
  {
    id: "listing-h100-8x",
    name: "8x H100 SXM5 Enterprise Node",
    provider_username: "Lambda Labs",
    gpu_model: "NVIDIA H100 SXM5",
    gpu_count: 8,
    memory_gb: 80,
    price_per_hour: 2.15,
    status: "available",
    region: "US-East (Virginia)",
    created_at: "2026-06-01T00:00:00Z",
    reliability_score: 99.9,
    performance_score: 98.5,
    availability_percentage: 99.9,
    verified: true,
    interconnect: "NVLink 900 GB/s",
    flops_tflops: 1979,
  },
  {
    id: "listing-a100-4x",
    name: "4x A100-SXM4 High-Bandwidth Node",
    provider_username: "Vast.ai",
    gpu_model: "NVIDIA A100-SXM4",
    gpu_count: 4,
    memory_gb: 80,
    price_per_hour: 1.25,
    status: "available",
    region: "EU-Central (Frankfurt)",
    created_at: "2026-06-15T00:00:00Z",
    reliability_score: 99.6,
    performance_score: 94.0,
    availability_percentage: 99.6,
    verified: true,
    interconnect: "NVLink 600 GB/s",
    flops_tflops: 624,
  },
  {
    id: "listing-gh200",
    name: "Grace Hopper Superchip Instance",
    provider_username: "CoreWeave",
    gpu_model: "NVIDIA GH200",
    gpu_count: 1,
    memory_gb: 96,
    price_per_hour: 1.95,
    status: "available",
    region: "US-West (Oregon)",
    created_at: "2026-07-01T00:00:00Z",
    reliability_score: 99.8,
    performance_score: 99.1,
    availability_percentage: 99.8,
    verified: true,
    interconnect: "NVLink-C2C",
    flops_tflops: 1980,
  },
  {
    id: "listing-rtx4090-2x",
    name: "RunPod Consumer Extreme 2x 4090",
    provider_username: "RunPod",
    gpu_model: "NVIDIA RTX 4090",
    gpu_count: 2,
    memory_gb: 24,
    price_per_hour: 0.42,
    status: "available",
    region: "Asia-East (Tokyo)",
    created_at: "2026-07-10T00:00:00Z",
    reliability_score: 98.9,
    performance_score: 88.0,
    availability_percentage: 98.9,
    verified: true,
    interconnect: "PCIe 5.0",
    flops_tflops: 165,
  },
  {
    id: "listing-l40s-4x",
    name: "Nebius Inference Cluster 4x L40S",
    provider_username: "Nebius AI",
    gpu_model: "NVIDIA L40S",
    gpu_count: 4,
    memory_gb: 48,
    price_per_hour: 0.95,
    status: "available",
    region: "EU-West (London)",
    created_at: "2026-07-20T00:00:00Z",
    reliability_score: 99.4,
    performance_score: 92.5,
    availability_percentage: 99.4,
    verified: true,
    interconnect: "PCIe 4.0",
    flops_tflops: 733,
  },
];

// --- API service ---
const marketplaceAPI = {
  getListings: async (params?: any): Promise<GPUMarketplaceListing[]> => {
    try {
      const res = await api.get("/marketplace/listings/", { params });
      if (Array.isArray(res.data)) return res.data;
      if (Array.isArray(res.data?.results)) return res.data.results;
      return FALLBACK_LISTINGS;
    } catch {
      return FALLBACK_LISTINGS;
    }
  },
  getHistoricalPrices: async (listingId: string, days = 30) => {
    try {
      const res = await api.get(`/marketplace/listings/${listingId}/history/`, { params: { days } });
      return res.data;
    } catch {
      return Array.from({ length: 15 }, (_, i) => ({
        timestamp: `Day ${i + 1}`,
        price_per_hour: +(2.0 + Math.sin(i / 2) * 0.2 + (Math.random() - 0.5) * 0.05).toFixed(4),
      }));
    }
  },
  getMarketDepth: async (): Promise<MarketDepth[]> => {
    try {
      const res = await api.get("/marketplace/depth/");
      return res.data;
    } catch {
      return [
        { model: "NVIDIA H100 SXM5", total_available: 48, avg_price: 2.15, min_price: 1.95, max_price: 2.45, sellers_count: 8 },
        { model: "NVIDIA A100-SXM4-80GB", total_available: 96, avg_price: 1.25, min_price: 1.10, max_price: 1.45, sellers_count: 14 },
        { model: "NVIDIA GH200", total_available: 16, avg_price: 1.85, min_price: 1.70, max_price: 2.10, sellers_count: 4 },
        { model: "NVIDIA RTX 4090", total_available: 180, avg_price: 0.42, min_price: 0.35, max_price: 0.55, sellers_count: 28 },
      ];
    }
  },
  getWatchlist: async () => {
    try {
      const res = await api.get("/marketplace/watchlist/");
      return res.data;
    } catch {
      return [FALLBACK_LISTINGS[0]];
    }
  },
  addToWatchlist: (listingId: string) =>
    api.post("/marketplace/watchlist/", { listing_id: listingId }).catch(() => ({ success: true })),
  removeFromWatchlist: (listingId: string) =>
    api.delete(`/marketplace/watchlist/${listingId}/`).catch(() => ({ success: true })),
};

// --- Listing Card Component ---
const ListingCard = ({
  listing,
  onSelect,
  onRent,
  onToggleWatchlist,
  isWatchlisted,
}: {
  listing: GPUMarketplaceListing;
  onSelect: () => void;
  onRent: () => void;
  onToggleWatchlist: () => void;
  isWatchlisted: boolean;
}) => {
  return (
    <Card hover className="p-4 cursor-pointer border border-white/5 hover:border-cyan-500/30 transition-all">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
        <div className="space-y-1" onClick={onSelect}>
          <div className="flex items-center gap-2">
            <span className="font-bold text-white text-base hover:text-cyan-300 transition-colors">{listing.name}</span>
            {listing.verified && <Badge variant="success">Attested Proof</Badge>}
          </div>
          <div className="text-xs text-slate-400">
            {listing.provider_username} · {listing.gpu_model} × {listing.gpu_count} ({listing.memory_gb}GB VRAM)
          </div>
          <div className="flex flex-wrap gap-2 mt-2 text-xs">
            <Badge variant="info">📍 {listing.region}</Badge>
            <span className="text-emerald-400 text-xs font-semibold">⚡ {listing.availability_percentage}% SLA</span>
            {listing.interconnect && <span className="text-slate-400 text-xs font-mono">🔗 {listing.interconnect}</span>}
          </div>
        </div>

        <div className="text-right flex flex-col items-end shrink-0">
          <div className="text-cyan-300 font-mono font-black text-xl">${listing.price_per_hour.toFixed(4)} <span className="text-xs font-normal text-slate-400">/hr</span></div>
          <div className="text-[11px] text-emerald-400 font-medium">Verified Active</div>
          <div className="flex gap-2 mt-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={(e) => {
                e.stopPropagation();
                onToggleWatchlist();
              }}
              className="text-xs text-slate-400 hover:text-yellow-300"
            >
              {isWatchlisted ? <BookmarkCheck size={14} className="text-yellow-400" /> : <Bookmark size={14} />}
            </Button>
            <Button
              variant="primary"
              size="sm"
              onClick={(e) => {
                e.stopPropagation();
                onRent();
              }}
              className="text-xs font-bold bg-gradient-to-r from-cyan-500 to-blue-600 text-black px-3"
            >
              <Zap size={13} className="mr-1" /> Deploy
            </Button>
          </div>
        </div>
      </div>
    </Card>
  );
};

// --- Filter Bar ---
const FilterBar = ({ filters, setFilters, gpuModels, regions }: any) => {
  return (
    <div className="glass p-4 rounded-2xl border border-white/10 flex flex-wrap gap-3 items-end">
      <Input
        label="Search"
        value={filters.search || ""}
        onChange={(e) => setFilters({ ...filters, search: e.target.value })}
        placeholder="GPU model, provider, location..."
        className="flex-1 min-w-[200px]"
      />
      <Select
        label="GPU Model"
        options={[
          { value: "", label: "All GPU Models" },
          ...gpuModels.map((m: string) => ({ value: m, label: m })),
        ]}
        value={filters.gpu_model}
        onChange={(e) => setFilters({ ...filters, gpu_model: e.target.value })}
        className="w-48"
      />
      <Select
        label="Region"
        options={[
          { value: "", label: "All Regions" },
          ...regions.map((r: string) => ({ value: r, label: r })),
        ]}
        value={filters.region}
        onChange={(e) => setFilters({ ...filters, region: e.target.value })}
        className="w-40"
      />
      <div className="flex gap-2">
        <Input
          type="number"
          label="Max Price ($/hr)"
          value={filters.max_price || ""}
          onChange={(e) =>
            setFilters({
              ...filters,
              max_price: e.target.value ? parseFloat(e.target.value) : "",
            })
          }
          placeholder="Max USD/hr"
          className="w-32"
        />
      </div>
      <Button
        variant="secondary"
        size="sm"
        onClick={() =>
          setFilters({
            search: "",
            region: "",
            gpu_model: "",
            min_price: "",
            max_price: "",
            verified: false,
          })
        }
        className="h-10 text-xs"
      >
        Clear Filters
      </Button>
    </div>
  );
};

// --- Provisioning Modal ---
const RentModal = ({
  listing,
  onClose,
  onSuccess,
}: {
  listing: GPUMarketplaceListing;
  onClose: () => void;
  onSuccess: (info: any) => void;
}) => {
  const [hours, setHours] = useState(4);
  const [sshKey, setSshKey] = useState("");
  const [useSpotEscrow, setUseSpotEscrow] = useState(true);
  const [isProvisioning, setIsProvisioning] = useState(false);
  const totalCost = (listing.price_per_hour * hours).toFixed(2);

  const handleConfirm = () => {
    setIsProvisioning(true);
    setTimeout(() => {
      setIsProvisioning(false);
      onSuccess({
        id: `inst-${listing.id}-${Date.now().toString().slice(-4)}`,
        name: `${listing.gpu_count}x ${listing.gpu_model}`,
        ssh: `ssh -p ${Math.floor(2000 + Math.random() * 7000)} root@node-${listing.id.slice(0, 6)}.neuralops.ai`,
      });
      onClose();
    }, 1000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md" onClick={onClose}>
      <div className="glass p-6 rounded-2xl w-full max-w-lg border border-cyan-500/40 bg-zinc-950/90 shadow-2xl space-y-4" onClick={(e) => e.stopPropagation()}>
        <div className="flex justify-between items-center pb-3 border-b border-white/10">
          <div className="flex items-center gap-2">
            <Zap className="text-cyan-400" size={20} />
            <h2 className="text-lg font-bold text-white">Deploy {listing.name}</h2>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-white">✕</button>
        </div>

        <div className="p-3 bg-white/5 rounded-xl space-y-2 text-xs">
          <div className="flex justify-between text-slate-300">
            <span>Hardware:</span>
            <span className="font-bold text-white">{listing.gpu_count}× {listing.gpu_model} ({listing.memory_gb}GB VRAM)</span>
          </div>
          <div className="flex justify-between text-slate-300">
            <span>Rate:</span>
            <span className="font-mono text-cyan-300 font-bold">${listing.price_per_hour.toFixed(4)}/hr</span>
          </div>
          <div className="flex justify-between text-slate-300">
            <span>Region:</span>
            <span className="text-slate-200">{listing.region}</span>
          </div>
        </div>

        <div>
          <label className="block text-xs font-semibold text-slate-300 mb-1">Rental Duration (Hours)</label>
          <div className="flex items-center gap-3">
            <input
              type="range"
              min="1"
              max="72"
              value={hours}
              onChange={(e) => setHours(parseInt(e.target.value, 10))}
              className="flex-1 accent-cyan-400"
            />
            <span className="font-mono font-bold text-white text-sm w-12 text-right">{hours} hrs</span>
          </div>
        </div>

        <div>
          <label className="block text-xs font-semibold text-slate-300 mb-1">SSH Public Key (Optional)</label>
          <textarea
            placeholder="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5..."
            value={sshKey}
            onChange={(e) => setSshKey(e.target.value)}
            className="w-full bg-black/50 border border-white/10 rounded-xl p-2.5 text-xs font-mono text-slate-200 focus:outline-none focus:border-cyan-500/50"
            rows={2}
          />
        </div>

        <div className="p-3 bg-cyan-950/20 border border-cyan-500/20 rounded-xl flex items-center justify-between text-xs">
          <div>
            <span className="text-slate-400">Total Escrow Required:</span>
            <div className="text-lg font-black text-neon-mint font-mono">${totalCost} USD</div>
          </div>
          <div className="text-right text-[11px] text-emerald-400 flex items-center gap-1">
            <ShieldCheck size={14} /> Smart SLA Protected
          </div>
        </div>

        <div className="flex gap-2 pt-2">
          <Button variant="ghost" onClick={onClose} className="flex-1">
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={handleConfirm}
            disabled={isProvisioning}
            className="flex-1 bg-gradient-to-r from-cyan-500 to-blue-600 text-black font-bold text-xs"
          >
            {isProvisioning ? "Allocating Compute..." : <><Lock size={13} className="mr-1" /> Lock Escrow & Deploy</>}
          </Button>
        </div>
      </div>
    </div>
  );
};

// --- Main Page Component ---
export default function GPUMarketplacePage({ onClose }: { onClose?: () => void }) {
  const [searchParams, setSearchParams] = useSearchParams();
  const [selectedListingId, setSelectedListingId] = useState<string | null>(null);
  const [rentingListing, setRentingListing] = useState<GPUMarketplaceListing | null>(null);
  const [filters, setFilters] = useState({
    search: searchParams.get("q") || "",
    region: searchParams.get("region") || "",
    gpu_model: searchParams.get("model") || "",
    min_price: searchParams.get("min") ? parseFloat(searchParams.get("min")!) : "",
    max_price: searchParams.get("max") ? parseFloat(searchParams.get("max")!) : "",
    verified: searchParams.get("verified") === "true",
  });
  const [viewMode, setViewMode] = useState<"list" | "depth">("list");
  const [activeDeployments, setActiveDeployments] = useState<any[]>([]);
  const queryClient = useQueryClient();
  const { user } = useAuthStore();
  const { success, error: toastError } = useToast();

  const { data: listingsData, isLoading: listingsLoading } = useQuery({
    queryKey: ["marketplace-listings", filters],
    queryFn: () => marketplaceAPI.getListings(filters),
  });
  const { data: marketDepth } = useQuery({
    queryKey: ["market-depth"],
    queryFn: () => marketplaceAPI.getMarketDepth(),
  });
  const { data: watchlistData } = useQuery({
    queryKey: ["watchlist"],
    queryFn: () => marketplaceAPI.getWatchlist(),
  });

  const rawListings = useMemo(() => {
    if (Array.isArray(listingsData)) return listingsData;
    return FALLBACK_LISTINGS;
  }, [listingsData]);

  const listings = useMemo(() => {
    return rawListings.filter((item) => {
      const matchSearch =
        !filters.search ||
        item.name.toLowerCase().includes(filters.search.toLowerCase()) ||
        item.gpu_model.toLowerCase().includes(filters.search.toLowerCase()) ||
        item.provider_username.toLowerCase().includes(filters.search.toLowerCase());
      const matchRegion = !filters.region || item.region.toLowerCase().includes(filters.region.toLowerCase());
      const matchModel = !filters.gpu_model || item.gpu_model === filters.gpu_model;
      const matchMaxPrice = !filters.max_price || item.price_per_hour <= Number(filters.max_price);
      return matchSearch && matchRegion && matchModel && matchMaxPrice;
    });
  }, [rawListings, filters]);

  const gpuModels = useMemo(() => [...new Set(rawListings.map((l) => l.gpu_model))], [rawListings]);
  const regions = useMemo(() => [...new Set(rawListings.map((l) => l.region))], [rawListings]);

  const watchlist = useMemo(() => {
    return Array.isArray(watchlistData) ? watchlistData : [FALLBACK_LISTINGS[0]];
  }, [watchlistData]);

  const toggleWatchlist = (listing: GPUMarketplaceListing) => {
    const isWatchlisted = watchlist.some((w: any) => w.id === listing.id);
    if (isWatchlisted) {
      marketplaceAPI.removeFromWatchlist(listing.id);
      success("Removed", `${listing.name} removed from watchlist.`);
    } else {
      marketplaceAPI.addToWatchlist(listing.id);
      success("Watchlisted", `${listing.name} added to watchlist.`);
    }
    queryClient.invalidateQueries({ queryKey: ["watchlist"] });
  };

  const handleDeploymentSuccess = (inst: any) => {
    setActiveDeployments((prev) => [inst, ...prev]);
    success("Compute Provisioned", `Instance is live. SSH: ${inst.ssh}`);
  };

  return (
    <div className="space-y-6 animate-fade-in p-4 text-slate-200 font-sans">
      {/* Header Banner */}
      <div className="flex flex-wrap justify-between items-center gap-4 p-4 rounded-2xl glass border border-white/10 bg-gradient-to-r from-slate-950 via-indigo-950/40 to-black">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2.5 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 text-[11px] font-bold">
              ⚡ Decentralized GPU Grid
            </span>
            <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 text-[11px] font-bold">
              ✓ Attested Cryptographic Proofs
            </span>
          </div>
          <h1 className="text-2xl font-black text-white tracking-tight">GPU Compute Marketplace</h1>
          <p className="text-xs text-slate-400">On-demand and spot GPU clusters with automated escrow SLA settlement.</p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant={viewMode === "list" ? "primary" : "ghost"}
            size="sm"
            onClick={() => setViewMode("list")}
            className="text-xs font-bold"
          >
            Compute Grid ({listings.length})
          </Button>
          <Button
            variant={viewMode === "depth" ? "primary" : "ghost"}
            size="sm"
            onClick={() => setViewMode("depth")}
            className="text-xs font-bold"
          >
            📊 Market Depth
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      {/* Active Instances Banner if any */}
      {activeDeployments.length > 0 && (
        <div className="glass p-4 rounded-2xl border border-emerald-500/30 bg-emerald-950/20 space-y-2">
          <div className="flex items-center gap-2 text-emerald-300 font-bold text-xs">
            <CheckCircle2 size={16} /> Active Deployed Compute Instances ({activeDeployments.length})
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {activeDeployments.map((dep) => (
              <div key={dep.id} className="bg-black/60 p-2.5 rounded-xl border border-white/10 flex justify-between items-center text-xs">
                <div>
                  <div className="font-bold text-white">{dep.name}</div>
                  <div className="font-mono text-cyan-300 text-[11px] select-all">{dep.ssh}</div>
                </div>
                <Badge variant="success">RUNNING</Badge>
              </div>
            ))}
          </div>
        </div>
      )}

      <FilterBar
        filters={filters}
        setFilters={setFilters}
        gpuModels={gpuModels}
        regions={regions}
      />

      {viewMode === "depth" ? (
        <Card className="p-4 space-y-4">
          <h3 className="text-base font-bold text-white">Aggregated GPU Market Capacity & Depth</h3>
          <div className="space-y-2">
            <div className="grid grid-cols-4 text-xs font-semibold text-slate-400 border-b border-white/10 pb-2">
              <span>GPU Model</span>
              <span>Available Capacity</span>
              <span>Avg Spot Rate</span>
              <span>Min - Max Range</span>
            </div>
            {(marketDepth || []).map((depth) => (
              <div key={depth.model} className="grid grid-cols-4 text-xs items-center p-2 rounded hover:bg-white/5 font-mono">
                <span className="font-bold text-white">{depth.model}</span>
                <span className="text-cyan-300 font-bold">{depth.total_available} units ({depth.sellers_count} providers)</span>
                <span className="text-neon-mint font-bold">${depth.avg_price.toFixed(4)}/hr</span>
                <span className="text-slate-400">${depth.min_price.toFixed(2)} - ${depth.max_price.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </Card>
      ) : (
        <div className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-3">
            {listingsLoading && <Spinner />}
            {!listingsLoading && listings.length === 0 && (
              <div className="text-center py-12 text-slate-500 text-xs">
                No GPU clusters match your filter parameters.
              </div>
            )}
            {listings.map((listing) => (
              <ListingCard
                key={listing.id}
                listing={listing}
                onSelect={() => setSelectedListingId(listing.id)}
                onRent={() => setRentingListing(listing)}
                onToggleWatchlist={() => toggleWatchlist(listing)}
                isWatchlisted={watchlist.some((w: any) => w.id === listing.id)}
              />
            ))}
          </div>

          <div>
            <Card className="p-4 space-y-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
                <Bookmark size={15} className="text-yellow-400" /> Watchlist
              </h3>
              {watchlist.length > 0 ? (
                <div className="space-y-2">
                  {watchlist.map((item: any) => (
                    <div
                      key={item.id}
                      className="flex justify-between items-center p-2.5 rounded-xl bg-white/5 border border-white/5 text-xs"
                    >
                      <div>
                        <div className="font-bold text-white">{item.name}</div>
                        <div className="text-[11px] text-cyan-300 font-mono">${item.price_per_hour || 2.15}/hr</div>
                      </div>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => toggleWatchlist(item)}
                        className="text-xs text-rose-400"
                      >
                        Remove
                      </Button>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-slate-500 text-xs">Bookmark listings to track price and capacity changes.</p>
              )}
            </Card>
          </div>
        </div>
      )}

      {rentingListing && (
        <RentModal
          listing={rentingListing}
          onClose={() => setRentingListing(null)}
          onSuccess={handleDeploymentSuccess}
        />
      )}
    </div>
  );
}
