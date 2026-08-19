import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { marketplaceAIApi } from "../services/api";
import { DynamicComparisonTable } from "../components/DynamicComparisonTable";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { Plus, Save, Trash2, Copy, Layers, Sparkles, Check } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";
import { ComparisonGroup } from "../types";

export default function ComparisonBuilder({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [selectedListings, setSelectedListings] = useState<string[]>([]);
  const [comparisonName, setComparisonName] = useState("");
  const [activeComparisonId, setActiveComparisonId] = useState<string | null>(null);
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: comparisons, isLoading: compsLoading } = useQuery({
    queryKey: ["comparison-groups"],
    queryFn: () => marketplaceAIApi.getComparisonGroups(),
  });

  const { data: listings, isLoading: listingsLoading } = useQuery({
    queryKey: ["marketplace-listings"],
    queryFn: () => marketplaceAIApi.getListings({ limit: 50 }),
  });

  const allComps = comparisons || [];
  useEffect(() => {
    if (allComps.length > 0 && !activeComparisonId) {
      setActiveComparisonId(allComps[0].id);
    }
  }, [allComps, activeComparisonId]);

  const createComparison = useMutation({
    mutationFn: () =>
      marketplaceAIApi.createComparisonGroup(
        comparisonName || "Custom Architecture Comparison",
        selectedListings,
      ),
    onSuccess: (newComp) => {
      queryClient.invalidateQueries({ queryKey: ["comparison-groups"] });
      setActiveComparisonId(newComp.id);
      setSelectedListings([]);
      setComparisonName("");
      success("Comparison Created", `${newComp.name} saved to your workspace.`);
    },
  });

  const deleteComparison = useMutation({
    mutationFn: (id: string) => marketplaceAIApi.deleteComparisonGroup(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["comparison-groups"] });
      success("Comparison Deleted");
    },
  });

  const activeComparison = allComps.find((c) => c.id === activeComparisonId);
  const allListings = listings?.results || [];

  if (compsLoading && listingsLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full overflow-y-auto p-4 space-y-4 font-sans text-slate-200">
      {/* Top Banner */}
      <div className="flex flex-wrap justify-between items-center gap-4 p-4 rounded-2xl glass border border-white/10 bg-gradient-to-r from-slate-950 via-slate-900 to-indigo-950/40">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2.5 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 text-[10px] font-bold flex items-center gap-1">
              <Sparkles size={12} /> Multi-Cluster Telemetry Matcher
            </span>
          </div>
          <h1 className="text-xl font-black text-white flex items-center gap-2">
            <Layers size={20} className="text-neon-mint" /> GPU Comparison & Benchmark Builder
          </h1>
          <p className="text-xs text-slate-400">Side-by-side hardware evaluation across VRAM, interconnects, SLA compliance, and pricing.</p>
        </div>

        <div className="flex items-center gap-2">
          {onClose && (
            <Button variant="ghost" size="sm" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      <div className="grid lg:grid-cols-12 gap-4">
        {/* Left Column: Comparison Groups & Creator */}
        <div className="lg:col-span-4 space-y-4">
          <TerminalCard className="p-4 space-y-3">
            <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
              <Layers size={14} className="text-cyan-400" /> Saved Comparisons ({allComps.length})
            </h3>
            <div className="space-y-1.5">
              {allComps.map((c) => (
                <div
                  key={c.id}
                  className={`flex justify-between items-center p-2.5 rounded-xl cursor-pointer transition-all border ${
                    activeComparisonId === c.id
                      ? "bg-cyan-950/30 border-cyan-500/40 text-white"
                      : "bg-white/5 border-transparent hover:bg-white/10 text-slate-300"
                  }`}
                  onClick={() => setActiveComparisonId(c.id)}
                >
                  <div>
                    <div className="text-xs font-bold">{c.name}</div>
                    <div className="text-[10px] text-slate-400 font-mono">{c.listings?.length || 0} hardware targets</div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteComparison.mutate(c.id);
                    }}
                    className="p-1 hover:bg-rose-500/20 rounded text-rose-400"
                    title="Delete Comparison"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
            </div>
          </TerminalCard>

          <TerminalCard className="p-4 space-y-3 bg-gradient-to-b from-slate-900 to-slate-950 border border-cyan-500/30">
            <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
              <Plus size={14} className="text-cyan-400" /> Create Comparison Group
            </h3>
            <Input
              placeholder="e.g. SVD Attention: 8x H100 vs 4x GH200"
              value={comparisonName}
              onChange={(e) => setComparisonName(e.target.value)}
            />

            <div>
              <div className="text-xs text-slate-400 mb-1 flex justify-between">
                <span>Selected Nodes ({selectedListings.length})</span>
                {selectedListings.length > 0 && (
                  <button onClick={() => setSelectedListings([])} className="text-rose-400 hover:text-rose-300 text-[10px]">
                    Clear
                  </button>
                )}
              </div>
              <div className="flex flex-wrap gap-1 max-h-24 overflow-y-auto">
                {selectedListings.map((id) => {
                  const listing = allListings.find((l) => l.id === id);
                  return listing ? (
                    <span
                      key={id}
                      className="px-2 py-0.5 rounded-lg bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 text-[11px] font-mono font-semibold"
                    >
                      {listing.gpu_specs?.model?.slice(0, 14)}
                    </span>
                  ) : null;
                })}
              </div>
            </div>

            <Button
              size="sm"
              className="w-full font-bold bg-gradient-to-r from-cyan-500 to-blue-600 text-black py-2.5 rounded-xl text-xs"
              onClick={() => createComparison.mutate()}
              disabled={selectedListings.length === 0}
            >
              <Save size={13} className="mr-1" /> Save & Compare
            </Button>
          </TerminalCard>
        </div>

        {/* Right Column: Dynamic Comparison Table & Hardware Grid */}
        <div className="lg:col-span-8 space-y-4">
          <TerminalCard className="p-4 space-y-3">
            <div className="flex justify-between items-center">
              <h3 className="text-sm font-bold text-white">
                {activeComparison ? activeComparison.name : "Hardware Comparison"}
              </h3>
            </div>
            {activeComparison && activeComparison.listings?.length ? (
              <DynamicComparisonTable listings={activeComparison.listings} />
            ) : (
              <div className="text-center text-slate-500 py-12 text-xs">
                Select or create a comparison group to see side‑by‑side performance metrics.
              </div>
            )}
          </TerminalCard>

          {/* Quick Hardware Catalog Selector */}
          <TerminalCard className="p-4 space-y-3">
            <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
              <Plus size={14} className="text-cyan-400" /> Add Hardware from Inventory
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-52 overflow-y-auto pr-1">
              {allListings.map((listing) => {
                const isSelected = selectedListings.includes(listing.id);
                return (
                  <div
                    key={listing.id}
                    className={`flex items-center justify-between p-2.5 rounded-xl border transition-all text-xs ${
                      isSelected ? "bg-cyan-950/20 border-cyan-500/40" : "bg-white/5 border-white/5"
                    }`}
                  >
                    <div>
                      <div className="font-bold text-white">{listing.gpu_specs?.model}</div>
                      <div className="text-[10px] text-slate-400">{listing.provider} · ${listing.price_per_hour}/hr</div>
                    </div>
                    <Button
                      size="sm"
                      variant={isSelected ? "secondary" : "ghost"}
                      onClick={() =>
                        setSelectedListings((prev) =>
                          prev.includes(listing.id)
                            ? prev.filter((id) => id !== listing.id)
                            : [...prev, listing.id]
                        )
                      }
                      className="text-xs font-semibold py-1 px-2.5 h-7"
                    >
                      {isSelected ? <Check size={12} className="text-emerald-400" /> : "Add"}
                    </Button>
                  </div>
                );
              })}
            </div>
          </TerminalCard>
        </div>
      </div>
    </div>
  );
}
