import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { marketplaceAIApi } from "../services/api";
import { DynamicComparisonTable } from "../components/DynamicComparisonTable";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Spinner } from "../../../components/ui/Spinner";
import { Plus, Save, Trash2, Copy } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function ComparisonBuilder({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [selectedListings, setSelectedListings] = useState<string[]>([]);
  const [comparisonName, setComparisonName] = useState("");
  const [activeComparisonId, setActiveComparisonId] = useState<string | null>(
    null,
  );
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

  const createComparison = useMutation({
    mutationFn: () =>
      marketplaceAIApi.createComparisonGroup(
        comparisonName || "Untitled Comparison",
        selectedListings,
      ),
    onSuccess: (newComp) => {
      queryClient.invalidateQueries({ queryKey: ["comparison-groups"] });
      setActiveComparisonId(newComp.id);
      setSelectedListings([]);
      setComparisonName("");
      success("Comparison created");
    },
  });
  const deleteComparison = useMutation({
    mutationFn: (id: string) => marketplaceAIApi.deleteComparisonGroup(id),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["comparison-groups"] }),
  });
  const addToListings = useMutation({
    mutationFn: (ids: string[]) =>
      marketplaceAIApi.addToListingsToComparison(activeComparisonId!, ids),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["comparison-groups"] }),
  });

  const activeComparison = comparisons?.find(
    (c) => c.id === activeComparisonId,
  );
  const allListings = listings?.results || [];

  if (compsLoading || listingsLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <h1 className="text-xl font-bold text-white">Comparison Builder</h1>
        {onClose && (
          <Button variant="ghost" size="sm" onClick={onClose}>
            ✕
          </Button>
        )}
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-6">
        <div className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 space-y-4">
            <TerminalCard className="p-4">
              <h3 className="text-sm font-bold text-white mb-3">
                Your Comparisons
              </h3>
              <div className="space-y-2">
                {comparisons?.map((c) => (
                  <div
                    key={c.id}
                    className={`flex justify-between items-center p-2 rounded cursor-pointer ${activeComparisonId === c.id ? "bg-neon-blue/20" : "hover:bg-white/5"}`}
                    onClick={() => setActiveComparisonId(c.id)}
                  >
                    <span className="text-sm text-white">{c.name}</span>
                    <div className="flex gap-1">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteComparison.mutate(c.id);
                        }}
                      >
                        <Trash2 size={12} className="text-red-400" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </TerminalCard>
            <TerminalCard className="p-4">
              <h3 className="text-sm font-bold text-white mb-2">Create New</h3>
              <Input
                placeholder="Comparison name"
                value={comparisonName}
                onChange={(e) => setComparisonName(e.target.value)}
              />
              <div className="mt-2">
                <div className="text-xs text-slate-400 mb-1">
                  Selected ({selectedListings.length})
                </div>
                <div className="flex flex-wrap gap-1 max-h-24 overflow-y-auto">
                  {selectedListings.map((id) => {
                    const listing = allListings.find((l) => l.id === id);
                    return listing ? (
                      <span
                        key={id}
                        className="px-2 py-0.5 rounded bg-neon-blue/10 text-xs"
                      >
                        {listing.gpu_specs.model}
                      </span>
                    ) : null;
                  })}
                </div>
              </div>
              <Button
                size="sm"
                className="mt-3 w-full"
                onClick={() => createComparison.mutate()}
                disabled={selectedListings.length === 0}
              >
                <Save size={12} className="mr-1" /> Save Comparison
              </Button>
            </TerminalCard>
          </div>
          <div className="lg:col-span-2">
            <TerminalCard className="p-4">
              {activeComparison ? (
                <DynamicComparisonTable listings={activeComparison.listings} />
              ) : (
                <div className="text-center text-slate-500 py-12">
                  Select or create a comparison to see side‑by‑side details
                </div>
              )}
            </TerminalCard>
          </div>
        </div>
        <TerminalCard className="p-4">
          <h3 className="text-sm font-bold text-white mb-3">Browse Listings</h3>
          <div className="grid grid-cols-2 gap-2 max-h-60 overflow-y-auto">
            {allListings.map((listing) => (
              <div
                key={listing.id}
                className="flex items-center justify-between p-2 bg-white/5 rounded"
              >
                <div>
                  <div className="text-sm text-white">
                    {listing.gpu_specs.model}
                  </div>
                  <div className="text-xs text-slate-400">
                    {listing.provider}
                  </div>
                </div>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() =>
                    setSelectedListings((prev) =>
                      prev.includes(listing.id)
                        ? prev.filter((id) => id !== listing.id)
                        : [...prev, listing.id],
                    )
                  }
                >
                  {selectedListings.includes(listing.id) ? "Remove" : "Add"}
                </Button>
              </div>
            ))}
          </div>
        </TerminalCard>
      </div>
    </div>
  );
}
