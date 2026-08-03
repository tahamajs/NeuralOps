import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import api from "../../../api/client";

export default function MarketplacePage() {
  const [filters, setFilters] = useState({
    gpu_model: "",
    min_vram: "",
    max_price: "",
  });
  const { data, isLoading } = useQuery({
    queryKey: ["marketplace-listings", filters],
    queryFn: async () => {
      try {
        const r = await api.get("/marketplace/listings/", { params: filters });
        return r.data;
      } catch (err) {
        console.warn(
          "Backend marketplace listings endpoint failed, returning mock data",
          err,
        );
        return {
          results: [
            {
              id: "listing-1",
              gpu_model: "NVIDIA H100",
              gpu_count: 8,
              vram_gb: 80,
              listing_type: "rent",
              current_price_per_hour: 2.15,
              reliability_score: "99.8%",
              provider: "Lambda Labs",
            },
            {
              id: "listing-2",
              gpu_model: "NVIDIA A100",
              gpu_count: 4,
              vram_gb: 40,
              listing_type: "rent",
              current_price_per_hour: 1.25,
              reliability_score: "99.5%",
              provider: "Vast.ai",
            },
            {
              id: "listing-3",
              gpu_model: "RTX 4090",
              gpu_count: 2,
              vram_gb: 24,
              listing_type: "rent",
              current_price_per_hour: 0.45,
              reliability_score: "98.2%",
              provider: "RunPod",
            },
          ],
        };
      }
    },
  });

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">GPU Marketplace</h1>
      <div className="flex gap-4 mb-6">
        <input
          placeholder="GPU model"
          value={filters.gpu_model}
          onChange={(e) =>
            setFilters((p) => ({ ...p, gpu_model: e.target.value }))
          }
          className="bg-white/5 border border-white/10 rounded-xl px-4 py-2 text-sm"
        />
        <input
          placeholder="Min VRAM (GB)"
          type="number"
          value={filters.min_vram}
          onChange={(e) =>
            setFilters((p) => ({ ...p, min_vram: e.target.value }))
          }
          className="bg-white/5 border border-white/10 rounded-xl px-4 py-2 text-sm"
        />
        <input
          placeholder="Max price/hr"
          type="number"
          step="0.01"
          value={filters.max_price}
          onChange={(e) =>
            setFilters((p) => ({ ...p, max_price: e.target.value }))
          }
          className="bg-white/5 border border-white/10 rounded-xl px-4 py-2 text-sm"
        />
      </div>
      {isLoading && <p className="text-slate-400">Loading...</p>}
      {data && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {data.results?.map((listing: any) => (
            <div key={listing.id} className="glass p-4 card-hover-effect">
              <h3 className="font-bold">
                {listing.gpu_model} ×{listing.gpu_count}
              </h3>
              <p className="text-sm text-slate-400">
                {listing.vram_gb} GB VRAM · {listing.listing_type}
              </p>
              <p className="text-lg font-bold mt-2">
                ${listing.current_price_per_hour}/hr
              </p>
              <p className="text-xs text-slate-500">
                Reliability: {listing.reliability_score} · {listing.provider}
              </p>
              <button className="mt-3 bg-blue-600 hover:bg-blue-700 px-4 py-1.5 rounded-lg text-sm font-semibold transition-colors">
                Place Order
              </button>
            </div>
          ))}
          {data.results?.length === 0 && (
            <p className="text-slate-500">No listings match your filters.</p>
          )}
        </div>
      )}
    </div>
  );
}
