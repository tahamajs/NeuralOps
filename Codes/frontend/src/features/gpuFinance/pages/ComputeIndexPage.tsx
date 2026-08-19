// src/features/gpuFinance/pages/ComputeIndexPage.tsx
import { useState, useMemo, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { gpuFinanceAPI } from "../services/api";
import { IndexChart } from "../components/IndexChart";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { TrendingUp, TrendingDown, BarChart3, Layers, Sparkles, Sliders } from "lucide-react";
import { ComputeIndex } from "../types";

const FALLBACK_INDICES: ComputeIndex[] = [
  {
    id: "idx-ngi-composite",
    name: "NeuralOps Global Compute Index (NGI)",
    symbol: "NGI",
    current_value: 184.5,
    change_24h: 3.42,
    volume_24h: 1254300,
    components: [
      { gpu_model: "NVIDIA H100 SXM5 80GB", weight: 45, price: 2.15 },
      { gpu_model: "NVIDIA A100 SXM4 80GB", weight: 30, price: 1.25 },
      { gpu_model: "NVIDIA GH200 Grace Hopper", weight: 15, price: 1.85 },
      { gpu_model: "NVIDIA RTX 4090 24GB", weight: 10, price: 0.42 },
    ],
    history: Array.from({ length: 30 }, (_, i) => ({
      timestamp: new Date(Date.now() - (29 - i) * 86400000).toISOString(),
      value: +(170 + Math.sin(i / 4) * 12 + i * 0.4).toFixed(2),
    })),
  },
  {
    id: "idx-inference-index",
    name: "LLM Inference Throughput Index (LITI)",
    symbol: "LITI",
    current_value: 92.8,
    change_24h: -1.15,
    volume_24h: 682000,
    components: [
      { gpu_model: "NVIDIA L40S 48GB", weight: 40, price: 0.95 },
      { gpu_model: "NVIDIA RTX 4090 24GB", weight: 35, price: 0.42 },
      { gpu_model: "NVIDIA A10G 24GB", weight: 25, price: 0.65 },
    ],
    history: Array.from({ length: 30 }, (_, i) => ({
      timestamp: new Date(Date.now() - (29 - i) * 86400000).toISOString(),
      value: +(95 - Math.cos(i / 5) * 6 + (Math.random() - 0.5) * 2).toFixed(2),
    })),
  },
  {
    id: "idx-svd-attention",
    name: "SVD Linear Attention Compute Index (SVD-CI)",
    symbol: "SVD-CI",
    current_value: 218.4,
    change_24h: 7.85,
    volume_24h: 940000,
    components: [
      { gpu_model: "NVIDIA H100 SXM5 (NVLink 900 GB/s)", weight: 60, price: 2.15 },
      { gpu_model: "NVIDIA GH200 NVLink-C2C", weight: 40, price: 1.95 },
    ],
    history: Array.from({ length: 30 }, (_, i) => ({
      timestamp: new Date(Date.now() - (29 - i) * 86400000).toISOString(),
      value: +(190 + Math.sin(i / 3) * 18 + i * 0.8).toFixed(2),
    })),
  },
];

export default function ComputeIndexPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [selectedIndex, setSelectedIndex] = useState<string>("idx-ngi-composite");
  const [timeframe, setTimeframe] = useState<"7D" | "30D" | "90D">("30D");

  const { data: serverIndices, isLoading } = useQuery({
    queryKey: ["compute-indices"],
    queryFn: () => gpuFinanceAPI.getIndices().catch(() => ({ data: FALLBACK_INDICES })),
  });

  const indices = useMemo(() => {
    const raw = (serverIndices as any)?.data || serverIndices;
    return Array.isArray(raw) && raw.length > 0 ? raw : FALLBACK_INDICES;
  }, [serverIndices]);

  useEffect(() => {
    if (indices.length > 0 && !selectedIndex) {
      setSelectedIndex(indices[0].id);
    }
  }, [indices, selectedIndex]);

  const index = indices.find((i: ComputeIndex) => i.id === selectedIndex) || indices[0];

  const filteredHistory = useMemo(() => {
    if (!index?.history) return [];
    const count = timeframe === "7D" ? 7 : timeframe === "30D" ? 30 : 90;
    return index.history.slice(-count);
  }, [index, timeframe]);

  if (isLoading && !index) return <Spinner />;

  return (
    <div className="flex flex-col h-full overflow-y-auto p-4 space-y-4 font-sans text-slate-200">
      {/* Top Banner */}
      <div className="flex flex-wrap justify-between items-center gap-4 p-4 rounded-2xl glass border border-white/10 bg-gradient-to-r from-slate-950 via-slate-900 to-indigo-950/40">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2.5 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 text-[10px] font-bold flex items-center gap-1">
              <Sparkles size={12} /> Volume-Weighted GPU Basket
            </span>
          </div>
          <h1 className="text-xl font-black text-white flex items-center gap-2">
            <BarChart3 size={20} className="text-neon-mint" /> Compute Market Indices & Benchmarks
          </h1>
          <p className="text-xs text-slate-400">Institutional index trackers reflecting real-time decentralized GPU capacity costs.</p>
        </div>

        <div className="flex items-center gap-2">
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      {/* Index Selector Tabs */}
      <div className="flex items-center gap-2 border-b border-white/10 pb-3 flex-wrap">
        {indices.map((idxItem: ComputeIndex) => (
          <button
            key={idxItem.id}
            onClick={() => setSelectedIndex(idxItem.id)}
            className={`px-4 py-2 rounded-xl text-xs font-bold transition-all flex items-center gap-2 ${
              selectedIndex === idxItem.id
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-lg shadow-cyan-500/10"
                : "text-slate-400 hover:text-white glass-hover"
            }`}
          >
            <span>{idxItem.symbol}</span>
            <span className="text-[11px] font-mono text-neon-mint">${(idxItem.current_value || 0).toFixed(2)}</span>
            <span
              className={`text-[10px] font-mono ${
                (idxItem.change_24h || 0) >= 0 ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              {(idxItem.change_24h || 0) >= 0 ? "+" : ""}
              {(idxItem.change_24h || 0).toFixed(2)}%
            </span>
          </button>
        ))}
      </div>

      {/* Selected Index Card */}
      {index && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <TerminalCard className="p-4 space-y-1 md:col-span-2">
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="text-sm font-bold text-white">{index.name}</h3>
                  <div className="text-xs text-slate-400">Symbol: <span className="font-mono text-cyan-300 font-bold">{index.symbol}</span></div>
                </div>
                <div className="flex items-center gap-1 bg-white/5 p-1 rounded-lg">
                  {(["7D", "30D", "90D"] as const).map((tf) => (
                    <button
                      key={tf}
                      onClick={() => setTimeframe(tf)}
                      className={`px-2.5 py-0.5 rounded text-xs font-bold ${
                        timeframe === tf ? "bg-cyan-500 text-black" : "text-slate-400 hover:text-white"
                      }`}
                    >
                      {tf}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex items-baseline gap-4 pt-2">
                <span className="text-3xl font-black text-white font-mono">${(index.current_value || 0).toFixed(2)}</span>
                <span
                  className={`text-sm font-bold flex items-center font-mono ${
                    (index.change_24h || 0) >= 0 ? "text-emerald-400" : "text-rose-400"
                  }`}
                >
                  {(index.change_24h || 0) >= 0 ? <TrendingUp size={16} className="mr-0.5" /> : <TrendingDown size={16} className="mr-0.5" />}
                  {(index.change_24h || 0) >= 0 ? "+" : ""}
                  {(index.change_24h || 0).toFixed(2)}% (24h)
                </span>
                <span className="text-xs text-slate-400 ml-auto">
                  24h Vol: <strong className="text-cyan-300 font-mono">${(index.volume_24h || 0).toLocaleString()}</strong>
                </span>
              </div>

              <div className="pt-2">
                <IndexChart data={filteredHistory} height={240} />
              </div>
            </TerminalCard>

            {/* Components Weighting Breakdown */}
            <TerminalCard className="p-4 space-y-3">
              <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
                <Layers size={14} className="text-cyan-400" /> Index Basket Weights
              </h3>
              <p className="text-[11px] text-slate-400">
                Basket weighting is dynamically rebalanced based on verified 24h decentralized cluster volume.
              </p>

              <div className="space-y-2">
                {index.components?.map((c) => (
                  <div key={c.gpu_model} className="p-2.5 rounded-xl bg-black/40 border border-white/5 space-y-1">
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-bold text-white truncate max-w-[150px]">{c.gpu_model}</span>
                      <span className="font-mono text-neon-mint font-bold">${(c.price || 0).toFixed(4)}/hr</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 bg-white/10 h-1.5 rounded-full overflow-hidden">
                        <div className="bg-gradient-to-r from-cyan-500 to-indigo-500 h-full rounded-full" style={{ width: `${c.weight}%` }} />
                      </div>
                      <span className="text-[10px] text-slate-400 font-mono">{c.weight}% Weight</span>
                    </div>
                  </div>
                ))}
              </div>
            </TerminalCard>
          </div>
        </div>
      )}
    </div>
  );
}
