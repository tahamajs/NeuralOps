import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";
import { cn } from "../../../lib/utils";
import { Zap, Sparkles, Layers } from "lucide-react";

export const MASTER_ATTENTIONS = [
  { id: "svd_linear", name: "SVD Linear (Stiefel)", category: "svd", speedup: "2.4×", badge: "bg-cyan-500/20 text-cyan-300 border-cyan-500/30" },
  { id: "dynamic_entropy_svd", name: "Dynamic Entropy SVD", category: "svd", speedup: "2.8×", badge: "bg-sky-500/20 text-sky-300 border-sky-500/30" },
  { id: "ressvd", name: "ResSVD Residual", category: "svd", speedup: "2.1×", badge: "bg-indigo-500/20 text-indigo-300 border-indigo-500/30" },
  { id: "flash_attention3", name: "FlashAttention-3 (FP8)", category: "linear", speedup: "2.2×", badge: "bg-rose-500/20 text-rose-300 border-rose-500/30" },
  { id: "griffin_ssd", name: "Griffin-SSD Recurrent", category: "linear", speedup: "3.1×", badge: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30" },
  { id: "gated_deltanet", name: "Gated DeltaNet", category: "linear", speedup: "2.6×", badge: "bg-teal-500/20 text-teal-300 border-teal-500/30" },
  { id: "mla_latent", name: "MLA Latent (DeepSeek)", category: "linear", speedup: "2.5×", badge: "bg-pink-500/20 text-pink-300 border-pink-500/30" },
  { id: "hymba_transx", name: "Hymba-TransX (SSM)", category: "hybrid", speedup: "2.7×", badge: "bg-amber-500/20 text-amber-300 border-amber-500/30" },
  { id: "jamba2", name: "Jamba2 (Mamba-2 MoE)", category: "hybrid", speedup: "3.0×", badge: "bg-yellow-500/20 text-yellow-300 border-yellow-500/30" },
  { id: "nsa_sparse", name: "NSA Native Sparse", category: "sparse", speedup: "2.9×", badge: "bg-purple-500/20 text-purple-300 border-purple-500/30" },
  { id: "infini_attention", name: "Infini-Attention (1M+)", category: "sparse", speedup: "2.8×", badge: "bg-blue-500/20 text-blue-300 border-blue-500/30" },
  { id: "retnet", name: "RetNet Retentive", category: "linear", speedup: "3.2×", badge: "bg-cyan-500/20 text-cyan-300 border-cyan-500/30" },
  { id: "rwkv6", name: "RWKV-6 Eagle", category: "linear", speedup: "3.4×", badge: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30" },
  { id: "performer", name: "Performer (FAVOR+)", category: "linear", speedup: "2.3×", badge: "bg-teal-500/20 text-teal-300 border-teal-500/30" },
  { id: "softmax_mha", name: "Softmax MHA (Standard)", category: "standard", speedup: "1.0×", badge: "bg-slate-500/20 text-slate-300 border-slate-500/30" },
  { id: "gqa", name: "Grouped Query (GQA)", category: "standard", speedup: "1.4×", badge: "bg-amber-500/20 text-amber-300 border-amber-500/30" },
];

export const AttentionPicker = ({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) => {
  const [category, setCategory] = useState<string>("all");
  const { data } = useQuery({
    queryKey: ["config/attentions"],
    queryFn: async () => (await api.get(ENDPOINTS.CONFIG.ATTENTIONS)).data,
  });

  const apiAttentions: string[] = data?.attentions || [];
  const displayList =
    apiAttentions.length > 0
      ? apiAttentions.map((id) => {
          const found = MASTER_ATTENTIONS.find((m) => m.id === id);
          return (
            found || {
              id,
              name: id.replace(/_/g, " "),
              category: "custom",
              speedup: "2.0×",
              badge: "bg-cyan-500/20 text-cyan-300 border-cyan-500/30",
            }
          );
        })
      : MASTER_ATTENTIONS;

  const filtered = displayList.filter(
    (item) => category === "all" || item.category === category
  );

  return (
    <div className="space-y-2.5">
      <div className="flex items-center justify-between">
        <label className="text-[10px] font-bold text-slate-400 uppercase tracking-widest flex items-center gap-1.5">
          <Sparkles size={12} className="text-cyan-400" /> Attention Architecture ({filtered.length})
        </label>
        <div className="flex items-center gap-1 text-[10px] font-mono">
          {[
            { id: "all", label: "All" },
            { id: "svd", label: "SVD" },
            { id: "linear", label: "Linear" },
            { id: "hybrid", label: "Hybrid" },
            { id: "sparse", label: "Sparse" },
            { id: "standard", label: "Standard" },
          ].map((cat) => (
            <button
              key={cat.id}
              type="button"
              onClick={() => setCategory(cat.id)}
              className={cn(
                "px-2 py-0.5 rounded-lg transition-colors cursor-pointer",
                category === cat.id
                  ? "bg-cyan-500 text-black font-bold"
                  : "bg-white/5 text-slate-400 hover:text-white"
              )}
            >
              {cat.label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 max-h-48 overflow-y-auto scrollbar-thin p-1 bg-black/30 rounded-2xl border border-white/5">
        {filtered.map((a) => {
          const isSelected = value === a.id;
          return (
            <button
              key={a.id}
              type="button"
              onClick={() => onChange(a.id)}
              className={cn(
                "p-2.5 rounded-xl text-left border transition-all cursor-pointer flex flex-col justify-between group",
                isSelected
                  ? "bg-cyan-500/20 border-cyan-500/60 shadow-lg shadow-cyan-500/10 text-white"
                  : "bg-white/[0.02] border-white/10 text-slate-300 hover:bg-white/10 hover:border-white/20"
              )}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[9px] font-mono text-cyan-400 font-bold px-1.5 py-0.2 rounded bg-cyan-950/80 border border-cyan-500/20">
                  {a.speedup}
                </span>
                {isSelected && <Zap size={12} className="text-cyan-400" />}
              </div>
              <span className="text-xs font-bold capitalize leading-tight block group-hover:text-cyan-300 transition-colors">
                {a.name}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
};
