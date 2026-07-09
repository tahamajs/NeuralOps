import React from "react";
import { CreditCard, Check, Zap, XCircle } from "lucide-react";
import { Button } from "../../../components/ui/Button";

interface PlanCardProps {
  plan: any;
  interval: "monthly" | "yearly";
  isCurrent: boolean;
  onSubscribe: () => void;
  onCancel: () => void;
}

export function PlanCard({
  plan,
  interval,
  isCurrent,
  onSubscribe,
  onCancel,
}: PlanCardProps) {
  const basePrice = parseFloat(plan.monthly_base_fee_usd || 0);
  const displayedPrice = interval === "yearly" ? basePrice * 0.8 * 12 : basePrice;

  return (
    <div className={`p-6 bg-slate-900/30 backdrop-blur-md border rounded-3xl flex flex-col justify-between h-[420px] transition-all duration-300 relative group overflow-hidden ${
      isCurrent
        ? "border-neon-blue bg-neon-blue/5 shadow-[0_0_20px_rgba(99,102,241,0.15)]"
        : "border-white/10 hover:border-slate-700 hover:bg-slate-900/40"
    }`}>
      {/* Background radial glow */}
      <div className="absolute -top-24 -right-24 w-48 h-48 bg-neon-blue/5 rounded-full blur-3xl pointer-events-none group-hover:bg-neon-blue/10 transition-all duration-500" />

      <div>
        <div className="flex justify-between items-start mb-4">
          <div>
            <h3 className="text-lg font-black text-white uppercase tracking-tight">{plan.name}</h3>
            <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">{plan.tier} Tier</span>
          </div>
          {isCurrent && (
            <span className="px-2.5 py-0.5 rounded-full bg-neon-blue/10 border border-neon-blue/20 text-[10px] font-black uppercase text-neon-blue">
              Active
            </span>
          )}
        </div>

        <div className="mb-6">
          <span className="text-3xl font-black text-white font-sans">${displayedPrice.toFixed(2)}</span>
          <span className="text-slate-500 text-xs font-semibold">
            /{interval === "yearly" ? "yr" : "mo"}
          </span>
        </div>

        {/* Features */}
        <ul className="space-y-2.5 text-xs text-slate-400">
          <li className="flex items-center gap-2">
            <Check size={14} className="text-neon-mint" />
            <span>GPU rental markup: <strong className="text-slate-200">{plan.gpu_markup_pct}%</strong></span>
          </li>
          <li className="flex items-center gap-2">
            <Check size={14} className="text-neon-mint" />
            <span>Max concurrent jobs: <strong className="text-slate-200">{plan.max_concurrent_jobs}</strong></span>
          </li>
          <li className="flex items-center gap-2">
            <Check size={14} className="text-neon-mint" />
            <span>Included GPU hours: <strong className="text-slate-200">{plan.included_gpu_hours} hrs</strong></span>
          </li>
          <li className="flex items-center gap-2">
            <Check size={14} className="text-neon-mint" />
            <span>Priority Support: <strong className="text-slate-200">{plan.priority_support ? "Yes" : "No"}</strong></span>
          </li>
        </ul>
      </div>

      <div className="pt-6 border-t border-white/5 flex flex-col gap-2 shrink-0">
        {isCurrent ? (
          plan.tier !== "free" ? (
            <Button
              onClick={onCancel}
              variant="outline"
              className="w-full border-rose-500/20 text-rose-400 hover:bg-rose-500/10 flex items-center justify-center gap-1.5"
            >
              <XCircle size={14} /> Cancel Plan
            </Button>
          ) : (
            <Button disabled className="w-full opacity-50 cursor-not-allowed">
              Default Tier
            </Button>
          )
        ) : (
          <Button
            onClick={onSubscribe}
            className="w-full bg-gradient-to-r from-neon-blue to-purple-600 hover:from-neon-blue/90 hover:to-purple-600/90 text-white flex items-center justify-center gap-1.5 shadow-glow-sm"
          >
            <Zap size={14} /> Upgrade Plan
          </Button>
        )}
      </div>
    </div>
  );
}
export default PlanCard;
