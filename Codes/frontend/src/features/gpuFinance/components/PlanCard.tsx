import React from "react";
import { CreditCard, Check, Zap, XCircle, Sparkles } from "lucide-react";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";

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
  const baseMonthlyPrice = parseFloat(plan.price_monthly ?? plan.monthly_base_fee_usd ?? 0);
  const annualBasePrice = parseFloat(plan.price_yearly ?? baseMonthlyPrice * 0.8 * 12);
  const displayedPrice = interval === "yearly" ? annualBasePrice : baseMonthlyPrice;

  return (
    <div
      className={`p-6 bg-slate-900/40 backdrop-blur-xl border rounded-3xl flex flex-col justify-between h-[460px] transition-all duration-300 relative group overflow-hidden ${
        isCurrent
          ? "border-cyan-400 bg-cyan-950/20 shadow-[0_0_30px_rgba(6,182,212,0.15)] ring-1 ring-cyan-500/30"
          : "border-white/10 hover:border-slate-700 hover:bg-slate-900/60"
      }`}
    >
      {/* Glow background */}
      <div className="absolute -top-24 -right-24 w-48 h-48 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none group-hover:bg-cyan-500/20 transition-all duration-500" />

      <div>
        <div className="flex justify-between items-start mb-3">
          <div>
            <h3 className="text-lg font-black text-white uppercase tracking-tight">
              {plan.name}
            </h3>
            <span className="text-[10px] text-cyan-300 font-bold uppercase tracking-widest">
              {plan.tier} Tier
            </span>
          </div>
          {isCurrent && (
            <span className="px-2.5 py-1 rounded-full bg-cyan-500/20 border border-cyan-500/40 text-[10px] font-black uppercase text-cyan-300 flex items-center gap-1">
              <Sparkles size={11} /> Current Plan
            </span>
          )}
        </div>

        <div className="mb-4">
          <div className="flex items-baseline gap-1">
            <span className="text-3xl font-black text-white font-mono">
              ${displayedPrice.toFixed(2)}
            </span>
            <span className="text-slate-400 text-xs font-semibold">
              /{interval === "yearly" ? "yr" : "mo"}
            </span>
          </div>
          {interval === "yearly" && baseMonthlyPrice > 0 && (
            <div className="text-[10px] text-emerald-400 font-bold mt-0.5">
              Saved ${(baseMonthlyPrice * 12 - displayedPrice).toFixed(2)} with annual billing
            </div>
          )}
        </div>

        {/* Feature List */}
        <ul className="space-y-2 text-xs text-slate-300">
          {(plan.features || []).map((feat: string, idx: number) => (
            <li key={idx} className="flex items-start gap-2">
              <Check size={14} className="text-emerald-400 shrink-0 mt-0.5" />
              <span>{feat}</span>
            </li>
          ))}
          {plan.included_gpu_hours !== undefined && (
            <li className="flex items-center gap-2">
              <Check size={14} className="text-emerald-400 shrink-0" />
              <span>
                Included compute: <strong className="text-white">{plan.included_gpu_hours} hrs/mo</strong>
              </span>
            </li>
          )}
          {plan.gpu_markup_pct !== undefined && (
            <li className="flex items-center gap-2">
              <Check size={14} className="text-emerald-400 shrink-0" />
              <span>
                Platform fee markup: <strong className="text-white">{plan.gpu_markup_pct}%</strong>
              </span>
            </li>
          )}
        </ul>
      </div>

      <div className="pt-4 border-t border-white/5 flex flex-col gap-2 shrink-0">
        {isCurrent ? (
          plan.tier !== "free" ? (
            <Button
              onClick={onCancel}
              variant="outline"
              className="w-full border-rose-500/30 text-rose-400 hover:bg-rose-500/10 flex items-center justify-center gap-1.5 text-xs font-bold"
            >
              <XCircle size={14} /> Cancel Subscription
            </Button>
          ) : (
            <div className="text-center text-slate-500 text-xs py-2">
              Active Community Tier
            </div>
          )
        ) : (
          <Button
            onClick={onSubscribe}
            className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-black font-bold flex items-center justify-center gap-1.5 shadow-lg shadow-cyan-500/20 text-xs py-2.5 rounded-xl"
          >
            <Zap size={14} /> Select {plan.name}
          </Button>
        )}
      </div>
    </div>
  );
}
export default PlanCard;
