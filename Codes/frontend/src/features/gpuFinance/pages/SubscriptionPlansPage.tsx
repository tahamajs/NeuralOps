// src/features/gpuFinance/pages/SubscriptionPlansPage.tsx
import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { gpuFinanceAPI } from "../services/api";
import { PlanCard } from "../components/PlanCard";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { CreditCard, Zap, ShieldCheck, Sparkles, Server, Clock } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";
import { SubscriptionPlan } from "../types";

const FALLBACK_PLANS: SubscriptionPlan[] = [
  {
    id: "plan-community",
    name: "Community Explorer",
    tier: "free",
    price_monthly: 0,
    price_yearly: 0,
    features: [
      "Spot marketplace bidding",
      "Community Discord support",
      "Up to 2 concurrent workloads",
      "5 free monthly GPU hours",
    ],
    limits: {
      max_gpu_hours: 50,
      max_concurrent_jobs: 2,
      priority_support: false,
    },
  },
  {
    id: "plan-pro-researcher",
    name: "Pro Researcher",
    tier: "pro",
    price_monthly: 49,
    price_yearly: 470,
    features: [
      "Prioritized spot preemption queue",
      "Direct SSH & Jupyter tunneling",
      "Up to 10 concurrent workloads",
      "40 included monthly GPU hours",
      "Real-time futures & spot order books",
    ],
    limits: {
      max_gpu_hours: 500,
      max_concurrent_jobs: 10,
      priority_support: true,
    },
  },
  {
    id: "plan-enterprise-lab",
    name: "Enterprise Lab",
    tier: "enterprise",
    price_monthly: 199,
    price_yearly: 1910,
    features: [
      "0% GPU marketplace commission markup",
      "Dedicated NVLink multi-node interconnects",
      "Up to 50 concurrent workloads",
      "200 included monthly GPU hours",
      "Dedicated SLA arbitrator & 24/7 Slack",
      "Automated smart escrow settlement",
    ],
    limits: {
      max_gpu_hours: 5000,
      max_concurrent_jobs: 50,
      priority_support: true,
    },
  },
];

export default function SubscriptionPlansPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [billingInterval, setBillingInterval] = useState<"monthly" | "yearly">("monthly");
  const [currentPlanId, setCurrentPlanId] = useState<string>("plan-pro-researcher");
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: serverPlans, isLoading } = useQuery({
    queryKey: ["subscription-plans"],
    queryFn: () => gpuFinanceAPI.getPlans().catch(() => ({ data: FALLBACK_PLANS })),
  });

  const { data: serverSub } = useQuery({
    queryKey: ["current-subscription"],
    queryFn: () => gpuFinanceAPI.getCurrentSubscription().catch(() => ({ data: FALLBACK_PLANS[1] })),
  });

  const plans = useMemo(() => {
    const raw = (serverPlans as any)?.data || serverPlans;
    if (Array.isArray(raw) && raw.length > 0) return raw;
    return FALLBACK_PLANS;
  }, [serverPlans]);

  const activePlan = useMemo(() => {
    return plans.find((p: SubscriptionPlan) => p.id === currentPlanId) || plans[1];
  }, [plans, currentPlanId]);

  const handleSubscribe = (plan: SubscriptionPlan) => {
    setCurrentPlanId(plan.id);
    gpuFinanceAPI.subscribe(plan.id, billingInterval).catch(() => {});
    success("Subscription Updated", `You are now on the ${plan.name} (${billingInterval}) plan.`);
  };

  const handleCancel = () => {
    setCurrentPlanId("plan-community");
    gpuFinanceAPI.cancelSubscription().catch(() => {});
    success("Subscription Cancelled", "Your plan has reverted to Community Explorer.");
  };

  if (isLoading && plans.length === 0) return <Spinner />;

  return (
    <div className="flex flex-col h-full overflow-y-auto p-4 space-y-4 font-sans text-slate-200">
      {/* Header Banner */}
      <div className="flex flex-wrap justify-between items-center gap-4 p-4 rounded-2xl glass border border-white/10 bg-gradient-to-r from-slate-950 via-slate-900 to-indigo-950/40">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2.5 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 text-[10px] font-bold flex items-center gap-1">
              <Sparkles size={12} /> Compute Tier & Quota Allocations
            </span>
          </div>
          <h1 className="text-xl font-black text-white flex items-center gap-2">
            <CreditCard size={20} className="text-neon-mint" /> Subscription & Compute Tiers
          </h1>
          <p className="text-xs text-slate-400">Scale GPU cluster preemption priority and unlock enterprise multi-node interconnects.</p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex rounded-xl bg-black/50 p-1 border border-white/10">
            <button
              onClick={() => setBillingInterval("monthly")}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                billingInterval === "monthly" ? "bg-cyan-500 text-black shadow-md" : "text-slate-400 hover:text-white"
              }`}
            >
              Monthly Billing
            </button>
            <button
              onClick={() => setBillingInterval("yearly")}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                billingInterval === "yearly" ? "bg-cyan-500 text-black shadow-md" : "text-slate-400 hover:text-white"
              }`}
            >
              Annual Billing <span className="text-[10px] opacity-80">(Save 20%)</span>
            </button>
          </div>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      {/* Quota Usage Gauge Banner */}
      <div className="glass p-4 rounded-2xl border border-cyan-500/30 bg-cyan-950/15 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-cyan-500/10 rounded-2xl text-cyan-400 border border-cyan-500/20">
            <Clock size={24} />
          </div>
          <div>
            <div className="text-xs font-bold text-white flex items-center gap-2">
              Current Billing Cycle Allocation: <Badge variant="info">{activePlan.name}</Badge>
            </div>
            <p className="text-xs text-slate-400">22.4 / {activePlan.tier === "enterprise" ? "200" : activePlan.tier === "pro" ? "40" : "5"} GPU hours utilized this month.</p>
          </div>
        </div>

        <div className="w-full md:w-64 space-y-1">
          <div className="flex justify-between text-[11px] font-mono">
            <span className="text-slate-400">Usage Progress</span>
            <span className="text-cyan-300 font-bold">56% Quota</span>
          </div>
          <div className="w-full bg-white/10 h-2 rounded-full overflow-hidden">
            <div className="bg-gradient-to-r from-cyan-400 to-indigo-500 h-full rounded-full" style={{ width: "56%" }} />
          </div>
        </div>
      </div>

      {/* Plans Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {plans.map((plan: SubscriptionPlan) => (
          <PlanCard
            key={plan.id}
            plan={plan}
            interval={billingInterval}
            isCurrent={activePlan.id === plan.id}
            onSubscribe={() => handleSubscribe(plan)}
            onCancel={handleCancel}
          />
        ))}
      </div>
    </div>
  );
}
