// src/features/gpuFinance/pages/SubscriptionPlansPage.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { gpuFinanceAPI } from "../services/api";
import { PlanCard } from "../components/PlanCard";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { CreditCard, Zap } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function SubscriptionPlansPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [billingInterval, setBillingInterval] = useState<"monthly" | "yearly">(
    "monthly",
  );
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: plans, isLoading } = useQuery({
    queryKey: ["subscription-plans"],
    queryFn: () => gpuFinanceAPI.getPlans(),
  });

  const { data: currentPlan } = useQuery({
    queryKey: ["current-subscription"],
    queryFn: () => gpuFinanceAPI.getCurrentSubscription(),
  });

  const subscribe = useMutation({
    mutationFn: ({
      planId,
      interval,
    }: {
      planId: string;
      interval: "monthly" | "yearly";
    }) => gpuFinanceAPI.subscribe(planId, interval),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["current-subscription"] });
      success("Subscription updated");
    },
  });

  const cancel = useMutation({
    mutationFn: () => gpuFinanceAPI.cancelSubscription(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["current-subscription"] });
      success("Subscription cancelled");
    },
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <CreditCard size={20} className="text-neon-mint" /> Subscription
            Plans
          </h1>
          <p className="text-xs text-slate-400">
            Choose the plan that fits your research needs
          </p>
        </div>
        <div className="flex gap-2">
          <div className="flex rounded bg-white/5 p-0.5">
            <button
              onClick={() => setBillingInterval("monthly")}
              className={`px-3 py-1 rounded text-sm ${billingInterval === "monthly" ? "bg-neon-blue text-white" : "text-slate-400"}`}
            >
              Monthly
            </button>
            <button
              onClick={() => setBillingInterval("yearly")}
              className={`px-3 py-1 rounded text-sm ${billingInterval === "yearly" ? "bg-neon-blue text-white" : "text-slate-400"}`}
            >
              Yearly (Save 20%)
            </button>
          </div>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4">
        <div className="grid md:grid-cols-3 gap-4">
          {plans?.map((plan) => (
            <PlanCard
              key={plan.id}
              plan={plan}
              interval={billingInterval}
              isCurrent={currentPlan?.id === plan.id}
              onSubscribe={() =>
                subscribe.mutate({ planId: plan.id, interval: billingInterval })
              }
              onCancel={cancel.mutate}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
