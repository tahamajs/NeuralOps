// src/features/llmHub/pages/CostDashboard.tsx
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { llmHubAPI } from "../services/api";
import { CostChart } from "../components/CostChart";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { DollarSign, TrendingUp, AlertTriangle } from "lucide-react";

export default function CostDashboard({ onClose }: { onClose?: () => void }) {
  const [timeRange, setTimeRange] = useState<"week" | "month" | "year">(
    "month",
  );
  const { data: costs, isLoading } = useQuery({
    queryKey: ["llm-costs", timeRange],
    queryFn: () => llmHubAPI.getCostEntries(),
  });
  const { data: summary } = useQuery({
    queryKey: ["llm-cost-summary"],
    queryFn: () => llmHubAPI.getCostSummary(),
  });

  if (isLoading) return <Spinner />;
  const totalCost = costs?.reduce((sum, c) => sum + c.cost_usd, 0) || 0;
  const providerTotals = costs?.reduce(
    (acc, c) => {
      acc[c.provider] = (acc[c.provider] || 0) + c.cost_usd;
      return acc;
    },
    {} as Record<string, number>,
  );

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <DollarSign size={20} className="text-yellow-400" /> Cost Dashboard
          </h1>
          <p className="text-xs text-slate-400">
            Monitor LLM spending across providers
          </p>
        </div>
        <div className="flex gap-2">
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <div className="grid md:grid-cols-3 gap-4">
          <TerminalCard className="p-3 text-center">
            <div className="text-2xl font-bold text-white">
              ${totalCost.toFixed(2)}
            </div>
            <div className="text-xs text-slate-400">Total Spend</div>
          </TerminalCard>
          <TerminalCard className="p-3 text-center">
            <div className="text-2xl font-bold text-white">
              {costs?.length || 0}
            </div>
            <div className="text-xs text-slate-400">Requests</div>
          </TerminalCard>
          <TerminalCard className="p-3 text-center">
            <div className="text-2xl font-bold text-yellow-400">
              ${(summary?.daily_average || 0).toFixed(2)}
            </div>
            <div className="text-xs text-slate-400">Daily Average</div>
          </TerminalCard>
        </div>
        <CostChart data={costs || []} />
        <TerminalCard className="p-4">
          <h3 className="text-sm font-bold text-white mb-2">By Provider</h3>
          <div className="space-y-2">
            {Object.entries(providerTotals).map(([provider, amount]) => (
              <div key={provider} className="flex justify-between items-center">
                <span className="text-white">{provider}</span>
                <span className="text-neon-mint">${amount.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </TerminalCard>
        <TerminalCard className="p-4">
          <h3 className="text-sm font-bold text-white mb-2 flex items-center gap-2">
            <AlertTriangle size={14} className="text-yellow-400" /> Budget
            Alerts
          </h3>
          <div className="text-slate-400 text-sm">
            Monthly budget: $500 · Remaining: ${(500 - totalCost).toFixed(2)}
          </div>
          <div className="w-full bg-white/10 rounded-full h-1.5 mt-2">
            <div
              className="bg-yellow-400 h-1.5 rounded-full"
              style={{ width: `${Math.min(100, (totalCost / 500) * 100)}%` }}
            />
          </div>
        </TerminalCard>
      </div>
    </div>
  );
}
