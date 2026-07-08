import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { recommenderAPI } from "../services/api";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { TrendingUp, RefreshCw } from "lucide-react";

export function OpportunityIndexWidget() {
  const queryClient = useQueryClient();
  const { data, isLoading, refetch } = useQuery({
    queryKey: ["opportunity-index"],
    queryFn: () => recommenderAPI.getOpportunityIndex(),
  });

  const recalc = useMutation({
    mutationFn: () => recommenderAPI.recalculateOpportunity(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["opportunity-index"] });
    },
  });

  if (isLoading) return <Spinner size="sm" />;
  if (!data) return null;

  return (
    <TerminalCard className="p-4">
      <div className="flex justify-between items-start">
        <div>
          <div className="flex items-center gap-2">
            <TrendingUp size={18} className="text-neon-mint" />
            <span className="text-sm font-bold text-white">
              Opportunity Index
            </span>
          </div>
          <div className="text-2xl font-bold text-neon-mint mt-1">
            {data.total_score.toFixed(0)} / 100
          </div>
          <div className="text-xs text-slate-400">Rank #{data.rank}</div>
        </div>
        <Button size="sm" variant="ghost" onClick={() => recalc.mutate()}>
          <RefreshCw size={14} /> Recalculate
        </Button>
      </div>
      <div className="mt-3">
        <div className="text-xs text-slate-400">Missing opportunities</div>
        <div className="flex flex-wrap gap-1 mt-1">
          {data.missing_attributes.slice(0, 5).map((attr) => (
            <span
              key={attr}
              className="px-2 py-0.5 rounded bg-yellow-500/20 text-yellow-400 text-xs"
            >
              {attr}
            </span>
          ))}
        </div>
      </div>
      <div className="mt-2">
        <div className="text-xs text-slate-400">High‑impact suggestions</div>
        <ul className="text-xs text-slate-300 mt-1 list-disc list-inside">
          {data.high_impact_suggestions.slice(0, 3).map((s) => (
            <li key={s}>{s}</li>
          ))}
        </ul>
      </div>
    </TerminalCard>
  );
}
