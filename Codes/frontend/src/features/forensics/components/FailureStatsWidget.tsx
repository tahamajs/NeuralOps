import { useQuery } from "@tanstack/react-query";
import { forensicsAPI } from "../services/api";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Spinner } from "../../../components/ui/Spinner";
import { AlertTriangle, CheckCircle, Clock } from "lucide-react";

export function FailureStatsWidget() {
  const { data, isLoading } = useQuery({
    queryKey: ["failure-stats"],
    queryFn: () => forensicsAPI.getFailureStats(),
    refetchInterval: 30000,
  });
  if (isLoading) return <Spinner size="sm" />;
  if (!data) return null;
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      <TerminalCard className="p-3 text-center">
        <AlertTriangle size={20} className="mx-auto text-red-400 mb-1" />
        <div className="text-2xl font-bold text-white">
          {data.total_failures}
        </div>
        <div className="text-xs text-slate-400">Total Failures</div>
      </TerminalCard>
      <TerminalCard className="p-3 text-center">
        <CheckCircle size={20} className="mx-auto text-green-400 mb-1" />
        <div className="text-2xl font-bold text-white">
          {data.resolved_failures}
        </div>
        <div className="text-xs text-slate-400">Resolved</div>
      </TerminalCard>
      <TerminalCard className="p-3 text-center">
        <AlertTriangle size={20} className="mx-auto text-yellow-400 mb-1" />
        <div className="text-2xl font-bold text-white">
          {data.unresolved_failures}
        </div>
        <div className="text-xs text-slate-400">Open</div>
      </TerminalCard>
      <TerminalCard className="p-3 text-center">
        <Clock size={20} className="mx-auto text-neon-blue mb-1" />
        <div className="text-2xl font-bold text-white">
          {data.avg_resolution_time_hours.toFixed(1)}h
        </div>
        <div className="text-xs text-slate-400">Avg. Resolution</div>
      </TerminalCard>
    </div>
  );
}
