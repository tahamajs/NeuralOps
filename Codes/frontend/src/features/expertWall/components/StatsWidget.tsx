import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Users, Briefcase, Award, TrendingUp } from "lucide-react";

interface Stats {
  total_experts: number;
  active_projects: number;
  avg_expertise: number;
  success_rate: number;
}

export function StatsWidget({ stats }: { stats?: Stats }) {
  if (!stats)
    return <div className="h-20 animate-pulse bg-white/5 rounded-xl" />;
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <TerminalCard className="p-3 text-center">
        <Users size={20} className="mx-auto text-neon-blue mb-1" />
        <div className="text-2xl font-bold text-white">
          {stats.total_experts.toLocaleString()}
        </div>
        <div className="text-xs text-slate-400">Experts</div>
      </TerminalCard>
      <TerminalCard className="p-3 text-center">
        <Briefcase size={20} className="mx-auto text-green-400 mb-1" />
        <div className="text-2xl font-bold text-white">
          {stats.active_projects}
        </div>
        <div className="text-xs text-slate-400">Active Projects</div>
      </TerminalCard>
      <TerminalCard className="p-3 text-center">
        <Award size={20} className="mx-auto text-yellow-400 mb-1" />
        <div className="text-2xl font-bold text-white">
          {stats.avg_expertise}
        </div>
        <div className="text-xs text-slate-400">Avg Expertise (years)</div>
      </TerminalCard>
      <TerminalCard className="p-3 text-center">
        <TrendingUp size={20} className="mx-auto text-purple-400 mb-1" />
        <div className="text-2xl font-bold text-white">
          {stats.success_rate}%
        </div>
        <div className="text-xs text-slate-400">Match Success</div>
      </TerminalCard>
    </div>
  );
}
