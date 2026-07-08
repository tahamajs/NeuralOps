import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Flame, Award, Calendar, TrendingUp } from "lucide-react";
import { StreakStats as StreakStatsType } from "../types";

export function StreakStats({ stats }: { stats?: StreakStatsType }) {
  if (!stats)
    return <div className="h-32 animate-pulse bg-white/5 rounded-xl" />;
  const expires = stats.streak_expires_at
    ? new Date(stats.streak_expires_at)
    : null;
  const hoursLeft = expires
    ? Math.max(
        0,
        Math.floor((expires.getTime() - Date.now()) / (1000 * 60 * 60)),
      )
    : 0;

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <TerminalCard className="p-4 text-center">
        <div className="flex justify-center gap-1 text-orange-400">
          <Flame size={24} />
          <Flame size={24} />
          <Flame size={24} />
        </div>
        <div className="text-3xl font-bold text-white mt-2">
          {stats.current_streak}
        </div>
        <div className="text-xs text-slate-400">Current Streak</div>
        {hoursLeft > 0 && (
          <div className="text-xs text-yellow-400 mt-1">
            {hoursLeft}h left to maintain
          </div>
        )}
      </TerminalCard>
      <TerminalCard className="p-4 text-center">
        <Award size={24} className="mx-auto text-yellow-400 mb-2" />
        <div className="text-3xl font-bold text-white">
          {stats.longest_streak}
        </div>
        <div className="text-xs text-slate-400">Longest Streak</div>
      </TerminalCard>
      <TerminalCard className="p-4 text-center">
        <Calendar size={24} className="mx-auto text-neon-blue mb-2" />
        <div className="text-3xl font-bold text-white">
          {stats.total_active_days.toLocaleString()}
        </div>
        <div className="text-xs text-slate-400">Active Days</div>
      </TerminalCard>
    </div>
  );
}
