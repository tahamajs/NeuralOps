import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { gamificationAPI } from "../services/api";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Avatar } from "../../../components/ui/Avatar";
import { Trophy, TrendingUp, TrendingDown, Minus } from "lucide-react";
import { useAuthStore } from "../../auth/store/authStore";

export default function ContributionLeaderboard({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [period, setPeriod] = useState<"weekly" | "monthly" | "all">("all");
  const { user } = useAuthStore();

  const { data: leaderboard, isLoading } = useQuery({
    queryKey: ["leaderboard", period],
    queryFn: () => gamificationAPI.getLeaderboard({ period }),
  });

  const { data: myStats } = useQuery({
    queryKey: ["my-gamification-stats"],
    queryFn: () => gamificationAPI.getMyStats(),
  });

  if (isLoading) return <Spinner />;

  const getRankIcon = (rank: number) => {
    if (rank === 1) return <Trophy size={20} className="text-yellow-400" />;
    if (rank === 2) return <Trophy size={20} className="text-slate-300" />;
    if (rank === 3) return <Trophy size={20} className="text-amber-600" />;
    return <span className="w-6 text-center font-mono">{rank}</span>;
  };

  const getTierBadge = (tier: string) => {
    const colors = {
      bronze: "bg-amber-600/20 text-amber-400",
      silver: "bg-slate-400/20 text-slate-300",
      gold: "bg-yellow-600/20 text-yellow-400",
      platinum: "bg-cyan-600/20 text-cyan-400",
      diamond: "bg-blue-600/20 text-blue-300",
    };
    return (
      <span
        className={`px-2 py-0.5 rounded text-xs ${colors[tier as keyof typeof colors]}`}
      >
        {tier}
      </span>
    );
  };

  const getChangeIcon = (change: number) => {
    if (change > 0)
      return (
        <span className="flex items-center gap-0.5 text-green-400">
          <TrendingUp size={12} /> +{change}
        </span>
      );
    if (change < 0)
      return (
        <span className="flex items-center gap-0.5 text-red-400">
          <TrendingDown size={12} /> {change}
        </span>
      );
    return <Minus size={12} className="text-slate-500" />;
  };

  const userEntry = leaderboard?.find((entry: any) => entry.user_id === (user as any)?.id);

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Trophy size={20} className="text-yellow-400" /> Contribution
            Leaderboard
          </h1>
          <p className="text-xs text-slate-400">
            Top contributors ranked by research impact
          </p>
        </div>
        <div className="flex gap-2">
          <Select
            options={[
              { value: "weekly", label: "Weekly" },
              { value: "monthly", label: "Monthly" },
              { value: "all", label: "All Time" },
            ]}
            value={period}
            onChange={(e) => setPeriod(e.target.value as any)}
            className="w-32"
          />
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-6">
        {/* User's rank card */}
        {myStats && (
          <TerminalCard className="p-4">
            <div className="flex justify-between items-center">
              <div>
                <div className="text-xs text-slate-400">Your Stats</div>
                <div className="text-2xl font-bold text-white">
                  {myStats.total_points} pts
                </div>
                <div className="text-sm text-neon-mint">
                  Level {myStats.level} {getTierBadge(myStats.tier)}
                </div>
              </div>
              <div className="text-right">
                <div className="text-xs text-slate-400">
                  Points to next level
                </div>
                <div className="text-xl font-bold text-yellow-400">
                  {myStats.points_to_next_level}
                </div>
                <div className="w-32 h-1 bg-white/10 rounded-full mt-1">
                  <div
                    className="bg-yellow-400 h-1 rounded-full"
                    style={{
                      width: `${(myStats.total_points / (myStats.total_points + myStats.points_to_next_level)) * 100}%`,
                    }}
                  />
                </div>
              </div>
            </div>
          </TerminalCard>
        )}

        {/* Leaderboard table */}
        <TerminalCard noPadding>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-slate-400 border-b border-white/5">
                  <th className="px-4 py-3">Rank</th>
                  <th>Contributor</th>
                  <th className="text-center">Points</th>
                  <th className="text-center">Level</th>
                  <th className="text-center">Tier</th>
                  <th className="text-center">Change</th>
                  <th className="text-right"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {(leaderboard || []).map((entry) => (
                  <tr
                    key={entry.user_id}
                    className={
                      entry.user_id === (user as any)?.id ? "bg-neon-blue/5 border-neon-blue/30" : "hover:bg-white/5"
                    }
                  >
                    <td className="px-4 py-3 font-mono">
                      <div className="flex items-center gap-1">
                        {getRankIcon(entry.rank)}
                        {entry.rank <= 3 && (
                          <span className="text-xs ml-0.5">
                            {entry.rank === 1
                              ? "🥇"
                              : entry.rank === 2
                                ? "🥈"
                                : "🥉"}
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-3">
                        <Avatar src={entry.avatar_url} size="sm" />
                        <div>
                          <div className="font-medium text-white">
                            {entry.username}
                          </div>
                          <div className="text-[10px] text-slate-400">
                            ID: {entry.user_id.slice(0, 8)}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-center font-bold text-neon-mint">
                      {entry.total_points}
                    </td>
                    <td className="px-4 py-3 text-center">{entry.level}</td>
                    <td className="px-4 py-3 text-center">
                      {getTierBadge(entry.tier)}
                    </td>
                    <td className="px-4 py-3 text-center">
                      {getChangeIcon(entry.change)}
                    </td>
                    <td className="px-4 py-3 text-right">
                      {entry.user_id === (user as any)?.id && (
                        <span className="text-xs text-neon-mint">You</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </TerminalCard>
      </div>
    </div>
  );
}
