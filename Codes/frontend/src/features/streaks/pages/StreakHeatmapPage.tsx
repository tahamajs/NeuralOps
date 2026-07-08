import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { streaksAPI } from "../services/api";
import { HeatmapCalendar } from "../components/HeatmapCalendar";
import { StreakStats } from "../components/StreakStats";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Flame, Calendar, Trophy, Users } from "lucide-react";
import { useAuthStore } from "../../auth/store/authStore";

export default function StreakHeatmapPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const { user } = useAuthStore();
  const [selectedYear, setSelectedYear] = useState(new Date().getFullYear());
  const [viewMode, setViewMode] = useState<"personal" | "leaderboard">(
    "personal",
  );

  const { data: heatmap, isLoading: heatmapLoading } = useQuery({
    queryKey: [
      "heatmap",
      selectedYear,
      viewMode === "personal" ? user?.username : undefined,
    ],
    queryFn: () =>
      streaksAPI.getHeatmap(
        viewMode === "personal" ? user?.username : undefined,
        selectedYear,
      ),
    enabled: viewMode === "personal" || !!user,
  });

  const { data: leaderboard, isLoading: leaderboardLoading } = useQuery({
    queryKey: ["streak-leaderboard"],
    queryFn: () => streaksAPI.getLeaderboard(),
    enabled: viewMode === "leaderboard",
  });

  const getYearOptions = () => {
    const currentYear = new Date().getFullYear();
    const options = [];
    for (let y = currentYear - 3; y <= currentYear + 1; y++) {
      options.push({ value: y.toString(), label: y.toString() });
    }
    return options;
  };

  if (heatmapLoading || (viewMode === "leaderboard" && leaderboardLoading))
    return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div className="flex items-center gap-2">
          <Flame size={20} className="text-orange-400" />
          <h1 className="text-xl font-bold text-white">Activity Streaks</h1>
        </div>
        <div className="flex gap-2">
          <div className="flex gap-1 mr-4">
            <button
              onClick={() => setViewMode("personal")}
              className={`px-3 py-1 rounded text-sm ${viewMode === "personal" ? "bg-neon-blue text-white" : "text-slate-400 hover:text-white"}`}
            >
              My Activity
            </button>
            <button
              onClick={() => setViewMode("leaderboard")}
              className={`px-3 py-1 rounded text-sm ${viewMode === "leaderboard" ? "bg-neon-blue text-white" : "text-slate-400 hover:text-white"}`}
            >
              Leaderboard
            </button>
          </div>
          <Select
            options={getYearOptions()}
            value={selectedYear.toString()}
            onChange={(e) => setSelectedYear(parseInt(e.target.value))}
            className="w-28"
          />
          <Button
            size="sm"
            variant="ghost"
            onClick={() => window.location.reload()}
          >
            Refresh
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-6">
        {viewMode === "personal" ? (
          <>
            <div className="grid md:grid-cols-3 gap-4">
              <StreakStats stats={heatmap?.streak_stats} />
              <TerminalCard className="p-4 text-center">
                <Trophy size={24} className="mx-auto text-yellow-400 mb-2" />
                <div className="text-2xl font-bold text-white">
                  {heatmap?.streak_stats?.rank || "—"}
                </div>
                <div className="text-xs text-slate-400">Global Rank</div>
              </TerminalCard>
              <TerminalCard className="p-4 text-center">
                <Users size={24} className="mx-auto text-neon-blue mb-2" />
                <div className="text-2xl font-bold text-white">
                  {heatmap?.total?.toLocaleString() || 0}
                </div>
                <div className="text-xs text-slate-400">
                  Total Contributions
                </div>
              </TerminalCard>
            </div>
            <HeatmapCalendar data={heatmap!} year={selectedYear} />
          </>
        ) : (
          <TerminalCard className="p-4">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-slate-400 border-b border-white/5">
                    <th className="px-4 py-3">Rank</th>
                    <th>User</th>
                    <th className="text-center">Current Streak</th>
                    <th className="text-center">Longest</th>
                    <th className="text-center">Active Days</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {(leaderboard || []).map((entry, idx) => (
                    <tr key={entry.user_id} className="hover:bg-white/5">
                      <td className="px-4 py-3 font-mono font-bold">
                        #{idx + 1}
                      </td>
                      <td className="px-4 py-3 font-medium text-white">
                        {entry.username}
                      </td>
                      <td className="px-4 py-3 text-center">
                        <span className="flex items-center justify-center gap-1">
                          <Flame size={12} className="text-orange-400" />{" "}
                          {entry.current_streak}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-center">
                        {entry.longest_streak}
                      </td>
                      <td className="px-4 py-3 text-center">
                        {entry.total_active_days}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </TerminalCard>
        )}
      </div>
    </div>
  );
}
