import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useAuthStore } from "../../../features/auth/store/authStore";
import { communityAPI } from "../../community/services/api";
import {
  Card,
  Button,
  Select,
  Spinner,
  Avatar,
  PaginationControls,
} from "../../../components/ui";
import {
  Trophy,
  Medal,
  TrendingUp,
  Award,
  Calendar,
  Flame,
  Users,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";

// ---------- Types ----------
type TimeRange = "all" | "month" | "week" | "day";

interface LeaderboardEntry {
  id: string;
  username: string;
  avatar_url: string | null;
  reputation: number;
  contribution_count: number;
  rank: number;
  rank_change: number;
  joined_at: string;
  total_upvotes_received: number;
  answers_given: number;
  questions_asked: number;
}

// ---------- Helper Components ----------
const getRankIcon = (rank: number) => {
  if (rank === 1)
    return <Trophy size={28} className="text-yellow-400 drop-shadow-glow" />;
  if (rank === 2)
    return <Medal size={28} className="text-slate-300 drop-shadow-glow" />;
  if (rank === 3)
    return <Medal size={28} className="text-amber-600 drop-shadow-glow" />;
  return (
    <span className="w-8 text-center text-lg font-bold text-slate-400">
      {rank}
    </span>
  );
};

const getRankChange = (change: number) => {
  if (change > 0)
    return <span className="text-green-400 text-xs">↑ {change}</span>;
  if (change < 0)
    return <span className="text-red-400 text-xs">↓ {Math.abs(change)}</span>;
  return <span className="text-slate-500 text-xs">–</span>;
};

// ---------- Main Component ----------
export default function LeaderboardPage() {
  const [timeRange, setTimeRange] = useState<TimeRange>("all");
  const [currentPage, setCurrentPage] = useState(1);
  const { user } = useAuthStore();
  const currentUser = user as any;
  const pageSize = 20;

  const { data: leaderboardData, isLoading } = useQuery({
    queryKey: ["leaderboard", { timeRange, page: currentPage }],
    queryFn: () =>
      (communityAPI as any).getLeaderboard({
        timeframe: timeRange,
        page: currentPage,
        page_size: pageSize,
      }),
  });

  const { data: currentUserRank } = useQuery({
    queryKey: ["leaderboard", "currentUserRank", timeRange],
    queryFn: () => (communityAPI as any).getCurrentUserRank(timeRange),
    enabled: !!currentUser,
  });

  const { data: activityData } = useQuery({
    queryKey: ["leaderboard", "activity", currentUser?.id],
    queryFn: () =>
      (communityAPI as any).getContributionHeatmap(currentUser?.id),
    enabled: !!currentUser,
  });

  if (isLoading) return <Spinner />;

  const entries = leaderboardData?.results || [];
  const totalPages = Math.ceil(leaderboardData?.count / pageSize) || 1;

  // Generate heatmap squares
  const weeks = activityData?.weeks || [];
  const getColor = (level: number) => {
    if (level === 0) return "bg-white/5";
    if (level === 1) return "bg-green-900";
    if (level === 2) return "bg-green-700";
    if (level === 3) return "bg-green-500";
    return "bg-green-300";
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header & Filters */}
      <div className="flex flex-wrap justify-between items-center gap-4">
        <div className="flex items-center gap-3">
          <Trophy size={28} className="text-yellow-400" />
          <div>
            <h1 className="text-2xl font-bold text-white">
              Community Leaderboard
            </h1>
            <p className="text-sm text-slate-400">
              Top contributors ranked by reputation and activity
            </p>
          </div>
        </div>
        <div className="flex gap-2">
          {(["all", "month", "week", "day"] as TimeRange[]).map((range) => (
            <button
              key={range}
              onClick={() => {
                setTimeRange(range);
                setCurrentPage(1);
              }}
              className={`px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                timeRange === range
                  ? "bg-neon-blue text-white shadow-glow"
                  : "bg-white/5 text-slate-400 hover:bg-white/10 hover:text-white"
              }`}
            >
              {range === "all"
                ? "All Time"
                : range === "month"
                  ? "This Month"
                  : range === "week"
                    ? "This Week"
                    : "Today"}
            </button>
          ))}
        </div>
      </div>

      {/* Current User Card */}
      {currentUser && currentUserRank && (
        <Card className="border-neon-blue/30 bg-neon-blue/5 p-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="relative">
                <Avatar src={currentUser.avatar_url} size="md" />
                {currentUserRank.rank <= 3 && (
                  <div className="absolute -top-2 -right-2">
                    {currentUserRank.rank === 1 && (
                      <Trophy size={16} className="text-yellow-400" />
                    )}
                    {currentUserRank.rank === 2 && (
                      <Medal size={16} className="text-slate-300" />
                    )}
                    {currentUserRank.rank === 3 && (
                      <Medal size={16} className="text-amber-600" />
                    )}
                  </div>
                )}
              </div>
              <div>
                <div className="text-sm text-slate-400">Your Rank</div>
                <div className="text-2xl font-bold text-white">
                  #{currentUserRank.rank}
                </div>
                <div className="text-xs text-slate-500">
                  {currentUserRank.reputation} reputation •{" "}
                  {currentUserRank.contribution_count} contributions
                </div>
              </div>
            </div>
            <div className="flex items-center gap-6">
              <div className="text-center">
                <div className="text-xl font-bold text-white">
                  {currentUserRank.rank_change}
                </div>
                <div className="text-[10px] text-slate-500">rank change</div>
              </div>
              <div className="text-center">
                <div className="text-xl font-bold text-white">
                  {currentUserRank.reputation_gained}
                </div>
                <div className="text-[10px] text-slate-500">rep gained</div>
              </div>
              <Button variant="secondary" size="sm">
                View My Profile
              </Button>
            </div>
          </div>
          {/* Activity Heatmap */}
          {weeks.length > 0 && (
            <div className="mt-4 pt-3 border-t border-white/10">
              <div className="flex items-center gap-2 text-xs text-slate-400 mb-2">
                <Calendar size={12} /> Contribution activity (past year)
              </div>
              <div className="flex gap-1 overflow-x-auto pb-2">
                {weeks.map((week: number[], weekIdx: number) => (
                  <div key={weekIdx} className="flex flex-col gap-1">
                    {week.map((level, dayIdx) => (
                      <div
                        key={dayIdx}
                        className={`w-2.5 h-2.5 rounded-sm ${getColor(level)}`}
                        title={`${weekIdx + dayIdx} contributions`}
                      />
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}
        </Card>
      )}

      {/* Top 3 Podium Cards */}
      {entries.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          {entries.slice(0, 3).map((entry: LeaderboardEntry, idx: number) => (
            <Card
              key={entry.id}
              hover
              className={`text-center p-6 ${
                idx === 0
                  ? "border-yellow-500/30 bg-gradient-to-b from-yellow-500/10 to-transparent"
                  : idx === 1
                    ? "border-slate-400/30 bg-gradient-to-b from-slate-400/10 to-transparent"
                    : "border-amber-600/30 bg-gradient-to-b from-amber-600/10 to-transparent"
              }`}
            >
              <div className="relative inline-block">
                <Avatar src={entry.avatar_url} size="lg" className="mx-auto" />
                <div className="absolute -top-2 -right-2">
                  {getRankIcon(entry.rank)}
                </div>
              </div>
              <h3 className="mt-3 text-xl font-bold text-white">
                {entry.username}
              </h3>
              <div className="flex justify-center gap-4 mt-2 text-sm">
                <div className="text-center">
                  <div className="font-bold text-neon-blue">
                    {entry.reputation}
                  </div>
                  <div className="text-[10px] text-slate-500">reputation</div>
                </div>
                <div className="text-center">
                  <div className="font-bold text-white">
                    {entry.contribution_count}
                  </div>
                  <div className="text-[10px] text-slate-500">
                    contributions
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      {/* Leaderboard Table */}
      <Card className="overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-slate-400 border-b border-white/5">
                <th className="px-6 py-4 w-16">#</th>
                <th className="px-6 py-4">Contributor</th>
                <th className="px-6 py-4 text-center">Reputation</th>
                <th className="px-6 py-4 text-center">Contributions</th>
                <th className="px-6 py-4 text-center">↑ Votes</th>
                <th className="px-6 py-4 text-center">Answers</th>
                <th className="px-6 py-4 text-right">Change</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {entries.map((entry: LeaderboardEntry) => (
                <tr
                  key={entry.id}
                  className={`hover:bg-white/5 transition-colors ${
                    entry.username === currentUser?.username
                      ? "bg-neon-blue/5"
                      : ""
                  }`}
                >
                  <td className="px-6 py-4 font-mono font-bold">
                    <div className="flex items-center gap-1">
                      {getRankIcon(entry.rank)}
                      {entry.rank <= 3 && (
                        <span className="text-xs ml-1">
                          {entry.rank === 1
                            ? "🥇"
                            : entry.rank === 2
                              ? "🥈"
                              : "🥉"}
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                      <Avatar src={entry.avatar_url} size="sm" />
                      <div>
                        <div className="font-medium text-white">
                          {entry.username}
                        </div>
                        <div className="text-[10px] text-slate-500">
                          joined{" "}
                          {formatDistanceToNow(new Date(entry.joined_at), {
                            addSuffix: true,
                          })}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-center font-mono font-bold text-neon-blue">
                    {entry.reputation.toLocaleString()}
                  </td>
                  <td className="px-6 py-4 text-center">
                    {entry.contribution_count.toLocaleString()}
                  </td>
                  <td className="px-6 py-4 text-center">
                    {entry.total_upvotes_received.toLocaleString()}
                  </td>
                  <td className="px-6 py-4 text-center">
                    {entry.answers_given}
                  </td>
                  <td className="px-6 py-4 text-right">
                    {getRankChange(entry.rank_change)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex justify-center pt-4">
          <PaginationControls
            currentPage={currentPage}
            totalPages={totalPages}
            onPageChange={setCurrentPage}
          />
        </div>
      )}
    </div>
  );
}
