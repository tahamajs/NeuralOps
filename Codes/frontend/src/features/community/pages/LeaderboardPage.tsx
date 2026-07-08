import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { communityAPI } from "../services/api";
import { Card, Spinner, Avatar, Select } from "../../../components/ui";
import { Trophy, Medal } from "lucide-react";

export default function LeaderboardPage() {
  const [timeframe, setTimeframe] = useState("all");
  const { data, isLoading } = useQuery({
    queryKey: ["leaderboard", timeframe],
    queryFn: () => communityAPI.getLeaderboard(timeframe),
  });

  if (isLoading) return <Spinner />;

  const getRankIcon = (rank: number) => {
    if (rank === 1) return <Trophy size={20} className="text-yellow-400" />;
    if (rank === 2) return <Medal size={20} className="text-slate-400" />;
    if (rank === 3) return <Medal size={20} className="text-amber-600" />;
    return <span className="w-5 text-center">{rank}</span>;
  };

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-white">Leaderboard</h1>
        <Select
          options={[
            { value: "all", label: "All Time" },
            { value: "month", label: "This Month" },
            { value: "week", label: "This Week" },
          ]}
          value={timeframe}
          onChange={(e) => setTimeframe(e.target.value)}
          className="w-32"
        />
      </div>

      <Card>
        <div className="space-y-2">
          {data?.map((user: any, idx: number) => (
            <div
              key={user.id}
              className="flex items-center gap-4 py-3 border-b border-white/5"
            >
              <div className="w-8 text-center">{getRankIcon(idx + 1)}</div>
              <Avatar src={user.avatar_url} size="sm" />
              <div className="flex-1">
                <div className="font-medium text-white">{user.username}</div>
                <div className="text-xs text-slate-400">
                  {user.reputation} reputation
                </div>
              </div>
              <div className="text-right text-sm text-neon-blue">
                {user.contributions} contributions
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
