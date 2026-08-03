import React from "react";
import { ArenaBattle } from "../types";

import { Trophy, Award } from "lucide-react";

interface ArenaLeaderboardProps {
  battles?: ArenaBattle[];
}

export function ArenaLeaderboard({ battles = [] }: ArenaLeaderboardProps) {
  // Aggregate stats
  const statsMap: Record<
    string,
    { wins: number; total: number; losses: number; ties: number }
  > = {};

  battles.forEach((b) => {
    if (!statsMap[b.model_a])
      statsMap[b.model_a] = { wins: 0, total: 0, losses: 0, ties: 0 };
    if (!statsMap[b.model_b])
      statsMap[b.model_b] = { wins: 0, total: 0, losses: 0, ties: 0 };

    if (b.winner) {
      statsMap[b.model_a].total++;
      statsMap[b.model_b].total++;

      if (b.winner === "A") {
        statsMap[b.model_a].wins++;
        statsMap[b.model_b].losses++;
      } else if (b.winner === "B") {
        statsMap[b.model_b].wins++;
        statsMap[b.model_a].losses++;
      } else {
        statsMap[b.model_a].ties++;
        statsMap[b.model_b].ties++;
      }
    }
  });

  const leaderData = Object.entries(statsMap)
    .map(([model, stats]) => ({
      model,
      wins: stats.wins,
      losses: stats.losses,
      ties: stats.ties,
      total: stats.total,
      winRate: stats.total > 0 ? (stats.wins / stats.total) * 100 : 0,
    }))
    .sort((a, b) => b.winRate - a.winRate || b.wins - a.wins);

  const getRankBadge = (idx: number) => {
    if (idx === 0) return <Trophy size={14} className="text-yellow-400" />;
    if (idx === 1) return <Award size={14} className="text-slate-400" />;
    if (idx === 2) return <Award size={14} className="text-amber-600" />;
    return (
      <span className="text-[10px] text-slate-500 font-mono">#{idx + 1}</span>
    );
  };

  return (
    <div className="bg-slate-900/30 border border-white/5 rounded-2xl p-4 flex flex-col gap-2 backdrop-blur-md font-sans text-xs">
      <div className="flex items-center gap-1.5 text-xs font-bold text-slate-400 uppercase tracking-widest mb-2 border-b border-white/5 pb-2">
        <Trophy size={14} className="text-yellow-400" />
        Model Arena Leaderboard
      </div>

      {leaderData.length === 0 ? (
        <div className="text-center py-6 text-slate-600 italic">
          No leaderboard data available. Submit some votes!
        </div>
      ) : (
        <div className="overflow-hidden border border-white/5 rounded-xl">
          <table className="w-full text-left border-collapse text-[11px]">
            <thead>
              <tr className="border-b border-white/10 bg-slate-900/60 text-slate-400 uppercase tracking-wider font-bold">
                <th className="p-2.5 w-12 text-center">Rank</th>
                <th className="p-2.5">Model</th>
                <th className="p-2.5 text-center">Wins</th>
                <th className="p-2.5 text-center">Losses</th>
                <th className="p-2.5 text-center">Ties</th>
                <th className="p-2.5 text-center">Win Rate</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 bg-slate-950/20 text-slate-300">
              {leaderData.map((row, idx) => (
                <tr key={row.model} className="hover:bg-white/5">
                  <td className="p-2.5 text-center">{getRankBadge(idx)}</td>
                  <td className="p-2.5 font-bold text-white">{row.model}</td>
                  <td className="p-2.5 text-center font-mono">{row.wins}</td>
                  <td className="p-2.5 text-center font-mono text-slate-500">
                    {row.losses}
                  </td>
                  <td className="p-2.5 text-center font-mono text-slate-500">
                    {row.ties}
                  </td>
                  <td className="p-2.5 text-center font-mono font-extrabold text-neon-mint">
                    {row.winRate.toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
