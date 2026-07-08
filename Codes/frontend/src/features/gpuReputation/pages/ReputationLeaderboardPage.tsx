import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card, Spinner, Table, Badge, Button } from "../../../components/ui";
import { Award, Shield, Clock } from "lucide-react";

export default function ReputationLeaderboardPage() {
  const { data: leaderboard, isLoading: lbLoading } = useQuery({
    queryKey: ["reputation-leaderboard"],
    queryFn: async () => (await api.get("/reputation/leaderboard/")).data,
  });
  const { data: myStakes, isLoading: stakeLoading } = useQuery({
    queryKey: ["my-stakes"],
    queryFn: async () => (await api.get("/stakes/")).data,
  });
  const { data: slashingHistory } = useQuery({
    queryKey: ["slashing-history"],
    queryFn: async () => (await api.get("/slashing/history/")).data,
  });

  if (lbLoading || stakeLoading) return <Spinner />;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">
        GPU Reputation & Staking
      </h1>

      {/* Leaderboard */}
      <Card>
        <h2 className="text-sm font-bold text-white flex items-center gap-2 mb-3">
          <Award size={16} className="text-yellow-400" /> Reputation Leaderboard
        </h2>
        <Table>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Provider</th>
              <th>Score</th>
              <th>Staked</th>
              <th>Verifications</th>
              <th>Slashing</th>
            </tr>
          </thead>
          <tbody>
            {leaderboard?.map((entry: any, idx: number) => (
              <tr key={entry.user_id}>
                <td>{idx + 1}</td>
                <td className="font-bold">{entry.username}</td>
                <td className="text-neon-blue">{entry.reputation_score}</td>
                <td>{entry.total_staked} tokens</td>
                <td>{entry.successful_verifications}</td>
                <td>
                  <Badge variant={entry.slashings > 0 ? "failed" : "success"}>
                    {entry.slashings > 0 ? "failed" : "success"}
                  </Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Card>

      {/* My Stakes */}
      <Card>
        <h2 className="text-sm font-bold text-white mb-3">My Stakes</h2>
        {myStakes?.length ? (
          <div className="space-y-2">
            {myStakes.map((stake: any) => (
              <div
                key={stake.id}
                className="flex justify-between border-b border-white/5 py-2"
              >
                <div>
                  <span className="text-white">{stake.amount} tokens</span>
                  <div className="text-xs text-slate-400">
                    Locked until{" "}
                    {new Date(stake.unbonding_end).toLocaleDateString()}
                  </div>
                </div>
                <Button size="sm" variant="ghost">
                  Unbond
                </Button>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-slate-500">No active stakes.</p>
        )}
        <Button size="sm" className="mt-3">
          Stake Tokens
        </Button>
      </Card>

      {/* Slashing History */}
      <Card>
        <h2 className="text-sm font-bold text-white mb-3">Slashing Events</h2>
        <Table>
          <thead>
            <tr>
              <th>Provider</th>
              <th>Reason</th>
              <th>Amount</th>
              <th>Date</th>
            </tr>
          </thead>
          <tbody>
            {slashingHistory?.map((event: any) => (
              <tr key={event.id}>
                <td>{event.provider_username}</td>
                <td>{event.reason}</td>
                <td className="text-red-400">-{event.slashed_amount}</td>
                <td>{new Date(event.created_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Card>
    </div>
  );
}
