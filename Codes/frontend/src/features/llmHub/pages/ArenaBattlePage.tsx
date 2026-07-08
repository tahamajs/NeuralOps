// src/features/llmHub/pages/ArenaBattlePage.tsx
import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { llmHubAPI } from "../services/api";
import { ArenaLeaderboard } from "../components/ArenaLeaderboard";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Trophy, RefreshCw } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function ArenaBattlePage({ onClose }: { onClose?: () => void }) {
  const [currentBattle, setCurrentBattle] = useState<any>(null);
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const {
    data: battles,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ["arena-battles"],
    queryFn: () => llmHubAPI.getArenaBattles(),
  });
  const voteMutation = useMutation({
    mutationFn: ({
      battleId,
      winner,
    }: {
      battleId: string;
      winner: "A" | "B" | "tie";
    }) => llmHubAPI.submitVote(battleId, winner),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["arena-battles"] });
      loadRandomBattle();
    },
  });

  const loadRandomBattle = () => {
    const pending = battles?.filter((b) => !b.winner);
    if (pending?.length)
      setCurrentBattle(pending[Math.floor(Math.random() * pending.length)]);
  };

  useEffect(() => {
    if (battles) loadRandomBattle();
  }, [battles]);

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Trophy size={20} className="text-yellow-400" /> Arena Battles
          </h1>
          <p className="text-xs text-slate-400">
            Vote for the best model response
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="ghost" onClick={() => refetch()}>
            <RefreshCw size={14} className="mr-1" /> Refresh
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 grid lg:grid-cols-2 gap-4">
        {currentBattle ? (
          <>
            <TerminalCard className="p-4 h-full">
              <div className="flex justify-between items-center mb-2">
                <h3 className="font-bold text-white">
                  Model A: {currentBattle.model_a}
                </h3>
                <Button
                  size="sm"
                  variant="primary"
                  onClick={() =>
                    voteMutation.mutate({
                      battleId: currentBattle.id,
                      winner: "A",
                    })
                  }
                >
                  Vote A
                </Button>
              </div>
              <div className="prose prose-invert max-w-none bg-black/30 p-3 rounded">
                {currentBattle.response_a}
              </div>
            </TerminalCard>
            <TerminalCard className="p-4 h-full">
              <div className="flex justify-between items-center mb-2">
                <h3 className="font-bold text-white">
                  Model B: {currentBattle.model_b}
                </h3>
                <Button
                  size="sm"
                  variant="primary"
                  onClick={() =>
                    voteMutation.mutate({
                      battleId: currentBattle.id,
                      winner: "B",
                    })
                  }
                >
                  Vote B
                </Button>
              </div>
              <div className="prose prose-invert max-w-none bg-black/30 p-3 rounded">
                {currentBattle.response_b}
              </div>
            </TerminalCard>
            <div className="text-center text-slate-400 mt-2">
              {currentBattle.prompt}
            </div>
          </>
        ) : (
          <div className="text-center text-slate-500 col-span-2 py-12">
            No battles available. Generate more comparisons!
          </div>
        )}
        <div className="lg:col-span-2">
          <ArenaLeaderboard battles={battles} />
        </div>
      </div>
    </div>
  );
}
