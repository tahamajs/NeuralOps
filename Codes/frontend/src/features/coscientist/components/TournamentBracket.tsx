// src/features/coscientist/components/TournamentBracket.tsx
import { Hypothesis, TournamentMatch } from "../types";
import { cn } from "../../../lib/utils";

interface TournamentBracketProps {
  matches: TournamentMatch[];
  hypotheses: Hypothesis[];
}

export function TournamentBracket({
  matches,
  hypotheses,
}: TournamentBracketProps) {
  const getHypothesisTitle = (id: string) =>
    hypotheses.find((h) => h.id === id)?.title.slice(0, 40) || "Unknown";

  const rounds = ["Quarterfinals", "Semifinals", "Finals"];
  const matchesByRound: Record<number, TournamentMatch[]> = {};
  matches.forEach((m) => {
    if (!matchesByRound[m.round]) matchesByRound[m.round] = [];
    matchesByRound[m.round].push(m);
  });

  return (
    <div className="overflow-x-auto">
      <div className="flex justify-center gap-6 min-w-[600px] p-4">
        {rounds.map((roundName, idx) => {
          const roundMatches = matchesByRound[idx + 1] || [];
          return (
            <div key={idx} className="flex-1">
              <div className="text-xs font-bold text-slate-400 text-center mb-3">
                {roundName}
              </div>
              <div
                className="space-y-4"
                style={{ height: `${roundMatches.length * 70}px` }}
              >
                {roundMatches.map((match) => {
                  const hypothesisA = getHypothesisTitle(match.hypothesis_a_id);
                  const hypothesisB = getHypothesisTitle(match.hypothesis_b_id);
                  const winnerId = match.winner_id;
                  return (
                    <div key={match.id} className="space-y-1">
                      <div
                        className={cn(
                          "p-2 rounded text-sm border",
                          winnerId === match.hypothesis_a_id
                            ? "bg-neon-mint/20 border-neon-mint text-white"
                            : "bg-white/5 border-white/10 text-slate-300",
                        )}
                      >
                        {hypothesisA}
                      </div>
                      <div className="text-center text-xs text-slate-500">
                        vs
                      </div>
                      <div
                        className={cn(
                          "p-2 rounded text-sm border",
                          winnerId === match.hypothesis_b_id
                            ? "bg-neon-mint/20 border-neon-mint text-white"
                            : "bg-white/5 border-white/10 text-slate-300",
                        )}
                      >
                        {hypothesisB}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
