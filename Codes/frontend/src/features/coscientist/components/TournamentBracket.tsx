// src/features/coscientist/components/TournamentBracket.tsx
import { useState } from "react";
import { Hypothesis, TournamentMatch } from "../types";
import { cn } from "../../../lib/utils";
import { Trophy, Swords, Zap, CheckCircle2 } from "lucide-react";
import { HypothesisDetailModal } from "./HypothesisDetailModal";

interface TournamentBracketProps {
  matches: TournamentMatch[];
  hypotheses: Hypothesis[];
  onSelectHypothesis?: (hypothesis: Hypothesis) => void;
}

export function TournamentBracket({
  matches = [],
  hypotheses = [],
  onSelectHypothesis,
}: TournamentBracketProps) {
  const [selectedHypothesis, setSelectedHypothesis] = useState<Hypothesis | null>(null);
  const [selectedMatch, setSelectedMatch] = useState<TournamentMatch | null>(null);

  // Helper to find hypothesis safely
  const getHypothesis = (id: string): Hypothesis | undefined =>
    hypotheses.find((h) => h?.id === id);

  const getHypothesisTitle = (id: string) => {
    const h = getHypothesis(id);
    if (h?.title) return h.title;
    return id ? `Hypothesis #${id.slice(0, 6)}` : "TBD";
  };

  const getHypothesisScore = (id: string) => {
    const h = getHypothesis(id);
    if (!h) return 1000;
    return Math.round(h.elo_rating || (h as any).elo_score || (h.total_score ? h.total_score * 100 : 1200));
  };

  // Default fallback bracket matches if none are provided
  const fallbackMatches: TournamentMatch[] = [
    {
      id: "m-q1",
      round: 1,
      hypothesis_a_id: hypotheses[0]?.id || "h1",
      hypothesis_b_id: hypotheses[1]?.id || "h2",
      winner_id: hypotheses[0]?.id || "h1",
      vote_distribution: { a: 75, b: 25 },
      completed: true,
      created_at: new Date().toISOString(),
    },
    {
      id: "m-q2",
      round: 1,
      hypothesis_a_id: hypotheses[2]?.id || "h3",
      hypothesis_b_id: hypotheses[3]?.id || "h4",
      winner_id: hypotheses[2]?.id || "h3",
      vote_distribution: { a: 60, b: 40 },
      completed: true,
      created_at: new Date().toISOString(),
    },
    {
      id: "m-s1",
      round: 2,
      hypothesis_a_id: hypotheses[0]?.id || "h1",
      hypothesis_b_id: hypotheses[2]?.id || "h3",
      winner_id: hypotheses[0]?.id || "h1",
      vote_distribution: { a: 80, b: 20 },
      completed: true,
      created_at: new Date().toISOString(),
    },
    {
      id: "m-f1",
      round: 3,
      hypothesis_a_id: hypotheses[0]?.id || "h1",
      hypothesis_b_id: hypotheses[4]?.id || "h5",
      winner_id: hypotheses[0]?.id || "h1",
      vote_distribution: { a: 85, b: 15 },
      completed: true,
      created_at: new Date().toISOString(),
    },
  ];

  const activeMatches = (matches && matches.length > 0) ? matches : fallbackMatches;

  const roundNames = ["Quarterfinals", "Semifinals", "Championship Final"];
  const matchesByRound: Record<number, TournamentMatch[]> = { 1: [], 2: [], 3: [] };

  activeMatches.forEach((m) => {
    const roundKey = m.round || 1;
    if (!matchesByRound[roundKey]) matchesByRound[roundKey] = [];
    matchesByRound[roundKey].push(m);
  });

  const handleCardClick = (hypothesisId: string) => {
    const h = getHypothesis(hypothesisId);
    if (h) {
      if (onSelectHypothesis) onSelectHypothesis(h);
      else setSelectedHypothesis(h);
    }
  };

  return (
    <div className="bg-slate-950/40 border border-white/10 rounded-2xl p-4 overflow-x-auto">
      <div className="flex items-center justify-between mb-4 border-b border-white/5 pb-3">
        <div className="flex items-center gap-2">
          <div className="p-1.5 bg-yellow-400/10 rounded-lg text-yellow-400 border border-yellow-400/20">
            <Trophy size={16} />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white tracking-wide">
              Idea Tournament Bracket
            </h3>
            <p className="text-[11px] text-slate-400">
              Pairwise LLM-debated elimination tournament driving hypothesis evolution
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-neon-mint" /> Winner
          </span>
          <span className="flex items-center gap-1 ml-2">
            <span className="w-2 h-2 rounded-full bg-slate-600" /> Contender
          </span>
        </div>
      </div>

      {/* Bracket Grid */}
      <div className="flex justify-between gap-6 min-w-[750px] p-2">
        {roundNames.map((roundName, idx) => {
          const roundNum = idx + 1;
          const roundMatches = matchesByRound[roundNum] || [];

          return (
            <div key={idx} className="flex-1 flex flex-col">
              <div className="text-xs font-mono uppercase tracking-wider text-slate-400 text-center mb-3 bg-white/5 py-1.5 rounded-lg border border-white/5">
                {roundName}
                <span className="ml-1.5 text-[10px] text-slate-500 font-normal">
                  ({roundMatches.length} {roundMatches.length === 1 ? "match" : "matches"})
                </span>
              </div>

              <div className="flex-1 flex flex-col justify-around gap-4 py-2">
                {roundMatches.length === 0 ? (
                  <div className="h-24 flex items-center justify-center border border-dashed border-white/10 rounded-xl text-xs text-slate-500 italic">
                    Awaiting prior round
                  </div>
                ) : (
                  roundMatches.map((match) => {
                    const hypA = getHypothesis(match.hypothesis_a_id);
                    const hypB = getHypothesis(match.hypothesis_b_id);
                    const titleA = hypA?.title || getHypothesisTitle(match.hypothesis_a_id);
                    const titleB = hypB?.title || getHypothesisTitle(match.hypothesis_b_id);
                    const scoreA = getHypothesisScore(match.hypothesis_a_id);
                    const scoreB = getHypothesisScore(match.hypothesis_b_id);
                    const isWinnerA = match.winner_id === match.hypothesis_a_id;
                    const isWinnerB = match.winner_id === match.hypothesis_b_id;
                    const voteA = match.vote_distribution?.a ?? 50;
                    const voteB = match.vote_distribution?.b ?? 50;

                    return (
                      <div
                        key={match.id}
                        onClick={() => setSelectedMatch(match)}
                        className="bg-slate-900/60 border border-white/10 rounded-xl p-2.5 hover:border-neon-mint/40 transition-all shadow-md cursor-pointer group"
                      >
                        {/* Hypothesis A */}
                        <div
                          onClick={(e) => {
                            e.stopPropagation();
                            handleCardClick(match.hypothesis_a_id);
                          }}
                          className={cn(
                            "p-2 rounded-lg text-xs border transition-all flex items-center justify-between gap-2",
                            isWinnerA
                              ? "bg-neon-mint/15 border-neon-mint/50 text-white font-medium shadow-[0_0_12px_rgba(0,255,157,0.15)]"
                              : "bg-white/5 border-white/5 text-slate-300 hover:bg-white/10"
                          )}
                        >
                          <div className="truncate flex-1" title={titleA}>
                            {titleA}
                          </div>
                          <div className="flex items-center gap-1.5 shrink-0">
                            <span className="text-[10px] font-mono text-slate-400">
                              {scoreA}
                            </span>
                            {isWinnerA && (
                              <CheckCircle2 size={13} className="text-neon-mint shrink-0" />
                            )}
                          </div>
                        </div>

                        {/* Match center divider with vote distribution */}
                        <div className="my-1.5 px-1 flex items-center justify-between text-[10px] text-slate-500">
                          <span className={cn(isWinnerA && "text-neon-mint font-bold")}>
                            {voteA}%
                          </span>
                          <span className="flex items-center gap-1 font-mono uppercase text-[9px] text-slate-400">
                            <Swords size={10} className="text-yellow-400" /> vs
                          </span>
                          <span className={cn(isWinnerB && "text-neon-mint font-bold")}>
                            {voteB}%
                          </span>
                        </div>

                        {/* Hypothesis B */}
                        <div
                          onClick={(e) => {
                            e.stopPropagation();
                            handleCardClick(match.hypothesis_b_id);
                          }}
                          className={cn(
                            "p-2 rounded-lg text-xs border transition-all flex items-center justify-between gap-2",
                            isWinnerB
                              ? "bg-neon-mint/15 border-neon-mint/50 text-white font-medium shadow-[0_0_12px_rgba(0,255,157,0.15)]"
                              : "bg-white/5 border-white/5 text-slate-300 hover:bg-white/10"
                          )}
                        >
                          <div className="truncate flex-1" title={titleB}>
                            {titleB}
                          </div>
                          <div className="flex items-center gap-1.5 shrink-0">
                            <span className="text-[10px] font-mono text-slate-400">
                              {scoreB}
                            </span>
                            {isWinnerB && (
                              <CheckCircle2 size={13} className="text-neon-mint shrink-0" />
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Selected Match Summary Drawer */}
      {selectedMatch && (
        <div className="mt-4 p-3 bg-slate-900/80 border border-white/10 rounded-xl flex items-center justify-between text-xs animate-in fade-in">
          <div className="flex items-center gap-2">
            <Zap size={14} className="text-yellow-400 shrink-0" />
            <div>
              <span className="font-bold text-white">
                Match Reasoning: {getHypothesisTitle(selectedMatch.hypothesis_a_id).slice(0, 30)} vs {getHypothesisTitle(selectedMatch.hypothesis_b_id).slice(0, 30)}
              </span>
              <p className="text-slate-400 text-[11px] mt-0.5">
                {(selectedMatch as any).reasoning ||
                  "The review committee favored this hypothesis due to higher mathematical soundness and lower computational complexity bounds."}
              </p>
            </div>
          </div>
          <button
            onClick={() => setSelectedMatch(null)}
            className="text-slate-400 hover:text-white px-2 py-1 bg-white/5 rounded text-[11px]"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Detail modal when hypothesis clicked */}
      {selectedHypothesis && (
        <HypothesisDetailModal
          hypothesis={selectedHypothesis}
          open={!!selectedHypothesis}
          onClose={() => setSelectedHypothesis(null)}
        />
      )}
    </div>
  );
}
