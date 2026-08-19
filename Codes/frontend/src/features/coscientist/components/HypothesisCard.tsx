// src/features/coscientist/components/HypothesisCard.tsx
import { useState } from "react";
import { Hypothesis } from "../types";
import { HypothesisDetailModal } from "./HypothesisDetailModal";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Badge } from "../../../components/ui/Badge";
import { FlaskConical, Star, Sparkles, TrendingUp, Trophy } from "lucide-react";

interface HypothesisCardProps {
  hypothesis: Hypothesis;
  onSelect?: (hypothesis: Hypothesis) => void;
  selected?: boolean;
}

export function HypothesisCard({
  hypothesis,
  onSelect,
  selected,
}: HypothesisCardProps) {
  const [showDetail, setShowDetail] = useState(false);

  const title = hypothesis.title || "Untitled Hypothesis";
  const description =
    hypothesis.description || (hypothesis as any).content || "No description provided.";
  const merit = Number(hypothesis.scientific_merit ?? 8.5);
  const novelty = Number(hypothesis.novelty_score ?? 9.0);
  const testability = Number(hypothesis.testability_score ?? 8.0);
  const impact = Number(hypothesis.impact_score ?? 8.8);
  const elo = Math.round(
    Number(hypothesis.elo_rating || (hypothesis as any).elo_score || 1200)
  );

  const calcScore =
    hypothesis.total_score !== undefined && hypothesis.total_score !== null
      ? Number(hypothesis.total_score)
      : (merit + novelty + testability + impact) / 4;

  const scoreFormatted = Number.isFinite(calcScore) ? calcScore.toFixed(1) : "8.5";
  const status = hypothesis.status || "proposed";
  const round = hypothesis.generation_round || (hypothesis as any).generation || 1;

  const handleClick = () => {
    if (onSelect) onSelect(hypothesis);
    else setShowDetail(true);
  };

  const getStatusBadgeVariant = (s: string) => {
    switch (s) {
      case "selected":
      case "finalist":
        return "success";
      case "evolved":
      case "refined":
        return "info";
      case "ranked":
      case "shortlisted":
        return "warning";
      default:
        return "secondary";
    }
  };

  return (
    <>
      <TerminalCard
        className={`p-3.5 cursor-pointer hover:border-neon-mint/60 transition-all ${
          selected ? "border-neon-mint bg-neon-mint/5 shadow-[0_0_15px_rgba(0,255,157,0.1)]" : ""
        }`}
        onClick={handleClick}
      >
        <div className="flex justify-between items-start gap-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <FlaskConical size={14} className="text-neon-mint shrink-0" />
              <span className="font-semibold text-white text-sm truncate" title={title}>
                {title}
              </span>
              <Badge variant={getStatusBadgeVariant(status)} className="text-[10px] uppercase">
                {status}
              </Badge>
            </div>
            <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed">
              {description}
            </p>

            <div className="flex flex-wrap items-center gap-3 mt-2.5 text-[11px] text-slate-400 font-mono">
              <span className="flex items-center gap-1">
                <Star size={11} className="text-yellow-400" /> Merit: {merit.toFixed(1)}
              </span>
              <span className="flex items-center gap-1">
                <Sparkles size={11} className="text-purple-400" /> Novelty: {novelty.toFixed(1)}
              </span>
              <span className="flex items-center gap-1">
                <TrendingUp size={11} className="text-neon-blue" /> Impact: {impact.toFixed(1)}
              </span>
              <span className="flex items-center gap-1 text-slate-500">
                Gen {round}
              </span>
            </div>
          </div>

          <div className="text-right shrink-0 flex flex-col items-end">
            <div className="text-base font-bold text-neon-mint font-mono">
              {scoreFormatted}
            </div>
            <div className="text-[10px] text-slate-500 font-mono flex items-center gap-1 mt-0.5">
              <Trophy size={10} className="text-yellow-400" /> {elo} ELO
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowDetail(true);
              }}
              className="mt-2 text-[10px] text-neon-blue hover:underline font-mono"
            >
              View details →
            </button>
          </div>
        </div>
      </TerminalCard>

      <HypothesisDetailModal
        hypothesis={hypothesis}
        open={showDetail}
        onClose={() => setShowDetail(false)}
      />
    </>
  );
}
