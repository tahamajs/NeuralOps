import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Badge } from "../../../components/ui/Badge";
import { Brain, TrendingUp, Star, Target, Zap, Trophy } from "lucide-react";

export function HypothesisCard({ hypothesis, onSelect, selected }: any) {
  if (!hypothesis) return null;

  const title = hypothesis.title || "Untitled Hypothesis";
  const desc = hypothesis.description || "No description provided.";
  const merit = Number(hypothesis.scientific_merit ?? 8.5);
  const novelty = Number(hypothesis.novelty ?? 8.8);
  const feasibility = Number(hypothesis.feasibility ?? 8.0);
  const impact = Number(hypothesis.impact ?? 9.0);

  const calcScore =
    hypothesis.total_score !== undefined && hypothesis.total_score !== null
      ? Number(hypothesis.total_score)
      : (merit + novelty + feasibility + impact) / 4;

  const scoreFormatted = Number.isFinite(calcScore) ? calcScore.toFixed(1) : "8.5";

  return (
    <TerminalCard
      className={`p-3.5 cursor-pointer transition-all ${
        selected
          ? "border-neon-mint bg-neon-mint/5 shadow-[0_0_15px_rgba(0,255,157,0.1)]"
          : "hover:border-white/20 bg-slate-950/40"
      }`}
      onClick={() => onSelect?.(hypothesis.id)}
    >
      <div className="flex justify-between items-start gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <Brain size={14} className="text-neon-mint shrink-0" />
            <h3 className="font-semibold text-white text-xs truncate" title={title}>
              {title}
            </h3>
            <Badge
              variant={
                hypothesis.status === "evolved"
                  ? "success"
                  : hypothesis.status === "ranked"
                  ? "info"
                  : "warning"
              }
              className="text-[9px] uppercase font-mono"
            >
              {hypothesis.status || "proposed"}
            </Badge>
          </div>
          <p className="text-xs text-slate-300 line-clamp-2 leading-relaxed">
            {desc}
          </p>
          <div className="flex flex-wrap items-center gap-3 mt-2 text-[10px] text-slate-400 font-mono">
            <span className="flex items-center gap-1">
              <Star size={10} className="text-yellow-400" /> Merit: {merit.toFixed(1)}
            </span>
            <span className="flex items-center gap-1">
              <TrendingUp size={10} className="text-neon-blue" /> Novelty: {novelty.toFixed(1)}
            </span>
            <span className="flex items-center gap-1">
              <Target size={10} className="text-emerald-400" /> Feasibility: {feasibility.toFixed(1)}
            </span>
            <span className="flex items-center gap-1">
              <Zap size={10} className="text-purple-400" /> Impact: {impact.toFixed(1)}
            </span>
          </div>
        </div>
        <div className="text-right shrink-0 flex flex-col items-end">
          <div className="text-base font-bold text-neon-mint font-mono">
            {scoreFormatted}
          </div>
          <div className="text-[9px] text-slate-500 font-mono">Score / 10</div>
        </div>
      </div>
    </TerminalCard>
  );
}
