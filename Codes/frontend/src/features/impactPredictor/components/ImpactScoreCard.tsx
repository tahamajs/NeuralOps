import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Badge } from "../../../components/ui/Badge";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import type { ImpactScore } from "../types";

export function ImpactScoreCard({ score }: { score: ImpactScore }) {
  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-green-400";
    if (score >= 60) return "text-yellow-400";
    return "text-red-400";
  };

  const getTrendIcon = (trend: "up" | "down" | "stable") => {
    switch (trend) {
      case "up":
        return <TrendingUp size={14} className="text-green-400" />;
      case "down":
        return <TrendingDown size={14} className="text-red-400" />;
      default:
        return <Minus size={14} className="text-yellow-400" />;
    }
  };

  const breakdownItems = [
    { label: "Novelty", value: score.breakdown.novelty },
    {
      label: "Technical Feasibility",
      value: score.breakdown.technical_feasibility,
    },
    { label: "Community Interest", value: score.breakdown.community_interest },
    { label: "Reproducibility", value: score.breakdown.reproducibility },
    {
      label: "Potential Citations",
      value: score.breakdown.potential_citations,
    },
  ];

  return (
    <TerminalCard className="p-4 space-y-4">
      <div className="text-center">
        <div className="text-xs text-slate-400 uppercase tracking-wider">
          Overall Impact Score
        </div>
        <div
          className={`text-5xl font-bold ${getScoreColor(score.overall_score)}`}
        >
          {score.overall_score}
        </div>
        <div className="flex justify-center items-center gap-2 mt-1">
          <Badge variant="info">#{score.percentile_rank} percentile</Badge>
          <div className="text-xs text-slate-400">
            CI: {score.confidence_interval[0]} – {score.confidence_interval[1]}
          </div>
        </div>
      </div>

      <div className="border-t border-white/10 pt-3">
        <div className="grid grid-cols-2 gap-2 text-center">
          <div>
            <div className="text-xs text-slate-400">Predicted h‑index</div>
            <div className="text-xl font-bold text-white">
              {score.predicted_h_index}
            </div>
          </div>
          <div>
            <div className="text-xs text-slate-400">5‑year Citations</div>
            <div className="text-xl font-bold text-white">
              {score.predicted_citations_5y.toLocaleString()}
            </div>
          </div>
        </div>
      </div>

      <div className="space-y-2">
        <div className="text-xs font-bold text-slate-400">Score Breakdown</div>
        {breakdownItems.map((item) => (
          <div key={item.label}>
            <div className="flex justify-between text-xs mb-0.5">
              <span>{item.label}</span>
              <span className="text-neon-mint">{item.value}</span>
            </div>
            <div className="w-full bg-white/10 rounded-full h-1">
              <div
                className="bg-neon-blue h-1 rounded-full"
                style={{ width: `${item.value}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </TerminalCard>
  );
}
