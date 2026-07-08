// src/features/coscientist/components/HypothesisCard.tsx
import { useState } from "react";
import { Hypothesis } from "../types";
import { HypothesisDetailModal } from "./HypothesisDetailModal";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Badge } from "../../../components/ui/Badge";

interface HypothesisCardProps {
  hypothesis: Hypothesis;
}

export function HypothesisCard({ hypothesis }: HypothesisCardProps) {
  const [showDetail, setShowDetail] = useState(false);

  const score = (
    hypothesis.total_score ||
    (hypothesis.scientific_merit +
      hypothesis.novelty_score +
      hypothesis.testability_score +
      hypothesis.impact_score) /
      4
  ).toFixed(2);

  return (
    <>
      <TerminalCard
        className="p-3 cursor-pointer hover:border-neon-mint/50 transition-colors"
        onClick={() => setShowDetail(true)}
      >
        <div className="flex justify-between items-start gap-2">
          <div className="flex-1">
            <div className="font-medium text-white">{hypothesis.title}</div>
            <div className="text-xs text-slate-400 mt-0.5 line-clamp-2">
              {hypothesis.description}
            </div>
          </div>
          <div className="text-right">
            <div className="text-sm font-bold text-neon-mint">{score}</div>
            <Badge variant="secondary" className="mt-1">
              {hypothesis.status}
            </Badge>
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
