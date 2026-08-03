// src/features/coscientist/components/HypothesisDetailModal.tsx
import { Hypothesis } from "../types";
import { Modal } from "../../../components/ui/Modal";
import { Badge } from "../../../components/ui/Badge";
import {
  TrendingUp,
  Star,
  FlaskConical,
  Target,
  Trophy,
  ExternalLink,
} from "lucide-react";

interface HypothesisDetailModalProps {
  hypothesis: Hypothesis;
  open: boolean;
  onClose: () => void;
}

export function HypothesisDetailModal({
  hypothesis,
  open,
  onClose,
}: HypothesisDetailModalProps) {
  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Hypothesis Details"
      className="max-w-4xl"
    >
      <div className="space-y-4">
        <div>
          <div className="text-sm text-slate-400">Description</div>
          <div className="text-white mt-1">{hypothesis.description}</div>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <ScoreCard
            label="Scientific Merit"
            value={hypothesis.scientific_merit}
            icon={<FlaskConical size={14} />}
          />
          <ScoreCard
            label="Novelty"
            value={hypothesis.novelty_score}
            icon={<Star size={14} />}
          />
          <ScoreCard
            label="Testability"
            value={hypothesis.testability_score}
            icon={<Target size={14} />}
          />
          <ScoreCard
            label="Impact"
            value={hypothesis.impact_score}
            icon={<TrendingUp size={14} />}
          />
        </div>
        <div className="flex justify-between items-center pt-2 border-t border-white/10">
          <div>
            <span className="text-sm text-slate-400">ELO Rating</span>
            <div className="text-xl font-bold text-yellow-400">
              {hypothesis.elo_rating}
            </div>
          </div>
          <Badge variant="info">Round {hypothesis.generation_round}</Badge>
        </div>
        {hypothesis.metrics?.relevant_papers &&
          hypothesis.metrics.relevant_papers.length > 0 && (
            <div>
              <div className="text-sm font-bold text-white mb-1">
                Related Literature
              </div>
              <ul className="space-y-1">
                {hypothesis.metrics.relevant_papers.map((paper, idx) => (
                  <li
                    key={idx}
                    className="text-xs text-slate-300 flex items-center gap-1"
                  >
                    <ExternalLink size={12} /> {paper}
                  </li>
                ))}
              </ul>
            </div>
          )}
      </div>
    </Modal>
  );
}

const ScoreCard = ({
  label,
  value,
  icon,
}: {
  label: string;
  value: number;
  icon: React.ReactNode;
}) => (
  <div className="bg-white/5 rounded-lg p-2">
    <div className="flex items-center gap-1 text-xs text-slate-400">
      {icon}
      {label}
    </div>
    <div className="text-lg font-bold text-white mt-0.5">
      {value.toFixed(2)}
    </div>
  </div>
);
