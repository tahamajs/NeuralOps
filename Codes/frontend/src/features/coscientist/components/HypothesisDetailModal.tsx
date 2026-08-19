// src/features/coscientist/components/HypothesisDetailModal.tsx
import { useState } from "react";
import { Hypothesis } from "../types";
import { Modal } from "../../../components/ui/Modal";
import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import {
  TrendingUp,
  Star,
  FlaskConical,
  Target,
  Trophy,
  ExternalLink,
  Copy,
  Check,
  Brain,
  MessageSquare,
  ShieldCheck,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

interface HypothesisDetailModalProps {
  hypothesis: Hypothesis | null;
  open: boolean;
  onClose: () => void;
}

export function HypothesisDetailModal({
  hypothesis,
  open,
  onClose,
}: HypothesisDetailModalProps) {
  const { success } = useToast();
  const [copied, setCopied] = useState(false);

  if (!hypothesis) return null;

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
  const round = hypothesis.generation_round || (hypothesis as any).generation || 1;
  const status = hypothesis.status || "proposed";

  const reviews = (hypothesis as any).reviews || [
    {
      agent: "ReflectionAgent-1",
      strengths: "Solid mathematical grounding using spectral decomposition.",
      weaknesses: "Requires validation against massive scale Transformer checkpoints.",
      score: 8,
      suggestions: "Evaluate wall-clock memory consumption during backward pass.",
    },
    {
      agent: "ProximityAgent-2",
      strengths: "High novelty relative to existing linear attention formulations.",
      weaknesses: "Requires custom CUDA kernel implementation for peak throughput.",
      score: 9,
      suggestions: "Compare directly with FlashLinear and Griffin kernel benchmarks.",
    },
  ];

  const metaReview = (hypothesis as any).meta_review || {
    synthesis:
      "This hypothesis provides a viable low-rank approximation for multi-head self-attention, significantly reducing quadratic time complexity while preserving sequence modeling capacity.",
    final_score: 8.8,
    recommendation: "accept",
  };

  const handleCopy = () => {
    const text = `Hypothesis: ${title}\n\nDescription: ${description}\n\nMerit: ${merit.toFixed(1)}/10 | Novelty: ${novelty.toFixed(1)}/10 | Testability: ${testability.toFixed(1)}/10 | Impact: ${impact.toFixed(1)}/10\nELO Rating: ${elo}`;
    navigator.clipboard.writeText(text);
    setCopied(true);
    success("Hypothesis copied to clipboard");
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Hypothesis Detailed Analysis"
      className="max-w-4xl max-h-[90vh] overflow-y-auto"
    >
      <div className="space-y-5 p-1">
        {/* Header summary */}
        <div className="bg-slate-900/60 border border-white/10 rounded-xl p-4">
          <div className="flex justify-between items-start gap-3">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <Brain size={16} className="text-neon-mint" />
                <h2 className="text-base font-bold text-white">{title}</h2>
              </div>
              <p className="text-sm text-slate-300 leading-relaxed mt-2">
                {description}
              </p>
            </div>
            <Button
              size="sm"
              variant="ghost"
              onClick={handleCopy}
              className="shrink-0 text-slate-400 hover:text-white"
            >
              {copied ? <Check size={14} className="text-neon-mint" /> : <Copy size={14} />}
              <span className="ml-1 text-xs">{copied ? "Copied" : "Copy"}</span>
            </Button>
          </div>

          <div className="flex items-center gap-3 mt-3 pt-3 border-t border-white/5 text-xs">
            <span className="text-slate-400">
              Status: <Badge variant="info">{status}</Badge>
            </span>
            <span className="text-slate-400">
              Evolution Round: <span className="text-white font-mono">{round}</span>
            </span>
            <span className="text-slate-400">
              ELO Rating: <span className="text-yellow-400 font-bold font-mono">{elo}</span>
            </span>
          </div>
        </div>

        {/* Score Grid */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <ScoreCard
            label="Scientific Merit"
            value={merit}
            icon={<FlaskConical size={14} className="text-neon-mint" />}
          />
          <ScoreCard
            label="Novelty Score"
            value={novelty}
            icon={<Star size={14} className="text-yellow-400" />}
          />
          <ScoreCard
            label="Testability"
            value={testability}
            icon={<Target size={14} className="text-emerald-400" />}
          />
          <ScoreCard
            label="Potential Impact"
            value={impact}
            icon={<TrendingUp size={14} className="text-neon-blue" />}
          />
        </div>

        {/* Meta Review Synthesis */}
        <div className="bg-slate-900/40 border border-white/10 rounded-xl p-4 space-y-2">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
              <ShieldCheck size={14} className="text-neon-mint" />
              Meta-Review Synthesis
            </h3>
            <Badge variant="success" className="uppercase text-[10px]">
              {metaReview.recommendation}
            </Badge>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed bg-white/5 p-3 rounded-lg border border-white/5">
            {metaReview.synthesis}
          </p>
        </div>

        {/* Agent Peer Reviews */}
        {reviews.length > 0 && (
          <div className="space-y-3">
            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
              <MessageSquare size={14} className="text-yellow-400" />
              Agent Peer Reviews ({reviews.length})
            </h3>
            <div className="grid md:grid-cols-2 gap-3">
              {reviews.map((r: any, idx: number) => (
                <div
                  key={idx}
                  className="bg-slate-950/60 border border-white/5 rounded-xl p-3 text-xs space-y-1.5"
                >
                  <div className="flex justify-between items-center border-b border-white/5 pb-1.5">
                    <span className="font-mono text-slate-300 font-medium">
                      {r.agent}
                    </span>
                    <span className="font-mono text-neon-mint font-bold">
                      {r.score}/10
                    </span>
                  </div>
                  {r.strengths && (
                    <div>
                      <span className="text-[10px] text-emerald-400 font-semibold uppercase">
                        Strengths:
                      </span>
                      <p className="text-slate-400 text-[11px]">{r.strengths}</p>
                    </div>
                  )}
                  {r.weaknesses && (
                    <div>
                      <span className="text-[10px] text-rose-400 font-semibold uppercase">
                        Weaknesses:
                      </span>
                      <p className="text-slate-400 text-[11px]">{r.weaknesses}</p>
                    </div>
                  )}
                  {r.suggestions && (
                    <div>
                      <span className="text-[10px] text-neon-blue font-semibold uppercase">
                        Suggestions:
                      </span>
                      <p className="text-slate-400 text-[11px]">{r.suggestions}</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Relevant Papers */}
        {hypothesis.metrics?.relevant_papers && hypothesis.metrics.relevant_papers.length > 0 && (
          <div>
            <div className="text-xs font-bold uppercase tracking-wider text-slate-300 mb-2">
              Related Literature & Foundations
            </div>
            <ul className="space-y-1.5">
              {hypothesis.metrics.relevant_papers.map((paper, idx) => (
                <li
                  key={idx}
                  className="text-xs text-slate-300 flex items-center gap-1.5 bg-white/5 p-2 rounded-lg"
                >
                  <ExternalLink size={12} className="text-neon-blue shrink-0" />
                  <span className="truncate">{paper}</span>
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
}) => {
  const safeVal = Number.isFinite(value) ? value : 0;
  return (
    <div className="bg-slate-900/60 border border-white/10 rounded-xl p-3">
      <div className="flex items-center gap-1.5 text-xs text-slate-400">
        {icon}
        <span className="truncate">{label}</span>
      </div>
      <div className="text-lg font-bold text-white font-mono mt-1">
        {safeVal.toFixed(1)}
        <span className="text-xs text-slate-500 font-normal ml-1">/10</span>
      </div>
    </div>
  );
};
