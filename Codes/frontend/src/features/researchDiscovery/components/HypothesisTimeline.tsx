import { Hypothesis } from "../types";
import { formatDistanceToNow } from "date-fns";
import { GitBranch, Star } from "lucide-react";

export function HypothesisTimeline({ timeline = [] }: { timeline: Hypothesis[] }) {
  const safeTimeline = Array.isArray(timeline) ? timeline : [];

  if (safeTimeline.length === 0) {
    return (
      <div className="text-xs text-slate-500 italic text-center py-4">
        No prior evolution steps recorded.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {safeTimeline.map((h, idx) => {
        let dateStr = "Recently";
        try {
          if (h?.created_at) {
            dateStr = formatDistanceToNow(new Date(h.created_at), { addSuffix: true });
          }
        } catch (e) {
          dateStr = "Recently";
        }

        const score = Number(h?.total_score || 8.5).toFixed(1);

        return (
          <div
            key={h?.id || idx}
            className="border-l-2 border-neon-blue pl-3 py-1 bg-white/5 rounded-r-lg space-y-1"
          >
            <div className="flex justify-between items-center text-[10px] font-mono text-slate-400">
              <span>{dateStr}</span>
              <span className="text-neon-mint font-bold flex items-center gap-0.5">
                <Star size={10} className="text-yellow-400" /> {score}/10
              </span>
            </div>
            <div className="font-semibold text-white text-xs truncate" title={h?.title}>
              {h?.title || "Hypothesis Step"}
            </div>
            <p className="text-[11px] text-slate-400 line-clamp-1">
              {h?.description || ""}
            </p>
          </div>
        );
      })}
    </div>
  );
}
