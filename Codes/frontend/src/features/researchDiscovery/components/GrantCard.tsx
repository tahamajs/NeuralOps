import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Badge } from "../../../components/ui/Badge";
import { Calendar, Award, ExternalLink, TrendingUp, DollarSign } from "lucide-react";

export function GrantCard({ grant }: any) {
  if (!grant) return null;

  let daysLeft = 30;
  let deadlineStr = "Rolling Deadline";

  try {
    if (grant.deadline) {
      const deadlineDate = new Date(grant.deadline);
      daysLeft = Math.max(
        0,
        Math.ceil((deadlineDate.getTime() - Date.now()) / (1000 * 60 * 60 * 24))
      );
      deadlineStr = deadlineDate.toLocaleDateString();
    }
  } catch (e) {
    daysLeft = 30;
  }

  const tags = Array.isArray(grant.tags) ? grant.tags : [];
  const matchPercent = Math.round((grant.match_score || 0.85) * 100);
  const amountFormatted = grant.amount ? `$${Number(grant.amount).toLocaleString()}` : "Variable Funding";

  return (
    <TerminalCard className="p-3.5 bg-slate-950/40 border-white/10 hover:border-white/20 transition-all">
      <div className="flex justify-between items-start gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <Award size={15} className="text-yellow-400 shrink-0" />
            <h3 className="font-bold text-white text-xs truncate" title={grant.title}>
              {grant.title}
            </h3>
            <Badge
              variant={grant.status === "open" ? "success" : "warning"}
              className="text-[9px] uppercase font-mono"
            >
              {grant.status || "open"}
            </Badge>
          </div>

          <div className="text-[11px] text-slate-400 font-mono">
            {grant.funding_body} · <strong className="text-white">{amountFormatted}</strong>
          </div>

          <div className="flex flex-wrap gap-1.5 mt-2">
            {tags.slice(0, 4).map((t: string) => (
              <span
                key={t}
                className="px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-[9px] font-mono text-slate-300"
              >
                {t}
              </span>
            ))}
          </div>

          <p className="text-xs text-slate-300 mt-2 line-clamp-2 leading-relaxed">
            {grant.description}
          </p>

          <div className="flex items-center gap-4 mt-2.5 text-[10px] font-mono text-slate-400">
            <span className="flex items-center gap-1">
              <Calendar size={11} /> Deadline: {deadlineStr}
            </span>
            <span
              className={`flex items-center gap-1 font-semibold ${
                daysLeft < 30 ? "text-rose-400" : "text-emerald-400"
              }`}
            >
              {daysLeft} days remaining
            </span>
          </div>
        </div>

        <div className="text-right shrink-0 flex flex-col items-end">
          <div className="flex items-center gap-1 text-neon-mint font-bold text-xs font-mono">
            <TrendingUp size={13} /> {matchPercent}% Match
          </div>
          <button
            onClick={() => window.open(grant.application_url || "https://grants.gov", "_blank")}
            className="mt-3 px-2.5 py-1 rounded bg-neon-blue/10 border border-neon-blue/30 text-neon-blue hover:bg-neon-blue/20 text-xs font-semibold flex items-center gap-1 transition-all"
          >
            Apply Now <ExternalLink size={11} />
          </button>
        </div>
      </div>
    </TerminalCard>
  );
}
