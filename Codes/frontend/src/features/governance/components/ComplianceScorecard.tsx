import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Progress } from "../../../components/ui/Progress";
import { ComplianceFramework } from "../types";
import { ShieldCheck, AlertTriangle, XCircle, CheckCircle, Clock } from "lucide-react";

export function ComplianceScorecard({
  framework,
}: {
  framework: ComplianceFramework;
}) {
  const compliant = framework.controls.filter(
    (c) => c.status === "compliant",
  ).length;
  const partial = framework.controls.filter(
    (c) => c.status === "partial",
  ).length;
  const nonCompliant = framework.controls.filter(
    (c) => c.status === "non_compliant",
  ).length;
  const score = framework.controls.length > 0
    ? ((compliant + partial * 0.5) / framework.controls.length) * 100
    : 100;

  const scoreColor =
    score >= 90
      ? "text-neon-mint"
      : score >= 75
        ? "text-yellow-400"
        : "text-red-400";

  return (
    <TerminalCard className="p-4 flex-1">
      <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-neon-mint/10 border border-neon-mint/30 rounded-xl text-center min-w-[80px]">
            <span className={`text-3xl font-black font-mono block ${scoreColor}`}>
              {Math.round(score)}%
            </span>
            <span className="text-[10px] text-slate-400 uppercase tracking-wide">
              {score >= 90 ? "PASSED" : score >= 75 ? "NEEDS WORK" : "FAILING"}
            </span>
          </div>
          <div>
            <h3 className="font-bold text-white text-base">{framework.name}</h3>
            <p className="text-xs text-slate-400 flex items-center gap-1.5 mt-0.5 font-mono">
              <span>Version: <strong>{framework.version}</strong></span>
              <span>·</span>
              <span className="flex items-center gap-1">
                <Clock size={11} /> Last Assessed: {new Date(framework.last_assessed).toLocaleDateString()}
              </span>
            </p>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-3 text-center min-w-[280px]">
          <div className="p-2 bg-green-500/10 border border-green-500/20 rounded">
            <div className="text-lg font-bold text-green-400 font-mono">{compliant}</div>
            <div className="text-[10px] text-slate-400 uppercase">Compliant</div>
          </div>
          <div className="p-2 bg-yellow-500/10 border border-yellow-500/20 rounded">
            <div className="text-lg font-bold text-yellow-400 font-mono">{partial}</div>
            <div className="text-[10px] text-slate-400 uppercase">Partial</div>
          </div>
          <div className="p-2 bg-red-500/10 border border-red-500/20 rounded">
            <div className="text-lg font-bold text-red-400 font-mono">{nonCompliant}</div>
            <div className="text-[10px] text-slate-400 uppercase">Non-compliant</div>
          </div>
        </div>
      </div>

      <div className="mt-3">
        <Progress value={score} className="h-1.5" />
      </div>
    </TerminalCard>
  );
}

