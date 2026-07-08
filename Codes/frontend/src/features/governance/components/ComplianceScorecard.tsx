import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Progress } from "../../../components/ui/Progress";
import { ComplianceFramework } from "../types";

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
  const score = (compliant / framework.controls.length) * 100;

  return (
    <TerminalCard className="p-4">
      <div className="grid md:grid-cols-4 gap-3 text-center">
        <div>
          <div className="text-2xl font-bold text-white">
            {score.toFixed(0)}%
          </div>
          <div className="text-xs text-slate-400">Compliance Score</div>
        </div>
        <div>
          <div className="text-2xl font-bold text-green-400">{compliant}</div>
          <div className="text-xs text-slate-400">Compliant</div>
        </div>
        <div>
          <div className="text-2xl font-bold text-yellow-400">{partial}</div>
          <div className="text-xs text-slate-400">Partial</div>
        </div>
        <div>
          <div className="text-2xl font-bold text-red-400">{nonCompliant}</div>
          <div className="text-xs text-slate-400">Non‑compliant</div>
        </div>
      </div>
      <Progress value={score} className="mt-3 h-1" />
    </TerminalCard>
  );
}
