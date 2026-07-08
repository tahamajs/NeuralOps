import { Card } from "../../../components/ui/Card";
import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { TrendingUp, Play } from "lucide-react";
import { useNavigate } from "react-router-dom";

interface Props {
  analysis: any;
  runId: string;
}

export default function TrainingAnalysisCard({ analysis, runId }: Props) {
  const navigate = useNavigate();
  const statusColor =
    {
      healthy: "completed",
      overfitting: "failed",
      plateau: "queued",
      underfitting: "queued",
      diverging: "failed",
    }[analysis.status] || "queued";

  return (
    <Card className="border border-yellow-500/20 bg-yellow-500/5">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-bold text-white flex items-center gap-2">
          <TrendingUp size={18} className="text-yellow-400" />
          Training Analysis
        </h3>
        <Badge status={statusColor}>{analysis.status}</Badge>
      </div>

      {analysis.issues?.length > 0 && (
        <div className="mb-3">
          <p className="text-xs text-slate-400 mb-1">Issues Detected:</p>
          <div className="flex flex-wrap gap-1">
            {analysis.issues.map((issue: string, i: number) => (
              <span
                key={i}
                className="px-2 py-0.5 bg-red-500/10 text-red-400 rounded text-[10px] uppercase tracking-wider"
              >
                {issue}
              </span>
            ))}
          </div>
        </div>
      )}

      {analysis.suggestions?.length > 0 && (
        <div className="mb-3">
          <p className="text-xs text-slate-400 mb-1">Suggestions:</p>
          <ul className="list-disc list-inside text-xs text-green-400 space-y-0.5">
            {analysis.suggestions.map((s: string, i: number) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="grid grid-cols-3 gap-2 mb-3 text-center text-xs">
        <div className="bg-cyber-800 rounded p-2">
          <div className="text-slate-400">Loss</div>
          <div className="text-white font-mono">
            {analysis.metrics_summary?.current_loss?.toFixed(4) || "—"}
          </div>
        </div>
        <div className="bg-cyber-800 rounded p-2">
          <div className="text-slate-400">Best Loss</div>
          <div className="text-white font-mono">
            {analysis.metrics_summary?.best_loss?.toFixed(4) || "—"}
          </div>
        </div>
        <div className="bg-cyber-800 rounded p-2">
          <div className="text-slate-400">Best Acc</div>
          <div className="text-white font-mono">
            {analysis.metrics_summary?.best_accuracy?.toFixed(4) || "—"}
          </div>
        </div>
      </div>

      <Button size="sm" onClick={() => navigate(`/experiments/${runId}`)}>
        <Play size={14} className="mr-1" /> View Experiment
      </Button>
    </Card>
  );
}
