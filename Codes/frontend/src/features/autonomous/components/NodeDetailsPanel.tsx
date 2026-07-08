import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { PipelineStep } from "../types";
import { Play, Loader2, CheckCircle, XCircle } from "lucide-react";

interface NodeDetailsPanelProps {
  step?: PipelineStep;
  currentStep?: PipelineStep;
  onRetry: () => void;
}

export function NodeDetailsPanel({
  step,
  currentStep,
  onRetry,
}: NodeDetailsPanelProps) {
  const displayStep = step || currentStep;
  if (!displayStep)
    return (
      <TerminalCard className="p-4 text-center text-slate-500">
        Select a step to view details
      </TerminalCard>
    );
  return (
    <TerminalCard className="p-4 space-y-3">
      <div className="flex justify-between items-start">
        <div>
          <div className="text-sm font-bold text-white">{displayStep.name}</div>
          <div className="text-xs text-slate-400 mt-0.5">
            ID: {displayStep.id.slice(0, 8)}
          </div>
        </div>
        <div className="flex gap-2">
          {displayStep.status === "failed" && (
            <Button size="sm" variant="ghost" onClick={onRetry}>
              <Play size={12} className="mr-1" /> Retry
            </Button>
          )}
        </div>
      </div>
      <div>
        <div className="text-xs text-slate-400">Status</div>
        <Badge
          variant={
            displayStep.status === "running"
              ? "info"
              : displayStep.status === "completed"
                ? "success"
                : displayStep.status === "failed"
                  ? "failed"
                  : "secondary"
          }
        >
          {displayStep.status}
        </Badge>
      </div>
      {displayStep.duration !== undefined && (
        <div>
          <div className="text-xs text-slate-400">Duration</div>
          <div className="text-white">{displayStep.duration} seconds</div>
        </div>
      )}
      {displayStep.started_at && (
        <div>
          <div className="text-xs text-slate-400">Started</div>
          <div className="text-white">
            {new Date(displayStep.started_at).toLocaleString()}
          </div>
        </div>
      )}
      {displayStep.completed_at && (
        <div>
          <div className="text-xs text-slate-400">Completed</div>
          <div className="text-white">
            {new Date(displayStep.completed_at).toLocaleString()}
          </div>
        </div>
      )}
      <div>
        <div className="text-xs text-slate-400">Dependencies</div>
        <div className="flex flex-wrap gap-1 mt-1">
          {displayStep.dependencies.length ? (
            displayStep.dependencies.map((depId) => (
              <span
                key={depId}
                className="px-2 py-0.5 rounded bg-white/10 text-xs"
              >
                {depId.slice(0, 8)}
              </span>
            ))
          ) : (
            <span className="text-slate-500">None</span>
          )}
        </div>
      </div>
      {displayStep.input_artifacts?.length > 0 && (
        <div>
          <div className="text-xs text-slate-400">Input Artifacts</div>
          {displayStep.input_artifacts.map((a) => (
            <div key={a.name} className="text-xs">
              {a.name} ({a.type})
            </div>
          ))}
        </div>
      )}
    </TerminalCard>
  );
}
