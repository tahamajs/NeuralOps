// src/features/autonomous/components/NodeDetailsPanel.tsx
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { PipelineStep } from "../types";
import { Play, CheckCircle2, Clock, RotateCw, FileCode, Layers } from "lucide-react";

interface NodeDetailsPanelProps {
  step?: PipelineStep;
  currentStep?: PipelineStep | null;
  onRetry: () => void;
}

export function NodeDetailsPanel({
  step,
  currentStep,
  onRetry,
}: NodeDetailsPanelProps) {
  const displayStep = step || currentStep;

  if (!displayStep) {
    return (
      <TerminalCard className="p-4 text-center text-slate-500 text-xs italic bg-slate-950/40">
        Click any DAG node to view step details & artifacts
      </TerminalCard>
    );
  }

  const dependencies = displayStep.dependencies || [];
  const inputArtifacts = displayStep.input_artifacts || [];
  const outputArtifacts = displayStep.output_artifacts || [];

  const getStatusVariant = (s: string) => {
    switch (s) {
      case "running":
        return "info";
      case "completed":
        return "success";
      case "failed":
        return "failed";
      default:
        return "secondary";
    }
  };

  return (
    <TerminalCard className="p-4 space-y-3 bg-slate-950/40 border-white/10">
      <div className="flex justify-between items-start gap-2">
        <div className="min-w-0">
          <div className="text-sm font-bold text-white truncate">
            {displayStep.name || "Step Details"}
          </div>
          <div className="text-[10px] text-slate-400 font-mono mt-0.5">
            ID: {displayStep.id ? displayStep.id.slice(0, 8) : "N/A"}
          </div>
        </div>
        <div className="flex gap-1.5 shrink-0">
          {displayStep.status === "failed" && (
            <Button size="sm" variant="ghost" onClick={onRetry} className="text-xs text-rose-400">
              <RotateCw size={12} className="mr-1" /> Retry
            </Button>
          )}
        </div>
      </div>

      <div className="flex items-center justify-between py-1 border-b border-white/5 text-xs">
        <span className="text-slate-400">Status</span>
        <Badge variant={getStatusVariant(displayStep.status) as any}>
          {displayStep.status}
        </Badge>
      </div>

      {displayStep.duration !== undefined && (
        <div className="flex items-center justify-between py-1 border-b border-white/5 text-xs">
          <span className="text-slate-400">Execution Time</span>
          <span className="text-white font-mono">{displayStep.duration}s</span>
        </div>
      )}

      {displayStep.started_at && (
        <div className="flex items-center justify-between py-1 border-b border-white/5 text-xs">
          <span className="text-slate-400">Started At</span>
          <span className="text-slate-300 font-mono text-[11px]">
            {new Date(displayStep.started_at).toLocaleTimeString()}
          </span>
        </div>
      )}

      {/* Dependencies */}
      <div>
        <div className="text-[11px] text-slate-400 font-mono mb-1">Dependencies</div>
        <div className="flex flex-wrap gap-1">
          {dependencies.length > 0 ? (
            dependencies.map((depId) => (
              <span
                key={depId}
                className="px-2 py-0.5 rounded bg-white/5 border border-white/10 text-[10px] font-mono text-slate-300"
              >
                {depId.slice(0, 8)}
              </span>
            ))
          ) : (
            <span className="text-[10px] text-slate-500 italic">None (Root Step)</span>
          )}
        </div>
      </div>

      {/* Input Artifacts */}
      {inputArtifacts.length > 0 && (
        <div>
          <div className="text-[11px] text-slate-400 font-mono mb-1">Input Artifacts</div>
          <div className="space-y-1">
            {inputArtifacts.map((a, i) => (
              <div
                key={i}
                className="text-[11px] bg-white/5 p-1.5 rounded flex items-center gap-1.5 text-slate-300"
              >
                <FileCode size={12} className="text-neon-blue shrink-0" />
                <span className="truncate">{a.name}</span>
                <span className="text-[9px] text-slate-500 font-mono ml-auto">({a.type})</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Output Artifacts */}
      {outputArtifacts.length > 0 && (
        <div>
          <div className="text-[11px] text-slate-400 font-mono mb-1">Output Artifacts</div>
          <div className="space-y-1">
            {outputArtifacts.map((a, i) => (
              <div
                key={i}
                className="text-[11px] bg-emerald-950/20 border border-emerald-500/20 p-1.5 rounded flex items-center gap-1.5 text-emerald-300"
              >
                <Layers size={12} className="text-neon-mint shrink-0" />
                <span className="truncate">{a.name}</span>
                <span className="text-[9px] text-emerald-500/70 font-mono ml-auto">({a.type})</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </TerminalCard>
  );
}
