// src/features/llmHub/pages/EvaluationSuiteRunner.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { llmHubAPI } from "../services/api";
import { EvaluationResultCard } from "../components/EvaluationResultCard";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { LineChart, Activity, Target } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function EvaluationSuiteRunner({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: runs, isLoading: runsLoading } = useQuery({
    queryKey: ["evaluation-runs"],
    queryFn: () => llmHubAPI.getEvaluationRuns(),
  });
  const { data: results, isLoading: resultsLoading } = useQuery({
    queryKey: ["evaluation-results", selectedRun],
    queryFn: () => llmHubAPI.getEvaluationResults(selectedRun!),
    enabled: !!selectedRun,
  });

  const createRun = useMutation({
    mutationFn: () =>
      llmHubAPI.createEvaluation({
        name: "New Evaluation",
        prompt_id: "prompt-1",
        models: ["gpt-4o", "claude-3.5"],
        metrics: ["accuracy", "fluency"],
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["evaluation-runs"] });
      success("Evaluation started");
    },
  });

  if (runsLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Target size={20} className="text-neon-mint" /> Evaluation Suite
          </h1>
          <p className="text-xs text-slate-400">
            Compare model performance across metrics
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" onClick={() => createRun.mutate()}>
            <Activity size={14} className="mr-1" /> New Evaluation
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <div className="grid md:grid-cols-3 gap-4">
          {runs?.map((run) => (
            <TerminalCard
              key={run.id}
              className={`p-3 cursor-pointer ${selectedRun === run.id ? "border-neon-blue" : ""}`}
              onClick={() => setSelectedRun(run.id)}
            >
              <div>
                <div className="flex justify-between">
                  <span className="font-bold text-white">{run.name}</span>
                  <Badge
                    variant={run.status === "completed" ? "success" : "warning"}
                  >
                    {run.status}
                  </Badge>
                </div>
                <div className="text-xs text-slate-400 mt-1">
                  Models: {run.models.join(", ")}
                </div>
                <div className="text-xs text-slate-400">
                  Metrics: {run.metrics.join(", ")}
                </div>
              </div>
            </TerminalCard>
          ))}
        </div>
        {selectedRun && results && (
          <TerminalCard className="p-4">
            <h3 className="text-sm font-bold text-white mb-3">Results</h3>
            <div className="space-y-3">
              {results.map((r) => (
                <EvaluationResultCard key={r.model} result={r} />
              ))}
            </div>
          </TerminalCard>
        )}
      </div>
    </div>
  );
}
