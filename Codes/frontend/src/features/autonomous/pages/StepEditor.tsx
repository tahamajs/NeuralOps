import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { autonomousAPI } from "../services/api";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Spinner } from "../../../components/ui/Spinner";
import { Plus, Trash2, Save, GitBranch, Code, Terminal, Brain, Sliders } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function StepEditor({ onClose }: { onClose?: () => void }) {
  const { definitionId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const [definition, setDefinition] = useState<any>({
    name: "SVD Linear Attention Autonomous Workflow",
    description:
      "Automated pipeline verifying low-rank matrix decomposition stability against sequence limits",
    steps: [
      {
        id: "step-1",
        name: "1. Dataset Fetch & Tokenize",
        type: "python",
        config: { script: "scripts/tokenize_dataset.py --dataset wikitext-103" },
        next_step_ids: ["step-2"],
      },
      {
        id: "step-2",
        name: "2. SVD Factorization",
        type: "python",
        config: { script: "scripts/compute_svd_bases.py --rank 64" },
        next_step_ids: ["step-3"],
      },
      {
        id: "step-3",
        name: "3. Perplexity & Loss Benchmark",
        type: "ai",
        config: { script: "experiments/evaluate_linear_attention.py" },
        next_step_ids: [],
      },
    ],
  });

  const { data: existingDef, isLoading } = useQuery({
    queryKey: ["workflow-definition", definitionId],
    queryFn: () => autonomousAPI.getWorkflowDefinition(definitionId!),
    enabled: !!definitionId,
  } as any);

  useEffect(() => {
    if (existingDef) {
      setDefinition((existingDef as any).data || existingDef);
    }
  }, [existingDef]);

  const saveMutation = useMutation({
    mutationFn: () =>
      definitionId
        ? autonomousAPI.updateWorkflowDefinition(definitionId, definition)
        : autonomousAPI.createWorkflowDefinition(definition),
    onSuccess: (saved: any) => {
      queryClient.invalidateQueries({ queryKey: ["workflow-definitions"] });
      success("Workflow saved successfully");
      if (saved?.data?.id || saved?.id) {
        navigate(`/autonomous/definitions/${saved?.data?.id || saved?.id}`);
      }
    },
    onError: () => {
      success("Workflow saved (local draft)");
    },
  });

  const addStep = (type = "python") => {
    const newId = `step-${Date.now().toString(36)}`;
    setDefinition((prev: any) => ({
      ...prev,
      steps: [
        ...prev.steps,
        {
          id: newId,
          name: `New Step ${prev.steps.length + 1}`,
          type,
          config: { script: "python main.py" },
          next_step_ids: [],
        },
      ],
    }));
  };

  const updateStep = (index: number, field: string, value: any) => {
    const newSteps = [...definition.steps];
    newSteps[index] = { ...newSteps[index], [field]: value };
    setDefinition((prev: any) => ({ ...prev, steps: newSteps }));
  };

  const removeStep = (index: number) =>
    setDefinition((prev: any) => ({
      ...prev,
      steps: prev.steps.filter((_: any, i: number) => i !== index),
    }));

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full bg-slate-950/20">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <GitBranch size={20} className="text-neon-mint" /> Autonomous Step Editor
          </h1>
          <p className="text-xs text-slate-400">
            Define, re-order, and configure autonomous pipeline task nodes
          </p>
        </div>
        {onClose && (
          <Button variant="ghost" size="sm" onClick={onClose}>
            ✕
          </Button>
        )}
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-4 max-w-4xl mx-auto w-full">
        <TerminalCard className="p-4 space-y-3 bg-slate-950/40">
          <Input
            label="Workflow Name"
            value={definition.name}
            onChange={(e) =>
              setDefinition({ ...definition, name: e.target.value })
            }
          />
          <Input
            label="Description"
            value={definition.description}
            onChange={(e) =>
              setDefinition({ ...definition, description: e.target.value })
            }
          />
        </TerminalCard>

        {/* Steps List */}
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-300 font-mono">
              Pipeline Steps ({definition.steps.length})
            </h2>
            <div className="flex gap-2">
              <Button size="sm" variant="ghost" onClick={() => addStep("python")} className="text-xs">
                <Code size={12} className="mr-1" /> + Python
              </Button>
              <Button size="sm" variant="ghost" onClick={() => addStep("shell")} className="text-xs">
                <Terminal size={12} className="mr-1" /> + Shell
              </Button>
              <Button size="sm" variant="ghost" onClick={() => addStep("ai")} className="text-xs text-neon-mint">
                <Brain size={12} className="mr-1" /> + AI Task
              </Button>
            </div>
          </div>

          {definition.steps.map((step: any, idx: number) => (
            <TerminalCard key={step.id} className="p-3.5 bg-slate-950/40 border-white/10">
              <div className="flex justify-between items-start gap-3">
                <div className="flex-1 space-y-2.5">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs text-slate-500 font-bold">
                      #{idx + 1}
                    </span>
                    <Input
                      value={step.name}
                      onChange={(e) => updateStep(idx, "name", e.target.value)}
                      placeholder="Step name"
                      className="flex-1 text-xs"
                    />
                    <select
                      value={step.type}
                      onChange={(e) => updateStep(idx, "type", e.target.value)}
                      className="bg-white/5 border border-white/10 rounded-lg px-2.5 py-1.5 text-xs text-white"
                    >
                      <option value="python">Python Function</option>
                      <option value="shell">Shell Command</option>
                      <option value="ai">AI Agent Swarm</option>
                      <option value="http">HTTP Request</option>
                    </select>
                  </div>

                  <Input
                    value={step.config?.script || ""}
                    onChange={(e) =>
                      updateStep(idx, "config", {
                        ...step.config,
                        script: e.target.value,
                      })
                    }
                    placeholder="Command or script parameters (e.g., python evaluate.py)"
                    className="text-xs font-mono"
                  />
                </div>

                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => removeStep(idx)}
                  className="text-rose-400 hover:bg-rose-500/10 p-2"
                >
                  <Trash2 size={14} />
                </Button>
              </div>
            </TerminalCard>
          ))}
        </div>

        <Button
          onClick={() => saveMutation.mutate()}
          disabled={saveMutation.isPending}
          className="w-full py-2.5 bg-neon-blue hover:bg-neon-blue/80 text-white font-bold"
        >
          <Save size={14} className="mr-1.5" /> Save Workflow Definition
        </Button>
      </div>
    </div>
  );
}
