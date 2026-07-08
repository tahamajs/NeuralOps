import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { autonomousAPI } from "../services/api";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Spinner } from "../../../components/ui/Spinner";
import { Plus, Trash2, Save, GitBranch } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function StepEditor({ onClose }: { onClose?: () => void }) {
  const { definitionId } = useParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { success, error } = useToast();
  const [definition, setDefinition] = useState<any>({
    name: "",
    description: "",
    steps: [],
  });

  const { data: existingDef, isLoading } = useQuery({
    queryKey: ["workflow-definition", definitionId],
    queryFn: () => autonomousAPI.getWorkflowDefinition(definitionId!),
    enabled: !!definitionId,
    onSuccess: (data) => setDefinition(data),
  });

  const saveMutation = useMutation({
    mutationFn: () =>
      definitionId
        ? autonomousAPI.updateWorkflowDefinition(definitionId, definition)
        : autonomousAPI.createWorkflowDefinition(definition),
    onSuccess: (saved) => {
      queryClient.invalidateQueries({ queryKey: ["workflow-definitions"] });
      success("Workflow saved");
      navigate(`/autonomous/definitions/${saved.id}`);
    },
  });

  const addStep = () =>
    setDefinition((prev) => ({
      ...prev,
      steps: [
        ...prev.steps,
        {
          id: crypto.randomUUID(),
          name: "New Step",
          type: "python",
          config: {},
          next_step_ids: [],
        },
      ],
    }));
  const updateStep = (index: number, field: string, value: any) => {
    const newSteps = [...definition.steps];
    newSteps[index] = { ...newSteps[index], [field]: value };
    setDefinition((prev) => ({ ...prev, steps: newSteps }));
  };
  const removeStep = (index: number) =>
    setDefinition((prev) => ({
      ...prev,
      steps: prev.steps.filter((_, i) => i !== index),
    }));

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <h1 className="text-xl font-bold text-white flex items-center gap-2">
          <GitBranch size={20} className="text-neon-mint" /> Step Editor
        </h1>
        {onClose && (
          <Button variant="ghost" size="sm" onClick={onClose}>
            ✕
          </Button>
        )}
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <TerminalCard className="p-4">
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
            className="mt-3"
          />
        </TerminalCard>
        <div className="space-y-3">
          <div className="flex justify-between items-center">
            <h2 className="text-sm font-bold text-white">Steps</h2>
            <Button size="sm" onClick={addStep}>
              <Plus size={14} className="mr-1" /> Add Step
            </Button>
          </div>
          {definition.steps.map((step, idx) => (
            <TerminalCard key={step.id} className="p-3">
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <Input
                    value={step.name}
                    onChange={(e) => updateStep(idx, "name", e.target.value)}
                    placeholder="Step name"
                  />
                  <select
                    value={step.type}
                    onChange={(e) => updateStep(idx, "type", e.target.value)}
                    className="mt-2 w-full bg-white/5 border border-white/10 rounded p-2 text-sm"
                  >
                    <option value="python">Python Function</option>
                    <option value="shell">Shell Command</option>
                    <option value="http">HTTP Request</option>
                    <option value="ai">AI Task</option>
                  </select>
                  <Input
                    value={step.config.script || ""}
                    onChange={(e) =>
                      updateStep(idx, "config", {
                        ...step.config,
                        script: e.target.value,
                      })
                    }
                    placeholder="Script / Command"
                    className="mt-2"
                  />
                </div>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => removeStep(idx)}
                >
                  <Trash2 size={14} className="text-red-400" />
                </Button>
              </div>
            </TerminalCard>
          ))}
        </div>
        <Button
          onClick={() => saveMutation.mutate()}
          disabled={saveMutation.isPending}
          className="w-full"
        >
          <Save size={14} className="mr-1" /> Save Workflow
        </Button>
      </div>
    </div>
  );
}
