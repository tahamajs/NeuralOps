import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { StepForward, StepBack, Play } from "lucide-react";

const ALGORITHMS = ["tpe", "random", "cmaes", "grid", "auto"];
const PRUNERS = ["hyperband", "median", "threshold", "none"];

interface SearchParam {
  name: string;
  type: "float" | "int" | "categorical";
  low?: number;
  high?: number;
  choices?: string;
  log?: boolean;
}

export default function SweepCreator({ onCreated }: { onCreated: () => void }) {
  const [step, setStep] = useState(1);
  const [form, setForm] = useState({
    name: "",
    project: "",
    algorithm: "tpe",
    direction: "maximize",
    metric: "eval_accuracy",
    max_trials: 20,
    parallel_workers: 2,
    pruner: "hyperband",
    searchSpace: [] as SearchParam[],
  });

  const createMut = useMutation({
    mutationFn: async () => {
      const params: any = {
        name: form.name,
        project: form.project || null,
        config: {
          base_config: {}, // would be populated from prefill
          search_space: form.searchSpace.reduce((acc: any, p) => {
            acc[p.name] = {
              type: p.type,
              low: p.low,
              high: p.high,
              choices: p.choices?.split(",").map((s) => s.trim()),
              log: p.log,
            };
            return acc;
          }, {}),
          algorithm: form.algorithm,
          direction: form.direction,
          metric: form.metric,
          max_trials: form.max_trials,
          parallel_workers: form.parallel_workers,
          pruner: form.pruner,
        },
      };
      await api.post("/sweeps/", params);
    },
    onSuccess: () => onCreated(),
  });

  const addParam = () => {
    setForm((prev) => ({
      ...prev,
      searchSpace: [
        ...prev.searchSpace,
        { name: "", type: "float", low: 0, high: 1 },
      ],
    }));
  };

  const updateParam = (idx: number, field: string, value: any) => {
    const updated = [...form.searchSpace];
    updated[idx] = { ...updated[idx], [field]: value };
    setForm({ ...form, searchSpace: updated });
  };

  return (
    <Card className="max-w-2xl mx-auto space-y-6">
      <h2 className="text-xl font-display font-bold text-white">
        Create Hyperparameter Sweep
      </h2>

      {/* Step indicator */}
      <div className="flex items-center gap-2 mb-4">
        {[1, 2, 3].map((s) => (
          <div
            key={s}
            className={`flex-1 h-1 rounded-full ${step >= s ? "bg-neon-blue" : "bg-white/10"}`}
          />
        ))}
      </div>

      {step === 1 && (
        <div className="space-y-4">
          <Input
            label="Sweep Name"
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
          />
          <Input
            label="Project ID (optional)"
            value={form.project}
            onChange={(e) => setForm({ ...form, project: e.target.value })}
          />
          <Select
            label="Algorithm"
            options={ALGORITHMS.map((a) => ({
              value: a,
              label: a.toUpperCase(),
            }))}
            value={form.algorithm}
            onChange={(e) => setForm({ ...form, algorithm: e.target.value })}
          />
          <Select
            label="Direction"
            options={[
              { value: "maximize", label: "Maximize" },
              { value: "minimize", label: "Minimize" },
            ]}
            value={form.direction}
            onChange={(e) => setForm({ ...form, direction: e.target.value })}
          />
          <Input
            label="Metric Name"
            value={form.metric}
            onChange={(e) => setForm({ ...form, metric: e.target.value })}
          />
          <div className="flex justify-end">
            <Button onClick={() => setStep(2)}>
              <StepForward size={16} className="mr-1" /> Next
            </Button>
          </div>
        </div>
      )}

      {step === 2 && (
        <div className="space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-sm font-bold text-white">Search Space</h3>
            <Button size="sm" variant="ghost" onClick={addParam}>
              + Add Parameter
            </Button>
          </div>
          {form.searchSpace.map((p, idx) => (
            <div
              key={idx}
              className="grid grid-cols-4 gap-2 items-center bg-cyber-800 p-3 rounded-xl"
            >
              <Input
                value={p.name}
                onChange={(e) => updateParam(idx, "name", e.target.value)}
                placeholder="Name"
              />
              <Select
                options={[
                  { value: "float", label: "Float" },
                  { value: "int", label: "Int" },
                  { value: "categorical", label: "Cat" },
                ]}
                value={p.type}
                onChange={(e) => updateParam(idx, "type", e.target.value)}
              />
              {p.type !== "categorical" ? (
                <>
                  <Input
                    type="number"
                    value={p.low}
                    onChange={(e) =>
                      updateParam(idx, "low", parseFloat(e.target.value))
                    }
                    placeholder="Low"
                  />
                  <Input
                    type="number"
                    value={p.high}
                    onChange={(e) =>
                      updateParam(idx, "high", parseFloat(e.target.value))
                    }
                    placeholder="High"
                  />
                </>
              ) : (
                <Input
                  value={p.choices}
                  onChange={(e) => updateParam(idx, "choices", e.target.value)}
                  placeholder="a,b,c"
                  className="col-span-2"
                />
              )}
            </div>
          ))}
          <div className="flex justify-between">
            <Button variant="ghost" onClick={() => setStep(1)}>
              <StepBack size={16} className="mr-1" /> Back
            </Button>
            <Button onClick={() => setStep(3)}>
              <StepForward size={16} className="mr-1" /> Next
            </Button>
          </div>
        </div>
      )}

      {step === 3 && (
        <div className="space-y-4">
          <Input
            label="Max Trials"
            type="number"
            value={form.max_trials}
            onChange={(e) =>
              setForm({ ...form, max_trials: parseInt(e.target.value) })
            }
          />
          <Input
            label="Parallel Workers"
            type="number"
            value={form.parallel_workers}
            onChange={(e) =>
              setForm({ ...form, parallel_workers: parseInt(e.target.value) })
            }
          />
          <Select
            label="Pruner"
            options={PRUNERS.map((p) => ({ value: p, label: p }))}
            value={form.pruner}
            onChange={(e) => setForm({ ...form, pruner: e.target.value })}
          />
          <div className="flex justify-between">
            <Button variant="ghost" onClick={() => setStep(2)}>
              <StepBack size={16} className="mr-1" /> Back
            </Button>
            <Button
              onClick={() => createMut.mutate()}
              disabled={createMut.isPending}
            >
              <Play size={16} className="mr-1" />{" "}
              {createMut.isPending ? "Creating..." : "Launch Sweep"}
            </Button>
          </div>
        </div>
      )}
    </Card>
  );
}
