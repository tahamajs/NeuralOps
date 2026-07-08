import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { hpoAPI } from "../services/api";
import { Modal, Input, Select, Button } from "../../../components/ui/Toast";

export function SweepCreator({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const [name, setName] = useState("");
  const [algorithm, setAlgorithm] = useState("bayesian");
  const [objective, setObjective] = useState("minimize");
  const queryClient = useQueryClient();

  const create = useMutation({
    mutationFn: () => hpoAPI.createSweep({ name, algorithm, objective }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sweeps"] });
      onClose();
      setName("");
    },
  });

  return (
    <Modal open={open} onClose={onClose} title="Create Hyperparameter Sweep">
      <div className="space-y-4">
        <Input
          label="Sweep Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g., ResNet50 Learning Rate Tuning"
        />
        <Select
          label="Algorithm"
          options={[
            { value: "grid", label: "Grid Search" },
            { value: "random", label: "Random Search" },
            { value: "bayesian", label: "Bayesian Optimization" },
            { value: "hyperband", label: "Hyperband" },
          ]}
          value={algorithm}
          onChange={(e) => setAlgorithm(e.target.value)}
        />
        <Select
          label="Objective"
          options={[
            { value: "minimize", label: "Minimize" },
            { value: "maximize", label: "Maximize" },
          ]}
          value={objective}
          onChange={(e) => setObjective(e.target.value)}
        />
        <Button
          onClick={() => create.mutate()}
          disabled={!name || create.isPending}
          className="w-full"
        >
          Create Sweep
        </Button>
      </div>
    </Modal>
  );
}
