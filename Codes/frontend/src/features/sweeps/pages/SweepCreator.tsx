import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import api from "../../../api/client";
import { Modal, Button, Input, Select } from "../../../components/ui";

export default function SweepCreator({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const queryClient = useQueryClient();
  const [name, setName] = useState("");
  const [algorithm, setAlgorithm] = useState("bayesian");
  const [objective, setObjective] = useState("maximize");
  const [params, setParams] = useState("");

  const create = useMutation({
    mutationFn: () =>
      api.post("/sweeps/", {
        name,
        algorithm,
        objective,
        parameters: JSON.parse(params || "{}"),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sweeps"] });
      onClose();
    },
  });

  return (
    <Modal open={open} onClose={onClose} title="Create New Sweep">
      <div className="space-y-4">
        <Input
          label="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <Select
          label="Algorithm"
          options={[
            { value: "random", label: "Random" },
            { value: "bayesian", label: "Bayesian" },
            { value: "grid", label: "Grid" },
          ]}
          value={algorithm}
          onChange={(e) => setAlgorithm(e.target.value)}
        />
        <Select
          label="Objective"
          options={[
            { value: "maximize", label: "Maximize" },
            { value: "minimize", label: "Minimize" },
          ]}
          value={objective}
          onChange={(e) => setObjective(e.target.value)}
        />
        <Input
          label="Parameters (JSON)"
          value={params}
          onChange={(e) => setParams(e.target.value)}
          placeholder='{"lr": {"min": 1e-5, "max": 1e-3}}'
        />
        <Button
          onClick={() => create.mutate()}
          disabled={create.isPending || !name}
        >
          Create Sweep
        </Button>
      </div>
    </Modal>
  );
}
