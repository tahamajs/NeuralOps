import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Button } from "../../../components/ui/Button";
import { Plus, Trash2 } from "lucide-react";

export function ResourceSelector({ protocol, onChange }: any) {
  const addResource = () => {
    const newResource = {
      id: crypto.randomUUID(),
      name: "",
      type: "dataset",
      required: true,
    };
    onChange({
      ...protocol,
      resources: [...(protocol.resources || []), newResource],
    });
  };
  const updateResource = (id: string, updates: any) => {
    onChange({
      ...protocol,
      resources: protocol.resources.map((r: any) =>
        r.id === id ? { ...r, ...updates } : r,
      ),
    });
  };
  const removeResource = (id: string) => {
    onChange({
      ...protocol,
      resources: protocol.resources.filter((r: any) => r.id !== id),
    });
  };
  return (
    <div className="space-y-3">
      <div className="flex justify-between items-center">
        <h3 className="text-sm font-bold text-white">Required Resources</h3>
        <Button size="sm" variant="ghost" onClick={addResource}>
          <Plus size={14} /> Add
        </Button>
      </div>
      {protocol.resources?.map((r: any) => (
        <div key={r.id} className="flex gap-2 p-2 rounded bg-white/5 items-end">
          <Input
            value={r.name}
            onChange={(e) => updateResource(r.id, { name: e.target.value })}
            placeholder="Resource name"
            className="flex-1"
          />
          <Select
            options={[
              { value: "dataset", label: "Dataset" },
              { value: "model", label: "Model" },
              { value: "code", label: "Code" },
              { value: "container", label: "Container" },
              { value: "external", label: "External" },
            ]}
            value={r.type}
            onChange={(e) => updateResource(r.id, { type: e.target.value })}
            className="w-28"
          />
          <Input
            placeholder="Version / URL"
            value={r.url || ""}
            onChange={(e) => updateResource(r.id, { url: e.target.value })}
            className="w-48"
          />
          <button onClick={() => removeResource(r.id)}>
            <Trash2 size={14} className="text-red-400" />
          </button>
        </div>
      ))}
    </div>
  );
}
