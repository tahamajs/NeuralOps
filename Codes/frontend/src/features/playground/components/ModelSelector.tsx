import { Select } from "../../../components/ui/Select";
import { ModelConfig } from "../types";

export function ModelSelector({ models, value, onChange }: any) {
  const providers = [...new Set(models.map((m: any) => m.provider))] as string[];
  return (
    <div className="space-y-2">
      <Select
        label="Provider"
        options={providers.map((p) => ({ value: p, label: p }))}
        value={value.provider}
        onChange={(e) => onChange({ ...value, provider: e.target.value })}
      />
      <Select
        label="Model"
        options={models
          .filter((m: any) => m.provider === value.provider)
          .map((m: any) => ({ value: m.model as string, label: m.model as string }))}
        value={value.model}
        onChange={(e) => onChange({ ...value, model: e.target.value })}
      />
    </div>
  );
}
