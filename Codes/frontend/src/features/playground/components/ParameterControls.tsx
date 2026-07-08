import { Input } from "../../../components/ui/Input";
import { ModelConfig } from "../types";

export function ParameterControls({ config, onChange }: any) {
  return (
    <div className="space-y-2">
      <Input
        label="Temperature"
        type="number"
        step="0.1"
        min="0"
        max="2"
        value={config.temperature}
        onChange={(e) =>
          onChange({ ...config, temperature: parseFloat(e.target.value) })
        }
      />
      <Input
        label="Max Tokens"
        type="number"
        value={config.max_tokens}
        onChange={(e) =>
          onChange({ ...config, max_tokens: parseInt(e.target.value) })
        }
      />
      <Input
        label="Top P"
        type="number"
        step="0.05"
        min="0"
        max="1"
        value={config.top_p}
        onChange={(e) =>
          onChange({ ...config, top_p: parseFloat(e.target.value) })
        }
      />
    </div>
  );
}
