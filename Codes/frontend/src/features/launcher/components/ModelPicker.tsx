import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";
import { Select } from "../../../components/ui/Select";

interface ModelEntry {
  name: string;
  variants?: string[];
}

export const ModelPicker = ({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) => {
  const { data } = useQuery({
    queryKey: ["config/models"],
    queryFn: async () => (await api.get(ENDPOINTS.CONFIG.MODELS)).data,
  });

  // Backend returns [{name, variants}, ...] OR ["string", ...]
  const rawModels: (ModelEntry | string)[] = data?.models ?? [];

  const options = rawModels.map((m) => {
    const name = typeof m === "string" ? m : m.name;
    return { value: name, label: name };
  });

  return (
    <Select
      label="Foundation Model"
      options={options}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    />
  );
};
