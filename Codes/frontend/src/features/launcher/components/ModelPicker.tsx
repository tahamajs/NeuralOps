import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";
import { Select } from "../../../components/ui/Select";

interface ModelEntry {
  name: string;
  variants?: string[];
}

export const FALLBACK_MODELS = [
  { value: "svd-linear-transformer-1.3b", label: "SVD Linear Transformer (1.3B Parameters)" },
  { value: "deepseek-ai/DeepSeek-V3", label: "DeepSeek-V3 (671B MLA MoE Architecture)" },
  { value: "meta-llama/Meta-Llama-3-8B-Instruct", label: "Meta LLaMA-3 (8B Parameters - GQA)" },
  { value: "mistralai/Mistral-7B-v0.1", label: "Mistral 7B (8-KV Head Grouped Query)" },
  { value: "google/griffin-3b", label: "Griffin-3B (Recurrent Linear RNN)" },
  { value: "nvidia/hymba-1.5b", label: "Hymba-1.5B (Hybrid Attention-SSM)" },
  { value: "bert-base-uncased", label: "BERT-Base (110M Parameters - Encoder)" },
  { value: "roberta-large", label: "RoBERTa-Large (355M Parameters)" },
  { value: "gpt2-large", label: "GPT-2 Large (774M Parameters - Autoregressive)" },
];

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

  const rawModels: (ModelEntry | string)[] = data?.models ?? [];
  const options =
    rawModels.length > 0
      ? rawModels.map((m) => {
          const name = typeof m === "string" ? m : m.name;
          return { value: name, label: name };
        })
      : FALLBACK_MODELS;

  return (
    <Select
      label="Foundation Model"
      options={options}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    />
  );
};
