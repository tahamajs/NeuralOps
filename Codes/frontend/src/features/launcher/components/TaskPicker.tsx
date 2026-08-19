import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { ENDPOINTS } from "../../../api/endpoints";
import { Select } from "../../../components/ui/Select";

export const FALLBACK_TASKS = [
  { value: "mrpc", label: "GLUE: MRPC (Paraphrase Identification)" },
  { value: "sst2", label: "GLUE: SST-2 (Stanford Sentiment Treebank)" },
  { value: "qnli", label: "GLUE: QNLI (Question Natural Language Inference)" },
  { value: "mnli", label: "GLUE: MNLI (Multi-Genre NLI Matched/Mismatched)" },
  { value: "cola", label: "GLUE: CoLA (Corpus of Linguistic Acceptability)" },
  { value: "qqp", label: "GLUE: QQP (Quora Question Pairs)" },
  { value: "rte", label: "GLUE: RTE (Recognizing Textual Entailment)" },
  { value: "lra_16k", label: "LRA: Long Range Arena (16,384 Sequence Context)" },
  { value: "needle_32k", label: "Needle-in-a-Haystack (32,768 Context Retrieval)" },
  { value: "gsm8k", label: "Reasoning: GSM8K (8.5K Grade School Math)" },
  { value: "wikitext103", label: "Pretraining: WikiText-103 Language Modeling" },
];

export const TaskPicker = ({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) => {
  const { data } = useQuery({
    queryKey: ["config/tasks"],
    queryFn: async () => (await api.get(ENDPOINTS.CONFIG.TASKS)).data,
  });
  const apiTasks: string[] = data?.tasks || [];
  const options =
    apiTasks.length > 0
      ? apiTasks.map((t: string) => ({
          value: t,
          label: t.toUpperCase(),
        }))
      : FALLBACK_TASKS;

  return (
    <Select
      label="Benchmark Dataset & Task"
      options={options}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    />
  );
};
