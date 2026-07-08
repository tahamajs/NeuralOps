import { useState } from "react";
import { useParams } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import api from "../../../api/client";
import { Button } from "../../../components/ui/Button";
import { Card } from "../../../components/ui/Card";

type EvalType = "fairness" | "robustness";

export default function EvaluationPage({ type }: { type: EvalType }) {
  const { runId } = useParams<{ runId: string }>();
  const [input, setInput] = useState("{}");
  const [result, setResult] = useState<any>(null);

  const mutation = useMutation({
    mutationFn: async () => {
      const endpoint =
        type === "fairness"
          ? `/api/runs/${runId}/fairness/`
          : `/api/runs/${runId}/robustness/`;
      const res = await api.post(endpoint, JSON.parse(input));
      return res.data;
    },
    onSuccess: setResult,
  });

  return (
    <div className="space-y-6 animate-fade-in max-w-4xl mx-auto">
      <h1 className="text-2xl font-display font-bold text-white capitalize">
        {type} Evaluation
      </h1>
      <Card>
        <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">
          Input Data (JSON)
        </label>
        <textarea
          className="w-full bg-white/5 border border-white/10 rounded-xl p-3 text-white h-48"
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <Button
          className="mt-4"
          onClick={() => mutation.mutate()}
          disabled={mutation.isPending}
        >
          Run {type} Check
        </Button>
        {result && (
          <pre className="mt-4 bg-white/5 p-4 rounded-lg text-xs text-slate-400">
            {JSON.stringify(result, null, 2)}
          </pre>
        )}
      </Card>
    </div>
  );
}
