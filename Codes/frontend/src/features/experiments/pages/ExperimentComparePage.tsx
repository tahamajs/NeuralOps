import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Spinner } from "../../../components/ui/Spinner";
import { Input } from "../../../components/ui/Input";

export default function ExperimentComparePage() {
  const [runIds, setRunIds] = useState<string[]>(["", ""]);
  const [result, setResult] = useState<any>(null);

  const { data: experiments = [] } = useQuery({
    queryKey: ["experiments"],
    queryFn: async () => (await api.get("/experiments/")).data,
  });

  const compareMutation = useMutation({
    mutationFn: async (ids: string[]) => {
      const res = await api.post("/experiments/compare/", { run_ids: ids });
      return res.data;
    },
    onSuccess: setResult,
  });

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        COMPARE <span className="text-neon-blue">EXPERIMENTS</span>
      </h1>
      <Card>
        <p className="text-slate-400 text-sm mb-4">
          Select two or more runs to compare.
        </p>
        <div className="grid grid-cols-2 gap-4">
          {runIds.map((id, idx) => (
            <select
              key={idx}
              className="bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-white"
              value={id}
              onChange={(e) => {
                const next = [...runIds];
                next[idx] = e.target.value;
                setRunIds(next);
              }}
            >
              <option value="">Select run...</option>
              {experiments.map((exp: any) => (
                <option key={exp.id} value={exp.id}>
                  {exp.name || exp.id.slice(0, 8)} ({exp.status})
                </option>
              ))}
            </select>
          ))}
        </div>
        <Button
          className="mt-4"
          onClick={() => compareMutation.mutate(runIds.filter(Boolean))}
          disabled={
            compareMutation.isPending || runIds.filter(Boolean).length < 2
          }
        >
          Compare
        </Button>
        {result && (
          <div className="mt-6 space-y-4">
            <h3 className="text-sm font-bold text-white">Metrics Comparison</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left text-slate-400">
                    <th className="py-2 px-3">Run</th>
                    {Object.keys(result.metrics_table[0] || {}).map((k) => (
                      <th key={k} className="py-2 px-3">
                        {k}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {result.metrics_table.map((row: any, i: number) => (
                    <tr key={i} className="border-b border-white/5">
                      {Object.values(row).map((val: any, j: number) => (
                        <td key={j} className="py-2 px-3">
                          {typeof val === "number" ? val.toFixed(4) : val}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
