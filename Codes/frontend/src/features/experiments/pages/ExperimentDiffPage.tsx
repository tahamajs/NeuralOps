import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";
import { Button } from "../../../components/ui/Button";

export default function ExperimentDiffPage() {
  const [runA, setRunA] = useState("");
  const [runB, setRunB] = useState("");
  const [diff, setDiff] = useState<any>(null);

  const { data: experiments = [] } = useQuery({
    queryKey: ["experiments"],
    queryFn: async () => (await api.get("/api/experiments/")).data,
  });

  const handleDiff = async () => {
    if (!runA || !runB) return;
    const res = await api.get(`/api/experiments/runs/${runA}/diff/${runB}/`);
    setDiff(res.data);
  };

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        EXPERIMENT <span className="text-neon-blue">DIFF</span>
      </h1>
      <Card>
        <div className="flex gap-4 mb-4">
          <select
            className="bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-white flex-1"
            value={runA}
            onChange={(e) => setRunA(e.target.value)}
          >
            <option value="">Run A</option>
            {experiments.map((e: any) => (
              <option key={e.id} value={e.id}>
                {e.name || e.id.slice(0, 8)}
              </option>
            ))}
          </select>
          <select
            className="bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-white flex-1"
            value={runB}
            onChange={(e) => setRunB(e.target.value)}
          >
            <option value="">Run B</option>
            {experiments.map((e: any) => (
              <option key={e.id} value={e.id}>
                {e.name || e.id.slice(0, 8)}
              </option>
            ))}
          </select>
          <Button onClick={handleDiff}>Compare</Button>
        </div>
        {diff && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h3 className="text-sm font-bold text-white">Config Diff</h3>
              <pre className="bg-cyber-800 p-3 rounded-lg text-xs text-slate-400 max-h-96 overflow-auto">
                {JSON.stringify(diff.config_diff, null, 2)}
              </pre>
            </div>
            <div>
              <h3 className="text-sm font-bold text-white">Metrics Diff</h3>
              <pre className="bg-cyber-800 p-3 rounded-lg text-xs text-slate-400 max-h-96 overflow-auto">
                {JSON.stringify(diff.metrics_diff, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
