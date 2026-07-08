import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";
import { LiveLossPlot } from "../../../components/widgets/LiveLossPlot";

export default function TrainingSnapshotsPage() {
  const { runId } = useParams<{ runId: string }>();
  const { data, isLoading } = useQuery({
    queryKey: ["trainingSnapshots", runId],
    queryFn: async () => (await api.get(`/api/runs/${runId}/snapshots/`)).data,
  });

  if (isLoading) return <Spinner />;
  const snapshots = data || [];

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        TRAINING <span className="text-neon-blue">SNAPSHOTS</span>
      </h1>
      <Card>
        {snapshots.length === 0 ? (
          <p className="text-slate-500">No snapshots recorded.</p>
        ) : (
          <>
            <LiveLossPlot
              metrics={{
                loss: snapshots.map((s: any) => s.train_loss),
                accuracy: snapshots.map((s: any) => s.eval_accuracy ?? 0),
              }}
            />
            <div className="mt-4 overflow-x-auto max-h-64 scrollbar-thin">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left text-slate-400">
                    <th className="py-1 px-2">Step</th>
                    <th className="py-1 px-2">Train Loss</th>
                    <th className="py-1 px-2">Eval Loss</th>
                    <th className="py-1 px-2">Accuracy</th>
                    <th className="py-1 px-2">LR</th>
                    <th className="py-1 px-2">CPU</th>
                  </tr>
                </thead>
                <tbody>
                  {snapshots.map((s: any) => (
                    <tr key={s.step} className="border-b border-white/5">
                      <td className="py-1 px-2">{s.step}</td>
                      <td className="py-1 px-2">{s.train_loss?.toFixed(4)}</td>
                      <td className="py-1 px-2">{s.eval_loss?.toFixed(4)}</td>
                      <td className="py-1 px-2">
                        {s.eval_accuracy?.toFixed(4)}
                      </td>
                      <td className="py-1 px-2">
                        {s.learning_rate?.toExponential(2)}
                      </td>
                      <td className="py-1 px-2">{s.cpu_percent}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </Card>
    </div>
  );
}
