import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";
import { LiveLossPlot } from "../../../components/widgets/LiveLossPlot";

export default function CustomMetricsPage() {
  const { id } = useParams<{ id: string }>();
  const { data, isLoading } = useQuery({
    queryKey: ["customMetrics", id],
    queryFn: async () => (await api.get(`/api/jobs/${id}/metrics/`)).data,
  });

  if (isLoading) return <Spinner />;
  const metrics = data || {};

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        CUSTOM <span className="text-neon-blue">METRICS</span>
      </h1>
      {Object.keys(metrics).length === 0 ? (
        <p className="text-slate-500">No custom metrics logged.</p>
      ) : (
        Object.entries(metrics).map(([name, group]: [string, any]) => (
          <Card key={name}>
            <h3 className="text-sm font-bold text-white mb-2">
              {name} ({group.type})
            </h3>
            {group.type === "scalar" && (
              <LiveLossPlot metrics={{ loss: group.values, accuracy: [] }} />
            )}
            {group.type === "text" && (
              <div className="bg-cyber-800 p-3 rounded-lg text-slate-400 text-xs max-h-48 overflow-auto">
                {group.values.join("\n")}
              </div>
            )}
            {group.type !== "scalar" && group.type !== "text" && (
              <p className="text-xs text-slate-400">
                Preview not available for {group.type}
              </p>
            )}
          </Card>
        ))
      )}
    </div>
  );
}
