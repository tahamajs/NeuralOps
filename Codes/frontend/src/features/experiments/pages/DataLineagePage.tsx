import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";

export default function DataLineagePage() {
  const { runId } = useParams<{ runId: string }>();
  const { data, isLoading } = useQuery({
    queryKey: ["dataLineage", runId],
    queryFn: async () => (await api.get(`/runs/${runId}/lineage/`)).data,
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        DATA <span className="text-neon-blue">LINEAGE</span>
      </h1>
      <Card>
        {!data || (!data.datasets?.length && !data.transformations?.length) ? (
          <p className="text-slate-500">No lineage data recorded.</p>
        ) : (
          <div className="space-y-4">
            {data.datasets && data.datasets.length > 0 && (
              <div>
                <h3 className="text-sm font-bold text-white mb-2">Datasets</h3>
                {data.datasets.map((ds: any, i: number) => (
                  <div key={i} className="bg-cyber-800 p-3 rounded mb-2 text-xs">
                    <p className="font-mono text-neon-blue">{ds.name || ds.label}</p>
                    <p className="text-slate-400">
                      Version: {ds.version || "N/A"} | Checksum:{" "}
                      {ds.checksum?.slice(0, 12) || "—"}
                    </p>
                  </div>
                ))}
              </div>
            )}
            {data.transformations && data.transformations.length > 0 && (
              <div>
                <h3 className="text-sm font-bold text-white mb-2">
                  Transformations
                </h3>
                {data.transformations.map((t: any, i: number) => (
                  <div key={i} className="bg-cyber-800 p-3 rounded mb-2 text-xs">
                    <p className="text-slate-400">
                      {t.name}: {t.input} → {t.output}
                    </p>
                  </div>
                ))}
              </div>
            )}
            {data.model_trained_on && data.model_trained_on.length > 0 && (
              <div>
                <h3 className="text-sm font-bold text-white mb-2">
                  Model Trained On
                </h3>
                {data.model_trained_on.map((m: any, i: number) => (
                  <div
                    key={i}
                    className="bg-cyber-800 p-3 rounded mb-2 text-xs"
                  >
                    <p>Model version: {m.model_version || "v1.0"}</p>
                    <p>Checksums: {Array.isArray(m.dataset_checksums) ? m.dataset_checksums.join(", ") : "—"}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </Card>
    </div>
  );
}
