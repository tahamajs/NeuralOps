import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";

export default function ModelLineagePage() {
  const { versionId } = useParams<{ versionId: string }>();
  const { data, isLoading } = useQuery({
    queryKey: ["modelLineage", versionId],
    queryFn: async () => (await api.get(`/models/${versionId}/lineage/`)).data,
  });

  if (isLoading) return <Spinner />;
  const nodes = data?.nodes || [];
  const edges = data?.edges || [];

  // Position nodes in a line; could be improved with a tree layout
  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white tracking-tighter">
        MODEL <span className="text-neon-blue">LINEAGE</span>
      </h1>
      <Card>
        {nodes.length === 0 ? (
          <p className="text-slate-500 text-center py-8">
            No lineage information available.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <svg width={nodes.length * 150} height="200">
              {nodes.map((n: any, i: number) => (
                <g key={n.id}>
                  <circle
                    cx={80 + i * 150}
                    cy={100}
                    r={40}
                    fill="#1e2330"
                    stroke="#3b82f6"
                    strokeWidth={2}
                  />
                  <text
                    x={80 + i * 150}
                    y={95}
                    textAnchor="middle"
                    fill="white"
                    fontSize="10"
                    fontWeight="bold"
                  >
                    {n.version}
                  </text>
                  <text
                    x={80 + i * 150}
                    y={115}
                    textAnchor="middle"
                    fill="#94a3b8"
                    fontSize="8"
                  >
                    {new Date(n.created_at).toLocaleDateString()}
                  </text>
                </g>
              ))}
              {edges.map((e: any, i: number) => {
                const fromIndex = nodes.findIndex((n: any) => n.id === e.from);
                const toIndex = nodes.findIndex((n: any) => n.id === e.to);
                if (fromIndex < 0 || toIndex < 0) return null;
                return (
                  <g key={i}>
                    <line
                      x1={80 + fromIndex * 150 + 40}
                      y1={100}
                      x2={80 + toIndex * 150 - 40}
                      y2={100}
                      stroke="#3b82f6"
                      strokeWidth={2}
                    />
                    <text
                      x={
                        (80 + fromIndex * 150 + 40 + 80 + toIndex * 150 - 40) /
                        2
                      }
                      y={90}
                      textAnchor="middle"
                      fill="#6b7b9a"
                      fontSize="8"
                    >
                      {e.relationship_type}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        )}
      </Card>
    </div>
  );
}
