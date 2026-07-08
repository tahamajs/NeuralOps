import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import api from "../../../api/client";
import Plot from "react-plotly.js";
import { Card } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Spinner } from "../../../components/ui/Spinner";
import { X, GitBranch } from "lucide-react";

interface Props {
  onClose: () => void;
}

export default function CitationGraphModal({ onClose }: Props) {
  const [paperId, setPaperId] = useState("");
  const [graph, setGraph] = useState<any>(null);

  const fetchMut = useMutation({
    mutationFn: async () => {
      const res = await api.post("/api/agents/papers/citation-graph/", {
        paper_ids: [paperId],
      });
      return res.data;
    },
    onSuccess: setGraph,
  });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 animate-fade-in">
      <Card className="w-full max-w-4xl max-h-[85vh] overflow-y-auto p-6 space-y-4">
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-display font-bold text-white flex items-center gap-2">
            <GitBranch size={20} className="text-neon-blue" /> Citation Graph
          </h2>
          <button onClick={onClose} className="text-slate-400 hover:text-white">
            <X size={20} />
          </button>
        </div>

        <div className="flex gap-3">
          <Input
            value={paperId}
            onChange={(e) => setPaperId(e.target.value)}
            placeholder="Enter Semantic Scholar paper ID"
            className="flex-1"
          />
          <Button
            onClick={() => fetchMut.mutate()}
            disabled={fetchMut.isPending}
          >
            Load Graph
          </Button>
        </div>

        {graph && graph.nodes?.length > 0 && (
          <Plot
            data={[
              {
                x: graph.nodes.map(
                  (_: any, i: number) =>
                    Math.cos((2 * Math.PI * i) / graph.nodes.length) * 100,
                ),
                y: graph.nodes.map(
                  (_: any, i: number) =>
                    Math.sin((2 * Math.PI * i) / graph.nodes.length) * 100,
                ),
                mode: "markers+text",
                type: "scatter",
                text: graph.nodes.map((n: any) => n.title.slice(0, 30)),
                textposition: "top center",
                marker: { size: 15, color: "#3b82f6" },
                hovertext: graph.nodes.map(
                  (n: any) => `${n.title}<br>Year: ${n.year}`,
                ),
              },
              ...graph.edges
                .map((e: any) => {
                  const fromIdx = graph.nodes.findIndex(
                    (n: any) => n.id === e.from,
                  );
                  const toIdx = graph.nodes.findIndex(
                    (n: any) => n.id === e.to,
                  );
                  if (fromIdx < 0 || toIdx < 0) return null;
                  return {
                    x: [
                      Math.cos((2 * Math.PI * fromIdx) / graph.nodes.length) *
                        100,
                      Math.cos((2 * Math.PI * toIdx) / graph.nodes.length) *
                        100,
                    ],
                    y: [
                      Math.sin((2 * Math.PI * fromIdx) / graph.nodes.length) *
                        100,
                      Math.sin((2 * Math.PI * toIdx) / graph.nodes.length) *
                        100,
                    ],
                    mode: "lines",
                    type: "scatter",
                    line: { color: "#ffffff30", width: 1 },
                    showlegend: false,
                  };
                })
                .filter(Boolean),
            ]}
            layout={{
              autosize: true,
              height: 500,
              showlegend: false,
              paper_bgcolor: "transparent",
              plot_bgcolor: "transparent",
              xaxis: { visible: false },
              yaxis: { visible: false },
              margin: { t: 10, r: 10, b: 10, l: 10 },
            }}
            style={{ width: "100%" }}
          />
        )}
      </Card>
    </div>
  );
}
