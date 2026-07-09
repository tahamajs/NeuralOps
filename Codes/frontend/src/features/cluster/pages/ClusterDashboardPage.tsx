import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";
import Plot from "react-plotly.js";
import { Cpu, Server } from "lucide-react";

export default function ClusterDashboardPage() {
  // Use existing load balancer endpoints
  const { data: gpuStatus } = useQuery({
    queryKey: ["gpuStatus"],
    queryFn: async () => (await api.get("/loadbalancer/gpu/status/")).data,
    refetchInterval: 5000,
  });
  const { data: gpuNodes } = useQuery({
    queryKey: ["gpuNodes"],
    queryFn: async () => (await api.get("/loadbalancer/gpu/nodes/")).data,
    refetchInterval: 5000,
  });
  const { data: schedulerStatus } = useQuery({
    queryKey: ["schedulerStatus"],
    queryFn: async () =>
      (await api.get("/loadbalancer/scheduler/status/")).data,
    refetchInterval: 5000,
  });

  const isLoading = !gpuStatus && !gpuNodes;
  if (isLoading) return <Spinner />;

  // Build node list from gpuNodes dict
  const nodes = gpuNodes
    ? Object.entries(gpuNodes).map(([hostname, gpus]: any) => ({
        hostname,
        gpus,
      }))
    : [];

  const totalGPUs = nodes.reduce(
    (sum: number, n: any) => sum + n.gpus.length,
    0,
  );

  // Heatmap data
  const heatmapZ = nodes.map((node: any) =>
    node.gpus.map((gpu: any) => gpu.utilization_pct || 0),
  );
  const heatmapY = nodes.map((n: any) => n.hostname);
  const heatmapX = nodes[0]?.gpus.map((_: any, i: number) => `GPU ${i}`) || [];

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-display font-bold text-white flex items-center gap-2">
            <Server size={24} className="text-neon-blue" /> GPU Cluster
          </h1>
          <p className="text-slate-400 text-sm">
            {nodes.length} nodes · {totalGPUs} GPUs
          </p>
        </div>
        <div className="flex gap-4">
          <Card className="px-4 py-2 text-center">
            <div className="text-xs text-slate-400">Active Jobs</div>
            <div className="text-xl font-bold text-white">
              {schedulerStatus?.running_jobs || 0}
            </div>
          </Card>
          <Card className="px-4 py-2 text-center">
            <div className="text-xs text-slate-400">Queued</div>
            <div className="text-xl font-bold text-white">
              {schedulerStatus?.queued_jobs || 0}
            </div>
          </Card>
        </div>
      </div>

      {heatmapZ.length > 0 && (
        <Card>
          <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
            <Cpu size={16} className="text-neon-blue" /> GPU Utilization Heatmap
          </h3>
          <Plot
            data={[
              {
                z: heatmapZ,
                y: heatmapY,
                x: heatmapX,
                type: "heatmap",
                colorscale: "RdYlBu_r",
                showscale: true,
                colorbar: { title: "Util %" },
              },
            ]}
            layout={{
              autosize: true,
              height: 200 + nodes.length * 30,
              margin: { t: 10, r: 20, b: 30, l: 100 },
              paper_bgcolor: "transparent",
              plot_bgcolor: "transparent",
              font: { color: "#94a3b8" },
            }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        </Card>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {nodes.map((node: any) => (
          <Card key={node.hostname} className="card-hover-effect">
            <h3 className="text-sm font-bold text-white mb-2">
              {node.hostname}
            </h3>
            <div className="grid grid-cols-2 gap-2">
              {node.gpus.map((gpu: any, i: number) => (
                <div key={i} className="bg-cyber-800 p-3 rounded-lg">
                  <div className="flex justify-between text-[10px] mb-1">
                    <span className="text-slate-400">GPU {i}</span>
                    <span className="text-neon-blue font-mono">
                      {gpu.utilization_pct || 0}%
                    </span>
                  </div>
                  <div className="text-[9px] text-slate-500">
                    Memory: {gpu.allocated_mb || 0}/{gpu.total_memory_mb || 0}{" "}
                    MB
                  </div>
                </div>
              ))}
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
