import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";

export default function ClusterOverviewPage() {
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
  const { data: queue } = useQuery({
    queryKey: ["schedulerQueue"],
    queryFn: async () => (await api.get("/loadbalancer/scheduler/queue/")).data,
    refetchInterval: 5000,
  });
  const { data: history } = useQuery({
    queryKey: ["schedulerHistory"],
    queryFn: async () =>
      (await api.get("/loadbalancer/scheduler/history/")).data,
  });
  const { data: clusterNodes } = useQuery({
    queryKey: ["clusterNodes"],
    queryFn: async () => (await api.get("/loadbalancer/cluster/nodes/")).data,
  });

  if (!gpuStatus) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        CLUSTER <span className="text-neon-blue">ORCHESTRATION</span>
      </h1>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <div className="text-xs text-slate-500 uppercase tracking-wider">
            GPUs
          </div>
          <div className="text-2xl font-bold text-white">
            {gpuStatus.total_gpus}{" "}
            <span className="text-neon-blue text-sm">
              / {gpuStatus.available_gpus} free
            </span>
          </div>
        </Card>
        <Card>
          <div className="text-xs text-slate-500 uppercase tracking-wider">
            Queue
          </div>
          <div className="text-2xl font-bold text-white">
            {schedulerStatus?.queued_jobs}
          </div>
        </Card>
        <Card>
          <div className="text-xs text-slate-500 uppercase tracking-wider">
            Running
          </div>
          <div className="text-2xl font-bold text-white">
            {schedulerStatus?.running_jobs}
          </div>
        </Card>
        <Card>
          <div className="text-xs text-slate-500 uppercase tracking-wider">
            Nodes
          </div>
          <div className="text-2xl font-bold text-white">
            {clusterNodes?.length}
          </div>
        </Card>
      </div>

      {/* GPU Nodes Detail */}
      {gpuNodes && (
        <Card>
          <h3 className="text-sm font-bold text-white mb-3">GPU Nodes</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(gpuNodes).map(([hostname, gpus]: any) => (
              <div key={hostname} className="bg-cyber-800 p-3 rounded-lg">
                <h4 className="text-white font-semibold text-sm mb-2">
                  {hostname}
                </h4>
                <div className="space-y-1">
                  {gpus.map((g: any, i: number) => (
                    <div key={i} className="flex justify-between text-xs">
                      <span className="text-slate-400">
                        GPU {g.index} ({g.name.split(" ")[0]})
                      </span>
                      <span className="text-neon-blue">
                        {g.utilization_pct}%
                      </span>
                      <span className="text-slate-500">
                        {g.allocated_mb}/{g.total_memory_mb} MB
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Job Queue */}
      {queue && (
        <Card>
          <h3 className="text-sm font-bold text-white mb-3">Job Queue</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-slate-400 uppercase tracking-wider border-b border-white/5">
                  <th className="py-2 px-3">Job ID</th>
                  <th className="py-2 px-3">Priority</th>
                  <th className="py-2 px-3">GPUs</th>
                  <th className="py-2 px-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {queue.map((j: any) => (
                  <tr key={j.job_id}>
                    <td className="py-2 px-3 font-mono text-neon-blue">
                      {j.job_id.slice(0, 8)}
                    </td>
                    <td className="py-2 px-3">{j.priority}</td>
                    <td className="py-2 px-3">{j.num_gpus}</td>
                    <td className="py-2 px-3">
                      <Badge status={j.state}>{j.state}</Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* History */}
      {history && (
        <Card>
          <h3 className="text-sm font-bold text-white mb-3">Recent History</h3>
          <ul className="space-y-1 max-h-64 overflow-y-auto scrollbar-thin">
            {history.map((h: any, idx: number) => (
              <li
                key={idx}
                className="text-xs text-slate-400 flex justify-between"
              >
                <span>{h.job_id.slice(0, 8)}</span>
                <Badge status={h.state}>{h.state}</Badge>
                <span>{h.num_gpus} GPU</span>
                <span>{new Date(h.start_time).toLocaleTimeString()}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}
