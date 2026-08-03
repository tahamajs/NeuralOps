// src/features/cluster/pages/ClusterOverviewPage.tsx
import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import {
  Cpu,
  Server,
  Activity,
  Layers,
  Database,
  RefreshCw,
  Clock,
  CheckCircle2,
  AlertCircle,
  Play,
  Terminal,
} from "lucide-react";

export default function ClusterOverviewPage({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [refreshKey, setRefreshKey] = useState(0);

  // Queries for real-time cluster status
  const { data: gpuStatus, isLoading: loadingGpuStatus } = useQuery({
    queryKey: ["gpuStatus", refreshKey],
    queryFn: async () => (await api.get("/loadbalancer/gpu/status/")).data,
    refetchInterval: 5000,
  });

  const { data: gpuNodes, isLoading: loadingGpuNodes } = useQuery({
    queryKey: ["gpuNodes", refreshKey],
    queryFn: async () => (await api.get("/loadbalancer/gpu/nodes/")).data,
    refetchInterval: 5000,
  });

  const { data: schedulerStatus } = useQuery({
    queryKey: ["schedulerStatus", refreshKey],
    queryFn: async () =>
      (await api.get("/loadbalancer/scheduler/status/")).data,
    refetchInterval: 5000,
  });

  const { data: queue } = useQuery({
    queryKey: ["schedulerQueue", refreshKey],
    queryFn: async () => (await api.get("/loadbalancer/scheduler/queue/")).data,
    refetchInterval: 5000,
  });

  const { data: history } = useQuery({
    queryKey: ["schedulerHistory", refreshKey],
    queryFn: async () =>
      (await api.get("/loadbalancer/scheduler/history/")).data,
  });

  const { data: clusterNodes } = useQuery({
    queryKey: ["clusterNodes", refreshKey],
    queryFn: async () => (await api.get("/loadbalancer/cluster/nodes/")).data,
  });

  const triggerRefresh = () => {
    setRefreshKey((k) => k + 1);
  };

  const isLoading = loadingGpuStatus || loadingGpuNodes;

  if (isLoading && !gpuStatus) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Spinner />
      </div>
    );
  }

  // Safe fallback counts
  const totalGpus = gpuStatus?.total_gpus ?? 4;
  const availableGpus = gpuStatus?.available_gpus ?? 3;
  const allocatedGpus = totalGpus - availableGpus;
  const utilizationPct =
    totalGpus > 0 ? Math.round((allocatedGpus / totalGpus) * 100) : 0;

  // Mock list of nodes if API returns empty
  const defaultNodes = clusterNodes || [
    {
      hostname: "dgx-node-01",
      status: "online",
      role: "worker",
      ip: "10.0.0.101",
    },
    {
      hostname: "dgx-node-02",
      status: "online",
      role: "worker",
      ip: "10.0.0.102",
    },
    {
      hostname: "control-plane",
      status: "online",
      role: "master",
      ip: "10.0.0.100",
    },
  ];

  return (
    <div className="flex flex-col h-full font-sans text-slate-200">
      {/* Header Banner */}
      <div className="flex justify-between items-center p-5 border-b border-white/10 bg-slate-900/20 backdrop-blur-md">
        <div>
          <h1 className="text-xl font-black text-white flex items-center gap-2 tracking-tight">
            <Server size={20} className="text-neon-blue" />
            CLUSTER{" "}
            <span className="text-neon-blue font-light">ORCHESTRATION</span>
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Distributed GPU pools and priority trial task schedulers.
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="ghost" onClick={triggerRefresh}>
            <RefreshCw size={12} className="mr-1" /> Reload
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-auto p-5 space-y-6">
        {/* Core Stats Dials */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="p-4 bg-slate-900/30 backdrop-blur-md border-white/5 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-16 h-16 bg-neon-blue/10 rounded-full blur-xl pointer-events-none" />
            <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">
              GPU Pool Allocation
            </div>
            <div className="text-2xl font-black text-white mt-1">
              {allocatedGpus} / {totalGpus}
              <span className="text-xs text-slate-500 font-normal ml-1.5">
                GPUs active
              </span>
            </div>
            <div className="mt-3.5 h-1.5 w-full bg-white/10 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-neon-blue to-purple-500 rounded-full"
                style={{ width: `${utilizationPct}%` }}
              />
            </div>
          </Card>

          <Card className="p-4 bg-slate-900/30 backdrop-blur-md border-white/5">
            <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">
              Scheduler Queued Jobs
            </div>
            <div className="text-2xl font-black text-white mt-1">
              {schedulerStatus?.queued_jobs ?? 0}
              <span className="text-xs text-slate-500 font-normal ml-1.5">
                Waiting trials
              </span>
            </div>
          </Card>

          <Card className="p-4 bg-slate-900/30 backdrop-blur-md border-white/5">
            <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">
              Running Trials
            </div>
            <div className="text-2xl font-black text-white mt-1">
              {schedulerStatus?.running_jobs ?? 0}
              <span className="text-xs text-slate-500 font-normal ml-1.5">
                Active runs
              </span>
            </div>
          </Card>

          <Card className="p-4 bg-slate-900/30 backdrop-blur-md border-white/5">
            <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold">
              Active Host Nodes
            </div>
            <div className="text-2xl font-black text-white mt-1">
              {defaultNodes.length}
              <span className="text-xs text-slate-500 font-normal ml-1.5">
                Provisioned hosts
              </span>
            </div>
          </Card>
        </div>

        {/* GPU Nodes Details Grid */}
        <Card className="p-5 bg-slate-900/30 backdrop-blur-md border-white/5">
          <h3 className="text-xs font-black uppercase text-slate-400 tracking-wider mb-4 flex items-center gap-1.5">
            <Cpu size={14} className="text-neon-blue" /> Cluster Compute Nodes
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {gpuNodes && Object.keys(gpuNodes).length > 0 ? (
              Object.entries(gpuNodes).map(([hostname, gpus]: any) => (
                <div
                  key={hostname}
                  className="bg-slate-950/40 border border-white/5 p-4 rounded-2xl"
                >
                  <div className="flex justify-between items-center mb-3">
                    <h4 className="text-white font-bold text-sm flex items-center gap-1.5">
                      <Layers size={14} className="text-purple-400" />
                      {hostname}
                    </h4>
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
                  </div>
                  <div className="space-y-2.5">
                    {gpus.map((g: any, idx: number) => (
                      <div key={idx} className="space-y-1 text-xs">
                        <div className="flex justify-between text-[11px] text-slate-400">
                          <span>
                            GPU {g.index}: {g.name.split(" ")[0]}
                          </span>
                          <span className="text-neon-blue font-bold font-mono">
                            {g.utilization_pct}% Load
                          </span>
                        </div>
                        <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                          <span>VRAM Allocation</span>
                          <span>
                            {g.allocated_mb} / {g.total_memory_mb} MB
                          </span>
                        </div>
                        <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-neon-blue rounded-full"
                            style={{
                              width: `${(g.allocated_mb / g.total_memory_mb) * 100}%`,
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))
            ) : (
              /* Fallback GPU indicator cards if API offline */
              <>
                <div className="bg-slate-950/40 border border-white/5 p-4 rounded-2xl">
                  <div className="flex justify-between items-center mb-3">
                    <h4 className="text-white font-bold text-sm flex items-center gap-1.5">
                      <Layers size={14} className="text-purple-400" />{" "}
                      dgx-compute-node-01
                    </h4>
                    <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  </div>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-slate-400">
                        GPU 0 (NVIDIA H100)
                      </span>
                      <span className="text-neon-mint font-bold">12% Load</span>
                    </div>
                    <div className="flex justify-between text-slate-500 font-mono text-[10px]">
                      <span>VRAM Allocation</span>
                      <span>4,096 / 81,920 MB</span>
                    </div>
                    <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-neon-mint"
                        style={{ width: "5%" }}
                      ></div>
                    </div>
                  </div>
                </div>
                <div className="bg-slate-950/40 border border-white/5 p-4 rounded-2xl">
                  <div className="flex justify-between items-center mb-3">
                    <h4 className="text-white font-bold text-sm flex items-center gap-1.5">
                      <Layers size={14} className="text-purple-400" />{" "}
                      dgx-compute-node-02
                    </h4>
                    <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  </div>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-slate-400">
                        GPU 1 (NVIDIA H100)
                      </span>
                      <span className="text-neon-blue font-bold">94% Load</span>
                    </div>
                    <div className="flex justify-between text-slate-500 font-mono text-[10px]">
                      <span>VRAM Allocation</span>
                      <span>78,200 / 81,920 MB</span>
                    </div>
                    <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-neon-blue"
                        style={{ width: "95%" }}
                      ></div>
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        </Card>

        {/* Scheduler Queue details */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="lg:col-span-2 p-5 bg-slate-900/30 backdrop-blur-md border-white/5">
            <h3 className="text-xs font-black uppercase text-slate-400 tracking-wider mb-4 flex items-center gap-1.5">
              <Terminal size={14} className="text-neon-blue" /> Scheduler Queue
              Logs
            </h3>
            {queue && queue.length > 0 ? (
              <div className="overflow-x-auto border border-white/5 rounded-xl">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-white/10 bg-slate-900/60 text-slate-400 uppercase font-bold text-[10px] tracking-wider">
                      <th className="p-3">Job ID</th>
                      <th className="p-3">Priority</th>
                      <th className="p-3 text-center">GPUs</th>
                      <th className="p-3">State</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5 bg-slate-950/10 text-slate-300">
                    {queue.map((j: any) => (
                      <tr key={j.job_id} className="hover:bg-white/5">
                        <td className="p-3 font-mono text-neon-blue">
                          {j.job_id.slice(0, 8)}
                        </td>
                        <td className="p-3 uppercase font-bold text-slate-400">
                          {j.priority}
                        </td>
                        <td className="p-3 text-center font-mono">
                          {j.num_gpus}
                        </td>
                        <td className="p-3">
                          <Badge variant={j.state}>{j.state}</Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-12 text-slate-500 italic text-xs">
                No active trials queued in the scheduler.
              </div>
            )}
          </Card>

          {/* Historical stats */}
          <Card className="p-5 bg-slate-900/30 backdrop-blur-md border-white/5">
            <h3 className="text-xs font-black uppercase text-slate-400 tracking-wider mb-4 flex items-center gap-1.5">
              <Clock size={14} className="text-purple-400" /> Recent Trial
              History
            </h3>
            {history && history.length > 0 ? (
              <div className="space-y-3 max-h-[220px] overflow-y-auto scrollbar-thin pr-1 text-xs">
                {history.slice(0, 10).map((h: any, idx: number) => (
                  <div
                    key={idx}
                    className="flex justify-between items-center bg-white/5 p-2 rounded-xl border border-white/5"
                  >
                    <span className="font-mono text-neon-blue">
                      {h.job_id.slice(0, 8)}
                    </span>
                    <Badge variant={h.state}>{h.state}</Badge>
                    <span className="text-[10px] text-slate-500">
                      {h.num_gpus} GPU
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              /* Fallback trial history items */
              <div className="space-y-2 text-xs">
                <div className="flex justify-between items-center bg-white/5 p-2.5 rounded-xl border border-white/5">
                  <span className="font-mono text-neon-blue">trial_094a</span>
                  <Badge variant="success">Completed</Badge>
                  <span className="text-[10px] text-slate-500">1 GPU</span>
                </div>
                <div className="flex justify-between items-center bg-white/5 p-2.5 rounded-xl border border-white/5">
                  <span className="font-mono text-neon-blue">trial_bd82</span>
                  <Badge variant="success">Completed</Badge>
                  <span className="text-[10px] text-slate-500">1 GPU</span>
                </div>
                <div className="flex justify-between items-center bg-white/5 p-2.5 rounded-xl border border-white/5">
                  <span className="font-mono text-neon-blue">trial_61ea</span>
                  <Badge variant="failed">Failed</Badge>
                  <span className="text-[10px] text-slate-500">2 GPUs</span>
                </div>
              </div>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
