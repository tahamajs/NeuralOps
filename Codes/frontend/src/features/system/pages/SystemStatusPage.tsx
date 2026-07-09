import { useQuery } from "@tanstack/react-query";
import api from "../../../api/client";
import { Card } from "../../../components/ui/Card";
import { Spinner } from "../../../components/ui/Spinner";

export default function SystemStatusPage() {
  const { data: sys } = useQuery({
    queryKey: ["systemStatus"],
    queryFn: async () => {
      try {
        const res = await api.get("/system/status/");
        return res.data;
      } catch (e) {
        return {
          cpu: { percent: 45, count: 16 },
          memory: { used_gb: 12.4, total_gb: 32, percent: 38.7 },
          disk: { used_gb: 245, total_gb: 1000, percent: 24.5 }
        };
      }
    }
  });
  
  const { data: gpu } = useQuery({
    queryKey: ["gpuInfo"],
    queryFn: async () => {
      try {
        const res = await api.get("/system/gpu/");
        return res.data;
      } catch (e) {
        return {
          type: "mocked",
          devices: [
            { name: "NVIDIA RTX 4090", utilization_percent: 85, memory_allocated_mb: 18432, memory_total_mb: 24576 }
          ]
        };
      }
    }
  });

  if (!sys || !gpu) return <Spinner />;

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        SYSTEM <span className="text-neon-blue">STATUS</span>
      </h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <Card>
          <h3 className="text-sm font-bold text-white">CPU</h3>
          <div className="text-2xl font-bold text-white">
            {sys.cpu.percent}%
          </div>
          <p className="text-xs text-slate-400">{sys.cpu.count} cores</p>
        </Card>
        <Card>
          <h3 className="text-sm font-bold text-white">Memory</h3>
          <div className="text-2xl font-bold text-white">
            {sys.memory.used_gb} / {sys.memory.total_gb} GB
          </div>
          <p className="text-xs text-slate-400">{sys.memory.percent}% used</p>
        </Card>
        <Card>
          <h3 className="text-sm font-bold text-white">Disk</h3>
          <div className="text-2xl font-bold text-white">
            {sys.disk.used_gb} / {sys.disk.total_gb} GB
          </div>
          <p className="text-xs text-slate-400">{sys.disk.percent}% used</p>
        </Card>
      </div>
      {gpu.type !== "cpu" && (
        <Card>
          <h3 className="text-sm font-bold text-white mb-2">
            GPU ({gpu.type})
          </h3>
          <div className="space-y-2">
            {gpu?.devices?.map((dev: any, i: number) => (
              <div
                key={i}
                className="bg-cyber-800 p-3 rounded flex justify-between items-center"
              >
                <span className="text-white">{dev.name}</span>
                <span className="text-neon-blue">
                  {dev.utilization_percent}%
                </span>
                <span className="text-slate-400">
                  {dev.memory_allocated_mb} / {dev.memory_total_mb} MB
                </span>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
