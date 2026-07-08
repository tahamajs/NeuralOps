import { useEffect, useState } from "react";
import api from "../../../api/client";

export default function SystemMonitorPanel() {
  const [sys, setSys] = useState<any>(null);
  const [gpu, setGpu] = useState<any>(null);

  useEffect(() => {
    const fetch = async () => {
      try {
        const [sysRes, gpuRes] = await Promise.all([
          api.get("/api/system/status/"),
          api.get("/api/system/gpu/"),
        ]);
        setSys(sysRes.data);
        setGpu(gpuRes.data);
      } catch {}
    };
    fetch();
    const interval = setInterval(fetch, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="p-4 space-y-4 bg-cyber-900 h-full overflow-y-auto">
      {sys && (
        <>
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-slate-400">
                CPU ({sys.cpu?.count} cores)
              </span>
              <span className="text-neon-blue font-mono">
                {sys.cpu?.percent}%
              </span>
            </div>
            <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
              <div
                className="h-full bg-neon-blue"
                style={{ width: `${sys.cpu?.percent}%` }}
              />
            </div>
          </div>
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-slate-400">Memory</span>
              <span className="text-neon-blue font-mono">
                {sys.memory?.used_gb}/{sys.memory?.total_gb} GB
              </span>
            </div>
            <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
              <div
                className="h-full bg-purple-500"
                style={{ width: `${sys.memory?.percent}%` }}
              />
            </div>
          </div>
          <div>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-slate-400">Disk</span>
              <span className="text-neon-blue font-mono">
                {sys.disk?.used_gb}/{sys.disk?.total_gb} GB
              </span>
            </div>
            <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
              <div
                className="h-full bg-cyan-500"
                style={{ width: `${sys.disk?.percent}%` }}
              />
            </div>
          </div>
          <div className="text-xs text-slate-500">
            Uptime: {sys.uptime_human} | Platform: {sys.platform}
          </div>
        </>
      )}
      {gpu &&
        gpu.type !== "cpu" &&
        gpu.devices?.map((dev: any, i: number) => (
          <div key={i} className="p-2 bg-cyber-800 rounded">
            <div className="flex justify-between text-xs mb-1">
              <span className="text-slate-400">{dev.name}</span>
              <span className="text-green-400 font-mono">
                {dev.utilization_percent}%
              </span>
            </div>
            <div className="text-[10px] text-slate-500 font-mono">
              VRAM: {dev.memory_allocated_mb}/{dev.memory_total_mb} MB
            </div>
          </div>
        ))}
    </div>
  );
}
