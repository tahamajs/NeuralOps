import { useEffect, useState } from "react";
import api from "../api/client";

export default function SystemResourcesWidget() {
  const [sys, setSys] = useState<any>(null);
  const [gpu, setGpu] = useState<any>(null);

  useEffect(() => {
    const fetch = async () => {
      try {
        const [sysRes, gpuRes] = await Promise.all([
          api.get("/system/status/"),
          api.get("/system/gpu/"),
        ]);
        setSys(sysRes.data);
        setGpu(gpuRes.data);
      } catch {}
    };
    fetch();
    const interval = setInterval(fetch, 3000);
    return () => clearInterval(interval);
  }, []);

  if (!sys) return null;

  return (
    <div className="glass-card">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] flex items-center gap-2">
          <div className="w-1 h-1 rounded-full bg-blue-500" />
          Hardware Telemetry
        </h2>
        <span className="text-[10px] font-mono text-blue-400">
          REFRESH_3000MS
        </span>
      </div>

      <div className="space-y-6">
        <div className="space-y-2">
          <div className="flex justify-between text-xs font-bold">
            <span className="text-slate-400">vCPU LOAD</span>
            <span className="text-white">{sys.cpu?.percent}%</span>
          </div>
          <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-500"
              style={{ width: `${sys.cpu?.percent}%` }}
            />
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between text-xs font-bold">
            <span className="text-slate-400">MEMORY_ALLOC</span>
            <span className="text-white">
              {sys.memory?.used_gb} / {sys.memory?.total_gb} GB
            </span>
          </div>
          <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
            <div
              className="h-full bg-indigo-500 transition-all duration-500"
              style={{ width: `${sys.memory?.percent}%` }}
            />
          </div>
        </div>

        {gpu && gpu.type !== "cpu" && (
          <div className="pt-4 border-t border-white/5 space-y-4">
            {gpu.devices?.map((dev: any, i: number) => (
              <div key={i} className="space-y-2">
                <div className="flex justify-between text-xs font-bold">
                  <span className="text-slate-500">
                    GPU_{i} ({dev.name.split(" ")[0]})
                  </span>
                  <span className="text-blue-400">
                    {dev.utilization_percent}%
                  </span>
                </div>
                <div className="flex justify-between text-[10px] font-mono text-slate-600">
                  <span>VRAM</span>
                  <span>
                    {dev.memory_allocated_mb} / {dev.memory_total_mb} MB
                  </span>
                </div>
              </div>
            ))}
            {gpu.type === "mps" && (
              <div className="text-xs font-bold flex justify-between">
                <span className="text-slate-500">METAL CORE (MPS)</span>
                <span className="text-green-400">ACTIVE</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
