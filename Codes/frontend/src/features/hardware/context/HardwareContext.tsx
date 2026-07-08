import { createContext, useContext, useEffect, useState } from "react";
import api from "../../../api/client";

interface HardwareMetrics {
  cpu: { percent: number };
  memory: { used_gb: number; total_gb: number; percent: number };
  gpu?: { type: string; devices: any[] };
}

const HardwareContext = createContext<HardwareMetrics | null>(null);

export const HardwareProvider = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  const [metrics, setMetrics] = useState<HardwareMetrics | null>(null);
  useEffect(() => {
    const fetch = async () => {
      try {
        const [sys, gpu] = await Promise.all([
          api.get("/system/status/"),
          api.get("/system/gpu/"),
        ]);
        setMetrics({
          cpu: sys.data.cpu,
          memory: sys.data.memory,
          gpu: gpu.data,
        });
      } catch {}
    };
    fetch();
    const interval = setInterval(fetch, 3000);
    return () => clearInterval(interval);
  }, []);
  return (
    <HardwareContext.Provider value={metrics}>
      {children}
    </HardwareContext.Provider>
  );
};
export const useHardware = () => useContext(HardwareContext);
