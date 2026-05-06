import React, { createContext, useContext, useEffect, useState } from 'react';
import api from '../api/client';

export interface HardwareMetrics {
  cpu: number;
  memory: number;
  gpu: number;
}

interface HardwareContextType {
  metrics: HardwareMetrics;
  isLoading: boolean;
  error: string | null;
}

const HardwareContext = createContext<HardwareContextType | undefined>(undefined);

export const HardwareProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [metrics, setMetrics] = useState<HardwareMetrics>({
    cpu: 0,
    memory: 0,
    gpu: 0,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<HardwareMetrics | null>(null);

  useEffect(() => {
    const pollInterval = setInterval(async () => {
      try {
        setIsLoading(true);
        const [systemResponse, gpuResponse] = await Promise.all([
          api.get('/system/status/'),
          api.get('/system/gpu/'),
        ]);

        const cpu = systemResponse.data.cpu?.percent || 0;
        const memory = systemResponse.data.memory?.percent || 0;
        const gpu = gpuResponse.data.devices?.[0]?.utilization_percent || 0;

        // Dirty-check: Only update if values have changed by >1%
        if (
          !lastUpdate ||
          Math.abs(cpu - lastUpdate.cpu) > 1 ||
          Math.abs(memory - lastUpdate.memory) > 1 ||
          Math.abs(gpu - lastUpdate.gpu) > 1
        ) {
          setMetrics({ cpu, memory, gpu });
          setLastUpdate({ cpu, memory, gpu });
        }

        setError(null);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setIsLoading(false);
      }
    }, 3000); // Poll every 3 seconds

    return () => clearInterval(pollInterval);
  }, [lastUpdate]);

  return (
    <HardwareContext.Provider value={{ metrics, isLoading, error }}>
      {children}
    </HardwareContext.Provider>
  );
};

export const useHardware = (): HardwareContextType => {
  const context = useContext(HardwareContext);
  if (!context) {
    throw new Error('useHardware must be used within HardwareProvider');
  }
  return context;
};
