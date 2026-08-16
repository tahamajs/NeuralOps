import React, { useEffect, useState } from 'react';
import {
  Cpu,
  Thermometer,
  Fan,
  Zap,
  HardDrive,
  Activity,
  RefreshCw,
  Sliders,
} from 'lucide-react';
import { ResponsiveContainer, RadialBarChart, RadialBar } from 'recharts';

export default function GPUDetails() {
  const [gpus, setGpus] = useState([]);
  const [useSimulation, setUseSimulation] = useState(false);

  const fetchGpu = async () => {
    if (window.neuralopsAPI) {
      try {
        const gpuInfo = await window.neuralopsAPI.getGpuInfo();
        if (gpuInfo && gpuInfo.length > 0) {
          setGpus(gpuInfo);
          return;
        }
      } catch (e) {
        console.warn('Error fetching GPU info:', e);
      }
    }

    if (useSimulation) {
      setGpus([
        {
          index: 0,
          name: 'NVIDIA GeForce RTX 4090',
          vramMb: 24576,
          utilPct: 68,
          tempC: 62,
          fanPct: 45,
          powerW: 285.5,
        },
      ]);
    } else {
      setGpus([]);
    }
  };

  useEffect(() => {
    fetchGpu();
    const interval = setInterval(fetchGpu, 3000);
    return () => clearInterval(interval);
  }, [useSimulation]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white">GPU Hardware Telemetry</h2>
          <p className="text-sm text-gray-400 mt-1">
            Real-time thermal sensors, VRAM allocation, clock speeds, and power metrics.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => {
              setUseSimulation(!useSimulation);
            }}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-xl border text-xs font-semibold transition-colors ${
              useSimulation
                ? 'border-primary-500/40 bg-primary-600/20 text-primary-300'
                : 'border-dark-border bg-dark-card text-gray-400 hover:text-white'
            }`}
          >
            <Sliders size={14} />
            {useSimulation ? 'Simulation Mode (Active)' : 'Preview Mode'}
          </button>

          <button
            onClick={fetchGpu}
            title="Refresh Sensors"
            className="p-2.5 rounded-xl border border-dark-border bg-dark-card hover:bg-dark-border text-gray-300 hover:text-white transition-colors"
          >
            <RefreshCw size={16} />
          </button>
        </div>
      </div>

      {gpus.length === 0 && (
        <div className="bg-dark-card border border-dark-border rounded-2xl p-12 text-center shadow-xl">
          <div className="w-16 h-16 rounded-2xl bg-dark-bg border border-dark-border flex items-center justify-center mx-auto mb-4 text-gray-500">
            <Cpu size={32} />
          </div>
          <h3 className="text-lg font-bold text-white">No NVIDIA GPU Detected</h3>
          <p className="text-gray-400 text-sm mt-1 max-w-md mx-auto">
            The agent queries <code className="text-primary-300 font-mono">nvidia-smi</code> for hardware telemetry. Ensure NVIDIA proprietary drivers and CUDA runtime are installed.
          </p>
          <button
            onClick={() => setUseSimulation(true)}
            className="mt-5 px-4 py-2 rounded-xl bg-primary-600 hover:bg-primary-700 text-white font-medium text-xs transition-colors"
          >
            Enable Preview Simulation
          </button>
        </div>
      )}

      <div className="space-y-6">
        {gpus.map((gpu, idx) => {
          const gaugeData = [
            {
              name: 'Usage',
              value: gpu.utilPct,
              fill: gpu.utilPct > 80 ? '#ef4444' : gpu.utilPct > 50 ? '#f59e0b' : '#22c55e',
            },
          ];

          return (
            <div key={idx} className="bg-dark-card border border-dark-border rounded-2xl p-6 shadow-xl space-y-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-dark-border pb-4">
                <div>
                  <div className="flex items-center gap-2">
                    <h3 className="text-xl font-bold text-white">{gpu.name}</h3>
                    <span className="text-xs font-mono bg-dark-bg border border-dark-border px-2 py-0.5 rounded text-gray-300">
                      PCIe GPU #{gpu.index ?? idx}
                    </span>
                  </div>
                  <p className="text-xs text-gray-400 mt-1">
                    {(gpu.vramMb / 1024).toFixed(1)} GB Total VRAM · Unified Virtual Memory Enabled
                  </p>
                </div>

                <div
                  className={`px-3 py-1 rounded-full text-xs font-semibold self-start sm:self-auto border ${
                    gpu.utilPct > 80
                      ? 'bg-rose-500/10 text-rose-400 border-rose-500/20'
                      : gpu.utilPct > 50
                      ? 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                      : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                  }`}
                >
                  {gpu.utilPct > 80 ? 'Heavy Load' : gpu.utilPct > 50 ? 'Active Compute' : 'Idle / Standby'}
                </div>
              </div>

              {/* Sensor Metric Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
                {[
                  {
                    icon: Activity,
                    label: 'GPU Core Load',
                    value: `${gpu.utilPct}%`,
                    color: 'text-blue-400',
                    bg: 'bg-blue-500/10',
                  },
                  {
                    icon: HardDrive,
                    label: 'Frame Buffer (VRAM)',
                    value: `${gpu.vramMb} MB`,
                    color: 'text-purple-400',
                    bg: 'bg-purple-500/10',
                  },
                  {
                    icon: Thermometer,
                    label: 'Die Temperature',
                    value: `${gpu.tempC}°C`,
                    color: gpu.tempC > 78 ? 'text-rose-400' : 'text-emerald-400',
                    bg: gpu.tempC > 78 ? 'bg-rose-500/10' : 'bg-emerald-500/10',
                  },
                  {
                    icon: Fan,
                    label: 'Fan Tachometer',
                    value: `${gpu.fanPct}%`,
                    color: 'text-cyan-400',
                    bg: 'bg-cyan-500/10',
                  },
                  {
                    icon: Zap,
                    label: 'Power Envelope',
                    value: `${gpu.powerW}W`,
                    color: 'text-amber-400',
                    bg: 'bg-amber-500/10',
                  },
                ].map(({ icon: Icon, label, value, color, bg }) => (
                  <div key={label} className="bg-dark-bg/70 border border-dark-border rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <div className={`p-1.5 rounded-lg ${bg} ${color}`}>
                        <Icon size={16} />
                      </div>
                      <span className="text-gray-400 text-xs font-medium">{label}</span>
                    </div>
                    <p className="text-xl font-bold text-white">{value}</p>
                  </div>
                ))}
              </div>

              {/* Gauge & Progress */}
              <div className="flex flex-col sm:flex-row items-center justify-between gap-6 bg-dark-bg/40 border border-dark-border/60 rounded-xl p-4">
                <div className="flex-1 w-full space-y-3">
                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-gray-400">Core Utilization</span>
                      <span className="font-semibold text-white">{gpu.utilPct}%</span>
                    </div>
                    <div className="w-full h-2 bg-dark-border rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary-500 rounded-full transition-all duration-500"
                        style={{ width: `${gpu.utilPct}%` }}
                      />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-gray-400">Thermal Threshold (Target: &lt;80°C)</span>
                      <span className="font-semibold text-white">{gpu.tempC}°C</span>
                    </div>
                    <div className="w-full h-2 bg-dark-border rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          gpu.tempC > 78 ? 'bg-rose-500' : 'bg-emerald-500'
                        }`}
                        style={{ width: `${Math.min(100, (gpu.tempC / 90) * 100)}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="w-28 h-28 shrink-0 flex items-center justify-center">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadialBarChart
                      cx="50%"
                      cy="50%"
                      innerRadius="65%"
                      outerRadius="95%"
                      barSize={8}
                      data={gaugeData}
                      startAngle={90}
                      endAngle={-270}
                    >
                      <RadialBar dataKey="value" cornerRadius={6} />
                    </RadialBarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}