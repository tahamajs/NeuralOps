import React, { useEffect, useState } from 'react';
import { Cpu, Thermometer, Fan, Zap, HardDrive, Activity } from 'lucide-react';
import { RadialBarChart, RadialBar, Legend, ResponsiveContainer } from 'recharts';

export default function GPUDetails() {
  const [gpus, setGpus] = useState([]);

  useEffect(() => {
    const fetchGpu = async () => {
      if (window.neuralopsAPI) {
        const gpuInfo = await window.neuralopsAPI.getGpuInfo();
        setGpus(gpuInfo);
      }
    };
    fetchGpu();
    const interval = setInterval(fetchGpu, 5000);
    return () => clearInterval(interval);
  }, []);

  if (gpus.length === 0) {
    return (
      <div className="space-y-4">
        <h2 className="text-2xl font-bold">GPU Details</h2>
        <div className="bg-dark-card border border-dark-border rounded-xl p-12 text-center">
          <Cpu size={48} className="mx-auto text-gray-600 mb-4" />
          <p className="text-gray-400 text-lg">No NVIDIA GPU detected</p>
          <p className="text-gray-500 text-sm mt-1">Install NVIDIA drivers and restart the agent</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">GPU Details</h2>
      {gpus.map((gpu, idx) => {
        const gaugeData = [{ name: 'Usage', value: gpu.utilPct, fill: gpu.utilPct > 80 ? '#ef4444' : gpu.utilPct > 50 ? '#f59e0b' : '#22c55e' }];
        return (
          <div key={idx} className="bg-dark-card border border-dark-border rounded-xl p-6">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h3 className="text-xl font-bold">{gpu.name}</h3>
                <p className="text-gray-400 text-sm">GPU {idx} · {gpu.vramMb} MB VRAM</p>
              </div>
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${gpu.utilPct > 80 ? 'bg-red-400/10 text-red-400' : gpu.utilPct > 50 ? 'bg-orange-400/10 text-orange-400' : 'bg-green-400/10 text-green-400'}`}>
                {gpu.utilPct > 80 ? 'High Load' : gpu.utilPct > 50 ? 'Moderate' : 'Idle'}
              </div>
            </div>

            <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
              {[
                { icon: Activity, label: 'GPU Usage', value: `${gpu.utilPct}%`, color: 'text-blue-400', bg: 'bg-blue-400/10' },
                { icon: HardDrive, label: 'VRAM', value: `${gpu.vramMb} MB`, color: 'text-purple-400', bg: 'bg-purple-400/10' },
                { icon: Thermometer, label: 'Temperature', value: `${gpu.tempC}°C`, color: gpu.tempC > 75 ? 'text-red-400' : 'text-green-400', bg: (gpu.tempC > 75 ? 'bg-red-400' : 'bg-green-400') + '/10' },
                { icon: Fan, label: 'Fan Speed', value: `${gpu.fanPct}%`, color: 'text-cyan-400', bg: 'bg-cyan-400/10' },
                { icon: Zap, label: 'Power Draw', value: `${gpu.powerW}W`, color: 'text-yellow-400', bg: 'bg-yellow-400/10' },
              ].map(({ icon: Icon, label, value, color, bg }) => (
                <div key={label} className={`${bg} rounded-xl p-4`}>
                  <div className="flex items-center gap-2 mb-2"><Icon size={16} className={color} /><span className="text-gray-400 text-xs">{label}</span></div>
                  <p className="text-xl font-bold">{value}</p>
                </div>
              ))}
            </div>

            <div className="mt-6 flex justify-center">
              <ResponsiveContainer width={200} height={200}>
                <RadialBarChart cx="50%" cy="50%" innerRadius="60%" outerRadius="90%" barSize={12} data={gaugeData} startAngle={90} endAngle={-270}>
                  <RadialBar dataKey="value" cornerRadius={8} />
                </RadialBarChart>
              </ResponsiveContainer>
            </div>
          </div>
        );
      })}
    </div>
  );
}