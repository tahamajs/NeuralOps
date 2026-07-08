import { useHardware } from "../context/HardwareContext";
import { Card } from "../../../components/ui/Card";

export const SystemResourcesWidget = () => {
  const { metrics, error } = useHardware();

  if (error) return null;

  return (
    <Card className="space-y-4" hover={false}>
      <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-widest flex items-center gap-2">
        <div className="w-1.5 h-1.5 rounded-full bg-neon-blue" />
        Hardware Telemetry
      </h3>

      <div className="space-y-3">
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-400">CPU</span>
            <span className="font-mono">{metrics.cpu.toFixed(1)}%</span>
          </div>
          <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
            <div
              className="h-full bg-neon-blue transition-all duration-500"
              style={{ width: `${metrics.cpu}%` }}
            />
          </div>
        </div>
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-400">Memory</span>
            <span className="font-mono">{metrics.memory.toFixed(1)}%</span>
          </div>
          <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
            <div
              className="h-full bg-purple-500 transition-all duration-500"
              style={{ width: `${metrics.memory}%` }}
            />
          </div>
        </div>
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span className="text-slate-400">GPU</span>
            <span className="font-mono">{metrics.gpu.toFixed(1)}%</span>
          </div>
          <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
            <div
              className="h-full bg-green-500 transition-all duration-500"
              style={{ width: `${metrics.gpu}%` }}
            />
          </div>
        </div>
      </div>
    </Card>
  );
};
