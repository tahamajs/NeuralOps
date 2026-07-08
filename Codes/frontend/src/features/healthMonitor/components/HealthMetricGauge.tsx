import { TerminalCard } from "../../../components/ui/TerminalCard";
import { PerformanceMetric } from "../types";

export function HealthMetricGauge({ metric }: { metric: PerformanceMetric }) {
  const value = metric.value;
  const color = value > 0.9 ? "#10b981" : value > 0.7 ? "#f59e0b" : "#ef4444";
  return (
    <TerminalCard className="text-center p-2">
      <div className="text-xs text-slate-400">{metric.metric_type}</div>
      <div className="text-lg font-bold text-white" style={{ color }}>
        {Math.round(value * 100)}%
      </div>
      <div className="w-full bg-white/10 rounded-full h-1 mt-1">
        <div
          className="h-1 rounded-full"
          style={{ width: `${value * 100}%`, backgroundColor: color }}
        />
      </div>
    </TerminalCard>
  );
}
