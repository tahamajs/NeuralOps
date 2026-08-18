// src/features/experimentExplorer/components/CorrelationMatrix.tsx
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { ParameterMetricCorrelation } from "../types";

export function CorrelationMatrix({
  correlations = [],
  targetMetric,
}: {
  correlations: ParameterMetricCorrelation[];
  targetMetric: string;
}) {
  const correlationList = Array.isArray(correlations) ? correlations : [];

  if (!correlationList.length)
    return (
      <div className="flex items-center justify-center h-full text-center text-slate-500 py-12 text-sm">
        Run more experiments to see correlation data for {targetMetric || "selected metric"}.
      </div>
    );

  const getColorClass = (value: number) => {
    if (isNaN(value)) return "text-slate-400 bg-white/5";
    if (value > 0.7) return "text-green-400 bg-green-500/10";
    if (value > 0.3) return "text-yellow-400 bg-yellow-500/10";
    if (value < -0.7) return "text-red-400 bg-red-500/10";
    if (value < -0.3) return "text-orange-400 bg-orange-500/10";
    return "text-slate-400 bg-white/5";
  };

  return (
    <div className="space-y-2 max-h-[450px] overflow-y-auto p-1">
      {correlationList.map((c, idx) => {
        const val = typeof c?.correlation === "number" && !isNaN(c.correlation) ? c.correlation : 0;
        const widthPercent = Math.min(100, Math.max(0, Math.abs(val) * 100));

        return (
          <div
            key={c?.parameter || idx}
            className="flex justify-between items-center p-2 rounded hover:bg-white/5 transition-colors"
          >
            <span className="text-sm text-white font-medium">{c?.parameter || `Param ${idx + 1}`}</span>
            <div className="flex items-center gap-2">
              <div className="w-32 h-2 bg-white/10 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${widthPercent}%`,
                    backgroundColor: val >= 0 ? "#3b82f6" : "#ef4444",
                  }}
                />
              </div>
              <span
                className={`px-2 py-0.5 rounded text-xs font-mono min-w-[56px] text-right ${getColorClass(val)}`}
              >
                {val > 0 ? "+" : ""}
                {val.toFixed(3)}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
