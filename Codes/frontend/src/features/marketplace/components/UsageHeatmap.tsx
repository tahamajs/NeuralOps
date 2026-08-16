import React from "react";
import { Card } from "../../../components/ui/Card";

export function UsageHeatmap({ usage, data = [] }: any) {
  // Generate 7x24 heatmap data based on usage details or simulated values
  const heatmapData = React.useMemo(() => {
    if (data && data.length) return data;

    const matrix = Array.from({ length: 7 }, () =>
      Array.from({ length: 24 }, () => 0.0),
    );
    for (let day = 0; day < 7; day++) {
      for (let hour = 0; hour < 24; hour++) {
        let val = 0.25 + Math.random() * 0.25;
        // Busy peak hours (10 AM to 8 PM)
        if (hour >= 10 && hour <= 20) {
          val += 0.35;
        }
        // Weekday versus weekend differentiation
        if (day === 5 || day === 6) {
          val -= 0.2;
        }
        matrix[day][hour] = Math.max(0.05, Math.min(0.95, val));
      }
    }
    return matrix;
  }, [data, usage]);

  const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

  return (
    <Card className="glass border-white/5 p-6">
      <h3 className="text-lg font-semibold text-white mb-4">
        Compute Usage Heatmap
      </h3>
      <div className="flex">
        {/* Y Axis Labels */}
        <div className="flex flex-col justify-between pr-4 py-1 text-xs text-slate-500 font-medium h-[200px]">
          {days.map((day) => (
            <span key={day}>{day}</span>
          ))}
        </div>

        {/* Heatmap Grid */}
        <div className="flex-1 flex flex-col justify-between h-[200px]">
          {heatmapData.map((dayRow: any, i: number) => (
            <div key={i} className="flex flex-1 gap-1 pb-1">
              {dayRow.map((val: number, j: number) => {
                // Color intensity based on value 0-1
                const intensity = Math.floor(val * 10);
                const opacity = Math.max(0.1, val);
                return (
                  <div
                    key={j}
                    className="flex-1 rounded-sm transition-colors hover:ring-2 hover:ring-neon-blue/50 cursor-pointer"
                    style={{
                      backgroundColor: `rgba(56, 189, 248, ${opacity})`,
                    }}
                    title={`${days[i]} ${j}:00 - Usage: ${(val * 100).toFixed(0)}%`}
                  />
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* X Axis Labels */}
      <div className="flex justify-between pl-12 mt-2 text-xs text-slate-500 font-medium">
        <span>00:00</span>
        <span>06:00</span>
        <span>12:00</span>
        <span>18:00</span>
        <span>23:59</span>
      </div>
    </Card>
  );
}
