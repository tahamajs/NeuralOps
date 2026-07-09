import React from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { CostEntry } from "../types";

interface CostChartProps {
  data: CostEntry[];
}

export function CostChart({ data }: CostChartProps) {
  // Group costs by date
  const chartDataMap = data.reduce((acc, entry) => {
    const date = new Date(entry.date).toLocaleDateString(undefined, { month: "short", day: "numeric" });
    acc[date] = (acc[date] || 0) + entry.cost_usd;
    return acc;
  }, {} as Record<string, number>);

  const chartData = Object.entries(chartDataMap).map(([date, cost]) => ({
    date,
    cost: parseFloat(cost.toFixed(4)),
  }));

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-950/90 border border-white/10 p-3 rounded-xl shadow-2xl backdrop-blur-md text-xs font-sans">
          <p className="text-slate-500 mb-1">{label}</p>
          <p className="font-bold text-white">
            Cost: <span className="text-neon-mint">${payload[0].value.toFixed(4)}</span>
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-slate-900/30 border border-white/5 rounded-2xl p-4 flex flex-col gap-2 backdrop-blur-md">
      <div className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-2">
        Spending Timeline
      </div>
      <div className="w-full h-48">
        {chartData.length === 0 ? (
          <div className="w-full h-full flex items-center justify-center text-xs text-slate-600 italic">
            No cost entries recorded.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
              <XAxis
                dataKey="date"
                stroke="#64748b"
                fontSize={9}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                stroke="#64748b"
                fontSize={9}
                tickLine={false}
                axisLine={false}
                tickFormatter={(value) => `$${value}`}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="cost" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
