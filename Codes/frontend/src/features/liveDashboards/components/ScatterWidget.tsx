import React from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export function ScatterWidget({ data, config }: any) {
  if (!data || data.length === 0) {
    return <div className="text-slate-500 text-xs p-4">No data available</div>;
  }

  const chartData = data.map((d, i) => ({
    x: d.x ?? i,
    y: d.y ?? d.value ?? 0,
    name: d.name ?? `Point ${i}`,
  }));

  return (
    <ResponsiveContainer width="100%" height={200}>
      <ScatterChart>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis type="number" dataKey="x" name="X" stroke="#475569" tick={{ fontSize: 10 }} />
        <YAxis type="number" dataKey="y" name="Y" stroke="#475569" />
        <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={{ backgroundColor: "#0a0f1a" }} />
        <Scatter name="Points" data={chartData} fill="#3b82f6" />
      </ScatterChart>
    </ResponsiveContainer>
  );
}
