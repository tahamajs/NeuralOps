// src/features/gpuFinance/components/IndexChart.tsx
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

interface HistoryPoint {
  timestamp: string;
  value: number;
}

interface IndexChartProps {
  data: HistoryPoint[];
  height?: number;
  color?: string;
}

export function IndexChart({
  data,
  height = 300,
  color = "#3b82f6",
}: IndexChartProps) {
  const chartData = data.map((d) => ({
    date: new Date(d.timestamp).toLocaleDateString(),
    value: d.value,
  }));

  const formatYAxis = (value: number) => `$${value.toFixed(2)}`;
  const formatTooltip = (value: number) => [
    `$${value.toFixed(2)}`,
    "Index Value",
  ];

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart
        data={chartData}
        margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
      >
        <defs>
          <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={color} stopOpacity={0.3} />
            <stop offset="95%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis
          dataKey="date"
          stroke="#94a3b8"
          tick={{ fontSize: 10 }}
          tickLine={false}
          axisLine={false}
        />
        <YAxis
          stroke="#94a3b8"
          tick={{ fontSize: 10 }}
          tickLine={false}
          axisLine={false}
          tickFormatter={formatYAxis}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#0a0f1a",
            border: "1px solid #1e293b",
            borderRadius: "8px",
            fontSize: "12px",
          }}
          labelStyle={{ color: "#94a3b8" }}
          formatter={formatTooltip}
        />
        <Area
          type="monotone"
          dataKey="value"
          stroke={color}
          strokeWidth={2}
          fill="url(#colorValue)"
          fillOpacity={1}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
