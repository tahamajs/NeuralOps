import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export function LineChartWidget({ data, config }: any) {
  const chartData = data.map((d) => ({
    timestamp: new Date(d.timestamp).toLocaleTimeString(),
    value: d.value,
  }));
  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={chartData}>
        <XAxis dataKey="timestamp" stroke="#475569" tick={{ fontSize: 10 }} />
        <YAxis stroke="#475569" />
        <Tooltip contentStyle={{ backgroundColor: "#0a0f1a" }} />
        <Line
          type="monotone"
          dataKey="value"
          stroke="#3b82f6"
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
