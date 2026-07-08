import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export type RevenuePoint = any;

export function RevenueChart({ data }: { data: RevenuePoint[] }) {
  const chartData = data.map((d) => ({
    date: new Date(d.date).toLocaleDateString(),
    amount: d.amount,
  }));
  return (
    <ResponsiveContainer width="100%" height={250}>
      <AreaChart data={chartData}>
        <XAxis dataKey="date" stroke="#475569" />
        <YAxis stroke="#475569" />
        <Tooltip contentStyle={{ backgroundColor: "#0a0f1a" }} />
        <Area
          type="monotone"
          dataKey="amount"
          stroke="#3b82f6"
          fill="#3b82f620"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
