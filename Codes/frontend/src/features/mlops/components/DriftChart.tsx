import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { DriftCheck } from "../types";

export function DriftChart({
  checks,
  threshold,
}: {
  checks: DriftCheck[];
  threshold: number;
}) {
  const data = checks.map((c) => ({
    time: new Date(c.timestamp).toLocaleDateString(),
    p_value: c.p_value,
    divergence: c.divergence,
  }));
  return (
    <ResponsiveContainer width="100%" height={250}>
      <LineChart data={data}>
        <XAxis dataKey="time" stroke="#475569" />
        <YAxis stroke="#475569" />
        <Tooltip contentStyle={{ backgroundColor: "#0a0f1a" }} />
        <ReferenceLine
          y={threshold}
          stroke="#ef4444"
          strokeDasharray="3 3"
          label="Threshold"
        />
        <Line
          type="monotone"
          dataKey="p_value"
          stroke="#3b82f6"
          strokeWidth={2}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
