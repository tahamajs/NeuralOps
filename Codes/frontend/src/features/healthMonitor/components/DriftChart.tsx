import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
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
    timestamp: new Date(c.timestamp).toLocaleString(),
    score: c.drift_score,
    status: c.status,
  }));
  return (
    <ResponsiveContainer width="100%" height={250}>
      <LineChart data={data}>
        <XAxis dataKey="timestamp" stroke="#475569" tick={{ fontSize: 10 }} />
        <YAxis stroke="#475569" domain={[0, 1]} />
        <Tooltip contentStyle={{ backgroundColor: "#0a0f1a" }} />
        <ReferenceLine
          y={threshold}
          stroke="#ef4444"
          strokeDasharray="3 3"
          label="Threshold"
        />
        <Area type="monotone" dataKey="score" fill="#3b82f620" stroke="none" />
        <Line
          type="monotone"
          dataKey="score"
          stroke="#3b82f6"
          strokeWidth={2}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
