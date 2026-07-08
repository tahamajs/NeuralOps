import { useQuery } from "@tanstack/react-query";
import { hpoAPI } from "../services/api";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export function OptimizationHistory({ sweepId }: { sweepId: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ["history", sweepId],
    queryFn: () => hpoAPI.getHistoryData(sweepId),
  });
  if (isLoading)
    return (
      <div className="text-slate-500 text-center py-8">Loading history...</div>
    );
  if (!data?.length)
    return (
      <div className="text-slate-500 text-center py-8">
        No trials completed yet
      </div>
    );
  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={data}>
        <XAxis dataKey="trial" stroke="#475569" />
        <YAxis stroke="#475569" />
        <Tooltip contentStyle={{ backgroundColor: "#0a0f1a" }} />
        <Line
          type="monotone"
          dataKey="value"
          stroke="#3b82f6"
          strokeWidth={2}
          dot={false}
          name="Objective"
        />
        <Line
          type="monotone"
          dataKey="best"
          stroke="#00ff9d"
          strokeWidth={2}
          dot={false}
          name="Best So Far"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
