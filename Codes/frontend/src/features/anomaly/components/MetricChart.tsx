import { useQuery } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { Spinner } from "../../../components/ui/Spinner";
import { request } from "../../../api/client";

export function MetricChart({
  runId,
  metricName = "loss",
}: {
  runId: string;
  metricName?: string;
}) {
  const { data: rawData, isLoading } = useQuery({
    queryKey: ["run-metrics", runId, metricName],
    queryFn: () =>
      request.get<any[]>(`/runs/${runId}/history/?metric=${metricName || "loss"}`),
    enabled: !!runId,
  });

  if (isLoading) return <Spinner size="sm" />;

  const dataList = Array.isArray(rawData) ? rawData : [];
  const normalizedData = dataList.map((d, index) => {
    const rawVal = d?.value ?? d?.[metricName] ?? d?.loss ?? d?.accuracy;
    const numVal = typeof rawVal === "number" ? rawVal : Number(rawVal) || 0;
    return {
      step: d?.step ?? index + 1,
      value: numVal,
    };
  });

  if (!normalizedData.length)
    return <div className="text-slate-500">No metric data available</div>;

  const firstValue = normalizedData[0]?.value;
  const threshold = typeof firstValue === "number" && !isNaN(firstValue) ? firstValue * 1.2 : undefined;

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={normalizedData}>
        <XAxis dataKey="step" stroke="#475569" />
        <YAxis stroke="#475569" />
        <Tooltip
          contentStyle={{
            backgroundColor: "#0a0f1a",
            borderColor: "rgba(255,255,255,0.1)",
            borderRadius: "0.5rem",
          }}
        />
        <Line
          type="monotone"
          dataKey="value"
          name={metricName}
          stroke="#3b82f6"
          strokeWidth={2}
          dot={false}
        />
        {threshold !== undefined && (
          <ReferenceLine
            y={threshold}
            stroke="#ef4444"
            strokeDasharray="3 3"
            label={{ value: "Anomaly threshold", fill: "#ef4444", fontSize: 12 }}
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}
