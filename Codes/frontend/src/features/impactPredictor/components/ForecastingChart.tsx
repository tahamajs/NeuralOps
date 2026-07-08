import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { ImpactHistory } from "../types";
import { TerminalCard } from "../../../components/ui/TerminalCard";

interface ForecastingChartProps {
  history: ImpactHistory[];
  predictedScore?: number;
}

export function ForecastingChart({
  history,
  predictedScore,
}: ForecastingChartProps) {
  const chartData = history.map((item) => ({
    date: new Date(item.date).toLocaleDateString(),
    score: item.score,
    h_index: item.h_index,
    citations: item.citations,
  }));

  return (
    <TerminalCard className="p-4">
      <h3 className="text-sm font-bold text-white mb-3">
        Historical Impact Trends
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <XAxis
            dataKey="date"
            stroke="#475569"
            tick={{ fill: "#94a3b8", fontSize: 10 }}
          />
          <YAxis yAxisId="left" stroke="#475569" tick={{ fill: "#94a3b8" }} />
          <YAxis
            yAxisId="right"
            orientation="right"
            stroke="#475569"
            tick={{ fill: "#94a3b8" }}
          />
          <Tooltip contentStyle={{ backgroundColor: "#0a0f1a" }} />
          <Legend />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="score"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
            name="Impact Score"
          />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="h_index"
            stroke="#00ff9d"
            strokeWidth={2}
            dot={false}
            name="h-index"
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="citations"
            stroke="#f59e0b"
            strokeWidth={2}
            dot={false}
            name="Citations"
          />
        </LineChart>
      </ResponsiveContainer>
      {predictedScore && (
        <div className="mt-3 text-xs text-slate-400 text-center">
          Predicted h‑index:{" "}
          <span className="text-neon-mint font-bold">{predictedScore}</span>
        </div>
      )}
    </TerminalCard>
  );
}
