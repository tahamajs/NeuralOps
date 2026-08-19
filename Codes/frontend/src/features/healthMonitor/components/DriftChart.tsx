import { useState } from "react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  Legend,
} from "recharts";
import { DriftCheck } from "../types";
import { TrendingUp, Layers } from "lucide-react";

export function DriftChart({
  checks,
  threshold,
}: {
  checks: DriftCheck[];
  threshold: number;
}) {
  const [viewMode, setViewMode] = useState<"timeline" | "histogram">("timeline");

  const latestCheck = checks[checks.length - 1] || checks[0];

  const data = checks.map((c) => ({
    time: new Date(c.timestamp).toLocaleDateString([], {
      month: "short",
      day: "numeric",
      hour: "2-digit",
    }),
    score: c.drift_score,
    status: c.status,
    sample_size: c.sample_size,
  }));

  const distData =
    latestCheck?.distribution_comparison?.reference_histogram?.map((refVal, idx) => ({
      bin: `Bin ${idx + 1}`,
      Reference: refVal,
      Current: latestCheck.distribution_comparison?.current_histogram?.[idx] ?? refVal,
    })) || [];

  return (
    <div className="space-y-3">
      <div className="flex justify-between items-center text-xs">
        <div className="flex gap-1 bg-white/5 p-0.5 rounded border border-white/10">
          <button
            onClick={() => setViewMode("timeline")}
            className={`px-2 py-0.5 rounded flex items-center gap-1 transition-colors ${
              viewMode === "timeline"
                ? "bg-neon-blue text-white font-bold"
                : "text-slate-400 hover:text-white"
            }`}
          >
            <TrendingUp size={11} /> Timeline
          </button>
          <button
            onClick={() => setViewMode("histogram")}
            className={`px-2 py-0.5 rounded flex items-center gap-1 transition-colors ${
              viewMode === "histogram"
                ? "bg-neon-blue text-white font-bold"
                : "text-slate-400 hover:text-white"
            }`}
          >
            <Layers size={11} /> Histogram
          </button>
        </div>

        <span className="text-[11px] font-mono text-slate-400">
          Threshold: <strong className="text-red-400">{threshold}</strong>
        </span>
      </div>

      {viewMode === "timeline" ? (
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={data}>
            <XAxis dataKey="time" stroke="#64748b" tick={{ fontSize: 10 }} />
            <YAxis stroke="#64748b" tick={{ fontSize: 10 }} domain={[0, "auto"]} />
            <Tooltip
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  const item = payload[0].payload;
                  return (
                    <div className="bg-slate-900 border border-slate-700 rounded p-2 text-xs shadow-xl space-y-1">
                      <div className="font-bold text-white">{label}</div>
                      <div className="text-neon-mint font-mono">
                        Divergence Score: <strong>{item.score}</strong>
                      </div>
                      <div className="text-slate-400 text-[11px]">
                        Status:{" "}
                        <span
                          className={`font-semibold uppercase ${
                            item.status === "critical"
                              ? "text-red-400"
                              : item.status === "warning"
                                ? "text-yellow-400"
                                : "text-green-400"
                          }`}
                        >
                          {item.status}
                        </span>
                      </div>
                      <div className="text-slate-500 text-[10px]">
                        Sample: {item.sample_size?.toLocaleString()}
                      </div>
                    </div>
                  );
                }
                return null;
              }}
            />
            <ReferenceLine
              y={threshold}
              stroke="#ef4444"
              strokeDasharray="3 3"
              label={{ value: `Alert (${threshold})`, fill: "#ef4444", fontSize: 10 }}
            />
            <Line
              type="monotone"
              dataKey="score"
              name="Drift Score"
              stroke="#38bdf8"
              strokeWidth={2}
              dot={{ r: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <div className="space-y-2">
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={distData}>
              <XAxis dataKey="bin" stroke="#64748b" tick={{ fontSize: 10 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0f172a",
                  borderColor: "#334155",
                  fontSize: "11px",
                }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="Reference" fill="#3b82f6" radius={[2, 2, 0, 0]} />
              <Bar dataKey="Current" fill="#f59e0b" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

