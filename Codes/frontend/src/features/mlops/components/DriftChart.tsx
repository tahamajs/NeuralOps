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
  Legend,
} from "recharts";
import { DriftCheck } from "../types";
import { BarChart2, TrendingUp, Layers } from "lucide-react";

export function DriftChart({
  checks,
  threshold,
}: {
  checks: DriftCheck[];
  threshold: number;
}) {
  const [viewMode, setViewMode] = useState<"time-series" | "distribution">("time-series");
  const [activeMetric, setActiveMetric] = useState<"p_value" | "divergence" | "both">("both");

  const latestCheck = checks[checks.length - 1] || checks[0];

  const data = checks.map((c) => ({
    time: new Date(c.timestamp).toLocaleDateString([], {
      month: "short",
      day: "numeric",
      hour: "2-digit",
    }),
    p_value: c.p_value,
    divergence: c.divergence,
    status: c.status,
    sample_size: c.sample_size,
  }));

  // Prepare histogram comparison data
  const distData =
    latestCheck?.reference_histogram?.map((refVal, idx) => ({
      bin: `Bin ${idx + 1}`,
      Reference: refVal,
      Current: latestCheck.current_histogram?.[idx] ?? refVal,
    })) || [];

  return (
    <div className="space-y-3">
      {/* Controls Bar */}
      <div className="flex justify-between items-center text-xs">
        <div className="flex gap-1 bg-white/5 p-0.5 rounded border border-white/10">
          <button
            onClick={() => setViewMode("time-series")}
            className={`px-2 py-1 rounded flex items-center gap-1 transition-colors ${
              viewMode === "time-series"
                ? "bg-neon-blue text-white font-bold"
                : "text-slate-400 hover:text-white"
            }`}
          >
            <TrendingUp size={12} /> Trend
          </button>
          <button
            onClick={() => setViewMode("distribution")}
            className={`px-2 py-1 rounded flex items-center gap-1 transition-colors ${
              viewMode === "distribution"
                ? "bg-neon-blue text-white font-bold"
                : "text-slate-400 hover:text-white"
            }`}
          >
            <Layers size={12} /> Distribution Compare
          </button>
        </div>

        {viewMode === "time-series" && (
          <div className="flex gap-1">
            <button
              onClick={() => setActiveMetric("p_value")}
              className={`px-2 py-0.5 rounded text-[11px] font-mono border ${
                activeMetric === "p_value"
                  ? "border-neon-blue bg-neon-blue/20 text-white font-bold"
                  : "border-white/10 text-slate-400"
              }`}
            >
              p-value
            </button>
            <button
              onClick={() => setActiveMetric("divergence")}
              className={`px-2 py-0.5 rounded text-[11px] font-mono border ${
                activeMetric === "divergence"
                  ? "border-yellow-400 bg-yellow-400/20 text-yellow-300 font-bold"
                  : "border-white/10 text-slate-400"
              }`}
            >
              divergence
            </button>
            <button
              onClick={() => setActiveMetric("both")}
              className={`px-2 py-0.5 rounded text-[11px] font-mono border ${
                activeMetric === "both"
                  ? "border-white/30 bg-white/10 text-white font-bold"
                  : "border-white/10 text-slate-400"
              }`}
            >
              all
            </button>
          </div>
        )}
      </div>

      {viewMode === "time-series" ? (
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data}>
            <XAxis
              dataKey="time"
              stroke="#64748b"
              tick={{ fontSize: 10 }}
              interval="preserveStartEnd"
            />
            <YAxis stroke="#64748b" tick={{ fontSize: 10 }} domain={[0, "auto"]} />
            <Tooltip
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  const item = payload[0].payload;
                  return (
                    <div className="bg-slate-900 border border-slate-700 rounded p-2 text-xs shadow-xl space-y-1">
                      <div className="font-bold text-white border-b border-white/10 pb-1">
                        {label}
                      </div>
                      <div className="text-neon-blue font-mono">
                        P-Value: <strong>{item.p_value}</strong>
                      </div>
                      <div className="text-yellow-400 font-mono">
                        Divergence Score: <strong>{item.divergence}</strong>
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
                        Sample Size: {item.sample_size.toLocaleString()}
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
              strokeDasharray="4 4"
              label={{
                value: `Threshold (${threshold})`,
                fill: "#ef4444",
                fontSize: 10,
                position: "right",
              }}
            />
            <Legend wrapperStyle={{ fontSize: 11, paddingTop: 4 }} />
            {(activeMetric === "p_value" || activeMetric === "both") && (
              <Line
                type="monotone"
                name="P-Value (KS/Hypothesis)"
                dataKey="p_value"
                stroke="#38bdf8"
                strokeWidth={2}
                dot={{ r: 2 }}
                activeDot={{ r: 5 }}
              />
            )}
            {(activeMetric === "divergence" || activeMetric === "both") && (
              <Line
                type="monotone"
                name="Divergence / PSI Score"
                dataKey="divergence"
                stroke="#facc15"
                strokeWidth={2}
                dot={{ r: 2 }}
                activeDot={{ r: 5 }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <div className="space-y-2">
          <div className="text-xs text-slate-400 flex items-center justify-between">
            <span>Reference Baseline vs. Current Ingestion Window</span>
            <span className="font-mono text-neon-mint text-[11px]">
              Latest sample: {latestCheck?.sample_size?.toLocaleString() || 5000} items
            </span>
          </div>
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

