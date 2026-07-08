import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { ImpactMetrics } from "../types";
import { TerminalCard } from "../../../components/ui/TerminalCard";

export function MetricsRadar({ metrics }: { metrics: ImpactMetrics }) {
  const radarData = [
    { subject: "h-index", value: metrics.h_index, fullMark: 100 },
    { subject: "i10-index", value: metrics.i10_index, fullMark: 100 },
    { subject: "Citations", value: metrics.total_citations, fullMark: 1000 },
    { subject: "Output Count", value: metrics.output_count, fullMark: 50 },
    {
      subject: "Collaborations",
      value: metrics.collaborations_count,
      fullMark: 20,
    },
    {
      subject: "Intl Collabs",
      value: metrics.international_collaborations,
      fullMark: 20,
    },
  ];

  return (
    <TerminalCard className="p-4">
      <h3 className="text-sm font-bold text-white mb-3">
        Research Impact Radar
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <RadarChart data={radarData}>
          <PolarGrid stroke="#334155" />
          <PolarAngleAxis
            dataKey="subject"
            tick={{ fill: "#94a3b8", fontSize: 10 }}
          />
          <PolarRadiusAxis tick={{ fill: "#94a3b8" }} domain={[0, "dataMax"]} />
          <Radar
            dataKey="value"
            stroke="#3b82f6"
            fill="#3b82f620"
            fillOpacity={0.6}
          />
          <Legend />
        </RadarChart>
      </ResponsiveContainer>
    </TerminalCard>
  );
}
