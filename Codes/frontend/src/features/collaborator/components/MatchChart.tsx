import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { MatchHistoryEntry } from "../types";

export function MatchChart({ matches }: { matches: MatchHistoryEntry[] }) {
  const statusCount = matches.reduce(
    (acc, m) => {
      acc[m.match_status] = (acc[m.match_status] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>,
  );
  const typeCount = matches.reduce(
    (acc, m) => {
      acc[m.collaboration_type] = (acc[m.collaboration_type] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>,
  );
  const pieData = Object.entries(statusCount).map(([name, value]) => ({
    name,
    value,
  }));
  const COLORS = {
    pending: "#f59e0b",
    accepted: "#3b82f6",
    connected: "#10b981",
    rejected: "#ef4444",
  };
  return (
    <div className="grid md:grid-cols-2 gap-4">
      <TerminalCard className="p-4">
        <h3 className="text-sm font-bold text-white mb-3">
          Collaboration Status
        </h3>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie
              data={pieData}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              outerRadius={80}
              label
            >
              {pieData.map((entry) => (
                <Cell
                  key={entry.name}
                  fill={COLORS[entry.name as keyof typeof COLORS] || "#94a3b8"}
                />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </TerminalCard>
      <TerminalCard className="p-4">
        <h3 className="text-sm font-bold text-white mb-3">
          Collaboration Types
        </h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart
            data={Object.entries(typeCount).map(([type, count]) => ({
              type,
              count,
            }))}
          >
            <XAxis dataKey="type" stroke="#475569" />
            <YAxis stroke="#475569" />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </TerminalCard>
    </div>
  );
}
