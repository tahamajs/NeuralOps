import { useQuery } from "@tanstack/react-query";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Progress } from "../../../components/ui/Progress";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  PieChart,
  Pie,
} from "recharts";
import { federatedAPI } from "../services/api";

interface Props {
  runId: string;
  onClose?: () => void;
}

export function FLDashboard({ runId, onClose }: Props) {
  const { data: globalMetrics, isLoading: globalLoading } = useQuery({
    queryKey: ["fl-global", runId],
    queryFn: () => federatedAPI.getGlobalMetrics(runId),
    refetchInterval: 5000,
  });
  const { data: clientsData, isLoading: clientsLoading } = useQuery({
    queryKey: ["fl-clients", runId],
    queryFn: () => federatedAPI.getClients(runId),
    refetchInterval: 10000,
  });

  const clients = clientsData?.clients || [];
  const totalClients = clients.length;
  const activeClients = clients.filter((c) => c.status === "active").length;
  const avgClientAccuracy =
    clients.reduce((sum, c) => sum + c.accuracy, 0) / totalClients || 0;
  const globalAccuracy = globalMetrics?.final_accuracy || 0;
  const history = globalMetrics?.history || [];

  const clientChartData = clients.map((c) => ({
    name: c.id.slice(0, 8),
    accuracy: c.accuracy,
  }));

  if (globalLoading || clientsLoading)
    return (
      <div className="flex items-center justify-center h-full text-slate-400">
        Loading FL dashboard...
      </div>
    );

  return (
    <div className="flex flex-col h-full overflow-y-auto p-4 space-y-4">
      <div className="flex justify-between items-center border-b border-white/10 pb-2">
        <h2 className="text-sm font-mono text-neon-mint">
          🔗 Federated Learning Dashboard
        </h2>
        {onClose && (
          <button onClick={onClose} className="text-slate-400 hover:text-white">
            ✕
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <TerminalCard className="p-4 text-center">
          <div className="text-2xl font-bold text-neon-mint">
            {totalClients}
          </div>
          <div className="text-xs text-slate-500">TOTAL CLIENTS</div>
        </TerminalCard>
        <TerminalCard className="p-4 text-center">
          <div className="text-2xl font-bold text-green-400">
            {activeClients}
          </div>
          <div className="text-xs text-slate-500">ACTIVE CLIENTS</div>
        </TerminalCard>
        <TerminalCard className="p-4 text-center">
          <div className="text-2xl font-bold text-blue-400">
            {globalAccuracy.toFixed(2)}%
          </div>
          <div className="text-xs text-slate-500">GLOBAL ACCURACY</div>
        </TerminalCard>
        <TerminalCard className="p-4 text-center">
          <div className="text-2xl font-bold text-yellow-400">
            {avgClientAccuracy.toFixed(2)}%
          </div>
          <div className="text-xs text-slate-500">AVG CLIENT ACCURACY</div>
        </TerminalCard>
      </div>

      <TerminalCard className="p-4">
        <div className="text-sm font-bold text-white mb-2">
          Global Accuracy over Rounds
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={history}>
            <XAxis dataKey="round" stroke="#475569" />
            <YAxis stroke="#475569" />
            <Tooltip contentStyle={{ backgroundColor: "#0a0f1a" }} />
            <Area
              type="monotone"
              dataKey="accuracy"
              stroke="#00ff9d"
              fill="#00ff9d20"
            />
          </AreaChart>
        </ResponsiveContainer>
      </TerminalCard>

      {clientChartData.length > 0 && (
        <TerminalCard className="p-4">
          <div className="text-sm font-bold text-white mb-2">
            Client Performance
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={clientChartData}>
              <XAxis dataKey="name" stroke="#475569" />
              <YAxis stroke="#475569" />
              <Tooltip contentStyle={{ backgroundColor: "#0a0f1a" }} />
              <Bar dataKey="accuracy" fill="#3b82f6">
                {clientChartData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.accuracy > 90 ? "#00ff9d" : "#f59e0b"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </TerminalCard>
      )}

      <TerminalCard className="p-4">
        <div className="flex justify-between items-center">
          <div>
            <div className="text-sm font-bold text-white">Current Round</div>
            <div className="text-2xl font-mono text-neon-mint">
              {globalMetrics?.current_round || 0}
            </div>
          </div>
          <div>
            <div className="text-sm font-bold text-white">Next Aggregation</div>
            <Progress
              value={globalMetrics?.next_aggregation_progress || 0}
              className="w-32 h-1"
            />
          </div>
        </div>
      </TerminalCard>
    </div>
  );
}
