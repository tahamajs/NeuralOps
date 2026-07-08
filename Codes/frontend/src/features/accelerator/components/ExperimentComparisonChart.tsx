import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { acceleratorAPI } from "../services/api";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ZAxis,
} from "recharts";
import { Search } from "lucide-react";

export function ExperimentComparisonChart() {
  const [experimentIds, setExperimentIds] = useState<string[]>([]);
  const [inputValue, setInputValue] = useState("");
  const { data, isLoading, refetch } = useQuery({
    queryKey: ["experiment-comparison", experimentIds],
    queryFn: () => acceleratorAPI.compareExperiments(experimentIds),
    enabled: experimentIds.length > 0,
  });

  const handleAdd = () => {
    if (inputValue && !experimentIds.includes(inputValue)) {
      setExperimentIds([...experimentIds, inputValue]);
      setInputValue("");
      refetch();
    }
  };

  const handleRemove = (id: string) =>
    setExperimentIds(experimentIds.filter((exp) => exp !== id));

  if (isLoading) return <Spinner size="sm" />;
  const experiments = data?.data || [];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        <div className="relative flex-1 min-w-[200px]">
          <Search
            size={14}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400"
          />
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAdd()}
            placeholder="Experiment ID to compare..."
            className="w-full bg-white/5 border border-white/10 rounded-xl pl-9 pr-3 py-2 text-sm text-white placeholder-slate-500"
          />
        </div>
        <button
          onClick={handleAdd}
          className="px-3 py-2 bg-neon-blue/20 rounded text-sm text-white"
        >
          Add
        </button>
      </div>
      {experimentIds.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {experimentIds.map((id) => (
            <span
              key={id}
              className="flex items-center gap-1 px-2 py-1 rounded bg-white/10 text-sm"
            >
              <span className="font-mono">{id.slice(0, 8)}</span>
              <button
                onClick={() => handleRemove(id)}
                className="text-slate-400 hover:text-red-400"
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={experiments}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="name" tick={{ fill: "#94a3b8" }} />
            <YAxis yAxisId="left" tick={{ fill: "#94a3b8" }} />
            <YAxis
              yAxisId="right"
              orientation="right"
              tick={{ fill: "#94a3b8" }}
            />
            <Tooltip contentStyle={{ backgroundColor: "#0a0f1a" }} />
            <Legend />
            <Bar
              yAxisId="left"
              dataKey="accuracy"
              fill="#3b82f6"
              name="Accuracy (%)"
            />
            <Bar yAxisId="right" dataKey="loss" fill="#ef4444" name="Loss" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
