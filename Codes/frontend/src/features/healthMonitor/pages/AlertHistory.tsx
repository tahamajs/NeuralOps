// src/features/healthMonitor/pages/AlertHistory.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { healthMonitorAPI } from "../services/api";
import { AlertTable } from "../components/AlertTable";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Bell, Filter, RefreshCw } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function AlertHistory({ onClose }: { onClose?: () => void }) {
  const [statusFilter, setStatusFilter] = useState("active");
  const [severityFilter, setSeverityFilter] = useState("");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const {
    data: alerts,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: [
      "alerts",
      { status: statusFilter, severity: severityFilter, page },
    ],
    queryFn: () =>
      healthMonitorAPI.getAlerts({
        status: statusFilter !== "all" ? statusFilter : undefined,
        severity: severityFilter || undefined,
      }),
  });

  const acknowledgeAlert = useMutation({
    mutationFn: (id: string) => healthMonitorAPI.acknowledgeAlert(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["alerts"] });
      success("Alert acknowledged");
    },
  });
  const resolveAlert = useMutation({
    mutationFn: (id: string) => healthMonitorAPI.resolveAlert(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["alerts"] });
      success("Alert resolved");
    },
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Bell size={20} className="text-yellow-400" /> Alert History
          </h1>
          <p className="text-xs text-slate-400">
            View and manage system alerts
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="ghost" onClick={() => refetch()}>
            <RefreshCw size={14} />
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <div className="flex flex-wrap gap-3 items-end">
          <Select
            options={[
              { value: "active", label: "Active" },
              { value: "acknowledged", label: "Acknowledged" },
              { value: "resolved", label: "Resolved" },
              { value: "all", label: "All" },
            ]}
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="w-32"
          />
          <Select
            options={[
              { value: "", label: "All Severities" },
              { value: "info", label: "Info" },
              { value: "warning", label: "Warning" },
              { value: "critical", label: "Critical" },
              { value: "fatal", label: "Fatal" },
            ]}
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
            className="w-36"
          />
          <Input
            placeholder="Search alerts..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-48"
          />
        </div>
        <AlertTable
          alerts={alerts || []}
          onAcknowledge={(id) => acknowledgeAlert.mutate(id)}
          onResolve={(id) => resolveAlert.mutate(id)}
        />
      </div>
    </div>
  );
}
