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
import {
  Bell,
  Filter,
  RefreshCw,
  Download,
  AlertTriangle,
  CheckCircle2,
  Flame,
  Search,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function AlertHistory({ onClose }: { onClose?: () => void }) {
  const [statusFilter, setStatusFilter] = useState("all");
  const [severityFilter, setSeverityFilter] = useState("");
  const [search, setSearch] = useState("");

  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const {
    data: alerts,
    isLoading,
    refetch,
  } = useQuery({
    queryKey: ["alerts", { status: statusFilter, severity: severityFilter }],
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
      success("Alert marked as resolved");
    },
  });

  const handleExportCSV = () => {
    if (!alerts || alerts.length === 0) return;
    const headers = ["ID", "Monitor", "Severity", "Status", "Title", "Message", "Timestamp"];
    const rows = alerts.map((a) => [
      a.id,
      `"${a.monitor_name || ""}"`,
      a.severity,
      a.status,
      `"${a.title.replace(/"/g, '""')}"`,
      `"${a.message.replace(/"/g, '""')}"`,
      a.created_at,
    ]);

    const csvContent = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `health_alerts_history_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    success("CSV Exported", "Alert history report generated");
  };

  const handleAcknowledgeAll = async () => {
    const activeAlerts = alerts?.filter((a) => a.status === "active") || [];
    if (activeAlerts.length === 0) return;
    for (const a of activeAlerts) {
      await healthMonitorAPI.acknowledgeAlert(a.id);
    }
    queryClient.invalidateQueries({ queryKey: ["alerts"] });
    success("Batch Acknowledged", `Acknowledged ${activeAlerts.length} active alerts`);
  };

  if (isLoading) return <Spinner />;

  const filteredAlerts =
    alerts?.filter((a) => {
      if (!search) return true;
      const q = search.toLowerCase();
      return (
        a.title.toLowerCase().includes(q) ||
        a.message.toLowerCase().includes(q) ||
        a.monitor_name.toLowerCase().includes(q)
      );
    }) || [];

  const totalCount = alerts?.length || 0;
  const activeCount = alerts?.filter((a) => a.status === "active").length || 0;
  const criticalCount = alerts?.filter((a) => a.severity === "critical" || a.severity === "fatal").length || 0;
  const resolvedCount = alerts?.filter((a) => a.status === "resolved").length || 0;

  return (
    <div className="flex flex-col h-full bg-slate-950">
      {/* Header */}
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Bell size={20} className="text-yellow-400" /> System Health Alert History
          </h1>
          <p className="text-xs text-slate-400">
            Real-time telemetry breaches, service degradation alerts & triage audit trail
          </p>
        </div>
        <div className="flex gap-2 items-center">
          <Button size="sm" variant="ghost" onClick={handleExportCSV}>
            <Download size={14} className="mr-1" /> Export CSV
          </Button>
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

      {/* Main Content */}
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {/* KPI Stats Bar */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <TerminalCard className="p-3">
            <div className="text-[11px] text-slate-400 font-mono">TOTAL ALERTS</div>
            <div className="text-xl font-bold text-white font-mono mt-1">{totalCount}</div>
          </TerminalCard>
          <TerminalCard className="p-3 border-yellow-500/30">
            <div className="text-[11px] text-yellow-400 font-mono">ACTIVE ISSUES</div>
            <div className="text-xl font-bold text-yellow-400 font-mono mt-1">{activeCount}</div>
          </TerminalCard>
          <TerminalCard className="p-3 border-red-500/30">
            <div className="text-[11px] text-red-400 font-mono">CRITICAL BREACHES</div>
            <div className="text-xl font-bold text-red-400 font-mono mt-1">{criticalCount}</div>
          </TerminalCard>
          <TerminalCard className="p-3 border-green-500/30">
            <div className="text-[11px] text-green-400 font-mono">RESOLVED RATE</div>
            <div className="text-xl font-bold text-green-400 font-mono mt-1">
              {totalCount > 0 ? `${Math.round((resolvedCount / totalCount) * 100)}%` : "100%"}
            </div>
          </TerminalCard>
        </div>

        {/* Filter and Action Bar */}
        <div className="flex flex-wrap justify-between items-center gap-3">
          <div className="flex flex-wrap gap-2 items-center">
            <div className="relative w-56">
              <Input
                placeholder="Search alerts by keyword..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full text-xs pl-7"
              />
              <Search
                size={12}
                className="absolute left-2 top-2.5 text-slate-500 pointer-events-none"
              />
            </div>

            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="bg-slate-900 border border-white/10 rounded px-2.5 py-1.5 text-xs text-slate-300 focus:outline-none focus:border-neon-mint"
            >
              <option value="all">All Statuses</option>
              <option value="active">Active Only</option>
              <option value="acknowledged">Acknowledged Only</option>
              <option value="resolved">Resolved Only</option>
            </select>

            <select
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value)}
              className="bg-slate-900 border border-white/10 rounded px-2.5 py-1.5 text-xs text-slate-300 focus:outline-none focus:border-neon-mint"
            >
              <option value="">All Severities</option>
              <option value="critical">Critical & Fatal</option>
              <option value="warning">Warning</option>
              <option value="info">Info</option>
            </select>
          </div>

          {activeCount > 0 && (
            <Button
              size="sm"
              variant="ghost"
              onClick={handleAcknowledgeAll}
              className="text-xs text-yellow-400 hover:bg-yellow-500/10"
            >
              <Bell size={13} className="mr-1" /> Acknowledge All Active ({activeCount})
            </Button>
          )}
        </div>

        {/* Alerts Table */}
        <AlertTable
          alerts={filteredAlerts}
          onAcknowledge={(id) => acknowledgeAlert.mutate(id)}
          onResolve={(id) => resolveAlert.mutate(id)}
        />
      </div>
    </div>
  );
}

