// src/features/governance/pages/AuditLogDashboard.tsx
import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { governanceAPI } from "../services/api";
import { AuditLogTable } from "../components/AuditLogTable";
import { StatCard } from "../components/StatCard";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { DateRangePicker } from "../../../components/ui/DateRangePicker";
import { Spinner } from "../../../components/ui/Spinner";
import {
  Download,
  Filter,
  AlertCircle,
  Users,
  Database,
  Shield,
  Search,
  CheckCircle2,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function AuditLogDashboard({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [dateRange, setDateRange] = useState<{ from?: Date; to?: Date }>({});
  const [userId, setUserId] = useState("");
  const [actionType, setActionType] = useState("");
  const [resourceType, setResourceType] = useState("");
  const [page, setPage] = useState(1);
  const { success, error } = useToast();

  const startDateStr = dateRange.from ? dateRange.from.toISOString() : "";
  const endDateStr = dateRange.to ? dateRange.to.toISOString() : "";

  const { data, isLoading, refetch } = useQuery({
    queryKey: [
      "audit-logs",
      {
        start_date: startDateStr,
        end_date: endDateStr,
        user_id: userId,
        action: actionType,
        resource_type: resourceType,
        page,
      },
    ],
    queryFn: () =>
      governanceAPI.getAuditLogs({
        start_date: startDateStr || undefined,
        end_date: endDateStr || undefined,
        user_id: userId || undefined,
        action: actionType || undefined,
        resource_type: resourceType || undefined,
        page,
      }),
  });

  const exportMutation = useMutation({
    mutationFn: () =>
      governanceAPI.exportAuditLogs({
        start_date: startDateStr,
        end_date: endDateStr,
        user_id: userId,
      }),
    onSuccess: (res) => {
      const url = URL.createObjectURL(res.data);
      const a = document.createElement("a");
      a.href = url;
      a.download = `audit_logs_${new Date().toISOString().slice(0, 19)}.csv`;
      a.click();
      URL.revokeObjectURL(url);
      success("Audit trail exported", "CSV audit log downloaded");
    },
  });

  if (isLoading) return <Spinner />;

  const logs = data?.results || [];
  const totalPages = Math.ceil((data?.count || 0) / 20) || 1;

  return (
    <div className="flex flex-col h-full bg-slate-950">
      {/* Header */}
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Shield size={20} className="text-yellow-400" /> Audit Log Dashboard
          </h1>
          <p className="text-xs text-slate-400">
            Immutable activity logs, actor attribution & compliance proof verification
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => exportMutation.mutate()}
            disabled={exportMutation.isPending}
          >
            <Download size={14} className="mr-1" /> Export CSV
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
        {/* KPI Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard
            icon={<Users size={16} />}
            label="Total Events"
            value={data?.count || 0}
          />
          <StatCard
            icon={<AlertCircle size={16} className="text-red-400" />}
            label="Critical Incidents"
            value={logs.filter((l) => l.severity === "critical").length}
          />
          <StatCard
            icon={<AlertCircle size={16} className="text-yellow-400" />}
            label="Security Warnings"
            value={logs.filter((l) => l.severity === "warning").length}
          />
          <StatCard
            icon={<Database size={16} className="text-neon-mint" />}
            label="Resources Affected"
            value={new Set(logs.map((l) => l.resource_id)).size}
          />
        </div>

        {/* Filter Controls Bar */}
        <TerminalCard className="p-3">
          <div className="flex flex-wrap gap-2.5 items-center">
            <DateRangePicker
              value={dateRange}
              onChange={setDateRange}
              className="text-xs"
            />
            <div className="relative w-40">
              <Input
                placeholder="User / Actor..."
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                className="w-full text-xs pl-7"
              />
              <Search
                size={12}
                className="absolute left-2 top-2.5 text-slate-500 pointer-events-none"
              />
            </div>
            <Select
              options={[
                { value: "", label: "All Actions" },
                { value: "create", label: "Create" },
                { value: "update", label: "Update" },
                { value: "delete", label: "Delete" },
                { value: "export", label: "Export" },
                { value: "share", label: "Share" },
                { value: "login", label: "Login" },
              ]}
              value={actionType}
              onChange={(e) => setActionType(e.target.value)}
              className="w-32 text-xs"
            />
            <Select
              options={[
                { value: "", label: "All Resources" },
                { value: "model", label: "Model" },
                { value: "dataset", label: "Dataset" },
                { value: "policy", label: "Policy" },
                { value: "contract", label: "Contract" },
                { value: "experiment", label: "Experiment" },
              ]}
              value={resourceType}
              onChange={(e) => setResourceType(e.target.value)}
              className="w-32 text-xs"
            />
            <Button size="sm" onClick={() => refetch()} className="text-xs h-8">
              Filter
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                setDateRange({});
                setUserId("");
                setActionType("");
                setResourceType("");
                refetch();
              }}
              className="text-xs h-8"
            >
              Reset
            </Button>
          </div>
        </TerminalCard>

        {/* Audit Log Table */}
        <AuditLogTable logs={logs} />

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex justify-center items-center gap-2 pt-2">
            <Button
              size="sm"
              variant="ghost"
              disabled={page === 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
            >
              Prev
            </Button>
            <span className="text-xs text-slate-400 font-mono">
              Page {page} of {totalPages}
            </span>
            <Button
              size="sm"
              variant="ghost"
              disabled={page === totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            >
              Next
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}

