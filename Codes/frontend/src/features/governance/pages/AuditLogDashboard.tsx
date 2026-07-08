// src/features/governance/pages/AuditLogDashboard.tsx
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
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
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function AuditLogDashboard({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [userId, setUserId] = useState("");
  const [actionType, setActionType] = useState("");
  const [resourceType, setResourceType] = useState("");
  const [page, setPage] = useState(1);
  const { success, error } = useToast();

  const { data, isLoading, refetch } = useQuery({
    queryKey: [
      "audit-logs",
      {
        start_date: startDate,
        end_date: endDate,
        user_id: userId,
        action: actionType,
        resource_type: resourceType,
        page,
      },
    ],
    queryFn: () =>
      governanceAPI.getAuditLogs({
        start_date: startDate,
        end_date: endDate,
        user_id: userId || undefined,
        action: actionType || undefined,
        resource_type: resourceType || undefined,
        page,
      }),
  });

  const exportMutation = useMutation({
    mutationFn: () =>
      governanceAPI.exportAuditLogs({
        start_date: startDate,
        end_date: endDate,
        user_id: userId,
      }),
    onSuccess: (res) => {
      const url = URL.createObjectURL(res.data);
      const a = document.createElement("a");
      a.href = url;
      a.download = `audit_logs_${new Date().toISOString().slice(0, 19)}.csv`;
      a.click();
      URL.revokeObjectURL(url);
      success("Export started");
    },
  });

  if (isLoading) return <Spinner />;

  const logs = data?.results || [];
  const totalPages = Math.ceil((data?.count || 0) / 20);

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Shield size={20} className="text-yellow-400" /> Audit Log Dashboard
          </h1>
          <p className="text-xs text-slate-400">
            Complete activity trail with compliance verification
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => exportMutation.mutate()}
          >
            <Download size={14} className="mr-1" /> Export
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <div className="grid md:grid-cols-4 gap-3">
          <StatCard
            icon={<Users size={16} />}
            label="Total Events"
            value={data?.count || 0}
          />
          <StatCard
            icon={<AlertCircle size={16} />}
            label="Critical"
            value={logs.filter((l) => l.severity === "critical").length}
          />
          <StatCard
            icon={<AlertCircle size={16} className="text-yellow-400" />}
            label="Warning"
            value={logs.filter((l) => l.severity === "warning").length}
          />
          <StatCard
            icon={<Database size={16} />}
            label="Resources Affected"
            value={new Set(logs.map((l) => l.resource_id)).size}
          />
        </div>
        <TerminalCard className="p-4">
          <div className="flex flex-wrap gap-3 items-end">
            <DateRangePicker
              startDate={startDate}
              endDate={endDate}
              onStartChange={setStartDate}
              onEndChange={setEndDate}
            />
            <Input
              placeholder="User ID"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              className="w-40"
            />
            <Select
              options={[
                { value: "", label: "All Actions" },
                { value: "create", label: "Create" },
                { value: "update", label: "Update" },
                { value: "delete", label: "Delete" },
                { value: "login", label: "Login" },
              ]}
              value={actionType}
              onChange={(e) => setActionType(e.target.value)}
              className="w-36"
            />
            <Select
              options={[
                { value: "", label: "All Resources" },
                { value: "dataset", label: "Dataset" },
                { value: "model", label: "Model" },
                { value: "experiment", label: "Experiment" },
              ]}
              value={resourceType}
              onChange={(e) => setResourceType(e.target.value)}
              className="w-36"
            />
            <Button size="sm" onClick={() => refetch()}>
              Apply
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                setStartDate("");
                setEndDate("");
                setUserId("");
                setActionType("");
                setResourceType("");
                refetch();
              }}
            >
              Reset
            </Button>
          </div>
        </TerminalCard>
        <AuditLogTable logs={logs} />
        {totalPages > 1 && (
          <div className="flex justify-center gap-2">
            <Button
              size="sm"
              variant="ghost"
              disabled={page === 1}
              onClick={() => setPage((p) => p - 1)}
            >
              Prev
            </Button>
            <span className="text-sm text-slate-400">
              Page {page} of {totalPages}
            </span>
            <Button
              size="sm"
              variant="ghost"
              disabled={page === totalPages}
              onClick={() => setPage((p) => p + 1)}
            >
              Next
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
