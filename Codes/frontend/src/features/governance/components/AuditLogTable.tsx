import { useState } from "react";
import { DataGrid } from "../../../components/ui/DataGrid";
import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { Modal } from "../../../components/ui/Modal";
import { AuditLogEntry } from "../types";
import { Eye, Shield, Terminal, User, Globe } from "lucide-react";

export function AuditLogTable({ logs }: { logs: AuditLogEntry[] }) {
  const [inspectLog, setInspectLog] = useState<AuditLogEntry | null>(null);

  const getActionVariant = (action: string) => {
    switch (action) {
      case "create":
        return "success";
      case "update":
        return "info";
      case "delete":
        return "failed";
      case "export":
        return "warning";
      case "login":
      case "logout":
        return "secondary";
      default:
        return "info";
    }
  };

  const columns = [
    {
      accessorKey: "timestamp",
      header: "Timestamp",
      cell: ({ row }: any) => (
        <span className="font-mono text-xs text-slate-300">
          {new Date(row.original.timestamp).toLocaleString([], {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
          })}
        </span>
      ),
    },
    {
      accessorKey: "username",
      header: "Actor",
      cell: ({ row }: any) => (
        <div className="flex items-center gap-1.5 font-mono text-xs text-white">
          <User size={12} className="text-neon-mint" />
          <span>{row.original.username}</span>
        </div>
      ),
    },
    {
      accessorKey: "action",
      header: "Action",
      cell: ({ row }: any) => (
        <Badge variant={getActionVariant(row.original.action) as any} className="text-[10px] uppercase font-bold">
          {row.original.action}
        </Badge>
      ),
    },
    {
      accessorKey: "resource_type",
      header: "Resource Type",
      cell: ({ row }: any) => (
        <span className="font-mono text-xs text-neon-blue uppercase">
          {row.original.resource_type}
        </span>
      ),
    },
    {
      accessorKey: "resource_name",
      header: "Resource Name",
      cell: ({ row }: any) => (
        <span className="text-xs text-slate-200 font-medium truncate max-w-xs block">
          {row.original.resource_name}
        </span>
      ),
    },
    {
      accessorKey: "ip_address",
      header: "IP Address",
      cell: ({ row }: any) => (
        <span className="font-mono text-xs text-slate-400">
          {row.original.ip_address}
        </span>
      ),
    },
    {
      accessorKey: "severity",
      header: "Severity",
      cell: ({ row }: any) => (
        <Badge
          variant={
            row.original.severity === "critical"
              ? "failed"
              : row.original.severity === "warning"
                ? "warning"
                : "info"
          }
          className="text-[10px] uppercase"
        >
          {row.original.severity}
        </Badge>
      ),
    },
    {
      id: "inspect",
      header: "",
      cell: ({ row }: any) => (
        <Button
          size="sm"
          variant="ghost"
          onClick={() => setInspectLog(row.original)}
          className="text-xs h-6 px-2 text-neon-mint"
        >
          <Eye size={12} className="mr-1" /> Inspect
        </Button>
      ),
    },
  ];

  return (
    <>
      <DataGrid data={logs} columns={columns} />

      {/* Inspect Event Payload Modal */}
      <Modal
        open={!!inspectLog}
        onClose={() => setInspectLog(null)}
        title={`Audit Event: ${inspectLog?.id || ""}`}
      >
        {inspectLog && (
          <div className="space-y-3.5 text-xs">
            <div className="grid grid-cols-2 gap-3">
              <div className="p-2.5 bg-white/5 rounded border border-white/10 space-y-1">
                <span className="text-slate-400 text-[10px] uppercase">Actor & IP</span>
                <div className="font-bold text-white font-mono">{inspectLog.username}</div>
                <div className="text-slate-400 font-mono text-[11px]">{inspectLog.ip_address}</div>
              </div>
              <div className="p-2.5 bg-white/5 rounded border border-white/10 space-y-1">
                <span className="text-slate-400 text-[10px] uppercase">Target Resource</span>
                <div className="font-bold text-white font-mono">{inspectLog.resource_name}</div>
                <div className="text-neon-blue font-mono text-[11px]">ID: {inspectLog.resource_id}</div>
              </div>
            </div>

            <div className="p-2.5 bg-white/5 rounded border border-white/10 space-y-1">
              <span className="text-slate-400 text-[10px] uppercase">User Agent / Client Fingerprint</span>
              <div className="font-mono text-slate-300 text-[11px] break-all">{inspectLog.user_agent}</div>
            </div>

            <div className="space-y-1">
              <span className="font-semibold text-slate-300 flex items-center gap-1">
                <Terminal size={12} className="text-neon-mint" /> Event Payload Details (JSON)
              </span>
              <pre className="p-3 bg-slate-950 rounded border border-white/10 text-neon-mint font-mono text-[11px] max-h-56 overflow-auto">
                {JSON.stringify(inspectLog.details, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </Modal>
    </>
  );
}

