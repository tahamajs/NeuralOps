import { useState } from "react";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import {
  AlertCircle,
  AlertTriangle,
  Info,
  CheckCircle,
  Bell,
  ChevronDown,
  ChevronUp,
  Clock,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { Alert } from "../types";

export function AlertCard({
  alert,
  onAcknowledge,
  onResolve,
}: {
  alert: Alert;
  onAcknowledge: () => void;
  onResolve: () => void;
}) {
  const [expanded, setExpanded] = useState(false);

  const severityConfig = {
    info: {
      color: "text-blue-400",
      bg: "bg-blue-500/10",
      border: "border-blue-500/30",
      icon: Info,
      badge: "info" as const,
    },
    warning: {
      color: "text-yellow-400",
      bg: "bg-yellow-500/10",
      border: "border-yellow-500/30",
      icon: AlertTriangle,
      badge: "warning" as const,
    },
    critical: {
      color: "text-red-400",
      bg: "bg-red-500/10",
      border: "border-red-500/30",
      icon: AlertCircle,
      badge: "danger" as const,
    },
  }[alert.severity] || {
    color: "text-slate-400",
    bg: "bg-slate-500/10",
    border: "border-slate-500/30",
    icon: Info,
    badge: "secondary" as const,
  };

  const statusConfig = {
    active: { label: "Active", variant: "danger" as const },
    acknowledged: { label: "Acknowledged", variant: "warning" as const },
    resolved: { label: "Resolved", variant: "success" as const },
  }[alert.status] || { label: alert.status, variant: "secondary" as const };

  const IconComponent = severityConfig.icon;

  return (
    <TerminalCard
      className={`p-3.5 border transition-all ${severityConfig.border} hover:bg-white/5`}
    >
      <div className="flex justify-between items-start gap-3">
        <div className="space-y-1.5 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <IconComponent size={16} className={severityConfig.color} />
            <span className="font-bold text-white text-sm">
              {alert.message}
            </span>
            <Badge variant={statusConfig.variant} className="text-[10px] py-0">
              {statusConfig.label}
            </Badge>
            <Badge variant={severityConfig.badge} className="text-[10px] py-0 uppercase">
              {alert.severity}
            </Badge>
          </div>

          <div className="flex items-center gap-3 text-xs text-slate-400">
            <span className="flex items-center gap-1">
              <Clock size={11} />
              {formatDistanceToNow(new Date(alert.created_at), {
                addSuffix: true,
              })}
            </span>
            {alert.acknowledged_at && (
              <span className="text-yellow-400/80">
                Acked {formatDistanceToNow(new Date(alert.acknowledged_at), { addSuffix: true })}
              </span>
            )}
            {alert.resolved_at && (
              <span className="text-green-400/80">
                Resolved {formatDistanceToNow(new Date(alert.resolved_at), { addSuffix: true })}
              </span>
            )}
          </div>
        </div>

        <div className="flex items-center gap-1">
          {alert.status === "active" && (
            <Button
              size="sm"
              variant="ghost"
              onClick={onAcknowledge}
              className="text-xs h-7 px-2 hover:text-yellow-400"
            >
              <Bell size={12} className="mr-1" /> Ack
            </Button>
          )}
          {alert.status !== "resolved" && (
            <Button
              size="sm"
              variant="ghost"
              onClick={onResolve}
              className="text-xs h-7 px-2 hover:text-green-400"
            >
              <CheckCircle size={12} className="mr-1" /> Resolve
            </Button>
          )}
          {alert.metadata && Object.keys(alert.metadata).length > 0 && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="p-1 text-slate-400 hover:text-white"
              title="Toggle Alert Details"
            >
              {expanded ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
            </button>
          )}
        </div>
      </div>

      {expanded && alert.metadata && (
        <div className="mt-3 pt-2.5 border-t border-white/10 text-xs">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {Object.entries(alert.metadata).map(([key, val]) => (
              <div key={key} className="p-2 bg-slate-900/80 rounded border border-white/5 font-mono">
                <span className="text-slate-500 text-[10px] block uppercase">
                  {key.replace(/_/g, " ")}
                </span>
                <span className="text-slate-200 text-xs truncate block font-bold">
                  {typeof val === "object" ? JSON.stringify(val) : String(val)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </TerminalCard>
  );
}

