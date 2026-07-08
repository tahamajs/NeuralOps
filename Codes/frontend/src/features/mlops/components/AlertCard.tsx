import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { AlertCircle, CheckCircle, Bell } from "lucide-react";
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
  const severityColor = {
    info: "blue-400",
    warning: "yellow-400",
    critical: "red-400",
  }[alert.severity];
  const statusConfig = {
    active: { label: "Active", color: "danger" },
    acknowledged: { label: "Acknowledged", color: "warning" },
    resolved: { label: "Resolved", color: "success" },
  }[alert.status];
  return (
    <TerminalCard className="p-3">
      <div className="flex justify-between">
        <div>
          <div className="flex items-center gap-2">
            <AlertCircle size={16} className={`text-${severityColor}`} />
            <span className="font-bold text-white">{alert.message}</span>
            <Badge variant={statusConfig.color as any}>
              {statusConfig.label}
            </Badge>
            <Badge variant="secondary">{alert.severity}</Badge>
          </div>
          <div className="text-xs text-slate-400 mt-1">
            {formatDistanceToNow(new Date(alert.created_at), {
              addSuffix: true,
            })}
          </div>
        </div>
        <div className="flex gap-1">
          {alert.status === "active" && (
            <Button size="sm" variant="ghost" onClick={onAcknowledge}>
              <Bell size={12} /> Acknowledge
            </Button>
          )}
          {alert.status !== "resolved" && (
            <Button size="sm" variant="ghost" onClick={onResolve}>
              <CheckCircle size={12} /> Resolve
            </Button>
          )}
        </div>
      </div>
    </TerminalCard>
  );
}
