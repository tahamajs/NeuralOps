import { useState } from "react";
import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { Modal } from "../../../components/ui/Modal";
import { Alert, AlertSeverity } from "../types";
import {
  CheckCircle2,
  Bell,
  AlertTriangle,
  Info,
  Flame,
  Clock,
  Eye,
  Check,
} from "lucide-react";

export function AlertTable({
  alerts,
  onAcknowledge,
  onResolve,
}: {
  alerts: Alert[];
  onAcknowledge: (id: string) => void;
  onResolve: (id: string) => void;
}) {
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);

  const getSeverityBadge = (severity: AlertSeverity) => {
    switch (severity) {
      case "critical":
      case "fatal":
        return <Badge variant="failed" className="text-[10px] uppercase font-bold py-0">{severity}</Badge>;
      case "warning":
        return <Badge variant="warning" className="text-[10px] uppercase font-bold py-0">{severity}</Badge>;
      case "info":
      default:
        return <Badge variant="info" className="text-[10px] uppercase font-bold py-0">{severity}</Badge>;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "resolved":
        return <Badge variant="success" className="text-[10px] py-0 font-bold uppercase">Resolved</Badge>;
      case "acknowledged":
        return <Badge variant="warning" className="text-[10px] py-0 font-bold uppercase">Acked</Badge>;
      case "active":
      default:
        return <Badge variant="failed" className="text-[10px] py-0 font-bold uppercase">Active</Badge>;
    }
  };

  return (
    <div className="space-y-2">
      <div className="rounded-lg border border-white/10 overflow-hidden bg-slate-900/60">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-900 text-slate-400 font-mono text-[11px] uppercase border-b border-white/10">
              <tr>
                <th className="p-3">Time</th>
                <th className="p-3">Monitor / Source</th>
                <th className="p-3">Alert Title & Reason</th>
                <th className="p-3">Severity</th>
                <th className="p-3">Status</th>
                <th className="p-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 font-mono text-slate-300">
              {alerts.map((a) => (
                <tr
                  key={a.id}
                  className="hover:bg-white/5 transition-colors cursor-pointer"
                  onClick={() => setSelectedAlert(a)}
                >
                  <td className="p-3 text-slate-400 whitespace-nowrap text-[11px]">
                    {new Date(a.created_at).toLocaleString([], {
                      month: "short",
                      day: "numeric",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </td>
                  <td className="p-3 whitespace-nowrap font-bold text-white text-xs">
                    {a.monitor_name || "System Detector"}
                  </td>
                  <td className="p-3 max-w-xs font-sans text-xs">
                    <div className="text-white font-medium truncate">{a.title}</div>
                    <div className="text-[11px] text-slate-400 truncate">{a.message}</div>
                  </td>
                  <td className="p-3 whitespace-nowrap">
                    {getSeverityBadge(a.severity)}
                  </td>
                  <td className="p-3 whitespace-nowrap">
                    {getStatusBadge(a.status)}
                  </td>
                  <td className="p-3 text-right whitespace-nowrap" onClick={(e) => e.stopPropagation()}>
                    <div className="flex gap-1.5 justify-end items-center">
                      {a.status === "active" && (
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => onAcknowledge(a.id)}
                          className="h-7 text-xs px-2 hover:text-yellow-400"
                        >
                          <Bell size={11} className="mr-1" /> Ack
                        </Button>
                      )}
                      {a.status !== "resolved" && (
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => onResolve(a.id)}
                          className="h-7 text-xs px-2 hover:text-green-400"
                        >
                          <CheckCircle2 size={11} className="mr-1" /> Resolve
                        </Button>
                      )}
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => setSelectedAlert(a)}
                        className="h-7 text-xs px-2"
                        title="View Details"
                      >
                        <Eye size={12} />
                      </Button>
                    </div>
                  </td>
                </tr>
              ))}

              {alerts.length === 0 && (
                <tr>
                  <td colSpan={6} className="text-center text-slate-500 py-12 font-sans">
                    <CheckCircle2 size={32} className="mx-auto mb-2 opacity-40 text-green-400" />
                    No health alerts found matching criteria.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Alert Details Modal */}
      {selectedAlert && (
        <Modal
          open={!!selectedAlert}
          onClose={() => setSelectedAlert(null)}
          title={`Alert Details: ${selectedAlert.id}`}
        >
          <div className="space-y-4 text-xs">
            <div className="flex justify-between items-center pb-2 border-b border-white/10">
              <div className="space-y-0.5">
                <h3 className="text-sm font-bold text-white">{selectedAlert.title}</h3>
                <p className="text-[11px] text-slate-400">Monitor: {selectedAlert.monitor_name}</p>
              </div>
              <div className="flex gap-1.5">
                {getSeverityBadge(selectedAlert.severity)}
                {getStatusBadge(selectedAlert.status)}
              </div>
            </div>

            <div className="p-3 bg-slate-900 rounded border border-white/10 text-slate-300 leading-relaxed font-sans">
              {selectedAlert.message}
            </div>

            {selectedAlert.metadata && (
              <div className="space-y-1.5">
                <label className="font-semibold text-slate-300 uppercase tracking-wide text-[10px]">
                  Observed Diagnostic Metadata
                </label>
                <div className="p-2.5 bg-black/40 rounded border border-white/5 font-mono text-[11px] text-neon-mint space-y-1">
                  {Object.entries(selectedAlert.metadata).map(([k, v]) => (
                    <div key={k} className="flex justify-between">
                      <span className="text-slate-400">{k}:</span>
                      <strong className="text-white">{String(v)}</strong>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="grid grid-cols-2 gap-2 text-[11px] font-mono text-slate-400 pt-2 border-t border-white/10">
              <div>
                Triggered: <span className="text-white">{new Date(selectedAlert.created_at).toLocaleString()}</span>
              </div>
              {selectedAlert.acknowledged_by && (
                <div>
                  Acked by: <span className="text-yellow-400">{selectedAlert.acknowledged_by}</span>
                </div>
              )}
              {selectedAlert.resolved_by && (
                <div>
                  Resolved by: <span className="text-green-400">{selectedAlert.resolved_by}</span>
                </div>
              )}
            </div>

            <div className="flex justify-end gap-2 pt-2 border-t border-white/10">
              {selectedAlert.status === "active" && (
                <Button
                  size="sm"
                  onClick={() => {
                    onAcknowledge(selectedAlert.id);
                    setSelectedAlert(null);
                  }}
                  className="bg-yellow-500 hover:bg-yellow-600 text-slate-950 font-bold"
                >
                  <Bell size={12} className="mr-1" /> Acknowledge Alert
                </Button>
              )}
              {selectedAlert.status !== "resolved" && (
                <Button
                  size="sm"
                  onClick={() => {
                    onResolve(selectedAlert.id);
                    setSelectedAlert(null);
                  }}
                  className="bg-green-500 hover:bg-green-600 text-slate-950 font-bold"
                >
                  <CheckCircle2 size={12} className="mr-1" /> Mark as Resolved
                </Button>
              )}
              <Button variant="ghost" size="sm" onClick={() => setSelectedAlert(null)}>
                Close
              </Button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
}

