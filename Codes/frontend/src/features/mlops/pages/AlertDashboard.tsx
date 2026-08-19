// src/features/mlops/pages/AlertDashboard.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { mlopsAPI } from "../services/api";
import { AlertCard } from "../components/AlertCard";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { Modal } from "../../../components/ui/Modal";
import { Input } from "../../../components/ui/Input";
import {
  Bell,
  Plus,
  Trash2,
  Download,
  Search,
  CheckCircle2,
  AlertTriangle,
  Send,
  Radio,
  Mail,
  Zap,
} from "lucide-react";
import { useToast } from "../../../components/ui/Toast";
import { AlertChannel } from "../types";

export default function AlertDashboard({ onClose }: { onClose?: () => void }) {
  const [statusFilter, setStatusFilter] = useState("all");
  const [severityFilter, setSeverityFilter] = useState("");
  const [search, setSearch] = useState("");
  const [showChannelModal, setShowChannelModal] = useState(false);
  const [newChannelType, setNewChannelType] = useState<
    "slack" | "email" | "webhook" | "pagerduty"
  >("slack");
  const [newChannelName, setNewChannelName] = useState("");
  const [newChannelConfig, setNewChannelConfig] = useState(
    JSON.stringify(
      { webhook_url: "https://hooks.slack.com/services/T00/B00/XXXX", channel: "#mlops-alerts" },
      null,
      2,
    ),
  );

  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: alerts, isLoading: alertsLoading } = useQuery({
    queryKey: ["alerts", statusFilter, severityFilter],
    queryFn: () =>
      mlopsAPI.getAlerts({
        status: statusFilter !== "all" ? statusFilter : undefined,
        severity: severityFilter || undefined,
      }),
  });

  const { data: channels } = useQuery({
    queryKey: ["alert-channels"],
    queryFn: () => mlopsAPI.getAlertChannels(),
  });

  const acknowledgeAlert = useMutation({
    mutationFn: (id: string) => mlopsAPI.acknowledgeAlert(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["alerts"] });
      success("Alert acknowledged");
    },
  });

  const resolveAlert = useMutation({
    mutationFn: (id: string) => mlopsAPI.resolveAlert(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["alerts"] });
      success("Alert marked as resolved");
    },
  });

  const acknowledgeAllActive = useMutation({
    mutationFn: async () => {
      const activeAlerts = alerts?.filter((a) => a.status === "active") || [];
      await Promise.all(activeAlerts.map((a) => mlopsAPI.acknowledgeAlert(a.id)));
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["alerts"] });
      success("All active alerts acknowledged");
    },
  });

  const createChannel = useMutation({
    mutationFn: () => {
      let parsedConfig = {};
      try {
        parsedConfig = JSON.parse(newChannelConfig);
      } catch {
        parsedConfig = { raw: newChannelConfig };
      }
      return mlopsAPI.createAlertChannel({
        type: newChannelType,
        name: newChannelName,
        config: parsedConfig,
        enabled: true,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["alert-channels"] });
      setShowChannelModal(false);
      setNewChannelName("");
      success("Alert channel created");
    },
    onError: (err: any) => {
      error("Failed to create channel", err?.message || "Invalid configuration");
    },
  });

  const deleteChannel = useMutation({
    mutationFn: (id: string) => mlopsAPI.deleteAlertChannel(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["alert-channels"] });
      success("Channel removed");
    },
  });

  const testChannel = useMutation({
    mutationFn: (id: string) => mlopsAPI.testAlertChannel(id),
    onSuccess: (res) => {
      success("Test ping sent", res.message);
    },
  });

  const handleExportAlerts = () => {
    if (!alerts || alerts.length === 0) return;
    const csvHeader = "ID,Severity,Status,Message,Created_At,Acknowledged_At,Resolved_At\n";
    const csvRows = alerts
      .map(
        (a) =>
          `"${a.id}","${a.severity}","${a.status}","${a.message.replace(/"/g, '""')}","${a.created_at}","${a.acknowledged_at || ""}","${a.resolved_at || ""}"`,
      )
      .join("\n");
    const blob = new Blob([csvHeader + csvRows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `alerts_export_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    success("Export complete", "Alerts history exported to CSV");
  };

  const handleChannelTypeChange = (type: "slack" | "email" | "webhook" | "pagerduty") => {
    setNewChannelType(type);
    if (type === "slack") {
      setNewChannelConfig(
        JSON.stringify(
          { webhook_url: "https://hooks.slack.com/services/T00/B00/XXXX", channel: "#mlops-alerts" },
          null,
          2,
        ),
      );
    } else if (type === "email") {
      setNewChannelConfig(
        JSON.stringify({ recipients: ["oncall@example.com", "lead@example.com"] }, null, 2),
      );
    } else if (type === "pagerduty") {
      setNewChannelConfig(
        JSON.stringify({ routing_key: "pd-service-key-1234", severity_override: "critical" }, null, 2),
      );
    } else {
      setNewChannelConfig(
        JSON.stringify({ url: "https://api.example.com/webhook", method: "POST" }, null, 2),
      );
    }
  };

  const filteredAlerts =
    alerts?.filter((a) => {
      const matchSearch =
        a.message.toLowerCase().includes(search.toLowerCase()) ||
        a.severity.toLowerCase().includes(search.toLowerCase());
      return matchSearch;
    }) || [];

  const totalAlerts = alerts?.length || 0;
  const activeCount = alerts?.filter((a) => a.status === "active").length || 0;
  const criticalCount = alerts?.filter((a) => a.severity === "critical").length || 0;
  const resolvedCount = alerts?.filter((a) => a.status === "resolved").length || 0;

  if (alertsLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full bg-slate-950">
      {/* Header */}
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Bell size={20} className="text-yellow-400" /> Alert Dashboard
          </h1>
          <p className="text-xs text-slate-400">
            Real‑time monitoring breaches, notification dispatchers & incident management
          </p>
        </div>
        <div className="flex gap-2">
          {activeCount > 0 && (
            <Button
              size="sm"
              variant="ghost"
              onClick={() => acknowledgeAllActive.mutate()}
              disabled={acknowledgeAllActive.isPending}
            >
              <CheckCircle2 size={14} className="mr-1 text-neon-mint" /> Ack All ({activeCount})
            </Button>
          )}
          <Button
            size="sm"
            variant="ghost"
            onClick={handleExportAlerts}
            title="Export CSV"
          >
            <Download size={14} className="mr-1" /> Export CSV
          </Button>
          <Button size="sm" onClick={() => setShowChannelModal(true)}>
            <Plus size={14} className="mr-1" /> Add Channel
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>

      {/* Main Container */}
      <div className="flex-1 overflow-auto p-4 space-y-4">
        {/* KPI Stat Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <TerminalCard className="p-3 text-center">
            <div className="text-xs text-slate-400">Total Alerts</div>
            <div className="text-2xl font-bold text-white mt-0.5">{totalAlerts}</div>
          </TerminalCard>
          <TerminalCard className="p-3 text-center">
            <div className="text-xs text-slate-400">Active Issues</div>
            <div className="text-2xl font-bold text-red-400 mt-0.5">{activeCount}</div>
          </TerminalCard>
          <TerminalCard className="p-3 text-center">
            <div className="text-xs text-slate-400">Critical Breaches</div>
            <div className="text-2xl font-bold text-yellow-400 mt-0.5">{criticalCount}</div>
          </TerminalCard>
          <TerminalCard className="p-3 text-center">
            <div className="text-xs text-slate-400">Resolved Rate</div>
            <div className="text-2xl font-bold text-neon-mint mt-0.5">
              {totalAlerts > 0 ? `${Math.round((resolvedCount / totalAlerts) * 100)}%` : "100%"}
            </div>
          </TerminalCard>
        </div>

        {/* Filter Toolbar */}
        <div className="flex flex-wrap gap-2 items-center justify-between">
          <div className="flex gap-2 flex-wrap items-center">
            <div className="relative w-56">
              <Input
                placeholder="Search alerts..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full pl-8 text-xs"
              />
              <Search
                size={13}
                className="absolute left-2.5 top-3 text-slate-500 pointer-events-none"
              />
            </div>
            <Select
              options={[
                { value: "all", label: "All Status" },
                { value: "active", label: "Active" },
                { value: "acknowledged", label: "Acknowledged" },
                { value: "resolved", label: "Resolved" },
              ]}
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="w-32 text-xs"
            />
            <Select
              options={[
                { value: "", label: "All Severities" },
                { value: "info", label: "Info" },
                { value: "warning", label: "Warning" },
                { value: "critical", label: "Critical" },
              ]}
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value)}
              className="w-36 text-xs"
            />
          </div>

          <div className="text-xs text-slate-400">
            Showing <strong>{filteredAlerts.length}</strong> alerts
          </div>
        </div>

        {/* Alerts List */}
        <div className="space-y-2.5">
          {filteredAlerts.map((a) => (
            <AlertCard
              key={a.id}
              alert={a}
              onAcknowledge={() => acknowledgeAlert.mutate(a.id)}
              onResolve={() => resolveAlert.mutate(a.id)}
            />
          ))}

          {filteredAlerts.length === 0 && (
            <div className="text-center text-slate-500 py-12">
              <CheckCircle2 size={32} className="mx-auto mb-2 opacity-40 text-green-400" />
              <p>No alerts matching current filters.</p>
            </div>
          )}
        </div>

        {/* Channels Section */}
        <TerminalCard className="p-4 mt-6">
          <div className="flex justify-between items-center mb-3">
            <div>
              <h3 className="text-sm font-bold text-white flex items-center gap-1.5">
                <Radio size={14} className="text-neon-mint" /> Notification & Pager Channels
              </h3>
              <p className="text-xs text-slate-400">
                Connected destinations for automated incident escalations
              </p>
            </div>
            <Button size="sm" variant="ghost" onClick={() => setShowChannelModal(true)}>
              <Plus size={12} className="mr-1" /> Add Channel
            </Button>
          </div>

          <div className="grid md:grid-cols-2 gap-3">
            {channels?.map((c) => (
              <div
                key={c.id}
                className="p-3 bg-white/5 rounded border border-white/10 flex justify-between items-center"
              >
                <div className="space-y-0.5">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-white text-xs">{c.name}</span>
                    <Badge variant={c.enabled ? "success" : "secondary"} className="text-[10px] py-0">
                      {c.type.toUpperCase()}
                    </Badge>
                  </div>
                  <div className="text-[11px] text-slate-400 font-mono truncate max-w-xs">
                    {c.type === "slack"
                      ? c.config.channel || "Webhook endpoint"
                      : c.type === "email"
                        ? Array.isArray(c.config.recipients)
                          ? c.config.recipients.join(", ")
                          : String(c.config.recipients || "Email list")
                        : c.type === "pagerduty"
                          ? `Routing: ${c.config.routing_key || "Primary"}`
                          : String(c.config.url || "Webhook")}
                  </div>
                </div>

                <div className="flex items-center gap-1">
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => testChannel.mutate(c.id)}
                    disabled={testChannel.isPending}
                    title="Send test message"
                    className="text-xs h-7 px-2 hover:text-neon-blue"
                  >
                    <Send size={11} className="mr-1" /> Test
                  </Button>
                  <button
                    onClick={() => deleteChannel.mutate(c.id)}
                    className="p-1 text-slate-500 hover:text-red-400 transition-colors"
                    title="Delete Channel"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </TerminalCard>
      </div>

      {/* Add Alert Channel Modal */}
      <Modal
        open={showChannelModal}
        onClose={() => setShowChannelModal(false)}
        title="Add Notification Channel"
      >
        <div className="space-y-4">
          <div className="grid grid-cols-4 gap-2">
            {(["slack", "pagerduty", "email", "webhook"] as const).map((type) => (
              <button
                key={type}
                type="button"
                onClick={() => handleChannelTypeChange(type)}
                className={`p-2 rounded text-xs text-center border font-bold capitalize transition-colors ${
                  newChannelType === type
                    ? "border-neon-mint bg-neon-mint/10 text-white"
                    : "border-white/10 bg-white/5 text-slate-400 hover:text-white"
                }`}
              >
                {type}
              </button>
            ))}
          </div>

          <Input
            label="Channel Display Name"
            value={newChannelName}
            onChange={(e) => setNewChannelName(e.target.value)}
            placeholder="e.g. #mlops-oncall-alerts"
          />

          <div className="space-y-1">
            <label className="text-xs font-mono text-slate-300">
              Channel Configuration (JSON)
            </label>
            <textarea
              rows={6}
              className="w-full bg-slate-900 border border-white/10 rounded p-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-neon-mint"
              value={newChannelConfig}
              onChange={(e) => setNewChannelConfig(e.target.value)}
            />
          </div>

          <div className="flex justify-end gap-2 pt-3 border-t border-white/10">
            <Button variant="ghost" onClick={() => setShowChannelModal(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => createChannel.mutate()}
              disabled={!newChannelName.trim() || createChannel.isPending}
            >
              {createChannel.isPending ? "Creating..." : "Save Channel"}
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}

