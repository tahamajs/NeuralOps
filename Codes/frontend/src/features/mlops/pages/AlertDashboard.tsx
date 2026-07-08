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
import { Bell, Plus, Settings, Trash2 } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function AlertDashboard({ onClose }: { onClose?: () => void }) {
  const [statusFilter, setStatusFilter] = useState("active");
  const [severityFilter, setSeverityFilter] = useState("");
  const [showChannelModal, setShowChannelModal] = useState(false);
  const [newChannelType, setNewChannelType] = useState<"email" | "slack" | "webhook">("slack");
  const [newChannelName, setNewChannelName] = useState("");
  const [newChannelConfig, setNewChannelConfig] = useState("");
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
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["alerts"] }),
  });
  const resolveAlert = useMutation({
    mutationFn: (id: string) => mlopsAPI.resolveAlert(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["alerts"] }),
  });
  const createChannel = useMutation({
    mutationFn: () =>
      mlopsAPI.createAlertChannel({
        type: newChannelType,
        name: newChannelName,
        config: JSON.parse(newChannelConfig),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["alert-channels"] });
      setShowChannelModal(false);
      success("Channel created");
    },
  });
  const deleteChannel = useMutation({
    mutationFn: (id: string) => mlopsAPI.deleteAlertChannel(id),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["alert-channels"] }),
  });

  if (alertsLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Bell size={20} className="text-yellow-400" /> Alert Dashboard
          </h1>
          <p className="text-xs text-slate-400">
            Real‑time alerts & notification channels
          </p>
        </div>
        <div className="flex gap-2">
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
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <div className="flex gap-2">
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
            ]}
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
            className="w-36"
          />
        </div>
        <div className="space-y-3">
          {alerts?.map((a) => (
            <AlertCard
              key={a.id}
              alert={a}
              onAcknowledge={() => acknowledgeAlert.mutate(a.id)}
              onResolve={() => resolveAlert.mutate(a.id)}
            />
          ))}
        </div>
        <TerminalCard className="p-4">
          <h3 className="text-sm font-bold text-white mb-2">Alert Channels</h3>
          <div className="space-y-2">
            {channels?.map((c) => (
              <div key={c.id} className="flex justify-between items-center">
                <span>
                  {c.name} ({c.type})
                </span>
                <div className="flex gap-2">
                  <Badge variant={c.enabled ? "success" : "secondary"}>
                    {c.enabled ? "Enabled" : "Disabled"}
                  </Badge>
                  <button onClick={() => deleteChannel.mutate(c.id)}>
                    <Trash2 size={14} className="text-red-400" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </TerminalCard>
      </div>
      <Modal
        open={showChannelModal}
        onClose={() => setShowChannelModal(false)}
        title="Add Alert Channel"
      >
        <div className="space-y-4">
          <Select
            label="Type"
            options={[
              { value: "slack", label: "Slack" },
              { value: "email", label: "Email" },
              { value: "webhook", label: "Webhook" },
            ]}
            value={newChannelType}
            onChange={(e) => setNewChannelType(e.target.value as "email" | "slack" | "webhook")}
          />
          <Input
            label="Name"
            value={newChannelName}
            onChange={(e) => setNewChannelName(e.target.value)}
          />
          <textarea
            rows={4}
            className="w-full bg-white/5 border border-white/10 rounded p-2 text-sm"
            placeholder='{"webhook_url": "https://..."}'
            value={newChannelConfig}
            onChange={(e) => setNewChannelConfig(e.target.value)}
          />
          <Button onClick={() => createChannel.mutate()}>Create</Button>
        </div>
      </Modal>
    </div>
  );
}
