// src/features/mlops/pages/DriftMonitorList.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { mlopsAPI } from "../services/api";
import { DriftChart } from "../components/DriftChart";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Select } from "../../../components/ui/Select";
import { Input } from "../../../components/ui/Input";
import { Modal } from "../../../components/ui/Modal";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { Activity, Plus, Eye, Trash2 } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function DriftMonitorList({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedMonitor, setSelectedMonitor] = useState<any>(null);
  const [newMonitorName, setNewMonitorName] = useState("");
  const [newFeatureName, setNewFeatureName] = useState("");
  const [newGroupId, setNewGroupId] = useState("");
  const [threshold, setThreshold] = useState(0.05);
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: monitors, isLoading } = useQuery({
    queryKey: ["drift-monitors"],
    queryFn: () => mlopsAPI.getDriftMonitors(),
  });
  const { data: groups } = useQuery({
    queryKey: ["feature-groups"],
    queryFn: () => mlopsAPI.getFeatureGroups(),
  });
  const { data: checks } = useQuery({
    queryKey: ["drift-checks", selectedMonitor?.id],
    queryFn: () => mlopsAPI.getDriftChecks(selectedMonitor?.id, 30),
    enabled: !!selectedMonitor?.id,
  });

  const createMonitor = useMutation({
    mutationFn: () =>
      mlopsAPI.createDriftMonitor({
        name: newMonitorName,
        feature_name: newFeatureName,
        feature_group_id: newGroupId,
        alert_threshold: threshold,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["drift-monitors"] });
      setShowCreateModal(false);
      success("Monitor created");
    },
  });
  const deleteMonitor = useMutation({
    mutationFn: (id: string) => mlopsAPI.deleteDriftMonitor(id),
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["drift-monitors"] }),
  });
  const runCheck = useMutation({
    mutationFn: (id: string) => mlopsAPI.runDriftCheck(id),
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: ["drift-checks", selectedMonitor?.id],
      }),
  });

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Activity size={20} className="text-yellow-400" /> Drift Monitors
          </h1>
          <p className="text-xs text-slate-400">
            Monitor feature & model drift
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" onClick={() => setShowCreateModal(true)}>
            <Plus size={14} className="mr-1" /> New Monitor
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 grid lg:grid-cols-2 gap-4">
        <div className="space-y-3">
          {monitors?.map((m) => (
            <TerminalCard
              key={m.id}
              className={`p-3 cursor-pointer ${selectedMonitor?.id === m.id ? "border-neon-blue" : ""}`}
              onClick={() => setSelectedMonitor(m)}
            >
              <div className="flex justify-between items-center">
                <div>
                  <h3 className="font-bold text-white">{m.name}</h3>
                  <div className="text-xs text-slate-400">
                    Feature: {m.feature_name}
                  </div>
                  <Badge
                    variant={
                      m.status === "normal"
                        ? "success"
                        : m.status === "warning"
                          ? "warning"
                          : "failed"
                    }
                    className="mt-1"
                  >
                    {m.status}
                  </Badge>
                </div>
                <div className="flex gap-1">
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={(e) => {
                      e.stopPropagation();
                      runCheck.mutate(m.id);
                    }}
                  >
                    Check
                  </Button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteMonitor.mutate(m.id);
                    }}
                  >
                    <Trash2 size={14} className="text-red-400" />
                  </button>
                </div>
              </div>
            </TerminalCard>
          ))}
        </div>
        <div className="space-y-4">
          {selectedMonitor && checks && (
            <TerminalCard className="p-4">
              <h3 className="text-sm font-bold text-white mb-2">
                Drift History: {selectedMonitor.name}
              </h3>
              <DriftChart
                checks={checks}
                threshold={selectedMonitor.alert_threshold}
              />
            </TerminalCard>
          )}
        </div>
      </div>
      <Modal
        open={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        title="Create Drift Monitor"
      >
        <div className="space-y-4">
          <Input
            label="Name"
            value={newMonitorName}
            onChange={(e) => setNewMonitorName(e.target.value)}
          />
          <Select
            label="Feature Group"
            options={groups?.map((g) => ({ value: g.id, label: g.name })) || []}
            value={newGroupId}
            onChange={(e) => setNewGroupId(e.target.value)}
          />
          <Input
            label="Feature Name"
            value={newFeatureName}
            onChange={(e) => setNewFeatureName(e.target.value)}
          />
          <Input
            label="Alert Threshold (p‑value)"
            type="number"
            step="0.01"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
          />
          <Button
            onClick={() => createMonitor.mutate()}
            disabled={!newMonitorName || !newGroupId || !newFeatureName}
          >
            Create
          </Button>
        </div>
      </Modal>
    </div>
  );
}
