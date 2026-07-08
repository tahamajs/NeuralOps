// src/features/healthMonitor/pages/DriftMonitorSetup.tsx
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { healthMonitorAPI } from "../services/api";
import { DriftChart } from "../components/DriftChart";
import { TerminalCard } from "../../../components/ui/TerminalCard";
import { Button } from "../../../components/ui/Button";
import { Input } from "../../../components/ui/Input";
import { Select } from "../../../components/ui/Select";
import { Spinner } from "../../../components/ui/Spinner";
import { Badge } from "../../../components/ui/Badge";
import { AlertTriangle, Plus, Trash2, RefreshCw, Play } from "lucide-react";
import { useToast } from "../../../components/ui/Toast";

export default function DriftMonitorSetup({
  onClose,
}: {
  onClose?: () => void;
}) {
  const [selectedMonitorId, setSelectedMonitorId] = useState<string | null>(
    null,
  );
  const [formData, setFormData] = useState({
    name: "",
    model_id: "",
    feature_name: "",
    method: "psi",
    alert_threshold: 0.1,
    enabled: true,
  });
  const [isCreating, setIsCreating] = useState(false);
  const queryClient = useQueryClient();
  const { success, error } = useToast();

  const { data: monitors, isLoading } = useQuery({
    queryKey: ["drift-monitors"],
    queryFn: () => healthMonitorAPI.getDriftMonitors(),
  });
  const { data: checks } = useQuery({
    queryKey: ["drift-checks", selectedMonitorId],
    queryFn: () => healthMonitorAPI.getDriftChecks(selectedMonitorId!, 30),
    enabled: !!selectedMonitorId,
  });

  const createMonitor = useMutation({
    mutationFn: () => healthMonitorAPI.createDriftMonitor(formData as any),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["drift-monitors"] });
      setIsCreating(false);
      setFormData({
        name: "",
        model_id: "",
        feature_name: "",
        method: "psi",
        alert_threshold: 0.1,
        enabled: true,
      });
      success("Monitor created");
    },
  });
  const updateMonitor = useMutation({
    mutationFn: (data: any) =>
      healthMonitorAPI.updateDriftMonitor(selectedMonitorId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["drift-monitors"] });
      success("Monitor updated");
    },
  });
  const deleteMonitor = useMutation({
    mutationFn: (id: string) => healthMonitorAPI.deleteDriftMonitor(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["drift-monitors"] });
      setSelectedMonitorId(null);
      success("Monitor deleted");
    },
  });
  const runCheck = useMutation({
    mutationFn: (id: string) => healthMonitorAPI.runDriftCheck(id),
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: ["drift-checks", selectedMonitorId],
      }),
  });

  const selectedMonitor = monitors?.find((m) => m.id === selectedMonitorId);

  if (isLoading) return <Spinner />;

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white">Drift Monitor Setup</h1>
          <p className="text-xs text-slate-400">
            Configure proactive drift detection for model features
          </p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" onClick={() => setIsCreating(true)}>
            <Plus size={14} className="mr-1" /> New Monitor
          </Button>
          {onClose && (
            <Button size="sm" variant="ghost" onClick={onClose}>
              ✕
            </Button>
          )}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4">
        <div className="grid lg:grid-cols-3 gap-4">
          <div className="space-y-2">
            {monitors?.map((m) => (
              <div
                key={m.id}
                className={`p-3 rounded-lg cursor-pointer border ${selectedMonitorId === m.id ? "border-neon-blue bg-neon-blue/10" : "border-white/10 hover:bg-white/5"}`}
                onClick={() => setSelectedMonitorId(m.id)}
              >
                <div className="flex justify-between">
                  <span className="font-bold text-white">{m.name}</span>
                  <Badge
                    variant={
                      m.status === "normal"
                        ? "success"
                        : m.status === "warning"
                          ? "warning"
                          : "failed"
                    }
                  >
                    {m.status}
                  </Badge>
                </div>
                <div className="text-xs text-slate-400">
                  Feature: {m.feature_name}
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  Threshold: {m.alert_threshold}
                </div>
              </div>
            ))}
          </div>
          <div className="lg:col-span-2 space-y-4">
            {selectedMonitor ? (
              <>
                <div className="flex justify-between items-center">
                  <div>
                    <h3 className="text-lg font-bold text-white">
                      {selectedMonitor.name}
                    </h3>
                    <div className="text-sm text-slate-400">
                      Feature: {selectedMonitor.feature_name} · Model:{" "}
                      {selectedMonitor.model_name}
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => runCheck.mutate(selectedMonitor.id)}
                    >
                      <Play size={14} />
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => deleteMonitor.mutate(selectedMonitor.id)}
                    >
                      <Trash2 size={14} className="text-red-400" />
                    </Button>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <Input
                    label="Name"
                    value={selectedMonitor.name}
                    onChange={(e) =>
                      updateMonitor.mutate({
                        ...selectedMonitor,
                        name: e.target.value,
                      })
                    }
                  />
                  <Input
                    label="Alert Threshold"
                    type="number"
                    step="0.01"
                    value={selectedMonitor.alert_threshold}
                    onChange={(e) =>
                      updateMonitor.mutate({
                        ...selectedMonitor,
                        alert_threshold: parseFloat(e.target.value),
                      })
                    }
                  />
                </div>
                <div>
                  <label className="text-xs text-slate-400">
                    Drift History
                  </label>
                  <DriftChart
                    checks={checks || []}
                    threshold={selectedMonitor.alert_threshold}
                  />
                </div>
              </>
            ) : isCreating ? (
              <TerminalCard className="p-4 space-y-4">
                <Input
                  label="Name"
                  value={formData.name}
                  onChange={(e) =>
                    setFormData({ ...formData, name: e.target.value })
                  }
                />
                <Input
                  label="Model ID"
                  value={formData.model_id}
                  onChange={(e) =>
                    setFormData({ ...formData, model_id: e.target.value })
                  }
                />
                <Input
                  label="Feature Name"
                  value={formData.feature_name}
                  onChange={(e) =>
                    setFormData({ ...formData, feature_name: e.target.value })
                  }
                />
                <Select
                  label="Detection Method"
                  options={[
                    { value: "psi", label: "Population Stability Index (PSI)" },
                    { value: "ks", label: "Kolmogorov‑Smirnov" },
                    { value: "emd", label: "Earth Mover Distance" },
                  ]}
                  value={formData.method}
                  onChange={(e) =>
                    setFormData({ ...formData, method: e.target.value })
                  }
                />
                <Input
                  label="Alert Threshold"
                  type="number"
                  step="0.01"
                  value={formData.alert_threshold}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      alert_threshold: parseFloat(e.target.value),
                    })
                  }
                />
                <Button onClick={() => createMonitor.mutate()}>Create</Button>
                <Button variant="ghost" onClick={() => setIsCreating(false)}>
                  Cancel
                </Button>
              </TerminalCard>
            ) : (
              <div className="text-center text-slate-500 py-12">
                Select a monitor or create a new one
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
